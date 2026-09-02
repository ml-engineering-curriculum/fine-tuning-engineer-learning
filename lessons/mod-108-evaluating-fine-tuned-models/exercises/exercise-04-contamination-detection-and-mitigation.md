# exercise-04: Contamination Detection and Mitigation

**Estimated effort:** 4 hours
**Companion chapters:** [`05-contamination-detection.md`](../05-contamination-detection.md), [`03-task-specific-held-out-evals.md`](../03-task-specific-held-out-evals.md), [`06-release-gate-eval-suite.md`](../06-release-gate-eval-suite.md)

## Objective

Run all three contamination lenses from chapter 5 — **text-overlap** (n-gram), **semantic-overlap** (embedding cosine), and at least one **model-signal** probe (canonical-continuation) — against a real SFT training set and a real eval suite. Hand-sample the false-positive rate, triage the confirmed hits against chapter 5's severity rubric, apply the appropriate fix (retrain / disclose / accept), and write the model-card disclosure section a reader could audit.

The point is calibrated skepticism: proving that your reported eval numbers are not inflated, disclosing what you cannot fix, and turning a scan into a decision.

## Prerequisites

- An SFT training set. Options:
  - Your training set from mod-102 / mod-103 (preferred — you already know the sources).
  - A public SFT dataset (`HuggingFaceH4/ultrachat_200k`, `Open-Orca/SlimOrca`, `teknium/OpenHermes-2.5`, or a comparable size).
- An eval suite. Options:
  - The `holdout_gate.jsonl` from exercise 02 (your task-specific eval).
  - Public academic evals downloaded through `lm-evaluation-harness` (MMLU test split, GSM8K test split, HumanEval, IFEval — you will need the raw items, not just the harness's task definitions).
  - Pick at least two evals; contamination looks different across them.
- A fine-tuned model whose SFT set you scan. If you do not have one, use a public fine-tune whose training data is public (e.g., an OpenHermes-derived model against the OpenHermes set) so the scan is meaningful.
- 1 GPU for the embedding pass and the model-signal probe (any card that fits your model in BF16).
- Python: `sentence-transformers`, `scikit-learn` or `faiss-cpu`, `datasets`, `transformers`, `numpy`, `pandas`.

## Requirements

### Part A — Assemble the corpora

Produce two files:

- `train_corpus.jsonl` — one line per SFT training row, with fields `{id, source, text}`. Concatenate the row's prompt + response into `text` for scanning; keep `source` populated (e.g., `ultrachat_200k`, `slim_orca_math`, `internal_prod_tickets_2026_07`) because chapter 5's fix step traces back to source, not row.
- `eval_corpus.jsonl` — one line per eval item, with fields `{id, eval_name, split, item_text, gold}`. `item_text` should be the full prompt the model sees (system + user + any context). For log-likelihood tasks, also include the gold-answer text; the scan checks both the question stem *and* the canonical answer.

Cap `train_corpus.jsonl` at a size you can process on your hardware — 50k–500k rows is a reasonable range. Down-sample per source with stratified sampling if the raw SFT set is larger, and record the sampling ratio in the report.

Write a small `assemble.py` that produces both files reproducibly from source paths.

### Part B — Lens 1: n-gram overlap

Write `scan_ngram.py` that:

1. Builds two n-gram indexes from `train_corpus.jsonl`: word 8-grams (conservative) and word 13-grams (strict near-verbatim). Use a hash set for membership queries; do not do naive nested loops.
2. For every eval item, checks whether any of its n-grams appear in either index.
3. Emits `ngram_hits.jsonl` with one line per (eval_item, train_row, matched_n, matched_ngram) tuple.
4. Emits `ngram_summary.md` with per-eval hit counts at both n = 8 and n = 13.

**Pre-filter** before scanning:

- Strip and skip lines that are obvious boilerplate (copyright headers, "The following is a multiple-choice question," dataset preambles). Log what you skipped and how many rows.
- Normalize whitespace and case for the n-gram key.

**Hand-sampling requirement.** Sample **at least 30 flagged hits** at n = 13 per eval, hand-label as `confirmed / boilerplate / spurious`, and report the confirmed-hit rate. Chapter 5 pitfall 6 is explicit: raw flag counts without a confirmed-hit rate mislead.

### Part C — Lens 2: semantic (embedding) overlap

Write `scan_embedding.py` that:

1. Embeds every eval item's `item_text` with a sentence encoder (`sentence-transformers/all-mpnet-base-v2`, `BAAI/bge-large-en-v1.5`, or `intfloat/e5-large-v2` — pick one and pin it).
2. Embeds every training row's `text` with the same encoder. Cache the training embeddings to disk — you will not want to redo this pass.
3. Computes cosine similarity via a normalized-embedding inner product (or FAISS if the corpus is large enough that a dense matmul is impractical). For each eval item, keeps the top-5 nearest training rows.
4. Emits `embedding_hits.jsonl` with one line per (eval_item, train_row, cosine_sim) tuple where `cosine_sim ≥ 0.85`.
5. Emits `embedding_summary.md` with per-eval hit counts at thresholds `0.95`, `0.85`, and `0.70`.

**Threshold calibration.** Hand-label 50 pairs across the similarity range (some ≥0.95, some 0.85–0.95, some 0.70–0.85, some <0.70) as `duplicate / paraphrase / same-topic / unrelated`. Report the calibration table in the writeup — the "right" threshold for your encoder and task is what falls out of this exercise, not a copy-pasted number from chapter 5.

Report the top-10 highest-similarity pairs per eval side-by-side (eval item + matched training row + similarity) so a reviewer can eyeball severity.

### Part D — Lens 3: canonical-continuation model-signal probe

Write `probe_continuation.py` that runs the chapter 5 recipe against your fine-tuned model:

1. For each eval item, tokenize the prompt, take the first 60% of tokens as the prefix, keep the remaining 40% as the gold suffix.
2. Greedy-generate `len(suffix) + 20` tokens from the prefix with `do_sample=False`.
3. Record `exact_ratio` (fraction of the gold suffix reproduced verbatim as a leading substring) and `prefix_match` (boolean, `exact_ratio ≥ 0.9`).
4. Repeat against the *base model* — same prompts, same prefix cut. The base establishes the "inherited contamination" baseline.
5. Repeat against a **known-clean control set** — 30–50 items you author yourself that could not be in any web corpus (recent internal notes, invented word problems, novel instructions). This is the "known-negative" reference distribution.

Emit `continuation_probe.jsonl` and a `continuation_summary.md` table:

| Corpus | Model | Items | Mean `exact_ratio` | `prefix_match` rate |
|---|---|---|---|---|
| MMLU test | base | ... | ... | ... |
| MMLU test | fine-tuned | ... | ... | ... |
| GSM8K test | base | ... | ... | ... |
| GSM8K test | fine-tuned | ... | ... | ... |
| clean control | base | ... | ... | ... |
| clean control | fine-tuned | ... | ... | ... |

A fine-tune-caused contamination signal is `prefix_match(fine-tuned) − prefix_match(base)` on the eval, significantly above `prefix_match(fine-tuned) − prefix_match(base)` on the clean control. Interpret and report.

### Part E — Triage and severity rubric

For each (eval, lens) pair, apply chapter 5's severity table and record the verdict in `triage.md`:

| Eval | Lens | Raw flags | Confirmed after hand-sample | Severity (Critical / High / Low / Inherited / Clean) | Action |
|---|---|---|---|---|---|

Actions are one of:

- `retrain` — the fine-tune introduced verbatim contamination on ≥5% of items; the score is invalid, retrain with an updated exclusion list.
- `disclose + clean-subset` — 1–5% paraphrase-level contamination; report both full-set and clean-subset numbers.
- `disclose only` — <1% or boilerplate-only; disclosure is enough.
- `disclose inherited` — the signal is present on the base model and does not grow on the fine-tune; note in the model card that the base is contaminated on this eval.
- `no action` — clean on all three lenses.

If any eval lands on `retrain`, describe what would change in the mod-102 pipeline to prevent recurrence (updated exclusion list, source filter, decontam step). You do not need to actually retrain — but you need to be able to point to the exact fix.

### Part F — Compute "clean subset" numbers

For each eval that landed on `disclose + clean-subset` or worse:

1. Take the subset of eval items that were NOT flagged by any of the three lenses.
2. Re-aggregate the eval score (accuracy, exact-match, rubric mean — whatever the eval reports) on the clean subset only.
3. Report both the "as-is" and "clean subset" numbers with confidence intervals.

Add these to `triage.md` so the reader can see how much of the reported score is attributable to items that survived every scan.

### Part G — Write the model-card disclosure section

Produce `disclosure.md` in the exact structure chapter 5 shows — one table per lens, plus a paragraph per confirmed contamination event summarizing severity and action. This is the section that would be appended to your fine-tune's model card.

Minimum contents (adapted from chapter 5):

1. What was scanned: training-set size, evals covered, encoder + threshold for the embedding scan, harness version + prompts used for the continuation probe.
2. Per-eval overlap table (n-gram, embedding).
3. Per-eval model-signal table (continuation probe against fine-tuned, base, clean control).
4. Per-eval action taken.
5. Full-set vs. clean-subset numbers for any eval with confirmed contamination.
6. "Base-model contamination is inherited from pretraining and is not something we can retract post-hoc" acknowledgment for any inherited signals.

The disclosure should read like the "Data contamination" section in a well-run open-model card (Llama 3, Mistral, Qwen, Gemma — see `resources.md`). Reference no numbers you did not compute.

### Part H — Save

Directory layout:

```
outputs/exercise-04/
├── assemble.py
├── train_corpus.jsonl                # or a manifest pointing at the source
├── eval_corpus.jsonl
├── scan_ngram.py
├── ngram_hits.jsonl
├── ngram_summary.md
├── scan_embedding.py
├── embedding_hits.jsonl
├── embedding_summary.md
├── embedding_calibration.md          # your 50-pair calibration
├── probe_continuation.py
├── continuation_probe.jsonl
├── continuation_summary.md
├── clean_control.jsonl               # your authored 30–50-item negative set
├── triage.md
├── clean_subset_numbers.md
└── disclosure.md
```

## Starter guidance

- **Start with n-gram.** It's the cheapest and it tells you what's coming. If n-gram finds nothing, embedding often finds a small number of paraphrases; if n-gram finds thousands, embedding will find more but the story is already clear.
- **Do the boilerplate pre-filter before you look at the n-gram counts.** Otherwise every eval will show ten thousand "hits" that are all `"Choose the best answer:"` and you will lose the signal.
- **Pin the encoder.** Do not swap `all-mpnet-base-v2` for `bge-large` halfway through — the thresholds are not portable. Chapter 5 pitfall 3.
- **Author the clean-control set before the probe.** It is easy to forget until you see your fine-tuned model's `prefix_match` numbers and have no reference distribution to compare against. Write ~50 items that could not plausibly be anywhere in a public corpus — recent-dated internal notes, fresh word problems with unusual specific numbers, invented named entities.
- **Cache embeddings and generations.** The training-set embedding pass is the slowest step; the continuation probe is the second slowest. Both are trivially cacheable — do it once and store the artifact.
- **Do not silently delete flagged rows.** Chapter 5 pitfall 6. Every removal is a decision; log it, and hand-check that the removal targets a real hit and not a boilerplate false positive.
- **Membership-inference (Oren et al. 2023, lens 3c) is a stretch goal.** It is the strongest lens but the implementation is nontrivial and depends on a specific probe library. If you can adopt an existing implementation, great. If not, the canonical-continuation probe is enough to complete this exercise.

## Acceptance criteria

You have completed the exercise when:

- `ngram_hits.jsonl` and `ngram_summary.md` exist, and a hand-sampled confirmed-hit rate at n = 13 is reported for every eval.
- `embedding_hits.jsonl` and `embedding_summary.md` exist, with per-threshold counts at 0.95 / 0.85 / 0.70 and the top-10 highest-similarity pairs per eval.
- `embedding_calibration.md` shows 50 hand-labeled pairs and the resulting "our-encoder, our-corpus" threshold justification.
- `continuation_probe.jsonl` and `continuation_summary.md` include the fine-tuned model, the base model, and a clean-control comparison — all three columns are required to interpret the signal.
- `triage.md` assigns a severity + action to every (eval, lens) pair using chapter 5's rubric.
- For every eval flagged `disclose + clean-subset` or worse, `clean_subset_numbers.md` reports the re-aggregated score on the un-flagged subset alongside the full-set score, both with CIs.
- `disclosure.md` reads like a model-card section: methods, thresholds, flagged counts, actions, inherited-contamination acknowledgments. A reader could pick it up and audit your claims.
- No number in the disclosure is invented — every figure traces back to a JSONL your scripts produced.

## Stretch goals

- **Implement the membership-inference probe (Oren et al. 2023).** Compare the fine-tuned model's log-likelihood of the canonical eval-item ordering against many shuffled orderings; report the p-value. This is the strongest lens with formal statistical guarantees.
- **Implement the guided-prompting probe (Golchin & Surdeanu 2024).** Prepend the item's dataset-and-split metadata as a prefix and measure the accuracy delta vs. no prefix; report on MMLU and GSM8K.
- **Time-lag control.** Run the continuation probe against the same evals with a base model whose training data cutoff is *before* the eval was published. The clean-baseline number falls out for free — this is the strongest possible negative control for "the base has seen it."
- **Cross-eval leakage.** Some evals share source material (MMLU has overlap with AGIEval, HellaSwag has overlap with SWAG). Run the embedding scan eval-to-eval and report any pairs above 0.85.
- **Automate the disclosure.** Generate `disclosure.md` from a Jinja template driven by the JSONL outputs of Parts B–D. This is what mod-111 will consume; a hand-typed disclosure is not a platform artifact.
- **Version the scan.** Emit a `scan_manifest.yaml` that pins the training-set snapshot hash, the eval snapshot hash, the encoder, the harness commit, the model-signal probe seeds, and the clean-control-set hash. Every future rerun references this manifest — chapter 5 pitfall 6 warns against unversioned disclosures.
- **Wire into the release-gate manifest.** Add a `contamination` block to your `release_gate.v3.yaml` (or the exercise-05 manifest) that names which scans are "fast lanes" (per-commit) and which are "slow lanes" (per-release-candidate), so the disclosure is regenerated automatically as part of the gate.
