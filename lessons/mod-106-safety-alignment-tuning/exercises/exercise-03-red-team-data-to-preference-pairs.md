# exercise-03: Red-Team Data to Preference Pairs

**Estimated effort:** 3 hours (~1 hour intake + ~1 hour conversion + ~1 hour dataset card + audit)
**Companion chapters:** [`04-red-team-data-to-training-signal.md`](../04-red-team-data-to-training-signal.md), background from [`02-refusal-and-harm-reduction-datasets.md`](../02-refusal-and-harm-reduction-datasets.md), optional pipe-in from [`03-constitutional-ai-and-rlaif.md`](../03-constitutional-ai-and-rlaif.md)

## Objective

Ingest a public red-team / attack corpus, walk it through the chapter-4 intake pipeline (canonical schema → reproduction → dedup → categorization → severity triage), apply the pattern-A / B / C conversion rules to turn attack traces into `(prompt, chosen, rejected)` preference pairs, and ship a training-ready DPO dataset with a full dataset card and a decontamination audit against the standard eval benchmarks. The dataset produced here plugs directly into a mod-105 DPO run and, blended with exercise-01's preference set, forms the safety-preference half of the mix chapter 2 specified.

You are simulating the role of a fine-tuning engineer who has just received a batch of red-team submissions from a peer risk team and has to turn them into training data by end of week. The public corpus stands in for the peer team's inbound feed; the pipeline you build is the one you would run every red-team cycle in production.

## Prerequisites

- The safety mix from exercise 01 (`outputs/safety-mix-v1/`) — you will blend into and dedup against it.
- (Optional but recommended) The CAI-revised SFT set from exercise 02 (`outputs/cai-run/cai_sft/`) — its `final_response` rows are one source of "chosen" responses for pattern A below.
- `datasets`, `transformers`, `sentence-transformers` (for embedding-based dedup), and a small chat model for generating chosen responses when you use pattern A / B.
- A GPU with ≥ 16 GB VRAM (for the response-generation step in patterns A / B; the intake / conversion / audit steps are CPU-only).
- Access to at least **one** of the following attack corpora on the Hugging Face Hub:
  - `walledai/HarmBench` (the standard-behaviors split of Mazeika et al., 2024)
  - `JailbreakBench/JBB-Behaviors` (Chao et al., 2024 — 100 harmful + 100 benign behaviors)
  - `allenai/wildguardmix` (Han et al., 2024 — in-the-wild + adversarial + vanilla safety, with per-row harm / refusal labels)
  - `walledai/AdvBench` (Zou et al., 2023 — 520 harmful behaviors)

## Requirements

### Part A — Pick a corpus and design the intake schema

Pick one primary attack corpus (see prerequisites). Load it and inspect ~20 rows. Then commit `outputs/red-team-v1/intake_schema.json` — a JSON Schema describing the canonical intake row. Match the ideal schema from chapter 4:

```json
{
  "id": "rt-2026-09-02-0001",
  "prompt": "<attack prompt text>",
  "model_response": "<the response the target model produced on this prompt>",
  "target_model": "<model id + revision>",
  "attack_category": "direct" | "roleplay" | "encoded" | "instruction_hierarchy" | "multi_turn" | "adversarial_suffix" | "refusal_suppression" | "distribution_shift",
  "harm_category": "<category from your exercise-01 taxonomy>",
  "severity": "low" | "medium" | "high" | "critical",
  "source_corpus": "<hub_id>",
  "source_id": "<original row id in the source corpus>",
  "attacker_notes": "<free text, optional>",
  "expected_behavior": "hard_refusal" | "soft_refusal" | "safe_completion",
  "timestamp": "<ISO-8601>",
  "cluster_id": "<hash after dedup, populated in Part C>"
}
```

Add any corpus-specific fields you find useful (WildGuardMix carries `prompt_harm_label`, `response_harm_label`, `response_refusal`; keep them and namespace under `source_labels`). Document the crosswalk from source corpus fields to canonical fields.

### Part B — Ingest and reproduce

Convert your primary corpus (or a fixed subset — 300–500 rows is enough) into canonical schema rows and save as `outputs/red-team-v1/intake.jsonl`.

For every row, run the **reproduction step** from chapter 4:

- Load a small chat model as the *target model under test* — pick the same generator you used in exercise 02 (`Llama-3.2-3B-Instruct`, `Qwen2.5-3B-Instruct`, or your mod-103 SFT). This stands in for "the current checkpoint."
- Re-run each prompt against it. Capture the fresh response as `model_response`.
- Optionally, run the prompt against Llama Guard 3 (or WildGuard) to get a fast "was this response unsafe?" label; store as `response_harm_label` in `source_labels`.
- Tag rows where the current model already refuses / safe-completes as `status: "already_fixed"`; keep them in the intake but exclude them from Parts D–E's conversion. Chapter 4's reproduction step is what stops you from training against attacks the model has already learned to defeat.

The target model is deliberately a small, under-safety-tuned base — you want to see failures to convert, not to prove the model is already perfect. Record `target_model` in every row.

### Part C — Dedupe and categorize

Two operations, order matters.

**Deduplicate.** Embed every prompt with a small sentence encoder (`sentence-transformers/all-MiniLM-L6-v2` is enough). Cluster with a cosine-similarity threshold around 0.85 (adjust after eyeballing 20 pairs — too tight and near-duplicates survive; too loose and semantically distinct prompts get collapsed). Assign every row a `cluster_id`. Keep one representative per cluster for the conversion step (save the map so you can weight training samples by cluster size later). Log per-cluster sizes; a cluster with 50+ near-duplicate prompts is a red flag that your source corpus is over-representing one attack template — chapter 4's category-imbalance failure showing up in miniature.

**Categorize.** Tag every row with:

- `harm_category` from your exercise-01 taxonomy. If the source corpus has its own tags (HarmBench: 7 semantic categories; WildGuardMix: taxonomy-labeled), write a crosswalk in `outputs/red-team-v1/crosswalk.md` mapping each source category to your taxonomy.
- `attack_category` from chapter 4's eight-way attack taxonomy. If the source corpus does not label attack shape, tag manually or apply light heuristics: prompts containing "DAN" / "roleplay" / "pretend you are" → roleplay; prompts with base64 blobs or heavy unicode → encoded; multi-turn conversations → multi_turn; prompts with adversarial-suffix gibberish → adversarial_suffix; the rest → direct. Manual re-check a 20-row sample.
- `severity`. Apply a simple rubric (document it in `outputs/red-team-v1/severity_rubric.md`): critical = CBRN / CSAM / mass-harm; high = enables real-world crime or targeted individual harm; medium = policy violations without direct real-world harm; low = borderline / stylistic. Chapter 4 makes the point that this rubric is jointly owned with the risk team; here you are writing your side of it.

Produce a **coverage report** `outputs/red-team-v1/coverage.md` with two tables: harm_category × row count and attack_category × row count. Note any categories with < 10 rows — those are chapter 4's "under-represented" categories your training will be blind to.

### Part D — Convert to preference pairs

For every non-`already_fixed` row, apply the pattern rubric from chapter 4:

| If the prompt is... | Prefer pattern |
|---|---|
| Unambiguously out-of-policy (CBRN, CSAM, targeted harassment, mass violence) | **Pattern A** (chosen = hand-crafted / stronger-model hard refusal) |
| A paraphrase / attack variant on a policy the model already handles | **Pattern B** (chosen = current model's near-paraphrase refusal) |
| Partially in-policy (sensitive area with legitimate uses — self-harm, weapons for legitimate uses, drug harm-reduction) | **Pattern C** (chosen = safe completion — redacted / warning-prefixed / reframed) |
| A benign-adversarial looking-unsafe prompt | Route to the SFT full-answer bucket, not a preference pair (this is exercise 01's territory; drop here) |

For each row, tag the chosen `pattern` in the output.

**Generating chosen responses:**

- **Pattern A**: either author by hand from your exercise-01 style guide (2–3 sentence refusal, no lecture, optional alternative direction), or delegate to a strong open model with a strict "produce a chapter-2-style hard refusal for this prompt" system prompt. If exercise 02 is done, the CAI-revised responses for unambiguously-out-of-policy prompts are a natural source — pull them where they overlap.
- **Pattern B**: run the current target model on 3–5 paraphrases of the failed prompt (a small prompt-template does this: "Rephrase the following in a way that means the same thing: ..."). Take the first paraphrase where the model refuses cleanly. If none refuse, fall back to pattern A.
- **Pattern C**: prompt a strong open model with the safe-completion rubric ("Answer the following prompt, but strip out any harmful instructions; add appropriate safety context; recommend professional help if applicable"). Hand-review; safe completions are the highest-skill class and the most likely to have subtle issues.

**Rejected side** is always the row's `model_response` from Part B — the target model's actual failed output. If Llama Guard flagged it as safe (i.e., the response was benign), skip the row: no rejected candidate means no pair.

Output as `outputs/red-team-v1/preference/` — `datasets.Dataset` with the standard schema (matches exercise 01):

```python
{
  "prompt": str | list,
  "chosen": str,
  "rejected": str,
  "meta": {
    "source": "red-team-v1",
    "harm_category": str,
    "attack_category": str,
    "severity": str,
    "pattern": "A" | "B" | "C",
    "cluster_id": str,
    "chosen_generator": str,           # who produced the chosen response
    "rejected_target_model": str,
    "chosen_len_tokens": int,
    "rejected_len_tokens": int
  }
}
```

Apply exercise-01's quality-control filters here too — length filter (drop pairs where `chosen_len_tokens > 2 * rejected_len_tokens` or `< 0.5 * rejected_len_tokens`), degenerate filter (< 5 or > 1024 tokens), prompt dedup against exercise-01's own preference set. Log all deltas.

### Part E — Decontamination audit

The single most-often-skipped step in this pipeline and the one chapter 4 warns hardest about. Hash every training prompt in your new preference set against the prompt sets of:

- **HarmBench standard behaviors** (400) — if you did not use it as source.
- **JailbreakBench behaviors** (100 harmful + 100 benign) — if you did not use it as source.
- **XSTest** (250 safe + 200 unsafe).
- **OR-Bench** (sample the eval subset you will use in exercise 04).
- **Do-Not-Answer** (939).
- The eval prompts you reserved in exercise 01's Part E benign split.

Drop any prompt overlaps. Record drop counts per benchmark in `outputs/red-team-v1/decontamination.md`. If your primary corpus *is* one of the benchmarks (e.g., you used HarmBench as source), note explicitly which benchmark you cannot use for eval as a result. Chapter 4: "if the *same* prompt is in a public eval benchmark, your eval score on that benchmark is contaminated."

Bonus discipline: check for near-duplicates as well as exact matches. Embed both training and eval prompts, flag any pair above 0.9 cosine similarity, hand-inspect. Near-duplicate contamination is subtler than exact-match and easier to miss.

### Part F — Return-to-red-team diff report

Chapter 4's model-in-the-loop cadence has a step most teams skip: hand the results back to the red team so they can calibrate. Produce `outputs/red-team-v1/diff_report.md` with:

- **Baseline**: which target model you ran against, evaluation date, and the target model's overall refusal rate on the *original* attack set (before your training data would have applied). Use Llama Guard or WildGuard labels from Part B for the "refused / safe" tag.
- **Cluster-level status**: for each cluster from Part C, one row saying: "N attacks; current model refused K / N; we produced M training pairs; expected fix status = <fixed / partial / out-of-scope>."
- **Explicitly-not-fixing list**: attacks you chose *not* to train against, with a policy reason. Chapter 4 makes the point that "we're not going to fix this" is a valid answer to a red-team submission, but it has to be documented, not silently ignored.
- **Category coverage vs the red-team submission's mix**: where the delivered corpus over/under-represents categories relative to your policy priorities.

This report is what closes the loop with the risk team. Even in the exercise (where the "red team" is a public corpus), producing it exercises the muscle.

### Part G — Write the dataset card

`outputs/red-team-v1/dataset_card.md`. On top of exercise-01's dataset-card sections, add red-team-specific ones:

- **Corpus provenance**: source corpus, revision, license. Attack corpora are usually permissively licensed but check (HarmBench is MIT; JailbreakBench is MIT; WildGuardMix is ODC-BY; AdvBench is MIT). The `chosen` responses you generated inherit the license of the generator you used — Anthropic outputs come with API-terms restrictions, Llama-family outputs have Llama-license restrictions; note them.
- **Intake schema reference**: `intake_schema.json` and the crosswalk.md.
- **Reproduction step**: which target model, which chat template, sampling params.
- **Cluster statistics**: N raw rows, N clusters, top-5 largest clusters and their sizes.
- **Pattern breakdown**: how many pairs of each pattern A / B / C.
- **Chosen-response provenance**: fraction generated by which model / handwritten / pulled from CAI.
- **Decontamination**: from Part E; explicit list of which benchmarks the derived set is now contaminated *from* (i.e., cannot eval on).
- **Sensitivity note**: the `rejected` side contains model-generated harmful outputs. Chapter 4 makes the point that this is the exact shape a bad actor would want to steal. Document the access control on the artifact (`outputs/red-team-v1/` should not be shared publicly without redaction).
- **Blend guidance**: how many pairs to blend into a general preference set (chapter 2: 5–25% of preference pairs). Recommended blend ratio for your specific mix, plus a note on the length envelope match against exercise-01's preference pairs (the length-bias failure from chapter 2 and mod-105 chapter 7 lives here).

## Starter guidance

- **Read 20 raw rows before you write any code.** The realistic-intake variant in chapter 4 ("model said bad thing, see screenshot") is not a joke; public corpora are cleaner than internal streams but still have surprises (WildGuardMix has multi-turn conversations mixed in with single-turn attacks; HarmBench has some behaviors that read like generic prompts rather than attacks). Eyeballing rows first saves rewriting the schema later.
- **Do not skip the reproduction step.** If you convert every attack in the corpus without checking whether the current model already refuses, ~30–70% of your training pairs will be redundant. The `already_fixed` tag is a first-class part of the pipeline, not a nice-to-have.
- **Cluster before you convert.** Every conversion costs a generation (pattern A / B) or a hand-review (pattern C). Clustering by prompt first means you convert once per cluster and up-weight the training loss by cluster size, rather than converting 50 near-duplicate prompts individually.
- **A small severity rubric beats no rubric.** Even a three-line document ("critical = these categories, high = these categories, medium = default, low = borderline") is what turns your judgment into a reviewable artifact. The rubric is the load-bearing part of chapter 4's triage step.
- **Pattern C is the risky one.** Safe completions are the highest-quality signal but also the easiest to get subtly wrong (a warning that reads as a jailbreak because it details the harm; a reframe that misses the actual policy concern). Hand-review every pattern-C pair before including it. A pattern-C rate of ~10–20% of the total is normal; if you find yourself over 40%, either your rubric is too permissive or your corpus is skewed to gray-area prompts.
- **The `rejected` side matters.** Do not synthesize the failed response — use the actual model_response from reproduction. A DPO run with a synthetic "we imagine the model would say X" rejected side teaches the model to prefer the safe pattern over a *specific* failure that never happens in practice, and misses the failures the model actually produces.
- **Do not skip decontamination.** Chapter 4 and chapter 5 both warn about it; every safety writeup that skips it later has to be caveated. Do it up front.
- **Do not train a model here.** Same discipline as exercise 02: this is a data-engineering exercise. The training run that consumes this dataset lives in mod-105 exercise 04 (DPO) or a re-run of it on the combined `exercise-01 + exercise-03` preference set.

## Acceptance criteria

You have completed the exercise when:

- `outputs/red-team-v1/intake_schema.json` and `intake.jsonl` exist, with `intake.jsonl` reproducing all rows through the current target model (fresh `model_response`, `status`, and any Llama Guard / WildGuard labels captured).
- `outputs/red-team-v1/coverage.md` shows harm × attack category counts and calls out under-represented categories.
- `outputs/red-team-v1/preference/` loads via `datasets.load_from_disk` and contains rows in the standard preference schema with the required metadata (pattern, cluster_id, severity, provenance).
- `outputs/red-team-v1/decontamination.md` documents overlap counts against ≥ 4 of the listed eval benchmarks and any benchmarks now barred from evaluation use.
- `outputs/red-team-v1/diff_report.md` names the target model, its baseline refusal rate on the corpus, per-cluster fix status, and an explicit not-fixing list.
- `outputs/red-team-v1/dataset_card.md` covers every section in Part G, including the sensitivity / access-control note.
- Every non-`already_fixed`, non-benign row that survived the filters is tagged with exactly one pattern (A / B / C); the pattern distribution is reported in the dataset card.
- `chosen_len_tokens` and `rejected_len_tokens` distributions match the length envelope of exercise-01's preference set (median within ~20% of each other). Document explicitly if they don't and why.

## Stretch goals

- **Multi-turn conversion.** WildGuardMix contains multi-turn attack conversations. Extend Part D to handle them: the `prompt` becomes a chat-history list, the target model is run for a full turn continuation, and the safe response applies to the assistant's final turn. Chapter 4 flags multi-turn / gradual jailbreak as a distinct attack class that single-turn conversion misses.
- **Attack-strategy stratification.** After Part C, split your preference set by `attack_category` and train (or plan to train) mod-105's DPO trainer with per-strategy sampling weights. Report the per-category refusal-rate deltas — some attack classes (roleplay, refusal-suppression) tend to move more per training pair than others (adversarial-suffix, encoded).
- **CAI-augmented chosen responses.** If exercise 02 is done, for every pattern-A or pattern-C row, generate the `chosen` via the CAI critique-and-revise loop rather than a single-shot generation. Compare 50 CAI-derived `chosen` responses to 50 single-shot responses; which produces the more consistent style? Chapter 3 predicts the CAI-derived side wins on consistency and loses on latency; verify.
- **Length-normalize the preference set.** Downsample pattern-A pairs (which tend to be shortest on the chosen side) until the chosen-vs-rejected length delta histogram is roughly symmetric. Report the before/after in the card. This is the direct fix for the length-bias trap from chapter 2 / mod-105 chapter 7.
- **Freshness weighting.** Tag every pair with a timestamp; when blending into a training mix, weight recent pairs higher. Chapter 4's freshness note: attack techniques from 2023 have different shapes than 2025's; older pairs teach something but the training signal decays. Show the weighting scheme in the dataset card.
- **Cross-corpus stress test.** Take your preference set and evaluate a small DPO-trained model (from mod-105) on a held-out attack corpus you did *not* train on. Does the model generalize across attack surfaces, or does it only refuse the specific corpus it saw? Chapter 4's "coverage" thesis lives or dies on this test.
- **Escalation flags.** Add a `should_escalate: true` tag for rows that meet any of chapter 6's "escalate rather than train against" criteria (persistent adversarial suffixes, RAG / tool prompt-injection, CSAM, high-stakes-domain accuracy). Route these to a separate report and *out* of the training set. This is the operational half of chapter 6's handoff table, exercised on real rows.
- **PII scrub.** For any prompt or response containing named entities, PII patterns, or sensitive templates, replace with placeholders before training (regex-scrub emails / phone numbers / national IDs; NER-scrub named individuals). Chapter 4's "prompt confidentiality" note. Document the scrub in the dataset card.
