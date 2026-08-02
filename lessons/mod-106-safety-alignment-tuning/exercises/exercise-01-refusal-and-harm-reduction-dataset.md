# exercise-01: Refusal and Harm-Reduction Dataset

**Estimated effort:** 3 hours
**Companion chapters:** [`01-the-safety-alignment-problem.md`](../01-the-safety-alignment-problem.md), [`02-refusal-and-harm-reduction-datasets.md`](../02-refusal-and-harm-reduction-datasets.md)

## Objective

Build a safety-training dataset that is ready to feed into an `SFTTrainer` (safe-completion side) and a `DPOTrainer` (safety-preference side), together with a **dataset card** that documents provenance, taxonomy, coverage, filters, and license. The set produced here is the input to the CAI augmentation loop in exercise 02, the red-team conversion pipeline in exercise 03, and the eval harness in exercise 04 — sloppy work here compounds through the module.

The exercise is a data-engineering exercise, not a training exercise. You will not train a model here. You will produce two `datasets.DatasetDict` artifacts and a written card that a chapter-5 reviewer could sign off on.

## Prerequisites

- `datasets`, `transformers`, and `tokenizers` at recent versions.
- A Hugging Face account, with `huggingface-cli login` completed (several of the source datasets require terms acceptance).
- The tokenizer of your intended downstream SFT model (Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct, or your mod-103 checkpoint — pick one; you need its chat template).
- ~2 GB of disk for cached datasets.
- The preference dataset from mod-105 exercise 01 (if available) — you will blend safety data into it.

## Requirements

### Part A — Pick a taxonomy and write it down

Choose the harm taxonomy your dataset will cover. Options (chapter 1):

- **MLCommons AILuminate** (12 categories) — recommended default; matches Llama Guard 3's output taxonomy, so exercise 04's evals will slot in cleanly.
- **Anthropic's HH taxonomy** — narrower, closer to what HH-RLHF was labeled with.
- **A custom taxonomy** derived from a specific product's Acceptable Use Policy — realistic but more effort.

Write the taxonomy to `taxonomy.md` in your output directory. Every downstream artifact references this file.

### Part B — Pull from at least two open corpora

You need at least two upstream sources — one dominant, one supplementary — to reduce single-source bias.

**Recommended primary sources** (pick one):

- `PKU-Alignment/PKU-SafeRLHF` (BeaverTails preference pairs). CC-BY-NC-4.0; explicitly non-commercial — call this out in your card if you would ship the derived set.
- `Anthropic/hh-rlhf` (`harmless-base` split). MIT / CC-BY-4.0.
- `allenai/wildguardmix` (adversarial + vanilla safety scenarios). ODC-BY.

**Recommended supplementary sources** (pick at least one):

- `LibrAI/do-not-answer` (939 prompts across 12 subtypes).
- `PKU-Alignment/BeaverTails` (the labeled QA-format source of PKU-SafeRLHF).
- `HuggingFaceH4/no_robots` — the "no_robots" refusal exemplars from Hugging Face.
- A safety subset you carve from the WildGuardMix training split.

Load each source, record its size, columns, license, and a sample of 10 rows in `sources.md`.

### Part C — Produce the two-shape split

Emit two artifacts:

**1. Safe-completion SFT set** — `datasets.Dataset` with rows:

```python
{
  "prompt": str | list,       # chat-messages list preferred for multi-turn
  "response": str,
  "meta": {
    "source": str,
    "taxonomy_category": str,       # from Part A
    "response_class": str,          # "hard_refusal" | "soft_refusal" | "safe_completion" | "full_answer"
    "prompt_len_tokens": int,
    "response_len_tokens": int,
  }
}
```

Populate `response_class` heuristically for now (see Part D for the audit); a simple classifier looking for refusal keywords + length is fine for a first pass. Include **full_answer** rows drawn from your benign parallels (Part E) so a downstream SFT run does not train exclusively on refusals.

**2. Safety-preference set** — `datasets.Dataset` with rows in the standard preference shape (mod-105 chapter 2):

```python
{
  "prompt": str | list,
  "chosen": str,
  "rejected": str,
  "meta": {
    "source": str,
    "taxonomy_category": str,
    "chosen_class": str,          # same enum as response_class
    "rejected_class": str,
    "chosen_len_tokens": int,
    "rejected_len_tokens": int,
  }
}
```

Token lengths must be computed with the *actual* tokenizer you will train against.

### Part D — Quality controls

Implement each filter as a separate `.filter(...)` so the row count before/after each is visible. At minimum:

1. **Length filter** (mod-105 chapter 2). Drop pairs where `chosen_len_tokens > 2 * rejected_len_tokens` OR `chosen_len_tokens < 0.5 * rejected_len_tokens`. Log the count.
2. **Empty / degenerate filter.** Drop rows where any response is under 5 tokens or over 1024 tokens.
3. **Prompt deduplication.** Dedup by prompt hash across both artifacts. If the same prompt appears in both SFT and preference sets, that's fine, but do not have duplicate `(prompt, response_class)` combinations in the SFT set.
4. **Response-style consistency filter.** For each `taxonomy_category` × `response_class` cell, hand-inspect ~10 samples. If the samples span radically different lengths / voices, mark the cell as "high variance" in the card — you will want to normalize this before training in a real pipeline, but for the exercise, documenting it is acceptable.
5. **PII scrub** (defensive; skip if your sources are already scrubbed). Regex-scrub obvious PII (email addresses, phone numbers, national-id patterns) from prompts and responses. Do not rely on the scrub for real-world guarantees — this is exercise hygiene.

Log every filter's row-count delta to `filter_log.txt`.

### Part E — Benign-adversarial parallels

The critical over-refusal control. Produce or curate **at least 300 benign-adversarial rows** and include them in the SFT set with `response_class="full_answer"`.

Sources you can draw from:

- **XSTest** — but hold out at least half of it for evaluation; do not train on the entire benchmark.
- **OR-Bench** — has enough volume that a training subset + eval subset can be safely disjoint.
- **Hand-generated parallels.** For 50 of your refusal rows, generate a benign parallel by prompting a strong open model (Qwen2.5-14B-Instruct, Llama-3.1-8B-Instruct) with something like: "Given this refusal prompt, write a superficially similar but genuinely benign prompt that shares the surface keywords." Manually review a sample.

For each benign parallel, produce a *full answer* — the model completes the request rather than refuses. This is the disambiguation signal.

Document the split between training and eval benign-adversarial prompts and commit the split hashes to `benign_split.json`.

### Part F — Coverage table

Produce a coverage table in your dataset card with rows = taxonomy categories, columns = counts of:

- `sft_refusal` (hard + soft + safe_completion combined)
- `sft_benign_parallel` (full_answer rows)
- `preference_pairs`
- `eval_prompts_reserved` (from Part E's benign split + any held-out safety-eval prompts)

Every non-zero category in your taxonomy needs a non-zero count in every column. If a category has a zero, document why (out-of-policy for the deployment context, coverage gap you're accepting, etc.).

### Part G — Decontamination check

Hash all your training prompts and compare against the *prompt sets* of the eval benchmarks you plan to use in exercise 04:

- HarmBench standard behaviors (400)
- JailbreakBench behaviors (100 harmful + 100 benign)
- XSTest (250 safe + 200 unsafe)
- OR-Bench sample (whichever slice you reserved as eval)
- Do-Not-Answer (939)

Drop any prompt overlaps from your training sets. Record the drop count per benchmark in `decontamination.md`. This is the single most-often-skipped step; do not skip it.

### Part H — Dataset card

Write `dataset_card.md` with:

- **Taxonomy** — reference to `taxonomy.md`.
- **Provenance** — sources, revisions, licenses.
- **Sizes** — raw source counts; per-filter deltas; final per-split counts.
- **Length statistics** — median and 90th-percentile token lengths for prompt / chosen / rejected / response; the chosen-vs-rejected length delta distribution.
- **Response-style distribution** — count by `response_class` in the SFT set and by `chosen_class` × `rejected_class` in the preference set.
- **Coverage table** — from Part F.
- **Benign-parallel provenance** — hand-authored vs XSTest-derived vs OR-Bench-derived counts; split hashes reference.
- **Over-refusal self-audit** — sample 50 benign-adversarial prompts from your eval split and check by hand whether your training data would have led to a refusal response (nearest-neighbor lookup against your refusal set is enough for a first-pass check). Report the estimated false-refusal risk in the card.
- **Decontamination** — from Part G.
- **License chain** — every source's license; the effective license of the derived set (most-restrictive wins unless re-licensed).
- **Known limitations** — categories with low coverage; sources of judgment that may not match your policy; anything else worth flagging.

### Part I — Save

Directory layout:

```
outputs/safety-mix-v1/
├── taxonomy.md
├── sources.md
├── dataset_card.md
├── filter_log.txt
├── benign_split.json
├── decontamination.md
├── sft/                          # save_to_disk of the safe-completion set
│   ├── dataset_dict.json
│   └── ...
├── preference/                   # save_to_disk of the safety-preference set
│   ├── dataset_dict.json
│   └── ...
└── coverage_table.csv            # machine-readable version of Part F
```

Both `sft/` and `preference/` should be splittable into `train` / `eval` (~90 / 10 by default). Set a fixed seed and dump split assignments to `splits.json` inside each.

## Starter guidance

- **Do not skip the benign parallels.** They are the single most-effective over-refusal control. A dataset without them will produce an over-refusing model regardless of how good your refusal set is.
- **Hand-inspect ~50 rows before committing.** Read the prompt, the response, the label. Ask "would I have picked this response class here?" A 15-minute review before training saves an hour of debugging after.
- **Non-commercial licenses.** PKU-SafeRLHF is CC-BY-NC-4.0. If your production surface is commercial, either drop it or accept that the derived set can only be used for research / evaluation.
- **Multi-turn coverage.** HH-RLHF has multi-turn conversations; PKU-SafeRLHF is mostly single-turn. If your production surface is chat (usually is), include some multi-turn rows or plan to add them in a later pass.
- **Do not train yet.** This exercise stops at the dataset. Training runs happen in mod-105's exercises 03–05 (DPO / KTO / PPO) and can consume your outputs directly.
- **Small is fine.** 2000 SFT rows + 3000 preference pairs + 300 benign parallels is enough to feel every step of the pipeline. Scaling up is the same code with more data.

## Acceptance criteria

You have completed the exercise when:

- `outputs/safety-mix-v1/` exists with the structure above.
- `dataset_card.md` fills every section listed in Part H.
- `filter_log.txt` shows row counts after each filter, matching final split sizes.
- `decontamination.md` documents an overlap check against at least three of the listed eval benchmarks, with drop counts.
- `coverage_table.csv` has non-zero counts in every non-excluded row of your taxonomy × every column.
- `load_from_disk("outputs/safety-mix-v1/sft/")` and `load_from_disk("outputs/safety-mix-v1/preference/")` on a fresh session return the expected datasets.
- Your response-class distribution is *not* 100% refusal — the SFT set includes benign-parallel `full_answer` rows.

## Stretch goals

- **Cross-license the set.** If your primary source is CC-BY-NC-4.0, produce a commercial-eligible variant by dropping the non-commercial sources and topping up with MIT / Apache / CC-BY equivalents. Report the coverage impact.
- **Multi-language.** Translate 20% of your English refusal rows into a second language of your choice (using a strong open translator) and add them to the SFT set. Verify with a native speaker if possible.
- **Length-normalize the preference set.** Downsample "chosen longer" pairs until the length-delta histogram is roughly symmetric. Report the before/after in the card.
- **Add a system-prompt variant.** For 20% of your SFT rows, prepend a policy-establishing system prompt and re-generate the response in that context. This trains instruction-hierarchy behavior (chapter 4).
- **Nearest-neighbor over-refusal audit.** Embed all your refusal-set prompts with a small sentence encoder; for each XSTest safe prompt, look up the K nearest refusals in your training set. A cluster of very-similar refusals on a benign-adversarial query is a strong predictor of over-refusal. Report the cluster stats.
- **Publish the card publicly.** If your license chain allows it, upload the derived set to the Hub with the card as the model page. Even if you do not upload, following the Hub's `README.md` conventions (`dataset_info` block, `license` field, `tags`) makes the artifact more legible.
