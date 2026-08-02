# exercise-02: Eval-Set Decontamination

**Estimated effort:** 3 hours
**Companion chapter:** [`03-eval-decontamination.md`](../03-eval-decontamination.md)

## Objective

Take the curated SFT set from exercise 01 (or any equivalent SFT dataset) and run a defensible decontamination scan against MMLU, HumanEval, GSM8K, and MT-Bench. By the end you will have (a) an n-gram overlap scanner with two thresholds, (b) an embedding-similarity backstop for short-eval items, (c) a per-eval flag report with hand-labeled false-positive rates, and (d) the `decontam` field on every surviving training row filled in.

This exercise is not about writing beautiful code. It is about producing the artifact that lets you look a reader of your future model card in the eye and answer "did you decontaminate against MMLU?" with a specific yes.

## Prerequisites

- Python 3.10+ with `datasets`, `transformers`, `sentence-transformers`, and either `datasketch` or `bloom-filter2` for scalable set operations (optional at this scale).
- The `curated.jsonl` from exercise 01, or any SFT dataset of at least 10k rows in the same schema.

Eval datasets to pull (all publicly accessible on the HF Hub):

- `cais/mmlu` (all subjects, `test` split — ~14k items)
- `openai_humaneval` (164 items)
- `gsm8k` `main` config, `test` split (~1.3k items)
- MT-Bench prompts: `HuggingFaceH4/mt_bench_prompts` or download the 80 first-turn prompts from the FastChat repo

## Requirements

### Part A — Build the eval-item corpora

For each eval, write a loader that returns a list of `{"eval": name, "item_id": ..., "prompt": <text>}` records where `prompt` is the text you will scan against. Decisions to make and record:

- **MMLU**: concatenate the stem and the four choices, or the stem only? Write down which and why. (Hint: the choices carry signal, but multiple-choice reformatting in synthetic data can strip them — scan on stem-only for a strict pass and stem+choices for a permissive pass.)
- **HumanEval**: concatenate the function signature and the docstring. Also collect the canonical solutions as a *second* eval corpus so you can scan for solution leaks in your responses.
- **GSM8K**: scan on the problem text only. The `answer` field (which contains the step-by-step solution) leaks separately and should be its own corpus.
- **MT-Bench**: scan on the first-turn prompt.

Print, per eval, the number of items and the average item length in words.

### Part B — Implement n-gram overlap scanning

Build two scanners:

1. **Strict** — 13-word n-gram overlap. A training row is flagged if it shares any 13-gram with any eval item.
2. **Conservative-catch** — 8-word n-gram overlap. A training row is flagged if ≥1% of its 8-grams overlap the eval set.

Both should:

- Normalize text with a *documented* function (lowercase, whitespace collapse, unicode NFC; whichever you choose, do it the same for eval and training text).
- Scan against the concatenation of the training row's user prompt(s) and assistant response(s). Flag if either side hits.
- Return, per training row per eval, the top-3 matching eval items (with `item_id`) so a reviewer can see what triggered the flag.

### Part C — Add an embedding-similarity backstop for short evals

For MT-Bench and GSM8K, n-gram overlap misses paraphrases. Additionally run:

1. Embed every eval prompt with a small sentence embedder (`sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`).
2. Index the vectors with FAISS (or a hand-rolled cosine loop if the eval set is small).
3. Embed every training row's user prompt; query the index for the top-1 nearest neighbor.
4. Flag if cosine similarity ≥ 0.9. Record the matching eval item ID.

Report separately from the n-gram flags.

### Part D — Hand-label the false-positive rate

For each eval, sample 20 flagged training rows (or all of them if fewer than 20). For each, judge by hand whether the flag is:

- **True positive** — the training row genuinely leaks the eval item.
- **False positive** — coincidental overlap (common English boilerplate, math formula fragments) or the eval item is trivial and any general text about the topic would hit it.
- **Grey area** — same topic, unclear whether the model would gain unfair advantage.

Report the counts. Do not skip this step. The threshold you choose is only meaningful if you have measured its false-positive rate on your specific set.

### Part E — Produce the decontamination report

Write `decontam_report.md` containing:

1. Per-eval item count, scanning parameters, threshold choices.
2. Per-eval flag counts at both n=8 and n=13, plus embedding-similarity flags.
3. Per-eval hand-labeled false-positive rate.
4. Per-eval decision: remove flagged rows, or annotate and keep? Justify with the false-positive rate.
5. Ten example flagged rows per eval (or all flagged if fewer), with the matching eval item side by side.
6. Aggregate: how many training rows are removed by the union of all eval decontam passes?

### Part F — Write `decontam` back onto surviving rows

Every training row that survives now has:

```jsonc
"decontam": {
  "mmlu":      "clean" | "flagged_ngram_13" | "flagged_ngram_8" | "flagged_removed",
  "humaneval": "clean" | "flagged_ngram_13" | "flagged_ngram_8" | "flagged_removed",
  "gsm8k":     "clean" | "flagged_ngram_13" | "flagged_ngram_8" | "flagged_embed" | "flagged_removed",
  "mtbench":   "clean" | "flagged_ngram_13" | "flagged_ngram_8" | "flagged_embed" | "flagged_removed"
}
```

Emit `curated_decontaminated.jsonl` as the next artifact in the chain.

## Starter guidance

- Use `set` intersections over shingle tuples for the n-gram check at this scale (up to a few million grams total). Do not reach for Bloom filters or LSH unless you actually need them.
- For MMLU, `cais/mmlu` is split into per-subject configs; use `datasets.load_dataset("cais/mmlu", "all", split="test")` if the aggregated config is available, else iterate subjects.
- The HumanEval canonical solutions live in the `canonical_solution` field of each item — do not forget them; solution leaks are just as bad as prompt leaks.
- MT-Bench prompts are English, distinctive, and short. Any 13-gram match on MT-Bench is almost certainly a real leak — do not over-think the threshold there.
- Read the appendix C of Brown et al. (2020, GPT-3) for the reference recipe you are implementing. It is short.

## Acceptance criteria

You have completed the exercise when:

- Your two-threshold n-gram scanner runs against all four evals and terminates in reasonable wall-clock time (minutes, not hours) on the target dataset.
- The embedding-similarity backstop runs against MT-Bench and GSM8K and reports its top-1 similarities.
- Your `decontam_report.md` includes per-eval flag counts, hand-labeled false-positive rates, decisions, and side-by-side flag examples.
- `curated_decontaminated.jsonl` exists; every row has a filled-in `decontam` object; the row count matches "input minus rows removed."
- A reviewer can read the report in ten minutes and answer "was this dataset decontaminated against the big four evals, and how do we know?"

## Stretch goals

- Reproduce the Llama 2 or Llama 3 model-card decontamination methodology exactly (word n-gram at the paper's `n`; same removal criterion). Cite the specific paper section. Compare their per-eval leak rate to yours.
- Add a **code-aware decontamination pass** for HumanEval: strip whitespace, tokenize the code portion of the response with a Python tokenizer, and detect canonical-solution AST fragments. Report additional flags versus your text-only pass.
- Extend to a fifth eval of your choice (BBH, IFEval, TruthfulQA, ARC). Document why you chose it and what its decontamination target should be.
- Publish a small `decontam` module — `pip install`-able — that any teammate could point at a JSONL SFT set and get back the same report. Take one hour to make it usable; save your team days later.
