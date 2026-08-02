# exercise-01: SFT Data Quality Audit

**Estimated effort:** 3 hours
**Companion chapters:** [`01-curation-and-quality-filtering.md`](../01-curation-and-quality-filtering.md), [`02-deduplication.md`](../02-deduplication.md)

## Objective

Take one real open SFT corpus, apply a written quality rubric with an explicit filter stack, and produce a before/after audit report that a colleague could hand to a compliance reviewer without embarrassment. By the end you will have (a) a rubric document, (b) a filter pipeline whose per-filter drop rates you can defend, and (c) a per-example metadata schema attached to every surviving row.

The point of this exercise is to make the abstract phrase "curated SFT set" refer to a specific file on your disk that you built and can walk any reader through.

## Prerequisites

- Python 3.10+ with `datasets`, `transformers`, `sentence-transformers`, and one language identifier (`fasttext` with `lid.176`, or `langdetect`, or a small classifier of your choice).
- Optional but recommended: `datasketch` (for MinHash), `detoxify` or a hosted moderation classifier, `presidio-analyzer` for PII.
- HF Hub access.

Choose ONE of the following corpora — or an equivalent open SFT set you already work with. Each is small enough to fully process on a laptop:

- `OpenAssistant/oasst1` — pick a single language (English) subset of assistant-preferred threads.
- `HuggingFaceH4/no_robots` — 10k hand-authored pairs.
- `GAIR/lima` — 1k pairs.
- `databricks/databricks-dolly-15k` — 15k pairs.

Do not switch corpora midway. The audit report must be a coherent picture of one set.

## Requirements

### Part A — Write the rubric first

Before touching data, write a rubric document (`rubric.md`) of no more than two pages that specifies, for the corpus you chose:

1. What counts as a well-formed example (structural rules — required fields, role alternation, non-empty content).
2. What counts as a high-quality response, using the dimensions in chapter 1 (clarity, correctness, completeness, format faithfulness, absence of teacher artifacts, safety fit).
3. What the language policy is (English-only, allow mixed?), the PII policy (drop vs. redact vs. keep), and the refusal policy (keep, drop, or route to a separate pile).
4. What tokenizer you will use for length measurements, and what your target `max_seq_length` is (so length filters have a defined threshold).

You will refer back to this document at the end of the exercise. Reviewers should be able to reproduce every decision from it.

### Part B — Implement the filter stack

Build a pipeline that takes the raw corpus and produces `curated.jsonl`. At each stage:

1. **Structural filter** — schema validity, non-empty roles, correct role alternation.
2. **Language filter** — accept only your target language(s); log the ones you drop.
3. **Length filter** — drop rows shorter than a documented floor or longer than your effective `max_seq_length` when rendered through `apply_chat_template` for your target base model (pick any Llama or Qwen instruct model to define the template).
4. **PII filter** — either drop rows with PII or redact in place, following your rubric.
5. **Toxicity / harmful-content filter** — score each message with any toxicity classifier; apply your rubric's threshold. Route benign-response-to-toxic-prompt rows to a separate pile if your rubric keeps them.
6. **Response-quality filter** — at minimum:
   - Repetition check (unique n-gram ratio in the response).
   - Prefix / template artifact check (regex for "As an AI language model", "Sure! Here is", leaked system fragments).
   - Format compliance for at least one prompt category (JSON, code block, etc.).
7. **Deduplication** — an exact-hash dedup at least; MinHash + LSH is a bonus.

Each stage emits both the surviving rows and a small sample of dropped rows to a `drops/` directory, so a reviewer can eyeball for false positives.

### Part C — Attach the per-row metadata schema

Every surviving row must be written out with the metadata envelope from chapter 1:

```jsonc
{
  "id": "…",
  "source": "…",
  "source_id": "…",
  "license": "…",
  "source_url": "…",
  "category": "…",
  "language": "…",
  "generator": null,               // populate if you know a row was model-generated
  "quality_score": …,              // optional but recommended
  "filters_passed": ["length", "language", "toxicity", "response_quality", "dedup"],
  "decontam": {"mmlu": "not_run", "humaneval": "not_run", "gsm8k": "not_run", "mtbench": "not_run"},
  "messages": [ … ]
}
```

`decontam` may be `not_run` in this exercise — exercise 02 fills it in.

### Part D — Produce the audit report

Write `audit_report.md` containing:

1. Counts in and out per filter, with drop rates.
2. Sampled examples of dropped rows per filter (5 per filter is enough), with your one-line judgment: "true positive" (filter was right) or "false positive" (filter was wrong; here is what to fix).
3. Post-filter distributions: language, message-length histogram (using the tokenizer you declared in the rubric), and category counts if the source has categories.
4. Any decisions where you deviated from the rubric mid-exercise, with the reason.
5. A "known issues" section listing the ways this dataset is still not audit-clean (e.g., "decontamination not run yet, exercise 02 covers it").

## Starter guidance

- Load `oasst1` with `datasets.load_dataset("OpenAssistant/oasst1")` and reduce to English `assistant`-terminating threads before you start filtering — that alone shrinks the dataset dramatically and makes the pipeline tractable.
- For the length filter, apply the target model's chat template with `tokenize=True` and check the resulting length — this is the number the SFT trainer will actually see, not the raw character count.
- For the response-quality repetition check, a compact heuristic is `len(set(word_ngrams(text, 4))) / max(1, len(word_ngrams(text, 4)))`. Below ~0.5 is suspicious on responses of any length.
- Read the source dataset card *before* filtering. Half of your rubric decisions will be made for you by what the card already documents.
- Do not over-engineer. A working pipeline with sampled drops is worth more than a beautifully abstracted one you never got to run end to end.

## Acceptance criteria

You have completed the exercise when:

- `rubric.md`, `pipeline.py` (or notebook), `curated.jsonl`, `drops/`, and `audit_report.md` all exist and are internally consistent.
- Every surviving row has the full metadata envelope; no `null`s in required fields.
- The audit report's per-filter drop rates match what a reader can independently compute from the drop samples and the input size.
- At least one filter's drop sample includes a **false positive** you identified and either fixed or explicitly acknowledged in the report. If you did not find any false positives, look again — most filters have some.
- The report can be read in five minutes and answers "what is in the curated set, and what was removed, and why."

## Stretch goals

- Add a MinHash + LSH near-dedup pass on `(prompt, response)` at a Jaccard threshold of 0.75. Report additional rows dropped compared to exact dedup only. Sample dropped pairs.
- Add a **judge-model quality filter** — use a small local model or a hosted API to score each response 1–5, and drop below 3. Compare the eval-time behavior of the model *if* you were to fine-tune on the pre- and post-judge sets (a thought experiment; no need to train). Which drops surprise you?
- Fold in a second source (e.g., add `no_robots` to your primary `oasst1` set). Union them, dedup across sources, and report cross-source collisions. This is the setup for the "cross-source dedup trap" from chapter 2.
- Publish your `curated.jsonl` + `rubric.md` as a private HF Hub dataset with a dataset card generated from your audit report. This is the shape of an actual internal handoff.
