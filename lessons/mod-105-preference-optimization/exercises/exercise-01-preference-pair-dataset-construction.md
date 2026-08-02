# exercise-01: Preference Pair Dataset Construction

**Estimated effort:** 3 hours
**Companion chapters:** [`02-preference-pair-datasets.md`](../02-preference-pair-datasets.md), [`03-reward-modeling.md`](../03-reward-modeling.md)

## Objective

Turn an existing open preference corpus into a training-ready `datasets.DatasetDict` with quality and length controls applied, and ship a **dataset card** that documents what you did. The set you produce here is what exercises 02–05 consume — an RM, a PPO run, a DPO run, and a modern-family survey all draw from it. Sloppy work here compounds downstream.

## Prerequisites

- `datasets`, `transformers`, and `tokenizers` installed at recent versions.
- Access to the Hugging Face Hub (many source datasets require accepting terms; sign in with `huggingface-cli login`).
- The SFT checkpoint from mod-103 exercise 01 (or a public SFT model of your choice) — you need its tokenizer for chat-template checks.
- ~2 GB of local disk for cached datasets.

## Requirements

### Part A — Pick a source corpus

Pick **one** primary source from the list below. If your production surface has a strong domain (code, safety, multi-turn agents), pick the one closest to it.

- `HuggingFaceH4/ultrafeedback_binarized` — 60k+ prompt-based pairs, GPT-4-judged. The Zephyr default.
- `Anthropic/hh-rlhf` — the classical RLHF corpus; helpful + harmless splits.
- `stanfordnlp/SHP` — 385k Reddit-derived preferences; larger, noisier.
- `PKU-Alignment/PKU-SafeRLHF` — safety-focused preference pairs.
- `argilla/dpo-mix-7k` — 7k curated blend; small enough for fast iteration.
- `Intel/orca_dpo_pairs` — 12k GPT-4-judged pairs on top of Orca-2.

Load the raw dataset and record its size, columns, and a sample of 20 rows.

### Part B — Convert to a standard schema

Produce a `datasets.Dataset` (or `DatasetDict`) where every row has:

```python
{
  "prompt": str,           # or list of messages for multi-turn
  "chosen": str,           # or list of messages
  "rejected": str,         # or list of messages
  "meta": {                # optional but recommended
    "source": str,
    "chosen_len_tokens": int,
    "rejected_len_tokens": int,
    "prompt_len_tokens": int,
  }
}
```

If your source is already in this shape (UltraFeedback-Binarized is), just add the `meta` fields. If it is not (SHP's `history` / `human_ref_A` / `human_ref_B` shape, for instance), write a small `map(...)` transformation to convert.

Token lengths must be computed with the *actual* tokenizer you plan to use downstream — not `len(chosen.split())`. Load the tokenizer once and use it.

### Part C — Apply quality and length controls

Implement each filter as a separate `dataset.filter(...)` call so the row count before/after each is visible in your dataset card. At minimum:

1. **Length-delta filter.** Drop pairs where `chosen_len_tokens > 2 * rejected_len_tokens` or `chosen_len_tokens < 0.5 * rejected_len_tokens`. Log how many pairs this removed.
2. **Absolute length filter.** Drop pairs where any of `prompt`, `chosen`, `rejected` exceed a reasonable max (suggested: 1024 tokens for the response side, 512 for the prompt). Alternatively, keep long pairs but plan to truncate them at tokenize time — document your choice.
3. **Empty / near-empty filter.** Drop pairs where either `chosen` or `rejected` is fewer than 10 tokens. These are usually degenerate.
4. **Deduplication.** Dedup by prompt hash. Keep one pair per unique prompt (highest-confidence if the source provides that signal; otherwise the first).
5. **Non-refusal-vs-refusal filter (optional but recommended).** If your source contains many "chosen is a full answer, rejected is a refusal" pairs on prompts that clearly warrant answering, decide whether to keep them (helpful/harmless training signal) or drop them (avoids training the model to prefer answering over refusing when refusing would be right). Document the call.

After each filter, log a one-line summary: `filter=<name>, rows_before=N, rows_after=M`.

### Part D — Split and shuffle

Split the filtered dataset three ways:

- **`train`** — ~90% of the filtered rows. Used by exercises 02, 03, 04, 05 for training.
- **`eval_loss`** — ~5%. Used for training-time eval loss / RM accuracy.
- **`eval_benchmark`** — ~5%, held out entirely; never seen by any training run.

Split *before* any tokenization so the tokenized copies you cache later inherit the splits.

Set a fixed seed and dump the split assignment to `splits.json`. If someone downloads your dataset, they must be able to reproduce the exact same rows in each split.

### Part E — Chat-template sanity check

Load your intended downstream tokenizer (the one from your SFT checkpoint) and *apply the chat template* to 20 pairs, printing both chosen and rejected side by side. Verify:

1. Chosen and rejected have *identical* prompt sections (byte-for-byte match up to the response boundary).
2. Both include the assistant EOS / end-of-turn marker.
3. The prompt does not accidentally contain the assistant's response (a common bug where the message-list-to-string conversion glues them).

Save the 20-pair transcript to `chat_template_sample.txt` and commit it. This is your evidence that tokenization is correct — the RM and DPO trainers assume it is.

### Part F — The dataset card

Author `dataset_card.md` with the following sections:

**Provenance.**

- Source dataset(s) with Hub name + revision + license.
- Original data pipeline: who generated the responses, who / what judged them.

**Sizes.**

- Rows in raw source; after each filter; final per-split counts.

**Length statistics.**

- Median and 90th-percentile `prompt`, `chosen`, `rejected` token lengths.
- Chosen-vs-rejected length delta distribution (a small histogram, or 25th/50th/75th percentiles).
- Fraction of pairs where `chosen` is longer than `rejected` (target: closer to 50% than 70%).

**Label statistics.**

- Judge model (if LLM-judged) and prompt template used.
- Inter-annotator agreement (if available for the source).

**Filters applied.**

- Ordered list, each with row count before/after.

**Splits.**

- `train`, `eval_loss`, `eval_benchmark` sizes; the seed used.

**Decontamination.**

- Whether you checked prompt overlap against any downstream eval sets (MT-Bench, Chatbot Arena Hard, AlpacaEval, RewardBench). For a first-pass exercise, a note like "not decontaminated; suitable for pedagogical exercises only" is acceptable but must be explicit.

**License / use restrictions.**

- Inherit from source; note any restrictions on redistribution.

### Part G — Save the artifact

`dataset.save_to_disk("outputs/preference-set-v1/")`. Also produce `outputs/preference-set-v1/README.md` (which can be a symlink or copy of `dataset_card.md`) so anyone who later `load_from_disk` sees the card.

Directory shape:

```
outputs/preference-set-v1/
├── dataset_dict.json
├── train/
├── eval_loss/
├── eval_benchmark/
├── dataset_card.md
├── splits.json
├── chat_template_sample.txt
└── filter_log.txt         # one line per filter applied
```

## Starter guidance

- **Do not train on the raw source.** Every skipped filter is a specific bug you will chase in exercise 03 or 04 and blame on the trainer.
- **Hand-inspect ~50 pairs before you trust the labels.** For an LLM-judged source, spot-check that the "chosen" is genuinely better. If your agreement with the judge is below ~85%, either pick a different source or plan a stronger filter.
- **Length statistics tell the truth.** If your final chosen/rejected length ratio is still >1.5×, you did not filter enough. Length bias is measurable *before* training; catching it now saves a wasted PPO run.
- **A dataset card that lies is worse than no card.** If you skipped a step, say so. "No decontamination performed" is fine as long as it is written down.
- **Cache tokenizations later, not now.** This exercise produces the *raw* dataset. Exercises 02–05 will tokenize according to their per-trainer needs; do not pre-tokenize here.
- **If your GPU is tiny**, use `argilla/dpo-mix-7k` — 7000 pairs is enough to feel every step of the pipeline and small enough that filters run in seconds.

## Acceptance criteria

You have completed the exercise when:

- `outputs/preference-set-v1/` exists with the structure above.
- `dataset_card.md` fills every section listed in Part F.
- `filter_log.txt` shows the row count after each filter, matching the split sizes.
- `chat_template_sample.txt` shows 20 correctly-templated pairs; you can visually confirm chosen/rejected share a byte-identical prompt.
- `load_from_disk("outputs/preference-set-v1/")` on a fresh session returns the expected three splits.

## Stretch goals

- **Merge two sources.** Combine UltraFeedback-Binarized with a safety-focused split from PKU-SafeRLHF, apply your filters to the union, and re-do the dataset card. Document the mix ratio.
- **Length-balance the training split.** Downsample "chosen longer" pairs until they roughly equal "chosen shorter" pairs. Compare the length-delta histogram before/after in your card.
- **Cross-judge validation.** For an LLM-judged source, re-label a 500-pair sample with a *different* judge (Claude, GPT-4, an open judge like `PairRM`). Report per-pair agreement. Under 80% is a source-quality problem worth flagging.
- **Produce a KTO-format variant.** Convert each pair to two rows (`{prompt, completion, label: True}` for chosen; `{prompt, completion, label: False}` for rejected). Save as a sibling dataset `outputs/preference-set-v1-kto/`. Exercise 05 can consume this if you attempt KTO.
- **Decontaminate.** Hash prompts and compare against MT-Bench, Chatbot Arena Hard, and AlpacaEval prompt sets; drop overlaps; report the count. This is what a real production preference set requires.
