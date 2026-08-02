# Detecting Benchmark Contamination at Eval Time

**Estimated read:** 35 minutes.
**You should be able to, by the end:** name the three ways contamination enters a fine-tuned model, run an n-gram overlap scan against your training data, run at least one model-signal contamination test (canonical-continuation, membership-inference, or guided prompting) against a fine-tuned model, and write the contamination-disclosure section that attaches to the model card.

## Why this chapter exists on top of mod-102 chapter 3

Mod-102 chapter 3 covered decontamination *at dataset-build time*: n-gram overlap between your SFT set and MMLU / HumanEval / GSM8K / MT-Bench, with a report of flagged rows and a decontam-recipe writeup. That is the primary defense and remains the highest-leverage place to catch contamination.

This chapter exists because that defense is not sufficient on its own. Three ways contamination reaches a fine-tuned model *despite* mod-102's build-time scan:

1. **The base model was contaminated.** MMLU, GSM8K, HumanEval, and most academic benchmarks are on the public web; every base model pretrained on Common Crawl or a derived corpus has seen fragments of them. Meta, Mistral, and Qwen model cards explicitly acknowledge this. Your SFT on top of that base cannot uncontaminate the base — but the reported score is still inflated by base-model memorization.
2. **A new eval was added.** Your build-time pipeline scanned against evals A, B, C. You now want to report on eval D, which was not scanned. Rebuilding the training set is not always feasible; scanning at eval time gives you the disclosure.
3. **The training data grew and the scan was not re-run.** Someone added a new synthetic-data source between decontam and the training run. Or added a public dataset. Or bumped a mixed-in corpus's version.

Eval-time contamination detection is the safety net. It does not replace mod-102's build-time scan; it catches what escaped it.

## The three lenses of contamination

Contamination detection at eval time uses three complementary lenses. No one lens catches everything.

### Lens 1 — Text-overlap contamination (still)

The n-gram overlap technique from mod-102 chapter 3 applied at eval time: compare the eval items against the training set and report matches. Everything from that chapter still applies:

- Word 8-grams for a conservative scan, 13-grams for a strict-near-verbatim scan.
- Scan both the prompt and the response of training rows against the eval prompt and (where relevant) gold answer.
- Report per-eval flag counts, hand-sampled false-positive rate, and the n and threshold used.

The addition at eval time is that you now have a *model* to interrogate as well, which the next two lenses use.

### Lens 2 — Semantic-overlap contamination

N-gram scans miss paraphrases. Semantic-overlap scans catch them. Embed every eval item and every training row (or a random sample of it if the training set is very large) with a sentence encoder, index the training set in a vector DB, and for each eval item look up its `K` nearest neighbors. Any eval item with a training neighbor above a similarity threshold is a candidate paraphrase.

Practical setup:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

eval_embs = encoder.encode([item["prompt"] for item in eval_items], normalize_embeddings=True)
train_embs = encoder.encode([row["prompt"] + " " + row["response"] for row in train_rows], normalize_embeddings=True)

# Cosine similarity via inner product (embeddings are L2-normalized).
sims = eval_embs @ train_embs.T                     # (E, T)
top_k_idx = np.argsort(-sims, axis=1)[:, :5]
top_k_sim = np.take_along_axis(sims, top_k_idx, axis=1)

for i, item in enumerate(eval_items):
    for rank, (t_idx, s) in enumerate(zip(top_k_idx[i], top_k_sim[i])):
        if s > 0.85:                                # tune per encoder + task
            print(f"EVAL {item['id']} sim={s:.3f} -> TRAIN {train_rows[t_idx]['id']}")
```

Thresholds to expect with a decent encoder (`all-mpnet-base-v2`, `bge-large`, `e5-large`, etc.):

- **>0.95** — near-duplicate. Almost always a real match.
- **0.85–0.95** — likely paraphrase. Hand-check the top 100.
- **0.70–0.85** — same topic, not necessarily contamination. Report the rate but not the individual items.

At scale (millions of training rows), use FAISS or a hosted vector DB; for the 5k–500k rows that most SFT sets sit at, a NumPy inner product like the snippet above is fine.

### Lens 3 — Model-signal contamination

The most powerful lens: interrogate the *model* directly to see if it behaves as if it has seen the eval. Three published techniques, in roughly increasing invasiveness:

#### 3a. Canonical-continuation / verbatim-completion probe

For each eval item, feed the model a *prefix* (say, the first 60% of the item's stem) and see whether it produces the exact remaining 40% verbatim. If it does — for many items — the model has memorized the eval.

The recipe (adapted from the OpenAI GPT-4 tech report's contamination check and Golchin & Surdeanu, 2023):

```python
def canonical_completion_probe(model, tokenizer, item, prefix_frac=0.6):
    tokens = tokenizer.encode(item["prompt"])
    cut = int(len(tokens) * prefix_frac)
    prefix, suffix = tokens[:cut], tokens[cut:]
    prompt_text = tokenizer.decode(prefix, skip_special_tokens=True)

    out = model.generate(
        tokenizer(prompt_text, return_tensors="pt").input_ids.cuda(),
        max_new_tokens=len(suffix) + 20,
        do_sample=False,
    )
    completion = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt_text):]

    gold_text = tokenizer.decode(suffix, skip_special_tokens=True)
    return {
        "id": item["id"],
        "gold_suffix": gold_text,
        "completion": completion,
        "prefix_match": completion.strip().startswith(gold_text.strip()),
        "exact_ratio": _longest_common_prefix_ratio(completion, gold_text),
    }
```

Report:

- Fraction of items with `prefix_match: true` (near-exact continuation).
- Distribution of `exact_ratio` (how much of the suffix the model got verbatim on average).

A clean model produces the *content* of the suffix, not the exact tokens. A contaminated model produces the exact tokens for a suspicious fraction of items. Baseline expectations vary by base model — always run the same probe against the *base* model and against a known-clean control (an item you wrote yourself that is definitely not in any corpus) so you have a reference distribution.

#### 3b. Guided-prompting membership probe (Golchin & Surdeanu 2024, "Time-Travel in LLMs")

Prompt the model with a metadata cue that only makes sense if it has seen the eval:

```
The following is a question from the MMLU dataset, subject "college_biology",
split "test", question 42:
```

For a clean model, this cue provides no useful information — the model has to attempt the question. For a contaminated model, the cue increases the probability the model produces the exact gold answer (or the exact question text).

Score: compare the model's accuracy or verbatim-match rate on the eval *with* vs. *without* the guiding metadata prefix. A large gap indicates the model has learned the metadata-to-content mapping, which is only possible if it saw the labeled data during training.

#### 3c. Membership-inference probe (Oren et al. 2023, "Proving Test Set Contamination in Black-Box Language Models")

The strongest of the three, and the only one with formal statistical guarantees. The intuition:

- The order of items in a benchmark's test file is arbitrary but fixed.
- A model that was trained on the file has (weakly) memorized the order — so the model prefers the *canonical* ordering of items when scoring the joint likelihood of the whole file, compared to a random permutation of the same items.
- Compare the model's log-likelihood of the canonical order against many shuffled orderings; a statistically significant preference for the canonical order is evidence of contamination.

This is powerful because it works even when the model does not verbatim-emit the items — the statistical fingerprint of training-set membership shows up in relative log-probabilities. Reference implementations exist for `lm-evaluation-harness` and some open evaluation stacks.

<!-- needs-research: verify whether a maintained implementation of Oren et al. lives inside the current lm-evaluation-harness release; the community has produced several and they diverge in detail. -->

## Contamination severity: what to do with a hit

Not every contamination signal is equal. A pragmatic triage rubric:

| Signal | Severity | Action |
|---|---|---|
| ≥5% of an eval's items appear verbatim in training text | Critical | Retrain without those items; do not report the eval number |
| 1–5% of items are paraphrases (embedding sim ≥ 0.85) | High | Disclose in the model card; report both the "as-is" and "clean subset" numbers |
| <1% items are paraphrases, or matches are on boilerplate | Low | Disclose in the model card; report the "as-is" number |
| Canonical-continuation probe hits a suspicious fraction on the base model | Inherited | Disclose that the base is contaminated on this eval; report as-is |
| Membership-inference probe rejects null at p < 0.01 | Confirmed contamination somewhere | Investigate; if the fine-tune added it, retrain; if the base has it, disclose |
| No signal on any of the three lenses | Clean | Report as-is with the disclosure "no contamination detected under X/Y/Z scans" |

The rule: **if a contamination signal is real and unfixable, report the eval on the "clean subset" of items that did not trip the signal, in addition to (never instead of) the full-set number.** A "clean subset" number with a disclosed methodology is more trustworthy than a "full set" number without disclosure, and letting a reader compare the two is the best defense against the accusation later.

## Fixing contamination at the source

When contamination is detected in the *fine-tune* (not just inherited from the base), the fix is upstream, in the SFT dataset:

1. **Identify the leaking training rows.** The overlap scans give you the row IDs.
2. **Trace the source.** Where did the row come from — a scraped ShareGPT dump, a Self-Instruct pipeline, a public dataset? Fix the source, not just the rows.
3. **Update the dataset build.** Add the eval's items and item-hashes to mod-102's decontam pipeline's exclusion list. Update the source filter to drop the leaking source or add a decontam step to it.
4. **Rebuild the training set.** With the new exclusion list. Verify the scan against the eval now returns zero hits.
5. **Retrain.** No shortcut. The contaminated fine-tune's number is invalid; there is no post-hoc correction.
6. **Update the decontam report.** The model card's decontam section names the eval, the exclusion methodology, and the residual overlap after the fix.

If the contamination is *inherited from the base* and the base is not something you retrained, the fix is disclosure, not retraining — but the model card must acknowledge it.

## The disclosure section

Every fine-tuned model that reports academic eval numbers owes the reader a contamination disclosure. Minimum contents, adapted from the Llama 2 / Llama 3 / Mistral / Qwen model-card conventions:

```markdown
## Data contamination

We scanned the SFT training set (~N rows) against the following evals for
contamination:

| Eval  | Overlap method     | Threshold | Flagged rows | Items affected | Action |
|-------|--------------------|-----------|--------------|----------------|--------|
| MMLU  | 13-gram + emb 0.85 | any hit   |  1,204       |  312 / 14,042 | Removed all flagged rows and retrained. Residual: 0 hits. |
| GSM8K | 13-gram + emb 0.85 | any hit   |    42        |   17 / 1,319  | Removed. Residual: 0 hits. |
| ...   | ...                | ...       | ...          | ...           | ... |

We also ran two model-signal probes against the fine-tuned model:

- Canonical-continuation probe (Golchin & Surdeanu 2023): 3.1% of MMLU items
  produced ≥90% suffix match on our fine-tuned model, vs. 2.9% on the base
  model. We attribute the residual to base-model contamination.
- Membership-inference probe (Oren et al. 2023): p = 0.14 on MMLU, p = 0.03 on
  GSM8K. We regard GSM8K as borderline-contaminated; the "clean subset"
  number (excluding items with any training-set n-gram overlap) is reported
  alongside the "as-is" number in the results table.

Base-model contamination is a known artifact of pretraining on Common Crawl
and is not something we can retract post-hoc; the fine-tune inherits it.
```

Once you have written this section, keep it up to date. Every new eval, every new dataset source, every base-model change is a reason to re-run the scans and update the report.

## What the scans cost

Rough cost per eval per SFT set (200k rows against a 15k-item eval):

| Scan | Wall-clock | Compute cost | Notes |
|---|---|---|---|
| N-gram (8- and 13-gram) | 5–20 min | 1 CPU | Trivial; can run on a laptop |
| Embedding (mpnet, cosine) | 20–60 min | 1 GPU | Dominated by embedding the training set once |
| Canonical-continuation probe | 30–90 min | 1 GPU | 15k generation calls |
| Guided-prompting probe | 30–90 min | 1 GPU | 15k × 2 generation calls |
| Membership-inference probe | 60–180 min | 1 GPU | Many log-likelihood passes per permutation |

The full stack costs 3–8 GPU-hours per eval per release-candidate model. Cache embeddings and rerun only the lenses affected by whatever changed (new training data → all lenses; new eval → just the new eval); the release gate can afford to run the fast lenses (n-gram, embedding) every commit and the model-signal lenses only at release-candidate boundaries.

## Common pitfalls

Six things that catch fine-tuning engineers running contamination scans:

1. **Only scanning the SFT set, not the pretraining set.** The base model's contamination is inherited. You cannot fix it; you must disclose it. Run the model-signal probes to detect it.
2. **Scanning only against English text of the eval.** MMLU has language-specific subject areas; TruthfulQA has been translated; some evals have multilingual splits. Scan against the eval variant you actually report.
3. **Trusting embedding similarity thresholds from one encoder for another.** `0.85` on `all-mpnet-base-v2` is not the same threshold as `0.85` on `bge-large`. Calibrate on your specific encoder by hand-labeling 100 pairs across the similarity range.
4. **Reporting only the aggregate hit count.** "42 flagged rows" tells the reader nothing about severity. Include example matches (5–10 per eval) with the eval item and the training row side by side.
5. **Removing flagged rows without checking why they matched.** Sometimes the overlap is spurious (boilerplate license header, common phrase, both texts quote the same public source). Removing without hand-checking either over-removes or reveals a systematic bug in the scanner.
6. **Not versioning the contamination report.** Every model release has its own contamination report, with the SFT set version and eval versions it was run against. Do not overwrite the report for the previous release; store both.

## When contamination discovery is a security-relevant event

If your model is being used in a regulated context (medical, legal, hiring, credit), a discovered contamination signal is not just a quality issue — it is a compliance issue. The disclosure section becomes part of your regulatory paper trail. Two additional steps in that setting:

- **Preserve the exact training-set snapshot and eval-set snapshot used** at contamination-discovery time. If audited, you need to reproduce the finding.
- **Notify downstream consumers of the model** (product, safety, legal) rather than silently fixing. A "we tightened decontam" changelog note is not sufficient for a real contamination event.

The rest of the compliance story lives in mod-102's data provenance chapter and in mod-111's governance discussion; this chapter's job is to detect the event and produce the audit-quality report.

## Exercise pointer

`exercises/exercise-04-contamination-detection-and-mitigation.md` walks you through running n-gram, embedding, and canonical-continuation scans against a real SFT set + eval suite, hand-sampling the false-positive rate, and writing the model-card disclosure section.

## Summary

- Build-time decontamination (mod-102 ch. 3) is the primary defense; eval-time detection is the safety net for base-model contamination, new evals, and pipeline drift.
- Three lenses catch different classes of contamination: **text-overlap** (n-gram, exact match), **semantic-overlap** (embedding similarity for paraphrases), **model-signal** (canonical-continuation, guided-prompting, membership-inference — interrogating the model directly).
- Model-signal probes distinguish "the model saw this" from "the model can answer this." Membership inference (Oren et al. 2023) is the strongest, with formal statistical guarantees; canonical-continuation is the easiest to implement and interpret.
- Triage by severity: critical (≥5% verbatim) triggers retrain; high (1–5% paraphrase) triggers disclosure + clean-subset number; low triggers disclosure only; inherited-from-base triggers disclosure and acknowledgement.
- Fix contamination at the source (SFT pipeline exclusion list), not by post-hoc score correction. Retrain when contamination was introduced by the fine-tune.
- Every reported eval owes a contamination-disclosure section on the model card: methods used, thresholds, flagged rows, action taken, residual. Version the report per model release.
- Scan costs ~3–8 GPU-hours per eval per release candidate. Fast lanes (n-gram, embedding) can run per-commit; model-signal lanes at release-candidate boundaries.
