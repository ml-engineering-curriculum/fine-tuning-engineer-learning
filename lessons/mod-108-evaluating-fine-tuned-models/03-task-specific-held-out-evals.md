# Building Task-Specific Held-Out Evals

**Estimated read:** 35 minutes.
**You should be able to, by the end:** name at least three failure modes of relying only on academic benchmarks, design the schema for a task-specific eval dataset, size a held-out set for a given effect you care about, stratify by axes that matter for your product, and enforce the isolation between eval and training that stops your gold labels leaking back into the model.

## Why the academic suite alone will lie to you

Chapter 2's six-task suite tells you whether the model can answer MMLU questions, do grade-school math, code in Python, resist common falsehoods, and follow verifiable instruction formats. These are *proxies*. They correlate with general capability but do not measure whether your model does *your* job.

Four concrete ways this shows up:

1. **Narrow-task fine-tunes gain nothing on MMLU.** You fine-tune a 7B on 15,000 customer-support tickets. MMLU is unchanged (fine — it wasn't training for it). GSM8K may even drop a point (also fine — you traded some general capacity for specialization). The academic suite tells you the fine-tune didn't hurt broad capability. It tells you nothing about whether it helps support agents.
2. **Distribution drift.** A fine-tune trained on synthetic customer questions may score identically on the academic suite as one trained on real customer transcripts, and be dramatically worse on real transcripts. Only a task-specific eval catches that.
3. **Style / persona regressions.** Your product's assistant is supposed to end every response with "Is there anything else I can help you with?" A fine-tune that stops doing this scores fine on MMLU. It regresses in production the first day.
4. **Long-tail failures.** MMLU's 14k items give a good average. Your product has a handful of question categories that matter disproportionately (billing disputes, cancellation flows, PII-handling requests). MMLU cannot distinguish "average good" from "average good, but tanks on the five categories that account for 40% of tickets."

Every fine-tuning project needs at least one task-specific held-out eval. If you cannot state what task the model is fine-tuned for in one sentence, you cannot build one — and you should question whether the fine-tune is scoped correctly.

## What "task-specific held-out eval" means precisely

Three properties, in order:

1. **Task-specific.** The eval items are examples of the actual task the model is trained to do. If the task is summarizing support tickets, items are (ticket, gold summary) pairs. If the task is answering billing questions, items are (customer question, gold answer) pairs. If the task is code generation in a proprietary DSL, items are (spec, gold code, test suite) tuples.
2. **Held-out.** The eval items are *not* in the training set. Not the exact items, not paraphrases of them, not the source documents they came from. Chapter 5 covers how to enforce this; the short version is that this must be true at the file / source level, not just at the row level.
3. **Eval, not test-and-move-on.** The dataset is stable across releases. You compare adapter N against adapter N−1 on the *same* eval. If you rebuild the eval every release, you have no basis for regression comparison.

## The dataset schema

Every task-specific eval item needs at least these fields. Missing any one of them costs you either interpretability, reproducibility, or the ability to slice results.

```json
{
  "id": "eval_2026_08_00042",
  "input": {
    "system": "You are a customer-support assistant for Acme.",
    "user": "My account was charged twice yesterday for order #4471."
  },
  "gold": {
    "type": "reference",
    "value": "Confirm the double-charge, explain the ~5-day refund window, provide the refund case number."
  },
  "metadata": {
    "stratum": "billing/dispute",
    "difficulty": "medium",
    "source": "prod_traffic_2026_07",
    "labeler": "alice",
    "labeled_at": "2026-07-14",
    "notes": "user was mildly upset; escalation-language check applies",
    "requires_tools": ["get_order", "refund_case"]
  },
  "grading": {
    "metric": "rubric_llm_judge",
    "rubric_id": "billing_dispute_v3",
    "min_pass": 3.5
  }
}
```

Field-by-field, and what it costs you if you skip it:

- **`id`** — stable across releases. Without it, you cannot compute per-item diffs (chapter 6). Do not reuse ids for new items; append a version if you must edit an item.
- **`input`** — exactly what gets fed to the model, in whatever structure your chat template expects. If the model needs a system prompt, put it here; do not assume the eval harness injects one.
- **`gold`** — a *reference* answer or a *check* that determines correctness. Types below.
- **`metadata.stratum`** — the axis you'll slice on. Chapter 6 uses this to compute per-slice regression. Without stratification you get a single average that hides everything.
- **`metadata.source`** — where the item came from. Critical for the contamination pipeline in chapter 5.
- **`metadata.difficulty`** — coarse (`easy/medium/hard`) is enough. Lets you separate "we regressed on the easy stuff, model is broken" from "we regressed on the hard stuff, this is a scoping call."
- **`metadata.labeler` / `labeled_at`** — provenance. If the eval has a bug, you need to know who to ask.
- **`grading`** — the metric that turns a model response into a per-item score. Not always a single number; see below.

Store the eval as JSONL. One item per line. Version the file itself (Git or DVC or S3 with an object hash). The eval is a data artifact; treat it with the same discipline as the training data (mod-102 chapter 7).

## The five gold-answer types

The `gold.type` field is the interesting design choice. Which type you pick determines what the eval can and cannot measure.

### 1. `exact_match` — the input has a single correct answer

For classification, extraction, closed-form Q&A. Item's `gold.value` is the expected string; score is 1 if the model's response equals it (after canonicalization: strip, lowercase, remove punctuation), 0 otherwise.

- **When to use:** classification tasks, structured extraction, math with a numeric answer, code with a canonical output.
- **When it fails:** any task where the model can be right in multiple wordings.
- **Cost:** cheap to build, cheap to run.

### 2. `regex` / `verifier` — the input has multiple correct forms

`gold.value` is a regex or a small function; a response passes if it matches / is accepted. IFEval is a large-scale example of this — its instructions are verifiable by rule (e.g., "your answer must contain the word 'sustainability' three times").

- **When to use:** format checks, containment checks, structured outputs (JSON with a required schema).
- **When it fails:** open-ended quality — you cannot regex "was this a good summary."
- **Cost:** cheap to build if you know the rules, expensive to keep the rules accurate as the task evolves.

### 3. `unit_test` — the input has a runnable specification

For code generation. `gold.value` is a suite of unit tests; a response passes if the generated code passes them. HumanEval is the canonical academic example.

- **When to use:** code generation, tool-use with executable checks, structured outputs with a schema you can validate.
- **When it fails:** the tests don't cover the spec — model produces trivially wrong code that happens to pass — or the spec is under-specified.
- **Cost:** high (writing good tests is real engineering) but the highest-fidelity eval you can get for code.

### 4. `rubric_llm_judge` — the answer is open-ended but scoreable

`gold.value` is either empty or a reference answer; scoring is a call to a judge model with a rubric. Chapter 4 is entirely about doing this correctly.

- **When to use:** writing, summarization, chat, roleplay, reasoning explanations — anything you cannot write a regex or a test for.
- **When it fails:** noisy at small sample sizes; expensive per item; biased if the judge is not calibrated.
- **Cost:** high per item ($0.02–$0.20 per judge call at current API prices), moderate to build the rubric.

### 5. `human_only` — the answer needs a person

For the highest-stakes evals — safety-adjacent, brand-tone-critical, novel task categories where no proxy is trusted. Score is a human rating on a fixed rubric.

- **When to use:** anchor / calibration sets for the LLM-as-judge rubric (chapter 4); final go/no-go items on regulated-use models; things a judge model will systematically get wrong.
- **When it fails:** hard to scale, slow, expensive, subject to rater drift.
- **Cost:** highest. Reserve for 50–200 anchor items.

## Sizing a held-out set

The bigger question of "how many items do I need" reduces to "what effect do I want to detect with what confidence." Two ways to think about it:

### The confidence-interval rule of thumb

For a binary metric (accuracy, exact-match, pass/fail), the 95% CI half-width for `N` items with observed rate `p` is roughly `1.96 · sqrt(p(1-p)/N)`. Solving for common targets:

| Half-width you want | Items you need (worst case `p=0.5`) |
|---|---|
| ±5 percentage points | ~385 |
| ±3 pp | ~1,067 |
| ±2 pp | ~2,400 |
| ±1 pp | ~9,600 |

For a rubric-graded (1–10 or 1–5) metric, the equivalent calculation uses the standard deviation of the score. In practice ~200–500 items is where most task-specific evals live: enough to detect ~3–5 pp movements, small enough to be affordable per release.

If you have a very stringent gate ("no regression greater than 1 pp"), you either need a much larger eval or you need to accept that the gate cannot fire on differences that small. Chapter 6 discusses the McNemar test for paired comparisons, which lets you detect smaller effects than the marginal CI implies because it uses per-item pairing.

### The stratum-coverage rule of thumb

Total size matters less than *per-stratum* size. If the aggregate is 500 items but the "billing/dispute" stratum has 12 of them, a 20% drop on that stratum is not statistically distinguishable from noise. Target:

- **≥30 items per stratum** for a stratum you want to report separately with any confidence.
- **≥100 items per stratum** for a stratum you want to gate on.
- **Any stratum with <30** — report it as a spotlight (list the individual items and their scores) rather than an aggregate.

### The "how many hours of labeling" rule of thumb

A labeler can produce roughly 20–60 gold labels per hour for a reference-answer task; 5–15 per hour for a rubric-labeled anchor set. Working backwards from a labeling budget:

- 1 person-day (8 hours) ≈ 160–480 reference labels or 40–120 rubric labels.
- 1 person-week ≈ 800–2400 reference labels or 200–600 rubric labels.

Most task-specific evals are built in 1–3 person-days by the fine-tuning engineer plus one product / domain expert. The domain expert's time is the binding constraint; treat it accordingly.

## Stratification — the axis question

Aggregate metrics hide the failures that matter most. Every task-specific eval should be stratified along at least these axes; add more as needed for your product:

1. **Intent / topic.** For a support model: billing, technical, cancellation, account-management, general-info. For a summarization model: type of document. For a code model: language, domain, difficulty. Rule of thumb: ~5–15 intent strata; each with ≥30 items.
2. **Difficulty.** Easy / medium / hard, judged by the labeler at label time. Not the model's confidence — the human's assessment of task difficulty.
3. **Input length.** Short / medium / long. LLMs regress differently on long inputs; you want to see that.
4. **User-visible severity.** Which items, if wrong, cause a user-facing incident? Which cost you brand? A separate `critical` stratum with ~20–50 items that gate the release regardless of aggregate.
5. **Time.** When was the item generated / labeled? If the eval is drawn from production traffic, stratifying by month lets you spot regression on newer traffic patterns.

The eval report (chapter 6) shows the aggregate *and* the per-stratum table. Aggregates alone are how "the model looks fine but users complain about billing" ships.

## Building the eval from production traffic

The best task-specific eval mirrors production. If you have production traffic, the recipe is:

1. **Sample.** Pull 2–5× your target size of anonymized production inputs (say 1,000–2,500 items for a 500-item eval). Random sampling is fine; stratified sampling by intent gets you coverage faster.
2. **Filter.** Remove items with PII you cannot redact, items that hit safety flags (they belong in the safety eval, not the general held-out), items with malformed inputs, and duplicates.
3. **Label.** A domain expert writes the gold answer or scores the model's answer. For rubric evals, the expert writes the rubric first and then labels a calibration set.
4. **Split.** Divide into three subsets:
   - `holdout_gate` — the items the release gate runs on every release. ~200–500 items.
   - `holdout_dev` — the items *you* look at during iteration. ~50–100 items. Because you look at them during dev, they can be contaminated by your prompt engineering; do not use them to gate.
   - `anchor` — the small human-labeled set used to calibrate the LLM judge (chapter 4). ~50–100 items, drawn to cover the full quality spectrum.
5. **Version.** Store the JSONL with a version string. When you add new items, bump the version and note the addition in the changelog. Never edit items in place (change breaks per-item comparability).
6. **Firewall from training.** Add the eval items' IDs (and, for text-based items, an n-gram signature) to the dataset-engineering pipeline's exclusion list from mod-102. Chapter 5 revisits this at eval time — the check must happen at both build time and eval time.

## Building the eval when you have no production traffic

Cold-start case: you are fine-tuning a model for a use case that has no production traffic yet. Options in decreasing quality order:

1. **Domain-expert-authored items.** A specialist writes 300–500 realistic items from scratch. Expensive but the eval is trustworthy.
2. **Existing public eval + adaptation.** Take a public dataset (e.g., PubMedQA for medical Q&A, LegalBench subsets for legal, MTEB for retrieval) and use the subset closest to your use case as the held-out. Cheap; the tradeoff is your production distribution may differ.
3. **Synthetic + human review.** Use a strong LLM to generate candidate items, then have a domain expert accept / edit / reject each one. Cheaper than fully human-authored, higher-fidelity than raw synthetic. The mod-102 chapter 5 pipeline applies; the ratio is usually 3 synthetic items in : 1 kept out.
4. **Pure synthetic.** A judge-model generates both the item and the gold answer. Fastest, lowest fidelity — a lower bound on quality. Useful as a scaffold before you have real production traffic.

Whatever you build cold-start, plan to *replace* it with production-traffic-derived items once you have three months of production traffic. Cold-start evals are a placeholder for a real eval, not a permanent artifact.

## The isolation contract

The single most important property of a held-out eval is that it *stays held out*. Three ways this contract breaks and how to prevent each:

### 1. Direct leakage — training on the eval items

The model was fine-tuned on data that included the eval items. Standard mistake: pulling both training and eval from the same production dump and forgetting to deduplicate.

Prevention: at dataset-build time (mod-102 chapter 3), the SFT pipeline takes the eval's item IDs as an explicit exclusion list. Fail the build if any excluded ID appears in training. This is a build-time invariant, not a run-time hope.

### 2. Indirect leakage — training on paraphrases of the eval items

The eval items were generated from source documents; the training set includes those source documents. The model has seen the underlying content and can regurgitate it even though the exact item text is not in training.

Prevention: exclusion at the *source-document* level, not just the item level. If the eval item was labeled from ticket #4471, ticket #4471 does not appear anywhere in training. Track the source-document ID in `metadata.source`.

### 3. Drift leakage — the eval gets edited to include things the model handles well

You look at eval items your model gets wrong, decide some of them are "unfair" or "poorly worded," and edit them out. Over time the eval drifts toward items the model handles, and the score creeps up without the model actually improving.

Prevention: **the eval is versioned and additive-only.** If you must remove or edit an item, bump the version and record the reason. Compare adapter N against adapter N−1 on the *same version* of the eval. When the eval version changes, run both adapters against the new version to establish a new baseline.

## Metrics beyond accuracy

For open-ended tasks, accuracy is a poor primary metric. A short list of alternatives, keyed to task shape:

- **Rubric-graded score (1–5 or 1–10).** From LLM-as-judge (chapter 4) or human. Report mean, median, and per-stratum breakdown. Report agreement metrics (chapter 4).
- **Pairwise win rate vs. reference.** For each item, the judge is shown model A's answer and model B's (base or previous adapter's) answer and picks a winner. Report win / lose / tie rates.
- **BLEU / ROUGE / METEOR.** Old-school n-gram-overlap metrics. Fast and cheap, moderately correlated with human quality on translation and summarization, poor for open chat. Include if you already report them, but do not gate on them for chat.
- **BERTScore / MoverScore / BLEURT.** Learned similarity metrics. Better than n-gram, still not human-level.
- **Task-specific metrics.** If your task has a real-world metric (Levenshtein distance on structured-extraction output, JSON schema validity rate, code compilability rate, tool-call success rate), include it.

Rule of thumb: the release gate reports at least two metrics per task-specific eval — one rubric-graded (or human) and one automated (regex / test / structural). If both agree, the signal is solid. If they disagree, chapter 4 or 6 explains why.

## Refreshing the eval

Task-specific evals decay. Production traffic drifts, product features change, user behavior evolves. Plan for refresh:

- **Additive refresh — quarterly cadence.** Add 20–50 new items per quarter drawn from recent production traffic. This does not break comparability with prior adapters, because the eval is additive.
- **Full refresh — annually or on major product change.** Retire old items that no longer reflect production (`deprecated_in_version: 5`); do not delete — keep them in the JSONL for historical replay.
- **Break-glass refresh — after a production incident.** Add the specific items that caused the incident to the `holdout_gate` set. Now the gate catches that class of failure on future adapters.

Each refresh is a version bump. Chapter 6 discusses running the release gate against the new version and reporting both the new-version baseline and the old-version continuation.

## Exercise pointer

`exercises/exercise-02-task-specific-held-out-eval-design.md` walks the full build: pick a use case, design the schema, build 300–500 items with stratification and gold labels, isolate them from your training data, and produce the report a peer reviewer would sign off on.

## Summary

- Academic benchmarks (chapter 2) tell you the model is not broken on general capability; they do not tell you it does *your* task. Every fine-tune needs at least one task-specific held-out eval.
- The eval item schema needs `id`, `input`, `gold`, and metadata (stratum, source, labeler, difficulty). Missing any of these forecloses a downstream question you will want to answer.
- Five gold-answer types: `exact_match`, `regex/verifier`, `unit_test`, `rubric_llm_judge`, `human_only`. Choose per task; most fine-tunes use two or three.
- Size for the smallest effect you want to detect: ~385 items for ±5 pp binary CI; ~200–500 items is the practical sweet spot. Stratify with ≥30 items per stratum you report and ≥100 for strata you gate on.
- Build from production traffic when you can; cold-start with domain-expert authorship + synthetic generation + human review when you cannot.
- Enforce isolation at build time: the eval's item IDs and source documents are on the SFT pipeline's exclusion list; the eval is additive-only across versions.
- Report at least two metrics per task-specific eval — one rubric-graded and one automated — so agreement / disagreement is itself a signal.
- Plan for refresh: additive quarterly, full annually, break-glass after every incident.
