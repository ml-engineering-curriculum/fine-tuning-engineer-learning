# exercise-02: Task-Specific Held-Out Eval Design

**Estimated effort:** 3 hours
**Companion chapters:** [`01-evaluation-landscape.md`](../01-evaluation-landscape.md), [`03-task-specific-held-out-evals.md`](../03-task-specific-held-out-evals.md), [`04-llm-as-judge.md`](../04-llm-as-judge.md)

## Objective

Design and build a **300–500-item task-specific held-out evaluation dataset** for a named production use case. Produce the JSONL, the schema, the stratification plan with per-stratum sizing justification, the metric definitions, and the isolation contract that prevents leakage into training. The exercise is largely a design and labeling exercise — you will run the eval in exercise 03 (with LLM-as-judge) and exercise 05 (as part of the gate).

The point is to build an eval that is *worth gating on* — one whose per-item scores you would defend in a release review because you designed the strata deliberately, sized each one deliberately, and can point to isolation between eval and training.

## Prerequisites

- Pick a task. Options — pick the closest to your fine-tune or invent your own:
  1. **Customer-support assistant.** Answer support questions with a specific tone and escalation policy.
  2. **Code review agent.** Given a PR diff, produce a review comment identifying issues.
  3. **Structured extraction.** Given a document (invoice, contract, email), extract fields into JSON.
  4. **Domain-specific summarization.** Summarize technical documents into an executive brief.
  5. **Your own use case.** If you have real production traffic or a real fine-tune from prior modules, use that.
- A source of items — real production traffic if available, or a public dataset you can adapt, or willingness to author from scratch.
- 2–3 hours of your own time as the "domain expert" for labeling. If you have a real domain expert, an hour of their time is a huge upgrade.

## Requirements

### Part A — Write the eval spec

Before writing any items, write `eval_spec.md`:

1. **Task statement.** One sentence: what does the model do?
2. **Consumers.** Who reads the eval numbers? What decisions do they make from them?
3. **Success looks like.** What score, on what metric, on what stratum, would you consider "ready to deploy"?
4. **Strata.** List the 5–10 strata you'll split items across. For each, a one-line rule for what belongs and an approximate target count.
5. **Metrics.** For each metric you'll compute, name it, define it precisely, and state whether it's automated (regex / test / structural) or judged (rubric / human).
6. **Isolation contract.** How you'll enforce that no eval item ends up in training. Source-level, item-level, or both. Reference mod-102 chapter 3 patterns.

The spec is a *design doc* — a peer reviewer should be able to read it and know exactly what you're building before you build it. Aim for ~1 page.

### Part B — Build the dataset

Produce `holdout.jsonl`, one item per line, following the schema below (adapted from ch. 3):

```json
{
  "id": "eval-2026-08-0001",
  "input": {
    "system": "…",              // if applicable
    "user": "…",
    "context": {"...": "..."}    // any tool outputs, retrieved docs, etc.
  },
  "gold": {
    "type": "exact_match" | "regex" | "unit_test" | "rubric_llm_judge" | "human_only",
    "value": "…"                 // reference answer, regex, tests, or empty for rubric-judged
  },
  "metadata": {
    "stratum": "billing/dispute",
    "difficulty": "easy|medium|hard",
    "source": "prod_2026_07 | authored | adapted_from_dataset:pubmedqa_v1 | synthetic_reviewed",
    "labeler": "your-name",
    "labeled_at": "YYYY-MM-DD",
    "notes": "any relevant context",
    "requires_tools": []          // if the task uses tool calls
  },
  "grading": {
    "metric": "…",
    "rubric_id": "…",             // for rubric-judged items
    "min_pass": 3.5                // if the metric has a pass threshold
  }
}
```

Sizing requirements:

- **300–500 items total.**
- **≥30 items per stratum you plan to report on.**
- **≥50 items per stratum you plan to *gate* on.**
- **≥10 items per stratum in the "critical" category** (items whose failure is a user-facing incident).

At least three gold-answer types should be represented — do not build an eval that is 100% `rubric_llm_judge`. Include at least some `exact_match` or `regex` or `unit_test` items that give you a cheap automated ground truth.

### Part C — Build the anchor set

Pull 50–100 items from `holdout.jsonl` (across the stratum and difficulty distribution) into `anchor.jsonl`. For each anchor item:

- Label the reference answer yourself (or have a human label it).
- On a rubric-judged item, also produce a human rubric score.
- Include a `notes.human` field explaining the rating.

The anchor set is what exercise 03 uses to calibrate the LLM judge. Aim for coverage of the score range (some clearly-great, some clearly-bad, some borderline).

### Part D — Split into gate / dev / anchor

Produce three files:

- `holdout_gate.jsonl` — the release-gate items. ~250–400 items. Do not look at these during model iteration; that would leak them.
- `holdout_dev.jsonl` — iteration items. ~50–100 items. You can look at these during dev; that means you cannot gate on them.
- `anchor.jsonl` — human-labeled anchor from Part C. ~50–100 items.

An item can appear in `anchor.jsonl` and *also* in `holdout_gate.jsonl` (the anchor is a labeled subset). An item should not appear in both `dev` and `gate`.

### Part E — Prove isolation from training

Write `isolation_check.py` that:

1. Reads `holdout_gate.jsonl` and extracts item IDs + 13-word n-grams of every item's `input`.
2. Reads your training set (from mod-102 / mod-103).
3. For every training row, checks:
   - Whether its metadata source overlaps with the eval's source (per the `source` field).
   - Whether any of its 13-grams overlap with any eval 13-gram.
4. Emits `isolation_report.md`:
   - Total training rows scanned.
   - Rows flagged by source match; rows flagged by n-gram overlap.
   - The 10 highest-overlap examples for review.
   - The final action: if any flagged, what to do (remove from training and rebuild; disclose and note in the eval report; etc.).

If flagged rows exist, either:

- Remove them and rebuild your training set (best), or
- Add the eval's item IDs and n-grams to the training pipeline's exclusion list going forward, then discard the current fine-tune from future eval consideration until a clean rebuild is done.

Do not proceed to exercise 03 with unresolved leakage.

### Part F — Write the design report

`design_report.md`, structured as:

1. **The task.** One paragraph summarizing what the model is being evaluated for.
2. **Stratification.** The table of strata with per-stratum counts and rationale.
3. **Gold-answer types.** For each type used, why you used it and where.
4. **Isolation.** The results of Part E; explicit "no residual leakage" statement or the exact disclosure of what remains.
5. **Refresh plan.** How you'd add items per quarter; what would trigger a break-glass refresh; who owns it.
6. **Known limits.** What this eval *cannot* tell you (production distribution drift, long-tail regressions outside the strata, capabilities orthogonal to the task).

### Part G — Save

Directory layout:

```
outputs/exercise-02/
├── eval_spec.md
├── holdout.jsonl               # full set (source of truth)
├── holdout_gate.jsonl
├── holdout_dev.jsonl
├── anchor.jsonl
├── rubrics/
│   └── <task>_rubric.v1.yaml   # for any rubric-judged strata
├── isolation_check.py
├── isolation_report.md
├── design_report.md
└── README.md                   # one-page pointer for future collaborators
```

## Starter guidance

- **Start with the eval spec.** Do not write items until the spec is done — the spec drives the stratification and the strata drive the item collection.
- **Steal from public datasets to bootstrap.** For a support-agent task, some rows of a public customer-support dataset (with heavy anonymization) is a fast way to get to 300 items. Note the source in `metadata.source`.
- **Author items in batches, not all at once.** Do 50–100 items, look at what you have, adjust the stratum breakdown, do the next 50–100. Fixing a stratum imbalance after 400 items is more work than adjusting after 100.
- **Rubrics are a first-class artifact.** For any `rubric_llm_judge` item, the rubric file (`rubrics/<task>_rubric.v1.yaml`) is what makes the item scoreable. Version the rubric; do not edit in place.
- **Do not skip the anchor set.** It is small (50–100 items) and it is what gives the LLM-as-judge results credibility in exercise 03. An eval without an anchor is an eval whose judge scores you have no way to validate.
- **Isolate at both source and n-gram level.** If your eval items came from tickets #1000–#1500 in production, no training row should be from those tickets. The n-gram check is a second layer that catches paraphrases and synthetic-data leaks.

## Acceptance criteria

You have completed the exercise when:

- `eval_spec.md` exists and covers task, consumers, success criteria, strata, metrics, isolation.
- `holdout_gate.jsonl` has 250–400 items with ≥30 per reported stratum, ≥50 per gated stratum, ≥3 gold-answer types represented, and all items conform to the schema.
- `anchor.jsonl` has 50–100 items with human labels (either reference answers or rubric scores).
- `isolation_report.md` shows zero unresolved leakage into your training data.
- `design_report.md` covers all six sections from Part F.
- A peer could pick up `outputs/exercise-02/` and run an eval against a fine-tuned model without asking you questions — the artifact is self-describing.

## Stretch goals

- **Add a `holdout_v2` refresh.** Simulate one quarter of production drift: swap in 30–50 new items and mark 10–20 old items as `deprecated_in_version: 2`. Run the isolation check against the refresh. This exercises the versioning discipline from chapter 3.
- **Author a `holdout_stress` stratum.** 20–30 items designed to be harder than average — long inputs, adversarial phrasing, edge cases. Track separately in your reports.
- **Write a labeling guide.** A short doc (~1 page) that would let another labeler produce items in the same style yours are. This is what turns your eval into a repeatable process.
- **Build a mini-viewer.** A `viewer.py` that reads `holdout.jsonl` and prints one item at a time with schema fields highlighted. Useful for review meetings.
- **Cross-annotate the anchor.** Have a second labeler independently score the anchor items and compute inter-labeler agreement (κ). If it's below 0.6, either the rubric is unclear or the items are too subjective — fix before running exercise 03.
- **Compute the CI half-width you can achieve.** For each stratum, given its size, compute what effect size (in pp for binary metrics, in rubric-score units for continuous) you can detect at 95% confidence. Include in `design_report.md` as a table.
