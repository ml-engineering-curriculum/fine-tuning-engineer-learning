# exercise-02: Dataset Budget Allocation Across the Objective Stack

**Estimated effort:** 3 hours
**Companion chapters:** [`01-from-product-goal-to-proposal.md`](../01-from-product-goal-to-proposal.md), [`03-dataset-budget-across-objectives.md`](../03-dataset-budget-across-objectives.md), [`02-base-model-selection.md`](../02-base-model-selection.md)

## Objective

For **one product goal and one fixed dataset / compute budget**, allocate that budget across SFT, preference optimization (DPO / KTO / ORPO), safety tuning, and — if the case supports it — classical PPO-RLHF or RFT-style workflows. Defend the allocation against **two named alternatives**, pin an ablation plan that would revise it, and land the artifact bundle a real fine-tuning team would attach to a `dataset-budget-<project>` decision.

The point is the trade-off argument, not code. **You are not fine-tuning anything.** A reviewer six months from now should be able to open your budget allocation and either agree with your allocation on the merits or challenge it on a specific slice — not on gut feel.

## Prerequisites

- Chapter 3 read in full; chapter 1 read so the proposal that owns this allocation is visible; chapter 2 skimmed so you can reference the base you are targeting.
- Exercise 01 completed *or* an equivalent base pick you can defend in one paragraph. This exercise assumes the base is chosen; if it is not, do exercise 01 first.
- Rough per-objective compute numbers you can cite. Mods 103 (SFT), 104 (LoRA / QLoRA), 105 (DPO / PPO / GRPO), and 106 (safety adapter) each carry back-of-envelope numbers; if you did not do those tracks, use documented references (Unsloth benchmarks, HuggingFace TRL docs, or a public vLLM / DeepSpeed throughput number). Cite whatever you use.
- No GPU required. No API budget required.
- A spreadsheet or Markdown-table environment for the allocation table.

## Requirements

### Part A — Pick and specify the product goal + budget

In `goal.md`, define exactly **one** product goal and its budget envelope. The allocation only makes sense with a hard budget; do not let the goal drift into "as much as needed."

Pin:

- **Product goal.** One paragraph, chapter 1 section-1 vocabulary: what behavior is changing, who observes the change, what the baseline is today. A single quantified target (e.g., "reduce escalation rate on billing tickets from 12% to ≤ 8%"; "raise pass@1 on internal SQL-refactor eval from 42% to ≥ 65%"; "raise JSON-schema-valid rate for tool-call outputs from 82% to ≥ 95%").
- **Base model.** Pinned as in exercise 01 (`<org>/<name>` @ revision `<sha>`). Managed candidate is fine; note that PPO / RLHF is off the table for most managed bases.
- **Total budget.** Choose one of the following envelopes so you have a real constraint to argue against, or specify your own with numbers as concrete:
  - **Small.** ~10k rows total (SFT + preference), ~$2k GPU + judge, 2 engineer-weeks.
  - **Medium.** ~50k rows total, ~$10k GPU + judge, 4 engineer-weeks.
  - **Large.** ~250k rows total, ~$60k GPU + judge, 8 engineer-weeks.
- **Data availability (rough).** What internal / synthetic / public row counts you can realistically acquire in the allocation window. Chapter 3's rule of thumb: if you can't name where the rows come from, they don't exist by month 2.
- **Success metric and eval-set anchor.** Cite the mod-108 chapter 3 held-out set that will measure it. If it doesn't exist yet, name who builds it and by when — the allocation is meaningless if the metric isn't measurable.
- **Safety floor.** The mod-106 refusal-rate / harmful-completion thresholds the release gate carries. Even if safety needs no new training rows, this floor is what determines whether the safety slice is zero or nonzero.

One page. Everything downstream anchors on it.

### Part B — Pick the stack shape

In `stack.md`, name **exactly one** stack from chapter 3's four defaults and defend the shape in three or four paragraphs:

- Stack A — SFT-only.
- Stack B — SFT then DPO (the modern default).
- Stack C — joint ORPO or KTO (budget-constrained).
- Stack D — SFT then PPO-RLHF (or RFT on o-series managed).

Chapter 3's "selection heuristic in one paragraph" is your reference. Argue the shape *before* the sizes; do not let the size drive the shape.

Explicitly rule out at least two shapes with one paragraph each — one you rejected quickly and one you considered seriously. Chapter 3's worked examples show the shape.

### Part C — Author the allocation table

Author `allocation.md`, using chapter 3's write-up template. Required table:

| Objective | Rows / samples | Compute (GPU-hours or vendor $) | Data cost ($) | Owner | Rationale |
|---|---|---|---|---|---|
| SFT — task-specific | ... | ... | ... | ... | ... |
| SFT — capability-preserving mix | ... | ... | ... | ... | ... |
| Preference (DPO / ORPO / KTO / — ) | ... | ... | ... | ... | ... |
| PPO / RFT (if Stack D) | ... | ... | ... | ... | ... |
| Safety | ... | ... | ... | ... | ... |
| **Total** | ... | ... | ... | | |

Constraints:

- **Every row cites a chapter 3 rule of thumb or a specific source.** "1 000–5 000 rows for format compliance" (chapter 3) is a legitimate source; "vibes" is not. If you diverge from chapter 3's numbers, defend the divergence in the `Rationale` column.
- **Every row has an owner.** Chapter 3 is emphatic: rows without owners do not exist by month 2. If your project has one owner (yourself), name that owner in every row and note that this is a bandwidth risk in `risks.md` (Part F).
- **Capability-preserving mix is present.** Chapter 3's 30–60% mix ratio when total SFT rows ≥ 20 000; below 20 000 total, mix ratios matter less but the ratio still gets recorded. A pure-task SFT with no mix is chapter 3's most common misallocation — if you skip mix, `stack.md` must justify why (e.g., very low base capability regressions expected, or below-20k total).
- **Compute is a real number.** Cite whatever back-of-envelope math you use (mod-103 numbers, Unsloth benchmark, published vLLM / DeepSpeed throughput). Compute is rarely the dominant cost in a PEFT-first stack; the exercise is calling that out honestly, not padding compute to make the total look bigger.
- **Total must fit inside Part A's envelope.** If it does not, you have exceeded budget — go back to Part B and pick a smaller stack shape, or slim the slices.

### Part D — Defend against two alternatives

Author `defended-alternatives.md`. Two sections:

- **Alternative 1.** A stack shape different from the one you chose (chapter 3 stack A / B / C / D). Show the alternative's allocation table (rough is fine — you are not committing to it) and argue in 2–3 paragraphs why your chosen stack beats it on the criteria that matter for Part A's goal. Cite the specific chapter 3 failure mode you are avoiding (e.g., "Stack C would lose the SFT-format phase we need because zero-shot on the base already ranks poorly — chapter 3, 'when it fails' bullet on ORPO").
- **Alternative 2.** A different *sizing* of the same stack. Common pairs:
  - Same stack, 2× the SFT slice, 0.5× the preference slice — argue the chapter 3 diminishing-returns curve.
  - Same stack, 0.5× the SFT slice, 2× the preference slice — argue the chapter 3 "SFT-alone floor" gap.
  - Same stack, 0 mix rows / more task rows — argue the chapter 3 catastrophic-forgetting risk (mod-103 ch5 handoff).
  - Same stack, a bought SFT dataset in place of an internal-labeled one — argue the chapter 3 judge-quality / anchor-set discussion (mod-108 ch4 handoff).

Each alternative must be defended by pointing at *what specifically it would spend more or less of the budget on* and *what the eval would show*.

### Part E — Author the ablation plan

Author `ablations.md` following chapter 3's template. A table of at least six "signal from eval → what we do" rows. At minimum cover:

- SFT-alone hits the target with margin — skip preference?
- SFT-alone hits target but capability guardrails regress — rebalance mix?
- SFT-alone misses target by ≥ 10 pp — rethink base or dataset?
- SFT + preference hits target but length-controlled win rate < 0.5 — length-control the preferences?
- Safety metric moves the wrong way after preference training — isolate the safety adapter?
- Judge-vs-human agreement (mod-108 ch4) drops after preference — anchor set / judge upgrade?
- Anything specific to Stack D if that's your pick — reward hacking, KL clip, process reward.

For each row: what the specific eval signal is (metric + threshold), what the reallocation is (rows / slices to change), and what the extra cost / time is. The ablation plan turns the allocation from a guess into a decision loop; chapter 3's summary.

### Part F — Escape hatches and open questions

Author `escape-hatches.md`. Two sections:

- **Escape hatches.** Which reallocations are *pre-authorized* by this document. Chapter 3's "escape hatches" section. Example: "if SFT alone hits target with margin, skip DPO — pre-approved; no separate approval needed." At least three escape hatches. Each names the trigger (an eval signal, a schedule signal, a resource signal), the reallocation, and the fine-tuning engineer authorized to execute.
- **Open questions.** The things this document does not resolve. Owner + due date for each. Chapter 3 does not spell this out but chapter 1 does — every proposal has an open-questions block.

Small risk-adjacent register in `risks.md` (5–8 rows, chapter 1 format): dataset shortfall, labeling velocity, judge drift, capability-guardrail regression, safety-metric regression, PPO reward-hacking (if Stack D), preference-noise (if judge / labelers unproven). Severity + owner + mitigation per row.

### Part G — Directory layout

Final tree:

```
outputs/exercise-02/
├── goal.md
├── stack.md
├── allocation.md
├── defended-alternatives.md
├── ablations.md
├── escape-hatches.md
├── risks.md
└── sources.md          # every URL cited, with retrieval date
```

`sources.md` records every URL or chapter reference you cite for a row-count rule of thumb, a compute number, a benchmark, or a public dataset. Chapter 3's row-count and compute rules can be cited as `chapter 3 — <section>`; external sources need URLs with retrieval dates.

## Starter guidance

- **Pick the shape before the slices.** Chapter 3's central discipline. If you start by sizing the SFT and then look at whether DPO fits, you have decided the shape by budget accident rather than by argument.
- **Do not build the allocation table without an eval-set anchor.** If Part A does not have a testable metric with a real held-out set, no allocation is defensible. Go back to chapter 1 section-2 vocabulary and pin one.
- **Use chapter 3's row-count buckets as anchors, not as rules.** "1 000–5 000 for format; 10 000–50 000 for skills; 100 000+ for whole-surface" is a starting point; your scenario may pull you inside or outside it. What is not defensible is landing outside without saying so.
- **Do the safety slice first.** Chapter 3 puts safety last in the write-up because most projects reuse an adapter — but the *decision* comes first: does the base + adapter chain we plan clear the mod-106 floor? If not, safety is not "0 new rows"; it is a real slice that must be sized before the other three are argued.
- **Compute is small; data is expensive.** Chapter 3's "compute cost is rarely the dominant slice in a modern PEFT-first stack" bears repeating. If your allocation table has compute > 50% of total budget, either you are on Stack D (fine — RLHF is compute-heavy) or you are over-provisioning; check.
- **The ablation plan is the load-bearing section.** An allocation with no ablation plan is a coin flip. If you have "we allocated X to SFT and Y to DPO" but do not have "if the eval shows Z, we reallocate to Q," the plan does not tolerate reality.
- **Escape hatches beat re-approval loops.** Every reallocation that would require a separate approval meeting to execute is a decision you have deferred. Pre-authorize the common ones; chapter 3's escape-hatches template.
- **Do NOT commit real customer data.** Row-count estimates for internal labeler pipelines are usually fine; specific ticket volumes or vendor pricing under NDA should be generalized.

## Acceptance criteria

You have completed the exercise when:

- `goal.md` pins the six goal / budget fields (product goal, base, budget envelope, data availability, success metric, safety floor).
- `stack.md` names one stack (A / B / C / D), argues the shape in three or four paragraphs, and rules out at least two alternatives.
- `allocation.md` has a filled table with rows for SFT-task, SFT-mix, preference (or joint), safety, and — if applicable — PPO / RFT; every row has a citation and an owner; total fits inside Part A's envelope.
- `defended-alternatives.md` argues two alternatives with concrete row-count trades and cites the specific chapter 3 failure mode being avoided.
- `ablations.md` covers at least six eval-signal → reallocation rows.
- `escape-hatches.md` lists at least three pre-authorized reallocations, each with a trigger and an owner.
- `risks.md` has at least five risks with severity + owner + mitigation.
- `sources.md` records every citation.

## Stretch goals

- **Cost the ablation plan.** For each row of `ablations.md`, add a column with the extra GPU / labeling / calendar cost of executing that reallocation. This is what a sponsor cares about at the halfway-review; chapter 3's "compute cost is rarely dominant" changes shape when the ablation includes new preference collection.
- **Model the alternative under a different budget envelope.** Take your chosen stack and re-run the allocation under the *next envelope up* (small → medium, medium → large). Where do the extra rows go — SFT-task, SFT-mix, preference, safety? What does the marginal-return curve look like? This is chapter 3's "when the budget doubles" question.
- **Add a distillation slice.** Chapter 3 lists distillation as an underused lever. If your base is a smaller model whose teacher is a larger frontier vendor, include an optional slice of "SFT rows synthesized by teacher X" with the licensing constraint chapter 3 flags on it (vendor terms about training competing models). Argue what fraction of the SFT slice would be distilled versus internal-labeled.
- **Draft two chapter-1 proposal §4 blocks.** Take your allocation and paste-fit it into chapter 1's proposal template's section 4 (Dataset Budget and Objective Stack) — then rewrite it, cutting to one page, as if a sponsor were reading it. The compression is the exercise: what is essential to defend the allocation, and what is background?
- **Sensitivity-test the allocation.** Vary the primary metric target by ±10% and re-run Parts C and D. Does the same allocation still make sense at "target from 8% to 6%" versus "target from 8% to 10%"? A stack that shrinks well and grows well is a stack whose ablation plan you can trust.
- **Wire the allocation to a spreadsheet.** Author `allocation.xlsx` or `allocation.csv` with rows keyed on objective and columns for rows / compute / data-cost / cumulative-cost. Build a small `sensitivity` tab that recomputes as slices change. This is the artifact a platform lead will actually work in.
