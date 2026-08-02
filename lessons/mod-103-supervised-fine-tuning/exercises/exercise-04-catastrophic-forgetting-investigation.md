# exercise-04: Catastrophic Forgetting Investigation

**Estimated effort:** 4 hours
**Companion chapter:** [`05-overfitting-forgetting-template-leak.md`](../05-overfitting-forgetting-template-leak.md)

## Objective

Measure the delta between a base model and its SFT'd version on a held-out benchmark that your SFT data did not target, quantify catastrophic forgetting if present, apply *one* mitigation from chapter 5's ladder, and demonstrate whether the mitigation helped. By the end you will have (a) two benchmark numbers per model (base, SFT baseline, SFT with mitigation), (b) a written interpretation, and (c) a clear "would I ship the mitigation?" answer.

The exercise is small on purpose. Twelve runs on twelve benchmark subsets does not teach you anything you cannot learn from three runs on one subset done carefully.

## Prerequisites

- Trained checkpoint from exercise 01 (`baseline_sft/`).
- The corresponding base model (revision-pinned) that you started from.
- Enough compute to do one more SFT run (`mitigation_sft/`) and to evaluate all three models on a benchmark subset.
- `lm-evaluation-harness` (recommended) or equivalent for running the benchmark. Manual evaluation loops are acceptable if you cannot install the harness, but you must clearly document the loop.

## Requirements

### Part A — Pick a "held-out" benchmark

The benchmark you evaluate on must be *unrelated to what your SFT set taught*. If your SFT set was customer-support-style conversations, MMLU (general knowledge) is a good pick. If your SFT set was code, HumanEval is a good pick — but only if you also add a general benchmark to catch broader forgetting.

Suggested benchmarks (any one is enough; a second is a stretch goal):

- **MMLU** (Hendrycks et al., 2021) — 57 subject-domain multiple choice, general knowledge. A 500-row random slice is fine for this exercise; full MMLU is 14k rows.
- **ARC-Easy / ARC-Challenge** (Clark et al., 2018) — science questions, multiple choice.
- **HellaSwag** (Zellers et al., 2019) — commonsense completion.
- **GSM8K** (Cobbe et al., 2021) — grade-school math, exact-match on final answer.
- **HumanEval** (Chen et al., 2021) — Python function synthesis.

**Do not** pick a benchmark that has any overlap with your SFT training set. Mod-102 chapter 3's decontamination check should have flagged it; run the check if you did not.

Document your choice, subset size, and subset selection method (random with seed, first-N, etc.) in `bench.md` at the top of your exercise directory.

### Part B — Measure the baseline delta

Run three evaluations:

1. **Base model** — pretrained, no SFT.
2. **SFT baseline** — the `baseline_sft/` checkpoint from exercise 01.

Report a small table in `results.md`:

```markdown
| model            | benchmark | subset | score | 95% CI (or SE) |
|------------------|-----------|--------|-------|----------------|
| base             | MMLU-500  | random-seed-42 | 42.0 | ±2.2 |
| baseline_sft     | MMLU-500  | random-seed-42 | 39.1 | ±2.2 |
```

Also report:

- **Delta** in percentage points.
- **Whether the delta is inside the confidence interval** (i.e., whether it is a real regression or benchmark noise).

If the delta is inside CI, the base and SFT model are indistinguishable on this benchmark — you may need a larger subset, a different benchmark, or the honest conclusion that "on this data / this benchmark, forgetting is undetectable." All three are valid outcomes; document which one applies.

### Part C — Apply one mitigation

Pick **exactly one** mitigation from chapter 5's catastrophic-forgetting ladder to re-run SFT with:

1. **Lower learning rate** (halve it).
2. **Fewer epochs** (only viable if your baseline used > 1 epoch).
3. **Switch to LoRA / QLoRA** (mod-104 covers the mechanics; you may need to preview that chapter).
4. **Add general-domain data** (mix in 10–20% of an unrelated open SFT set — e.g., a slice of Tülu 3 mix or No Robots — to your training data).

Whichever you pick, change *only that one thing* from your exercise-01 configuration. Save the new checkpoint as `mitigation_sft/`. Emit a fresh manifest for the mitigation run (exercise 05 will formalize this).

Evaluate `mitigation_sft/` on the same benchmark subset with the same seed. Add the row to your results table.

```markdown
| model            | benchmark | subset | score | 95% CI |
|------------------|-----------|--------|-------|--------|
| base             | MMLU-500  | random-seed-42 | 42.0 | ±2.2 |
| baseline_sft     | MMLU-500  | random-seed-42 | 39.1 | ±2.2 |
| mitigation_sft   | MMLU-500  | random-seed-42 | 41.3 | ±2.2 |
```

### Part D — Cross-check: did the mitigation hurt the SFT-targeted capability?

The trap: your mitigation may recover MMLU by giving up some of your SFT target. Evaluate both `baseline_sft` and `mitigation_sft` on your SFT eval slice (chapter 4's eval loss, and a short qualitative generation sample).

Report:

- Eval loss on the SFT eval slice: baseline vs. mitigation.
- Generation quality on 5 canned SFT-domain prompts: baseline vs. mitigation, judged by you (or a fast LLM judge) on a 1–5 scale.

If the mitigation recovered general capability at meaningful cost to the SFT target, that is a *tradeoff*, not a win. Say so.

### Part E — Ship-decision writeup

Write `writeup.md` (one page) containing:

1. The three-model results table with delta interpretation.
2. The mitigation you chose and why.
3. The Part D tradeoff analysis: did the mitigation hurt the SFT target?
4. Your ship decision: `baseline_sft`, `mitigation_sft`, or "neither, needs another iteration" — with a one-sentence justification.
5. What you would try next, in one sentence.

## Starter guidance

- **Use the same eval harness for all three models.** Manually-implemented eval loops introduce implementation drift; use `lm-evaluation-harness` and pin the version.
- **Report confidence intervals.** A 2-point MMLU delta on 100 rows is noise; a 2-point delta on 5000 rows is real. Chapter 5's mitigations only make sense against real regressions.
- **Do not run all mitigations in parallel.** The point is measuring the effect of *one* change, not producing the best possible model.
- **Chat template consistency across evals.** Evaluate the base with the *same* chat template the SFT models expect. Otherwise you are comparing "how well does the model do without any chat framing" vs "how well does the model do with the SFT framing," which is a different (also interesting) question.
- **On small GPUs.** Full MMLU eval on a 1B model takes ~30 minutes with `lm-evaluation-harness`. A 500-row subset takes ~2 minutes. Both are fine at this scale; the subset is what you use to iterate the exercise and the full run is the stretch goal.

## Acceptance criteria

You have completed the exercise when:

- `bench.md` documents your benchmark choice, subset size, and subset seed.
- `results.md` has the three-model results table with confidence intervals or standard errors, and the deltas.
- `mitigation_sft/` exists as a real checkpoint from a real run whose config differs from `baseline_sft/` in exactly one line.
- Part D reports both the general-benchmark and SFT-target numbers for both SFT models.
- `writeup.md` names the ship candidate and defends it in a paragraph.
- A reader who has never seen your project can read `writeup.md` and, given the numbers, agree or explicitly disagree with your ship call.

## Stretch goals

- **Multi-benchmark eval.** Run the same three models on MMLU *and* GSM8K *and* IFEval. The three-way table clarifies which capabilities forgetting hit and which it did not.
- **Full MMLU.** Run all 14k rows, not a subset. Confidence intervals shrink; deltas that looked like noise become clearer.
- **Second mitigation.** Do one more run with a different mitigation from the ladder. Compare the two mitigations' tradeoffs.
- **Longer runs.** Repeat the baseline with `num_train_epochs=3`. Chapter 5 predicts forgetting gets worse with more epochs; check it.
- **Cross-language evaluation.** If your base is multilingual (Qwen 2.5, SmolLM2) and your SFT data is English-only, evaluate on a non-English benchmark (Flores subset, XNLI, TyDiQA). Non-target-language capability is a specific and easily-measured forgetting axis.
