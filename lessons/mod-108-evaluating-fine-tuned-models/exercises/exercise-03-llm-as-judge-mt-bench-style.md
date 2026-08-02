# exercise-03: LLM-as-Judge, MT-Bench Style

**Estimated effort:** 4 hours
**Companion chapters:** [`04-llm-as-judge.md`](../04-llm-as-judge.md), [`03-task-specific-held-out-evals.md`](../03-task-specific-held-out-evals.md), [`06-release-gate-eval-suite.md`](../06-release-gate-eval-suite.md)

## Objective

Build a working LLM-as-judge pipeline that scores your held-out set (from exercise 02) with both **single-answer rubric grading** and **pairwise comparison**, controls for **position bias**, measures **judge-vs-human agreement** against your anchor set, and reports the ~10 metadata fields chapter 4 lists as non-negotiable in a release-gate report.

The point is judge discipline: producing scores you can defend, not just producing scores. A judged eval whose biases are uncontrolled is worse than no judged eval — it produces confident-looking numbers that mislead.

## Prerequisites

- The `holdout_gate.jsonl`, `anchor.jsonl`, and rubric YAML from exercise 02.
- Two model responses per item — for exercise 03 you need at least (base or previous adapter) vs. (fine-tuned) on the same items. Generate these first with the same sampling settings (T=0 greedy for reproducibility, matched `max_new_tokens`, matched chat template).
- API access to at least one strong judge model (GPT-4o, Claude Sonnet, or a similar-class open model). For the ideal setup you want two different judge families (e.g., GPT-4o + Claude Sonnet).
- A budget of $10–50 for judge API calls; less if you cap eval size or use a cheaper judge.

## Requirements

### Part A — Generate model responses

Write `generate_responses.py` that takes `holdout_gate.jsonl` and one model (base and tuned in two runs) and produces `responses_<model>.jsonl` with one row per input:

```json
{"id": "eval-2026-08-0001", "model": "acme-support-v2.5", "response": "…", "meta": {"latency_ms": 340, "tokens": 187, "seed": 1234}}
```

Requirements:

- **Same sampling parameters** for both models. Greedy (T=0) unless you have a specific reason otherwise.
- **Same chat template** for both, using the tokenizer's `apply_chat_template`.
- **Same `max_new_tokens`** for both.
- **Record `latency_ms` and `tokens`**. Chapter 4's verbosity-bias analysis needs the token count.
- **Deterministic** — rerun should produce identical `response` fields.

Save `responses_base.jsonl` and `responses_tuned.jsonl`.

### Part B — Single-answer rubric grading

Write `grade_single.py`. For each item in `holdout_gate.jsonl` and each of the two response files, call the judge with a rubric-grading prompt and record the score.

Prompt template requirements (chapter 4):

1. Include the item's full input (system prompt + user prompt + any context).
2. Include the model's response.
3. Include the rubric verbatim from `rubrics/<task>_rubric.v1.yaml`.
4. **Require the judge to write 2–3 sentences of reasoning before the score.**
5. **Force a structured final line:** `Score: <integer>`.
6. Judge temperature = 0, seed = fixed.

Store, for each judged item:

```json
{
  "id": "eval-2026-08-0001",
  "model": "acme-support-v2.5",
  "judge": "gpt-4o-2024-08-06",
  "rubric_version": "support_agent_v3",
  "prompt_template_hash": "sha256:…",
  "score": 4,
  "reasoning": "…",
  "raw_judge_response": "…",
  "judge_seed": 1234
}
```

Save `judge_single_<model>_<judge>.jsonl`.

Aggregate metrics per model:

- Mean score, median score, per-stratum breakdown.
- 95% bootstrap CI on the mean.
- Wilcoxon signed-rank paired test between the two models' per-item scores.

### Part C — Pairwise comparison with position-bias correction

Write `grade_pairwise.py`. For each item, call the judge **twice**:

- Once with `response_A = base`, `response_B = tuned`.
- Once with `response_A = tuned`, `response_B = base`.

Prompt template requirements (chapter 4):

1. Show both responses labeled `A` and `B`.
2. Instruct the judge to consider correctness, helpfulness, and clarity; explicitly tell it to ignore superficial length differences.
3. Reasoning before verdict.
4. Structured final line: `Verdict: A` | `Verdict: B` | `Verdict: Tie`.

Collapse the two verdicts into a final verdict per item using the rule from chapter 4:

- Both orderings pick A (tuned or base) → that side wins.
- Orderings disagree → tie.
- Both say tie → tie.

Compute:

- Raw pairwise win rate (using one of the two orderings alone — the biased number for comparison).
- Position-bias-corrected win rate (using the collapsed rule).
- **Position-bias fraction:** % of items where the two orderings disagreed.

Save `judge_pairwise_<judge>.jsonl` with both raw verdicts and the collapsed final verdict per item.

### Part D — Length-controlled correction

Compute the mean response length for both models. If the two models differ by ≥15%, apply length-controlled correction:

- **Cheap version:** bin items by length quartile and report the pairwise win rate per bin. If tuned only wins in the top length quartile, verbosity is doing the work.
- **Rigorous version:** implement the length-controlled AlpacaEval correction (Dubois et al. 2024) or use the `alpaca_eval` library.

Report both raw and length-controlled win rates in the final table.

### Part E — Judge-vs-human agreement on the anchor

For every item in `anchor.jsonl`:

- You have (from exercise 02) the human single-answer score and (if you did the stretch) a human pairwise verdict.
- Extract the corresponding judge scores / verdicts from Parts B and C.

Compute:

- **Single-answer:** Spearman ρ between human scores and judge scores across the anchor.
- **Pairwise:** Cohen's κ between human verdicts and judge (collapsed) verdicts.

Report both. **A κ below 0.4 or ρ below 0.5 means the judge is not usable for this task at this rubric** — either revise the rubric, switch judge models, or fall back to human labels for the affected strata. If your numbers land there, describe your remediation plan in the report.

### Part F — Multi-judge agreement (stretch, but recommended)

Re-run Parts B and C with a *second* judge model from a different family (Claude if you used GPT; GPT if you used Claude). Compute inter-judge agreement:

- Spearman ρ between the two judges' single-answer scores.
- Cohen's κ between the two judges' pairwise verdicts.

If inter-judge agreement is high (ρ ≥ 0.7, κ ≥ 0.6), average their scores for the final report. If low, investigate — the disagreement patterns often reveal a systematic bias in one judge you can address.

### Part G — Write the report

`report.md`, structured as:

1. **Setup.** Judge model(s), version, temperature, seed, rubric version, prompt template hash, hardware (for response generation), sampling parameters for response generation.
2. **Single-answer table.** Per-model mean, median, per-stratum breakdown, CIs, paired-test p-value.
3. **Pairwise table.** Raw win rate, position-bias-corrected win rate, position-bias fraction, length-controlled win rate, mean response length per model.
4. **Agreement.** Judge-vs-human (κ / ρ) on the anchor. If run, inter-judge (κ / ρ) between the two judges.
5. **Bias controls applied.** Position (yes/no), verbosity (yes/no + method), self-preference (which judge families you used vs. the model being evaluated), style (rubric specificity notes, reference-answer usage).
6. **Findings.** One paragraph: does the tuned model genuinely improve on the held-out set per this judged eval, and how confident are you? What would you tell the release captain?

### Part H — Save

```
outputs/exercise-03/
├── generate_responses.py
├── grade_single.py
├── grade_pairwise.py
├── responses_base.jsonl
├── responses_tuned.jsonl
├── judge_single_base_<judge>.jsonl
├── judge_single_tuned_<judge>.jsonl
├── judge_pairwise_<judge>.jsonl
├── agreement.md                     # anchor & inter-judge agreement calc
├── prompt_templates/
│   ├── single_answer.v1.md
│   └── pairwise.v1.md
└── report.md
```

## Starter guidance

- **Version your prompt templates and hash them.** Every change to the prompt template invalidates prior judge scores; the hash in every result row is what lets you notice a drift.
- **Cache judge calls.** A tuple of `(prompt, response_a, response_b, rubric_version, judge_model, judge_seed)` should map to a single judge call ever. If you rerun, hit the cache. This makes iteration on your grading code cheap.
- **Start with 30 items.** Get the pipeline working end-to-end on a subset before running the full anchor + gate. The bugs (missing structured verdict, prompt-template escaping, ordering mismatches) surface fast.
- **Store the raw judge response.** Do not just parse and discard. When a per-item score seems wrong, the reasoning trace is what tells you why.
- **Use structured output where the judge API supports it.** OpenAI's JSON mode, Anthropic's structured output, or a strict regex parser at minimum. Free-form "the score is 4/5" text is a bug factory.
- **Never judge your fine-tune with the same model family.** If your fine-tune is a Llama derivative, don't use another Llama as judge. Use GPT or Claude. Chapter 4 is emphatic about this.

## Acceptance criteria

You have completed the exercise when:

- `responses_base.jsonl` and `responses_tuned.jsonl` exist, are deterministic (rerun byte-identical), and were generated with matched sampling settings.
- Single-answer and pairwise grading pipelines run end-to-end and produce the JSONL result files with all metadata fields.
- Pairwise pipeline runs *both* orderings per item and produces the collapsed final verdict.
- Position-bias fraction is reported.
- Length-controlled correction is applied (or you have justified in the report why it was unnecessary — e.g., <5% length difference).
- Judge-vs-human agreement (κ / ρ) on the anchor set is computed and reported.
- The final report includes all 10 metadata fields from chapter 4's "what to report" list.
- All numbers in the report have confidence intervals or agreement metrics; no bare point estimates.

## Stretch goals

- **Run two judges.** Complete Part F with GPT-4o and Claude Sonnet, average their scores, and report inter-judge agreement.
- **Ablate the rubric.** Run single-answer grading with (a) the full rubric, (b) a stripped rubric ("rate 1–5, higher is better"), (c) the rubric plus a reference answer. Compare agreement with the anchor. This is the receipt for chapter 4's rubric-specificity claims.
- **Ablate the reasoning-before-verdict requirement.** Run once with "reasoning first, verdict last" and once with "verdict only, no reasoning." Compare agreement. Reasoning should improve agreement; if it does not, the judge is probably ignoring it.
- **Compute cost.** Track total judge tokens (input + output) across the runs and back out dollars. Compare vs. what a human labeling shop would charge for 400 items. Include in the report.
- **Two-pass judging.** Run all items with a cheap judge (Haiku, GPT-4o-mini). Re-judge only the items where the cheap judge's confidence is low (verdict = tie, or score near the middle of the scale) with an expensive judge. Compare total cost and agreement vs. running the expensive judge on everything.
- **Add self-preference ablation.** If your fine-tune is a Llama, run a third judge from a Llama derivative (Llama-3.1-70B-Instruct via a hosted endpoint) and compare it to GPT/Claude. Confirm the Llama judge over-rates the Llama-derivative outputs.
