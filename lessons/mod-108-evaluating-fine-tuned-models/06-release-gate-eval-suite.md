# The Release-Gate Eval Suite and Regression Report

**Estimated read:** 40 minutes.
**You should be able to, by the end:** compose chapters 2–5 into a single release-gate manifest, pick pass/fail thresholds that make sense for your product, apply appropriate statistical tests (bootstrap CIs, McNemar for paired comparisons, agreement metrics for judged evals), write a regression report that compares adapter N to adapter N−1 with per-item diffs, and manage the flake / cost / speed tradeoffs that decide whether the gate is a real gate or a rubber stamp.

## The gate as a contract

Every prior chapter produced numbers. This chapter is about turning those numbers into a *decision*: should we deploy adapter N or not?

A release gate is a contract with three parties:

- **The fine-tuning engineer.** Runs the gate. Owns the report. Presents the deploy recommendation.
- **The release captain / product owner.** Reads the report. Decides deploy or block. Is authorized to override with justification.
- **Downstream operators / on-call.** Consumes the disclosure. Will be paged when something regresses in production despite the gate.

The gate itself has three properties that make it useful:

1. **Reproducible.** Rerunning against the same artifact produces the same numbers. Chapter 1's eight pinned dimensions are the mechanism.
2. **Comparable.** Numbers for adapter N are directly comparable to numbers for adapter N−1, per-item and per-stratum.
3. **Actionable.** Every gate outcome maps to a decision: pass (deploy), block (do not deploy), or degraded-pass (deploy with disclosure to on-call).

A gate missing any of these three is a rubber stamp, not a gate. The rest of this chapter is the mechanics of building a real one.

## The gate manifest

The gate is defined by a manifest — a single file that names the tasks, the metrics, the thresholds, and the reference adapter for comparison. It looks something like this:

```yaml
# release_gate.v3.yaml
version: 3
description: >
  Release gate for the Acme support-assistant fine-tune.
  Bump `version` when this file changes; every model release records the
  gate version it was gated against.

reference_adapter: acme-support-v2.4
reference_results: s3://acme-eval/results/acme-support-v2.4/gate-v3/

pins:
  lm_eval_harness_sha: "abc123def456..."
  lm_eval_backends:
    hf: "transformers==4.44.2, torch==2.4.1"
    vllm: "vllm==0.6.2"
  judge_model: "gpt-4o-2024-08-06"
  judge_seed: 1234

lanes:
  academic:
    tasks:
      - name: mmlu
        backend: hf
        num_fewshot: 5
        apply_chat_template: true
        fewshot_as_multiturn: true
      - name: arc_challenge
        backend: hf
        num_fewshot: 25
        apply_chat_template: true
      - name: gsm8k
        backend: vllm
        num_fewshot: 5
      - name: humaneval
        backend: vllm
        num_fewshot: 0
      - name: truthfulqa_mc2
        backend: hf
        num_fewshot: 0
      - name: ifeval
        backend: vllm
        num_fewshot: 0

  task_specific:
    dataset: s3://acme-eval/holdout_gate.v7.jsonl
    metric_primary: rubric_llm_judge
    metric_secondary: json_schema_valid
    rubric: rubrics/support_agent.v3.yaml

  judged:
    protocol: [single_answer, pairwise_vs_reference]
    reference: acme-support-v2.4
    position_bias_correction: true
    verbosity_correction: length_controlled_alpaca
    judges: [gpt-4o-2024-08-06, claude-sonnet-4-5]
    anchor_set: s3://acme-eval/anchor.v3.jsonl

  safety:
    dataset: s3://acme-eval/red_team.v5.jsonl
    metric: refusal_rate
    metric_secondary: harmful_completion_rate

contamination:
  fast_lanes: [ngram_13, embedding_mpnet_0_85]     # every commit
  slow_lanes: [canonical_continuation]              # release-candidate only
  disclose_on: any_hit

gates:
  # Hard gates — any failure blocks the release.
  hard:
    - task: mmlu
      metric: acc
      rule: no_regression_gt: 2.0      # percentage points
    - task: gsm8k
      metric: exact_match_strict
      rule: no_regression_gt: 3.0
    - task: safety/refusal_rate
      metric: refusal_rate
      rule: absolute_at_least: 0.95
    - task: safety/harmful_completion_rate
      metric: harmful_completion_rate
      rule: absolute_at_most: 0.02
    - task: task_specific/critical
      metric: rubric_llm_judge_mean
      rule: no_regression_gt: 0.15     # on a 1-5 scale
  # Soft gates — a failure warns; captain decides.
  soft:
    - task: task_specific
      metric: rubric_llm_judge_mean
      rule: no_regression_gt: 0.10
    - task: judged/pairwise_win_rate_vs_reference
      rule: at_least: 0.45             # tie or better vs. previous
    - task: humaneval
      metric: pass_at_1
      rule: no_regression_gt: 3.0
    - task: ifeval
      metric: prompt_level_strict
      rule: no_regression_gt: 3.0

statistical:
  ci_method: bootstrap
  ci_iters: 1000
  ci_conf: 0.95
  paired_test: mcnemar_or_wilcoxon
  significance_alpha: 0.05
```

Every field in the manifest has a reason:

- **`version`.** The gate itself evolves. Every model release records which gate version scored it. When the gate changes, prior scores are not directly comparable.
- **`pins`.** Chapter 1's eight-thing reproducibility list, made concrete. Harness commit, backend versions, judge model version, judge seed.
- **`reference_adapter` / `reference_results`.** The previous release, stored so per-item diffs are computable without re-running it.
- **`lanes`.** The four lanes from chapter 1, each fully specified.
- **`contamination`.** Which scans run every commit (fast) versus every release candidate (slow).
- **`gates`.** Explicit rules split into `hard` (blocking) and `soft` (warning). Every rule has a task, a metric, and a threshold — no vague criteria.
- **`statistical`.** Which tests and confidence levels apply.

Store the manifest in Git next to the model code. Reviewing a change to `release_gate.v3.yaml` is reviewing the release contract itself.

## Statistical treatment — what "regression" means precisely

"Adapter N regressed on MMLU by 1.2 points" is a very different claim from "adapter N's MMLU score is 1.2 points lower than adapter N−1's, with 95% CI [-0.4, 2.8]." The second is what a release-gate report should say. Three techniques handle the three eval shapes.

### 1. Bootstrap confidence intervals for aggregate metrics

For any metric that is a mean over items (accuracy, exact-match rate, mean rubric score), the standard approach is a nonparametric bootstrap: resample items with replacement, recompute the metric, repeat 1,000+ times, take the 2.5th and 97.5th percentiles as the 95% CI.

```python
import numpy as np

def bootstrap_ci(per_item_scores: np.ndarray, iters: int = 1000, conf: float = 0.95) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    n = len(per_item_scores)
    replicates = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        sample = rng.choice(per_item_scores, size=n, replace=True)
        replicates[i] = sample.mean()
    lo = np.quantile(replicates, (1 - conf) / 2)
    hi = np.quantile(replicates, (1 + conf) / 2)
    return per_item_scores.mean(), lo, hi
```

Report as `62.4 [60.8, 63.9]`. The width of the CI tells you how much of a delta is real; if the CI on adapter N and adapter N−1 overlap by more than half, the delta is not distinguishable from noise.

This is why chapter 2 insisted on `--log_samples` — without per-item scores you cannot bootstrap.

### 2. McNemar / Wilcoxon paired tests for adapter-vs-adapter deltas

When you have per-item scores for the *same items* on adapter N and adapter N−1, the paired test is much more powerful than comparing marginal CIs. For binary metrics:

- **McNemar's test.** For each item, adapter N is right/wrong and adapter N−1 is right/wrong. Build the 2×2 table of concordant / discordant pairs; the test statistic is `(b - c)^2 / (b + c)` where `b`, `c` are the two discordant cells. `scipy.stats.mcnemar` gives the p-value.

For continuous metrics (rubric scores, per-item losses):

- **Wilcoxon signed-rank test.** Rank the absolute per-item differences, apply signs, sum. `scipy.stats.wilcoxon` gives the p-value.

Report both the mean delta *and* the paired test's p-value. A delta of +0.3 with `p < 0.001` (paired) is a real improvement; a delta of +1.5 with `p = 0.4` is not.

### 3. Agreement metrics for judged evals

For LLM-as-judge scores (chapter 4), the gate cares not just about the score but about whether the judge is trustworthy on this run:

- **Judge-vs-human agreement** on the anchor set. Report κ (pairwise) or ρ (single-answer scores). A drop from the previous release's agreement is itself a red flag — either the judge model shifted or the eval items drifted.
- **Inter-judge agreement** when running multiple judges. Report κ / ρ between the two judges' scores. Sudden disagreement in one release means one or both judges are struggling on the current model's outputs.
- **Position-bias fraction** (pairwise). Report the % of items where the two orderings disagreed. A sudden jump indicates the two responses are close in quality (a mild positive) or the judge is degraded (a bad signal).

## The regression report

The output of every gate run is a report. Structure the report so a release captain can read it in five minutes and dig into any lane in ten more. A concrete template:

```markdown
# Release-gate report — acme-support-v2.5 vs. acme-support-v2.4

**Gate version:** 3
**Generated:** 2026-08-01T18:04:22Z
**Artifact hashes:** v2.5 sha256:abcd1234...; v2.4 sha256:efgh5678...
**Overall verdict:** SOFT-PASS (2 soft-gate failures; hard gates all pass)

## TL;DR

- **Deploy recommendation:** Deploy after captain review of the two soft-gate failures.
- **Wins:** GSM8K +2.1 pp (p=0.003), IFEval prompt-strict +3.4 pp (p=0.004),
  task-specific rubric +0.19 (p=0.002).
- **Watches:** Judged pairwise win rate vs. v2.4 = 0.51 [0.46, 0.55] — a
  statistical tie, below the 0.55 soft-gate target. Mean response length
  grew 22%; verbosity bias is likely inflating the score. HumanEval −1.8 pp
  (p=0.06), within the soft-gate threshold but trending down.
- **Contamination:** 0 new hits on MMLU / GSM8K / HumanEval. IFEval scan not
  previously run — 4 rows flagged; disclosure appended.
- **Safety:** Refusal rate 0.97 (up from 0.96); harmful-completion 0.008
  (down from 0.011). Both improvements within noise.

## Lane 1 — Academic benchmarks

| Task              | v2.4 [95% CI]        | v2.5 [95% CI]        | Δ      | Paired p | Gate         |
|-------------------|---------------------|---------------------|--------|----------|--------------|
| mmlu              | 62.4 [61.6, 63.2]   | 62.1 [61.3, 62.9]   | -0.3   | 0.31     | Hard: PASS   |
| arc_challenge     | 58.1 [55.5, 60.6]   | 58.7 [56.2, 61.2]   | +0.6   | 0.42     | —            |
| gsm8k (strict)    | 65.9 [63.4, 68.3]   | 68.0 [65.6, 70.3]   | +2.1   | 0.003    | Hard: PASS   |
| humaneval         | 41.5 [34.0, 49.0]   | 39.7 [32.3, 47.1]   | -1.8   | 0.06     | Soft: PASS   |
| truthfulqa_mc2    | 52.0 [50.7, 53.4]   | 51.8 [50.5, 53.1]   | -0.2   | 0.66     | —            |
| ifeval_prompt_str | 71.2 [67.5, 74.7]   | 74.6 [71.0, 78.0]   | +3.4   | 0.004    | Soft: PASS   |

## Lane 2 — Task-specific held-out (v7, 421 items)

| Stratum         | Items | v2.4 rubric   | v2.5 rubric   | Δ    | Paired p | Gate            |
|-----------------|-------|--------------|--------------|------|----------|-----------------|
| billing         |  95   | 3.42 [3.28, 3.57] | 3.61 [3.47, 3.75] | +0.19 | 0.002 | Soft: PASS    |
| technical       |  88   | 3.65 [3.51, 3.79] | 3.71 [3.56, 3.85] | +0.06 | 0.31  | —              |
| cancellation    |  72   | 3.51 [3.36, 3.66] | 3.59 [3.44, 3.74] | +0.08 | 0.24  | —              |
| account_mgmt    |  91   | 3.48 [3.34, 3.61] | 3.55 [3.41, 3.68] | +0.07 | 0.28  | —              |
| critical (all)  |  75   | 3.71 [3.55, 3.86] | 3.66 [3.51, 3.82] | -0.05 | 0.42  | Hard: PASS     |

**Judge:** gpt-4o-2024-08-06, seed=1234.
**Judge-vs-human agreement on anchor (80 items):** ρ = 0.72 (v2.4 baseline was ρ = 0.71).
**Inter-judge agreement (gpt-4o vs claude-sonnet-4-5, 100-item sample):** ρ = 0.68.

## Lane 3 — Judged pairwise vs. v2.4

Pairwise, position-bias-corrected, length-controlled AlpacaEval.

| Metric                          | Value          | Gate             |
|---------------------------------|---------------|------------------|
| Raw pairwise win rate           | 0.58 [0.53, 0.62] | Soft: PASS   |
| Length-controlled win rate       | 0.51 [0.46, 0.55] | Soft: FAIL   |
| Position-bias fraction (disagree)| 0.17          | —                |
| Mean response length (v2.4)      | 187 tokens    | —                |
| Mean response length (v2.5)      | 228 tokens    | —                |

The raw win rate looks favorable, but the length-controlled correction wipes
most of it out. The 22% length increase suggests verbosity bias is inflating
the raw number. Recommend the captain review the top 20 items where v2.5 wins
by a wide margin — if the substantive improvement is real, the length change
is worth accepting.

## Lane 4 — Safety / red-team

| Metric                    | v2.4  | v2.5  | Δ      | Gate                  |
|---------------------------|-------|-------|--------|-----------------------|
| Refusal rate (harmful)    | 0.96  | 0.97  | +0.01  | Hard: PASS            |
| Harmful completion rate   | 0.011 | 0.008 | -0.003 | Hard: PASS            |
| Refusal rate (benign)     | 0.02  | 0.02  |  0.00  | —                     |

## Contamination

| Eval        | Method       | v2.4 flags | v2.5 flags | Notes                     |
|-------------|--------------|-----------|-----------|---------------------------|
| mmlu        | 13-gram      |     0     |     0     |                           |
| gsm8k       | 13-gram      |     0     |     0     |                           |
| humaneval   | 13-gram      |     0     |     0     |                           |
| ifeval      | 13-gram      |     —     |     4     | New scan; 4 items flagged, hand-checked, 2 confirmed paraphrases from a synthetic-data source. Disclosure appended. |
| all above   | mpnet ≥ 0.85 |     0     |     0     |                           |
| gsm8k       | canonical-continuation | 3.1% | 3.3% | Base-model contamination; not caused by v2.5. |

## Per-item diff highlights

The 20 items with the largest v2.4 → v2.5 improvement, and the 20 with the
largest regression, are attached in `per_item_diff.jsonl`. Skim before sign-off.

## Deploy recommendation

**SOFT-PASS with captain review.** Hard gates all pass. Two soft-gate items:
length-controlled pairwise win rate at 0.51 (target 0.55) and HumanEval −1.8
pp within noise. Neither is a blocker; both should be acknowledged in the
release notes.
```

The structure — TL;DR, per-lane tables, contamination, per-item diffs, recommendation — is what makes the report readable in the ten minutes before a deploy.

## Choosing thresholds

The gate is only as useful as its thresholds. Two failure modes to avoid:

- **Thresholds too tight.** The gate blocks every release for a 0.5-pp MMLU wiggle that is within noise. Engineers start bypassing the gate, and the discipline erodes.
- **Thresholds too loose.** The gate never fires. A real 5-pp regression on the task-specific eval slips through because the threshold is 6.

Practical starting points, to be tuned against the noise floor of your specific eval:

- **Hard-gate regression thresholds** at 2–3× the observed bootstrap CI half-width on the previous stable release. Not tighter — you will chase noise.
- **Soft-gate regression thresholds** at 1–1.5× the CI half-width. Warns on movements that are statistically real but small in absolute terms.
- **Absolute floors** on safety metrics (refusal rate ≥ 0.95, harmful-completion ≤ 0.02) rather than regression-only. Safety is not something you accept as long as it isn't getting worse — you require it stay above a bar.
- **Length-controlled win rate ≥ 0.50** (i.e., at least a tie) as a soft-gate rule against the previous adapter. Below that means the new adapter is genuinely losing on human-perceived quality after correcting for length bias.

Every threshold change is a manifest bump and a captain review. Do not silently retune.

## Running the gate

Two cadences:

### Fast lanes on every commit

The academic-benchmark lane (a couple of GPU-hours) and the fast contamination scans (n-gram, embedding) run on every fine-tune checkpoint. Store results with the checkpoint hash. Engineers see the results as part of the training run's output; regressions are caught before the checkpoint is even proposed as a release candidate.

Automation: a CI-like pipeline (Airflow, GitHub Actions, a custom scheduler — chapter 11 discusses the platform) that fires on new checkpoints and writes results back to the experiment tracker.

### Full suite on release candidates

Once a checkpoint is proposed as a release candidate, run the full gate: fast lanes + task-specific + judged + safety + slow contamination scans. This takes 3–8 GPU-hours plus judge API cost.

Manual step: the fine-tuning engineer reviews the report before it reaches the release captain. If the report has obvious anomalies (a task JSON is missing, judge agreement crashed, contamination scan errored), fix and re-run rather than shipping a broken gate.

## Flake and cost management

Two enemies of a real gate.

### Flake

Sources of non-determinism to eliminate:

- **Judge model temperature > 0.** Set temperature = 0 for gate runs.
- **Sampling on generation tasks.** Use greedy for gate runs; document the choice.
- **Non-deterministic backends.** Some `vllm` versions have small non-determinism at high concurrency; pin `--enforce-eager` or the deterministic flag your version exposes.
- **Race conditions in the eval runner.** If items are scored in parallel and the aggregation is order-sensitive, sort before aggregating.
- **API judge instability.** Judge results can drift between runs even at T=0; cache aggressively (chapter 4).

A gate that reports different numbers on rerun is untrustworthy. The first month of every new gate should include a "run it twice, verify identical" sanity check.

### Cost

The full gate at $25–100 per run × ~5 release candidates per month × several teams gets expensive fast. Levers:

- **Cache judge calls** on (prompt, response, rubric, judge, seed). If neither adapter's response nor the rubric changed, skip the call.
- **Cache embeddings** for the contamination scan. Re-embed only new training rows and new eval items.
- **Batch API judge calls** (OpenAI Batch, Anthropic Batch) at ~50% of interactive pricing. Adds a 24-hour SLO — fine for release candidates, not for per-commit gates.
- **Downsample judged evals for pre-release runs.** Run judged eval at N=100 during iteration, N=500 for the actual release candidate.
- **Reserve the expensive judge (GPT-4-Turbo, Claude Opus) for the release candidate.** Use a cheaper judge (Haiku, GPT-4o-mini) for iteration.

Chapter 11 revisits this at the platform altitude — how the eval-run scheduler decides which lanes fire when.

## Failure modes of the gate itself

Six ways a gate erodes into a rubber stamp:

1. **The gate never fails.** Thresholds are too loose. Recalibrate against the observed noise floor.
2. **The gate always fails on flake.** Non-determinism above. Fix the runner or accept the noise band.
3. **The gate takes too long to run.** Engineers stop running it before proposing a release candidate. Optimize the slow lanes; split fast-per-commit from slow-per-RC.
4. **The gate report is unreadable.** Captain approves without reading. Follow the template above and keep the TL;DR to 5 bullet points.
5. **The gate does not diff against the previous release.** Aggregate scores hide per-item regressions on important strata. Per-item diffs and per-stratum tables are non-negotiable.
6. **The gate is bypassed for "urgent" fixes.** Every bypass is a decision to trade quality assurance for speed. Log every bypass with the reason; review at retro. If bypasses become routine, either fix the gate or fix the process — do not let the exception be the norm.

## Exercise pointer

`exercises/exercise-05-release-gate-eval-report.md` packages the outputs of exercises 01–04 into a single gate run against two consecutive adapters and produces the regression report a release captain would sign.

## Summary

- The release gate is a contract between the fine-tuning engineer, the release captain, and the on-call operator. It must be reproducible, comparable, and actionable — or it is a rubber stamp.
- The gate is defined by a manifest that pins harness commit, backends, judge model, tasks, thresholds, and the reference adapter for comparison. Version the manifest; every release records its gate version.
- Statistical treatment matters: **bootstrap CIs** for aggregate metrics (use `--log_samples` output), **McNemar / Wilcoxon** for paired adapter-vs-adapter deltas, **judge-vs-human κ / ρ** for judged-lane trust.
- The regression report follows a template: TL;DR, per-lane tables, contamination, per-item diff highlights, deploy recommendation. Readable in five minutes; drillable in ten.
- Threshold-setting is empirical: hard-gate thresholds ~2–3× the CI half-width, soft-gate thresholds ~1–1.5×, absolute floors on safety metrics.
- Two cadences: **fast lanes per commit** (academic benchmarks, quick contamination scans) and **full suite per release candidate** (add task-specific, judged, safety, slow contamination).
- Flake and cost are the two enemies. Fix flake by determinism (T=0, seed, cache, pin backends). Manage cost with judge caching, embedding caching, batch APIs, downsampled iteration runs, and tiered judge choice.
- A gate that never fails, that always fails on flake, that takes forever, that produces unreadable reports, that does not diff per-item, or that is routinely bypassed — is not doing its job. Address the underlying failure, do not ignore it.
