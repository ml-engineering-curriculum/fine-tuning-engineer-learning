# exercise-05: Release-Gate Eval Suite and Regression Report

**Estimated effort:** 5 hours
**Companion chapters:** [`06-release-gate-eval-suite.md`](../06-release-gate-eval-suite.md), [`02-lm-evaluation-harness.md`](../02-lm-evaluation-harness.md), [`03-task-specific-held-out-evals.md`](../03-task-specific-held-out-evals.md), [`04-llm-as-judge.md`](../04-llm-as-judge.md), [`05-contamination-detection.md`](../05-contamination-detection.md), [`07-eval-pitfalls-and-handoff.md`](../07-eval-pitfalls-and-handoff.md)

## Objective

Package the outputs of exercises 01–04 into a **single release-gate suite**, defined by a versioned YAML manifest, run it end-to-end against **two consecutive adapter versions** (adapter N and adapter N−1), and produce the regression report a release captain would sign. The report must let a reader decide `pass / soft-pass / block` in under five minutes and drill into any lane in ten more.

The point is integration: turning four separate eval capabilities into one gate that is reproducible, comparable, and actionable — the three properties chapter 6 requires. A gate that only exists as a collection of scripts and shared spreadsheets is not a gate.

## Prerequisites

- All four prior exercises completed. Specifically:
  - `outputs/exercise-01/` — the `lm-evaluation-harness` runs on base and tuned.
  - `outputs/exercise-02/` — the task-specific `holdout_gate.jsonl`, `anchor.jsonl`, rubric YAML.
  - `outputs/exercise-03/` — LLM-as-judge single-answer and pairwise scoring code.
  - `outputs/exercise-04/` — contamination scanners and the disclosure template.
- **Two consecutive adapter versions** to compare. Options:
  - Your own — take an intermediate checkpoint from mod-103/104/105 and the final checkpoint as N−1 and N.
  - Two public adapters — e.g., `Llama-3.1-8B-Instruct` (as N−1) and a public fine-tune of it (as N). If both are instruct, this simulates a "v1.4 → v1.5" release.
  - If you only have one adapter, use `(base_model, adapter)` as `(N−1, N)`. The mechanics are identical; interpretation is different (you are evaluating "did fine-tuning help" not "did the latest release improve").
- API budget of $25–100 for the judged lane (per full gate run × 2 adapters = one comparison). If tight, downsample per Part D below.
- 1 GPU-day of wall-clock budget for the academic lane on both adapters plus contamination fast lanes.

## Requirements

### Part A — Author the gate manifest

Write `release_gate.v1.yaml` following the template in chapter 6. Required top-level keys:

- `version` — start at `1`.
- `description` — one sentence identifying the fine-tune and the product context.
- `reference_adapter` — the id/hash of adapter N−1.
- `reference_results` — the path to N−1's cached results (from exercise 01 / 03).
- `pins` — chapter 1's eight reproducibility dimensions made concrete:
  - `lm_eval_harness_sha`
  - backend versions (`transformers`, `torch`, `vllm`)
  - `judge_model` with its dated version tag (e.g., `gpt-4o-2024-08-06`)
  - `judge_seed`
  - tokenizer version / sha per adapter
- `lanes` — the four lanes:
  - `academic` — the six-task list from exercise 01 with per-task backend, `num_fewshot`, and chat-template flags.
  - `task_specific` — path to `holdout_gate.jsonl`, primary metric, rubric version.
  - `judged` — protocols (`single_answer`, `pairwise_vs_reference`), reference adapter, judges list, `position_bias_correction: true`, verbosity-correction method, `anchor_set` path.
  - `safety` — reuse a red-team set if you have one from mod-106; otherwise mark this lane `deferred_to: mod-106` in the manifest and document the placeholder in the report.
- `contamination` — `fast_lanes` (n-gram + embedding, from exercise 04), `slow_lanes` (canonical-continuation), `disclose_on: any_hit`.
- `gates` — split into `hard` (blocking) and `soft` (warning). At minimum specify:
  - Hard: no MMLU regression > 2 pp; no GSM8K regression > 3 pp; task-specific `critical` stratum no regression > 0.15 (on your rubric scale); safety refusal rate ≥ 0.95 (if lane included).
  - Soft: task-specific aggregate no regression > 0.10; length-controlled pairwise win rate ≥ 0.45; HumanEval and IFEval no regression > 3 pp.
- `statistical` — `ci_method: bootstrap`, `ci_iters: 1000`, `ci_conf: 0.95`, `paired_test: mcnemar_or_wilcoxon`, `significance_alpha: 0.05`.

Store the manifest in the exercise output directory and treat it as the release contract. Justify every threshold in an inline YAML comment or in `manifest_rationale.md` — no magic numbers.

### Part B — Build the gate runner

Write `run_gate.py` that consumes the manifest and orchestrates the run:

1. Reads `release_gate.v1.yaml`.
2. For each adapter (N and N−1), fires the four lanes:
   - Academic → shells out to `lm_eval` per exercise 01 (two invocations for `hf` and `vllm` backends).
   - Task-specific → generates responses, runs the automated + rubric metrics.
   - Judged → uses exercise 03's single-answer and pairwise pipelines.
   - Contamination → runs the fast lanes from exercise 04 on the training snapshot + eval snapshot; slow lanes only if `--full` was passed.
3. Writes every lane's raw JSON to `results/<adapter>/<lane>/`.
4. Idempotent — a rerun with the same manifest, same artifact, same seeds produces byte-identical results JSONs (log-likelihood tasks) or same aggregates to 3 decimal places (generation tasks with greedy).
5. Records `run_manifest.json` that stamps: manifest hash, adapter hashes, harness commit, backend versions, wall-clock per lane, cost per lane (judge tokens × price), and a `PASS/SOFT-PASS/BLOCK` per gate rule.

Do not paper over lane failures. If the judged lane errors halfway, `run_gate.py` should surface it, not silently proceed to the report.

### Part C — Compute statistics

Write `compute_stats.py` (or extend `run_gate.py`) that reads the per-item scores dumped by each lane and produces:

1. **Bootstrap 95% CI** on every aggregate metric — 1,000 resamples of items with replacement (reuse or lift the snippet from chapter 6).
2. **McNemar's test** for every binary metric (MMLU, ARC accuracy, GSM8K exact-match, IFEval prompt-strict, HumanEval pass@1) on paired per-item scores.
3. **Wilcoxon signed-rank test** for every continuous metric (task-specific rubric scores, TruthfulQA-mc2, HumanEval pass@1 if you averaged over samples).
4. **Position-bias fraction** for the pairwise judged lane.
5. **Length-controlled pairwise win rate** for the judged lane (bin-by-length + AlpacaEval correction if implemented in exercise 03).
6. **Judge-vs-human ρ / κ** on the anchor from exercise 02, reported for both adapters' judged runs.

Emit `stats.json` keyed by `(lane, task, metric)` with fields `{value, ci_lo, ci_hi, delta_vs_ref, paired_p, verdict}` where `verdict ∈ {"improved", "regressed", "no-signif-change", "insufficient-signal"}`.

### Part D — Compute per-item diffs

Write `diff_items.py` that reads `stats.json` and the per-item score dumps for both adapters, and emits `per_item_diff.jsonl`:

```json
{
  "eval": "task_specific/billing",
  "item_id": "eval-2026-08-0087",
  "score_ref": 3,
  "score_new": 5,
  "delta": +2,
  "prompt_preview": "...",
  "response_ref": "...",
  "response_new": "..."
}
```

Also emit `top_regressions.md` (the 20 items where adapter N regressed the most vs. N−1) and `top_improvements.md` (the 20 items where adapter N improved the most). These are what a release captain skims before sign-off; chapter 6 lists them as the per-lane report's most-consulted artifact.

### Part E — Assemble the regression report

Write `render_report.py` that consumes `stats.json`, the per-item diffs, and the manifest, and produces `report.md` in the exact chapter 6 structure:

1. **Header** — gate version, timestamp (from the run manifest — do NOT compute at report-render time; the report should be reproducible from the JSONs), adapter hashes, overall verdict (`PASS` / `SOFT-PASS` / `BLOCK`).
2. **TL;DR** — 5–7 bullets. Deploy recommendation, wins, watches, contamination, safety.
3. **Lane 1 — Academic benchmarks** — the six-task table with `v[N-1] [CI]`, `v[N] [CI]`, Δ, paired p, gate status.
4. **Lane 2 — Task-specific held-out** — per-stratum rubric-mean table with CIs and paired p, plus the `critical` stratum with hard-gate outcome.
5. **Lane 3 — Judged** — single-answer means (with CIs and paired p); pairwise raw win rate; pairwise length-controlled win rate; position-bias fraction; mean response length per adapter; judge-vs-human agreement on anchor; inter-judge agreement (if run).
6. **Lane 4 — Safety** — actual numbers if you ran mod-106's evals, or an explicit "safety lane deferred to mod-106" placeholder with the rationale.
7. **Contamination** — the exercise-04 table diffed between the two adapters (any *new* hits are highlighted).
8. **Per-item diff highlights** — pointer to `top_regressions.md` and `top_improvements.md`.
9. **Deploy recommendation** — the verdict and the specific soft-gate items (if any) the captain must sign off on.

The report should be readable end-to-end in five minutes; every table should stand on its own.

### Part F — Run the gate twice and prove reproducibility

Run `run_gate.py` end-to-end against both adapters. Then rerun it — same manifest, same artifacts, same seeds, same hardware.

For every metric in `stats.json`, verify:

- Log-likelihood metrics (MMLU / ARC / TruthfulQA-mc2) match to bitwise identity.
- Greedy-generation metrics (GSM8K strict, HumanEval pass@1 at 1 sample, IFEval prompt-strict) match to 3 decimal places.
- Judged metrics match within one item at both position orderings (judge APIs have residual non-determinism at T=0; expect ±1 item wobble on ~500 items).

Emit `repro_check.md` with the max deviation per metric. If anything moves more than the acceptable band, chase it down before submitting — the gate is only credible if it produces the same numbers on rerun.

### Part G — Write the deploy note

Produce `deploy_note.md`, one page, intended for the release captain and the on-call rotation:

- **Recommendation.** `Deploy` / `Deploy with disclosure` / `Hold`.
- **Rationale.** Three sentences citing the strongest evidence.
- **Watches for on-call.** Any soft-gate failure that on-call should page on if it manifests in production (e.g., "if pairwise win rate against v[N-1] on production traffic drops below the offline number by more than 10 pp, page").
- **Rollback trigger.** The concrete metric that would cause a rollback in the first 48 hours.

This is the artifact the release captain actually acts on. The full `report.md` is the audit trail; the deploy note is the decision.

### Part H — Save

Directory layout:

```
outputs/exercise-05/
├── release_gate.v1.yaml
├── manifest_rationale.md
├── run_gate.py
├── compute_stats.py
├── diff_items.py
├── render_report.py
├── results/
│   ├── v[N-1]/
│   │   ├── academic/loglik/results_*.json
│   │   ├── academic/gen/results_*.json
│   │   ├── task_specific/scored.jsonl
│   │   ├── judged/single_*.jsonl
│   │   ├── judged/pairwise_*.jsonl
│   │   └── contamination/*.md
│   └── v[N]/
│       └── ... (same shape)
├── run_manifest.json                     # stamps, hashes, wall-clock, cost
├── stats.json                            # per (lane, task, metric) verdict
├── per_item_diff.jsonl
├── top_regressions.md
├── top_improvements.md
├── report.md                             # the full regression report
├── repro_check.md                        # rerun-vs-first-run deviations
└── deploy_note.md                        # one-page decision doc
```

## Starter guidance

- **Do not skip the manifest.** The temptation is to write ad-hoc scripts that hardcode paths and thresholds. The manifest *is* the release contract; treat it as the first-class artifact. Every threshold you set is a promise about how you will read the numbers.
- **Reuse everything from prior exercises.** `run_gate.py` should mostly shell out to (or import) the scripts from exercises 01–04. If you find yourself re-implementing something, back up — the point of this exercise is integration, not re-derivation.
- **Cache aggressively.** Chapter 6's cost section. Judge calls keyed on `(prompt, response, rubric_version, judge_model, judge_seed)` should hit a cache on the second adapter's run wherever the response is identical to the first (rare, but non-zero on very deterministic outputs). Embeddings for the contamination scan should be cached across all runs of this exercise.
- **Fix all sources of non-determinism up front.** Chapter 7 pitfalls 2, 4, 5. Fix seeds, fix sampling to greedy, fix judge temperature to 0, disable server-side prompt caching for the gate runs, pin backend versions in the manifest.
- **Downsample on the first end-to-end pass.** Run the whole gate with `--limit 30` on every lane first; verify the report renders and the numbers are sensible; then remove the limit and pay for the real numbers. Chapter 6's cost table warns about this.
- **Include the `deferred_to: mod-106` placeholder in the safety lane if you don't have a red-team set.** Do not silently skip. A gate without safety is a gate that failed to name safety as a requirement.
- **The report is the deliverable, not the JSON.** A reviewer will read `report.md` and `deploy_note.md`, then maybe dip into `top_regressions.md`. The JSONs exist for the platform (mod-111) to consume. Optimize the two Markdown files for a human reader.
- **Write the deploy recommendation last.** After you have looked at every lane's numbers and skimmed the per-item diffs. Do not decide `SOFT-PASS` before seeing the top regressions.

## Acceptance criteria

You have completed the exercise when:

- `release_gate.v1.yaml` exists, contains all four lanes (`safety` may be `deferred_to`), pins the eight reproducibility dimensions, and specifies at minimum four `hard` and four `soft` gate rules with justifications.
- `run_gate.py` executes end-to-end against both adapters and writes results into `results/v[N-1]/` and `results/v[N]/`.
- `stats.json` has bootstrap CIs and paired-test verdicts for every gate rule in the manifest.
- `per_item_diff.jsonl`, `top_regressions.md`, and `top_improvements.md` exist with 20 items each per lane.
- `report.md` follows the chapter 6 structure exactly, has a header + TL;DR + one section per lane + contamination + per-item diff pointer + deploy recommendation, and cites the adapter hashes and gate version.
- `repro_check.md` shows a rerun of the gate with per-metric deviation and passes the acceptable-drift bands.
- `deploy_note.md` fits on one page and names a rollback trigger.
- The overall verdict in `report.md` is one of `PASS`, `SOFT-PASS`, or `BLOCK`, and the reason maps mechanically to the manifest's rules — no verdicts that don't trace back to a rule.

## Stretch goals

- **Wire the gate into CI.** Make `run_gate.py` fail (exit non-zero) if any hard gate blocks; add a smaller "fast-lane" script that runs the academic lane + contamination fast lanes on every commit and posts a summary to your Git provider. Chapter 6's "fast lanes per commit" pattern.
- **Compute the gate's own noise floor.** Run the full gate three times against the *same* adapter with three different seeds. Report the observed per-metric variance. Use it to justify (or tighten) your gate thresholds — chapter 6's "thresholds at 2–3× the CI half-width" prescription becomes empirically grounded rather than a guess.
- **Two-judge ensemble.** Run the judged lane with two judges from different families (GPT-4o + Claude Sonnet). Add inter-judge agreement to the report. If they disagree markedly, investigate; the disagreement pattern often reveals a systematic issue with the fine-tune's output style.
- **Batch-API judging for cost.** Run the judged lane through OpenAI Batch or Anthropic Batch API for ~50% cost savings; report the cost delta in `run_manifest.json`. Acceptable because the release gate can tolerate a 24-hour SLO.
- **Diff two gate versions.** Write a `release_gate.v2.yaml` that tightens one threshold (e.g., MMLU regression from 2 pp to 1 pp), re-run the reports against both adapters, and note which rules would have fired under v2 but not v1. This is chapter 6's "every threshold change is a manifest bump" workflow.
- **Manifest schema validation.** Author a JSON Schema for the manifest and validate on load. Chapter 6 emphasizes the manifest as a contract; a schema is what enforces the contract mechanically.
- **Regression tracker.** Store `stats.json` from every gate run in a small SQLite / DuckDB file. Add a `history.md` command that shows adapter-over-time trends per metric. This is a mini-version of what mod-111 will build at the platform altitude.
- **Downstream artifact contract.** Package the outputs of this exercise as the interface handoff to mod-109 (quantization) — a directory containing the manifest, the results JSONs, the per-item scores, and the disclosure. Chapter 7 documents this interface; producing it here is a real deliverable for the next module.
