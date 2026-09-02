# exercise-03: Shared Eval Harness Across Vendors

**Estimated effort:** 3 hours
**Companion chapters:** [`07-eval-handoff-across-backends.md`](../07-eval-handoff-across-backends.md), [`02-openai-end-to-end.md`](../02-openai-end-to-end.md), [`05-cost-privacy-ip-residency.md`](../05-cost-privacy-ip-residency.md)

## Objective

Extend the mod-108 release-gate manifest so that a **single command** runs the same tasks, at the same sampling settings, against **one self-hosted candidate** (vLLM / TGI / SGLang serving your mod-104 or mod-105 adapter) **and at least one hosted candidate** (the OpenAI `ft:...` from exercise-01, or another managed fine-tune you already have). Reconcile the settings so the resulting leaderboard is genuinely like-for-like. Produce a comparison report a release captain could sign — the artifact bundle that turns "we have a managed fine-tune and a self-hosted fine-tune" into "we have a defensible answer to which one to ship."

The point of this exercise is the reconciliation. The trap it teaches you to avoid: a leaderboard that says the managed candidate wins by 8 points on MMLU when the truth is that the self-hosted candidate ran MMLU-logprob and the managed candidate ran MMLU-generative. Chapter 7 has the pitfalls; this exercise is where you catch them.

## Prerequisites

- Exercise-01 completed — you have a `ft:...` OpenAI model ID, its manifest, and eval outputs at `results/openai-ft/` and `results/base/`. If you skipped exercise-01, substitute any hosted fine-tune you can call (a Vertex, Mistral, or Bedrock one you already have, or a Cohere fine-tune) — the mechanics are the same.
- A **self-hosted candidate** on the same task: either your mod-104 QLoRA adapter or your mod-105 DPO checkpoint, served via vLLM / TGI / SGLang in OpenAI-compatible mode (`vllm serve --model ... --served-model-name your-model`). If you have no self-hosted candidate, the base model served locally is an acceptable substitute — you are then measuring "hosted FT vs. self-hosted base" instead of "hosted FT vs. self-hosted FT."
- The pinned `lm-evaluation-harness` install from mod-108 exercise-01 (same commit SHA — reproducibility carries forward).
- A judge model for the mod-108 chapter-4 lane (Claude Sonnet 4.5+, GPT-4.1, or another strong non-candidate model). Do not use a model in the candidate set as its own judge.
- LiteLLM (or another OpenAI-compat gateway) installed if you plan to add a third vendor (Anthropic / Vertex / Bedrock) as a stretch. Not required for the mandatory two-candidate baseline.
- API budget: expect $5–$50 for the hosted-side eval (the fast lane on a task-specific held-out + IFEval + GSM8K is usually the low end; a full academic-suite run is the high end). Budget is set by your fast-lane subset (Part B).

## Requirements

### Part A — Define the shared gate manifest

Author `gate.yaml`. This is the single source of truth for the run. Required top-level shape (chapter 7's snippet is the reference):

```yaml
gate_manifest:
  name: <your-project>-cross-backend-gate
  version: 2026-09-02
  sampling:
    temperature: 0
    top_p: 1.0
    max_tokens: 1024
    seed: 42
    samples_per_problem: 1
  chat_template_policy: server_side_only    # do NOT --apply_chat_template with hf-native tokens
  tasks: [ ... ]
  candidates: [ ... ]
```

Required task list — at minimum:

- `<your_task_held_out>` — the task-specific held-out set from mod-108 exercise-02. Judged by your rubric or the mod-108 chapter-4 judge.
- `ifeval` — instruction-following, generation-based, portable across every backend.
- `gsm8k` — arithmetic reasoning, generation-based, portable.
- ONE academic task that is generation-based across backends (e.g., `bbh_generative`, `mmlu_generative`, or `humaneval`). Do NOT include a log-likelihood-only task in the shared gate — chapter 7's log-likelihood-gap discussion. If you want the log-likelihood scores, put them in a separate self-hosted-only lane; keep them out of the cross-backend comparison.

Candidates (at minimum two, both defined in the manifest):

- `self_hosted_<name>` — `base_url: http://localhost:8000/v1`, `model: <your-served-model-name>`.
- `openai_ft_<name>` — `base_url: https://api.openai.com/v1`, `model: ft:...`.

Every threshold on which the gate would `PASS` / `SOFT-PASS` / `BLOCK` also belongs in `gate.yaml`, following mod-108 chapter 6's rubric. This is optional for the exercise but strongly recommended — a gate without thresholds is a leaderboard, not a gate.

### Part B — Author the fast-lane subset

Author `fast_lane.yaml`. Chapter 7's third cost lever: run the full gate rarely, the fast lane per candidate. Requirements:

- A subset of `gate.yaml`'s tasks that runs in under 15 minutes on the self-hosted side and under $5 on the hosted side.
- Deterministic subset selection — pin the exact question IDs, not "20 random items" (which would re-sample per run and break comparability).
- Includes at least the task-specific held-out set at full size (or a stratified subset covering all mod-108 chapter 3 strata) — this is the primary signal.
- Includes at least one academic task at reduced size (e.g., first 200 items of `gsm8k`).

Every metric that appears in `gate.yaml`'s thresholds must also be computable from `fast_lane.yaml`. Otherwise the fast lane can pass while the full gate would fail — the exact failure mode fast lanes are supposed to prevent.

### Part C — Build the runner

Author `run_gate.py` (or extend the runner from mod-108 exercise-05 if you did it). Requirements:

1. Reads `gate.yaml` (or `fast_lane.yaml` with `--fast`).
2. For each candidate, drives `lm-evaluation-harness` with:
   - The correct backend per candidate (`local-chat-completions` for self-hosted vLLM in OpenAI-compat mode; `openai-chat-completions` for OpenAI direct; `anthropic-chat-completions` for Anthropic direct if you add it).
   - The sampling block from `gate.yaml`, passed as `--gen_kwargs`.
   - `--apply_chat_template --fewshot_as_multiturn` where the backend expects chat formatting (chapter 7's chat-template reconciliation section is the reference — server-side templating is authoritative for hosted models).
   - `--log_samples --write_out` for every task, always. The sample dumps are how you diagnose the mismatches Part E hunts.
3. Writes results to `results/<candidate_id>/<task>/`.
4. Idempotent: rerunning on the same manifest, same seed, same model IDs re-uses the cached per-item scores (mod-108 chapter 6's discipline). Only re-runs items whose (candidate, task, item, sampling_hash) tuple is not cached.
5. On completion, writes `results/run_manifest.json` stamping: `gate.yaml` hash, per-candidate model IDs, per-candidate lm-eval command line, wall-clock per task, cost per task (rough — token count × current price, from a shared `prices.yaml`).

The runner must NOT paper over failures. If the OpenAI backend errors on IFEval, `run_gate.py` exits nonzero and surfaces the error. Silent-skip is worse than fail-loud.

### Part D — Compute stats and produce the leaderboard

Author `leaderboard.py` that consumes `results/` and produces:

1. **Score table** — one row per candidate, one column per task, with bootstrap 95% CIs (mod-108 chapter 3's 1000-resample recipe). Include per-request cost estimate (input+output tokens × current price) as a final column.
2. **Delta table** — pairwise deltas between candidates on every task, with paired significance tests (McNemar's for exact-match tasks, Wilcoxon signed-rank for continuous). Chapter 7 rules apply: a 1-point delta with overlapping CIs is not a signal.
3. **Refusal rate** — count of items where the candidate refused / declined to answer (empty output, safety-refusal string). Chapter 7 pitfall: silently scoring refusal as "wrong" hides a real quality signal.
4. **Length statistics** — mean output length per candidate per task, sample dumps of the longest and shortest outputs. Judged lanes are subject to length bias (mod-108 chapter 4's verbosity discussion); this table is where you catch it.

Write output as `leaderboard.md` (Markdown table for humans) and `leaderboard.json` (structured, for the next tool to consume).

### Part E — Reconciliation checks (the point of the exercise)

Author `reconcile.md`. For every task in `gate.yaml`, walk through the six-check reconciliation:

1. **Same prompt.** Pull one item from `results/<candidate>/<task>/samples.jsonl` per candidate for the same item ID. Confirm the *prompt-as-sent* is byte-identical modulo chat template. If they differ (extra system prompt on one, different few-shot exemplars, different newlines), stop and fix `run_gate.py`.
2. **Same sampling.** Grep the recorded lm-eval command lines in `run_manifest.json` for `--gen_kwargs`. Confirm every candidate got the same temperature / top_p / max_tokens / seed. This is the most-common reconciliation bug; do it every time.
3. **Same chat template application.** Confirm no candidate double-applied the chat template (self-hosted `hf` tokens wrapping an API-chat message; a client-side template on top of the vendor's server-side one). Chapter 7's chat-template rule: **let the vendor apply the chat template server-side; do not apply one yourself on API backends.**
4. **Same task variant.** Confirm you did NOT mix MMLU-logprob and MMLU-generative across candidates (chapter 7's log-likelihood gap). If any candidate is scored on a log-likelihood variant, every candidate must be — or drop the task from the shared gate.
5. **Same scorer.** Confirm the same scorer (`exact_match_strict`, `prompt_level_strict_acc`, `judge_correct_rate`) ran on every candidate for the same task. lm-eval will complain if a task's scorer is not compatible with a backend; do not silently fall back.
6. **Same judge, same judge seed.** For the judged lane, confirm the same judge model, judge system prompt, and judge seed were used across candidates. Position bias correction (mod-108 chapter 4) must have been applied uniformly.

Every check that fails must be logged with the candidate + task + observed mismatch, and either fixed (rerun that candidate/task) or documented as a caveat in the final report.

### Part F — Write the comparison report

Author `report.md`. Sections:

1. **Setup.** The candidates, the tasks, the sampling, the judge, the pinned lm-eval SHA. Point at `gate.yaml`.
2. **Reconciliation.** The `reconcile.md` summary — the six checks, per task, per candidate, with any caveats. This is the "why this leaderboard is trustworthy" section.
3. **Leaderboard.** The Part D table, with a two-paragraph interpretation. Which candidate wins on which axis? Are the wins statistically significant?
4. **Cost profile.** For each candidate, the per-request cost. If the candidate is self-hosted, use your mod-109 (or exercise-02 in this module) cost-per-request number. If hosted, use `token_count × current_price` from your pricing file. This is the numerator; the report's decision must reference both quality and cost together.
5. **Refusal and length behavior.** The Part D tables. Any candidate refusing >2% of items warrants a paragraph on why.
6. **Recommendation.** One paragraph. Ship candidate X, because (quality, cost, refusal, portability). If you cannot recommend one, say so and name the specific signal that would break the tie.
7. **Contamination disclosure.** Chapter 7's shape: fine-tuning data scanned (Y/N + result), base-model pretraining data (scannable per candidate: Y for self-hosted open, N for hosted closed), model-signal probes run on academic tasks.
8. **Released-artifact records.** Chapter 7's `released_artifact:` YAML block for every candidate (self-hosted → weight hash; hosted → model_id + base_pinned_version + fine_tune_job_id + training_file_sha256 + eval_gate_manifest_version). Save under `artifacts/`.

Keep it under 2000 words. The reader is a release captain who has read the leaderboard once and wants to know why to trust it.

### Part G — Directory layout

Final tree:

```
outputs/exercise-03/
├── gate.yaml
├── fast_lane.yaml
├── prices.yaml
├── run_gate.py
├── leaderboard.py
├── results/
│   ├── run_manifest.json
│   ├── self_hosted_<name>/
│   │   └── <task>/samples.jsonl, metrics.json
│   └── openai_ft_<name>/
│       └── <task>/samples.jsonl, metrics.json
├── leaderboard.md
├── leaderboard.json
├── reconcile.md
├── report.md
└── artifacts/
    ├── released_artifact_self_hosted.yaml
    └── released_artifact_openai_ft.yaml
```

## Starter guidance

- **Do Part E before Part F.** If the leaderboard was produced without the six-check reconciliation, the report is a fiction. Reconciliation is the load-bearing step — the leaderboard is only meaningful downstream of it.
- **Serve the self-hosted candidate in OpenAI-compat mode.** `vllm serve --model /path/to/adapter --served-model-name my-ft` produces an OpenAI-compat endpoint at `localhost:8000/v1`. Then both candidates use the same `local-chat-completions` (or `openai-chat-completions`) backend shape — one lm-eval invocation per candidate with only `base_url` and `model` different.
- **Dump prompts on every run.** `--write_out` is not optional; the six-check reconciliation depends on being able to diff the actual-prompt-sent per candidate. If the disk cost is a concern, keep only the first 5 samples per task per candidate — enough to eyeball.
- **Use a shared `prices.yaml`.** One file, one row per model ID, with `input_per_1M`, `output_per_1M`, and `retrieved_date`. `leaderboard.py` reads this file; when prices shift, updating this one file re-computes every cost cell.
- **Cache aggressively.** For a rerun after a Part E fix, only the affected (candidate, task) tuple should re-hit the API. Mod-108 chapter 6's cache discipline; a naive re-run without caching costs 2× the money for no additional signal.
- **Do the fast lane first.** Run `fast_lane.yaml` on both candidates before running the full `gate.yaml`. If the fast lane shows anomalies (a candidate scoring 30 points below expected, refusal rate >5%), diagnose before spending on the full run.
- **Do NOT include MMLU-logprob and MMLU-generative in the same gate.** Chapter 7's log-likelihood-gap discussion. Pick one; use it everywhere; note the choice in the report.
- **Let the vendor apply the chat template.** Do not hand-roll a chat template client-side for the hosted candidate — the vendor's server-side template is authoritative and often differs from any local reproduction. Chapter 7's chat-template reconciliation rule.
- **Bring the OpenAI Batch API in on Part D re-runs.** Chapter 7's cost lever 2. If your gate runs weekly, batch the eval — synchronous is fine for the first run, batch pays for itself starting on run 2.

## Acceptance criteria

You have completed the exercise when:

- `gate.yaml` and `fast_lane.yaml` are complete, with pinned sampling, pinned chat-template policy, at least the four required tasks, and at least two candidates.
- `run_gate.py` produces byte-identical `results/` on a re-run against the cache.
- `results/run_manifest.json` records the exact `lm_eval` command lines used per candidate — so a reviewer can reproduce every score without reading `run_gate.py`.
- `leaderboard.md` presents the score table with bootstrap CIs, the delta table with paired significance tests, and the per-candidate refusal rate and length stats.
- `reconcile.md` walks all six checks for every task × candidate cell, with any caveats explicitly logged.
- `report.md` includes all eight sections and names a specific recommendation.
- The `released_artifact_*.yaml` records under `artifacts/` are complete per chapter 7's template — each with a model identifier, a base pin, and a pointer to the training-side manifest.
- The gate reruns cleanly on cache (a `run_gate.py` invocation with no code changes produces `SAME` on every task × candidate cell).

## Stretch goals

- **Add a third vendor via a compatibility gateway.** Stand up LiteLLM pointing at Anthropic (or Vertex, or Bedrock), add a third candidate to `gate.yaml`, and re-run. Chapter 7's Vertex / Bedrock / Azure section. Which vendor's chat-template quirks trip the six-check reconciliation? Document.
- **Add a production-mirror sampling lane.** Duplicate `gate.yaml` as `gate_production_mirror.yaml` with `temperature=0.7` (or whatever your production sampling actually is) and `samples_per_problem=5`. Re-run. Are the winners the same at production sampling? If not, which sampling should your release-gate use? Mod-108 chapter 7's production-mirror discussion.
- **Add the Batch API lever.** Wrap `run_gate.py` with a `--batch` mode that submits every OpenAI request via the Batch API. Compare wall-clock and cost against synchronous mode. Chapter 7's cost lever 2.
- **Add a contamination probe on the base models.** Chapter 7's lens-3 model-signal probe (mod-108 chapter 5). Run it on GSM8K for both candidates and report the delta. Does the closed model show more signal of contamination than the open one on public benchmarks?
- **Add a "released-artifact hash" verifier.** Author `verify_released_artifact.py` that takes a `released_artifact.yaml`, re-runs the fast lane, and produces `PASS / FAIL` based on whether the current scores match the recorded scores within CI. This is what a "did the model change under us" watchdog looks like — a routine cron on both self-hosted and hosted candidates.
- **Add mod-111 hand-off.** Emit `registry_entry.yaml` per candidate, following whatever shape mod-111 chapter 4 defines for the model registry (or, if mod-111 is unwritten in your track, draft the shape yourself). This makes the exercise's output ingestable by the platform layer that comes next.
