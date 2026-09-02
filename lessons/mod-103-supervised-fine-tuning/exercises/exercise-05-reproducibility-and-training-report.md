# exercise-05: Reproducibility and Training Report

**Estimated effort:** 3 hours
**Companion chapter:** [`06-reproducibility-and-the-training-report.md`](../06-reproducibility-and-the-training-report.md)

## Objective

Take an SFT run you already produced in a previous exercise (exercise 01's baseline is the default choice; the exercise-03 sweep winner or exercise-04 mitigation run are equally valid), produce a full six-section reproducibility manifest for it, produce the one-page training report chapter 6 specifies, and then **prove** the run reproduces from the manifest alone — a colleague (or you, six months from now) can re-launch the run and land within a small tolerance on the same loss curve and eval numbers.

The deliverable is not "a manifest and a report exist." The deliverable is a manifest and report a stranger can act on, plus a second run whose numbers match the first within the tolerance you defined. If the reproduction fails, that failure — and the diagnosis of the reproducibility gap — is the exercise.

## Prerequisites

- At least one completed SFT run from an earlier exercise. Exercise 01's `outputs/final/` baseline is the reference case; exercise-03's `pick.md` winner or exercise-04's `mitigation_sft/` also work.
- The original training script and config you used for that run.
- Enough compute to run the same job one more time (the "reproduction" run). Budget the same wall-clock as the original.
- `pyyaml`, `hashlib` (stdlib), and whichever logging backend (WandB / TensorBoard / JSONL) you have been using across exercises.

## Requirements

### Part A — Emit the manifest from the training script

Author (or refactor) a `build_manifest.py` — or fold the logic into `train.py` — that emits `manifest.yaml` **before** `trainer.train()` starts. The manifest must have all six sections from chapter 6:

1. **`base_model` / `tokenizer`** — repo, revision hash (from `model.config._commit_hash` or `huggingface_hub.HfApi().repo_info(...).sha`), precision, `attn_implementation`, and the hash of the chat template string.
2. **`dataset`** — for every JSONL / Parquet / CSV file the trainer reads, capture SHA-256, row count, and a link back to the mod-102-owned build pipeline (a commit SHA of your data repo is fine). Do not settle for "the HF dataset name" — that is not a hash.
3. **`training`** — the frozen output of `trainer.args.to_dict()` (or the `SFTConfig.to_dict()` you passed in). YAML-serialize it as-is; do not hand-transcribe.
4. **`env`** — Python version, CUDA version, driver, GPU model and count, and the full `pip freeze` (or `uv pip freeze`) output. Ship `requirements.lock` (or `uv.lock`) alongside the manifest.
5. **`determinism`** — every seed you set (`seed`, `data_seed`), whether `torch.use_deterministic_algorithms(True)` was called, `torch.backends.cudnn.benchmark`, and `torch.backends.cudnn.deterministic`.
6. **`outputs`** — populated *after* the run by a small post-run script that hashes the final checkpoint, the tokenizer, and the loss-log files, and merges those hashes back into `manifest.yaml`.

Emit the manifest *before* the first optimizer step. If the run crashes at step 50, you still have a manifest describing what you attempted.

### Part B — Produce the one-page training report

Write `report.md` at the end of the run, following chapter 6's six-section template exactly:

1. **Run identity and provenance** — base model + revision short-hash, dataset name + row counts + short hash, trainer/framework versions, hardware, run duration, optimizer steps, link to the manifest.
2. **Objective and dataset summary** — one paragraph. What behavior you were inducing, dataset row count and category breakdown, link to the mod-102 dataset card if you have one.
3. **Training-loop numbers** — the two-plot view from chapter 4 (train + eval loss) as `report_images/loss.png`, plus the small numeric table (step-0 loss, final loss, best eval loss and its step, final gradient norm, final LR, effective batch in tokens). One-sentence diagnosis against chapter 4's shape guide.
4. **Held-out benchmark deltas (base → SFT)** — a table of at least one general benchmark and one targeted-capability benchmark, with the delta and a one-sentence interpretation. If you did exercise 04, reuse those numbers.
5. **Template smoke test** — the 20-generation smoke test from chapter 5, section 3: count of responses that ended with the expected terminator, count containing stray template tokens, count that truncated at `max_new_tokens`. Include one flagged example verbatim if any surfaced.
6. **Known issues and next steps** — the honest section. What you did not evaluate, what would be your next config change, what is queued.

Ship the report as `report.md` in the run directory. A safety or governance reviewer should be able to read it in five minutes and know what they need to know; if you find yourself needing more than a page, cut, don't append.

### Part C — Reproduce the run from the manifest

This is the load-bearing part of the exercise. You are proving the manifest is enough.

1. In a **fresh working directory** (or a fresh git worktree, or on a different machine — anything that guarantees you are not carrying hidden state from the original), install the exact package versions from `requirements.lock`.
2. Re-download the base model and tokenizer, **pinned to the revision hash from the manifest** (`AutoModelForCausalLM.from_pretrained(REPO, revision=REV_HASH)`), not to the branch alias.
3. Re-materialize the dataset files. Verify their SHA-256 against the manifest before continuing. If a hash mismatches, stop — your dataset is not what you thought it was.
4. Reconstruct `SFTConfig` from the manifest's `training` section. Set the same seeds. Set the same determinism flags.
5. Launch training. Log to a new run directory (`outputs/reproduction/`).

Define the tolerance **before** you look at the reproduction results:

- **Loss-curve tolerance:** train and eval loss at every logged step within X% of the original. Pick X honestly — 5% is a reasonable default at BF16 on non-deterministic CUDA; 1% is only achievable with `cuda_deterministic: true` and same-hardware.
- **Eval-number tolerance:** the held-out benchmark scores within Y percentage points of the original. Y should be roughly the benchmark's own standard error at your subset size.

### Part D — Report the reproduction

Add a `reproduction.md` (short — one page) to your run directory containing:

1. **Tolerance definitions** for loss and eval numbers, decided *before* you ran the reproduction.
2. **Side-by-side numeric comparison** of the two runs: step-0 loss, mid-training loss, final train loss, best eval loss, held-out benchmark scores.
3. **Overlay plot** (`reproduction_loss.png`) with the original and reproduction curves on the same axes, distinguishable by color.
4. **Verdict**: PASS (within tolerance on all metrics) or FAIL (miss on any). If FAIL, a one-paragraph diagnosis: what did the manifest fail to capture? Nondeterministic CUDA kernel? Missing environment variable? Silent tokenizer revision change? Missing dataset row?
5. **If FAIL**, the concrete change you would make to `build_manifest.py` so a future reproduction would succeed.

A first-attempt reproduction that fails and is diagnosed is a stronger deliverable than a lucky pass with no diagnosis. Do not massage the tolerances after the fact.

## Starter guidance

- **Do not hand-write the manifest.** Every field must come from an API (`trainer.args.to_dict()`, `subprocess.check_output(["pip", "freeze"])`, `hashlib.sha256`), not from your memory. Hand-written manifests drift from reality within a single dev cycle.
- **Pin the base-model revision.** The most common quiet reproducibility failure is: someone pushed a new `tokenizer_config.json` to `meta-llama/Llama-3.2-1B` between your original run and the reproduction, the chat template changed, the assistant mask changed, and your loss curve moved a percent. Pinning the revision hash is one line and it prevents this class of failure entirely.
- **Reproduce on the same hardware first.** If your reproduction is on a different GPU family, some of the delta you observe is CUDA-kernel variance, not a manifest bug. Do the same-hardware reproduction first; the different-hardware run is a stretch goal.
- **`cuda_deterministic` is expensive.** Chapter 6 recommends leaving it `false` for most runs. If your reproduction is failing loss tolerance by a lot, turning it on for one experimental reproduction can help you separate "manifest bug" from "kernel nondeterminism."
- **The report is written for a reviewer, not for you.** If the section titles read like "loss table" and "eval table" it will read fine to you and be useless to anyone else. Use chapter 6's exact section titles; a governance reviewer scanning several runs benefits from consistency.
- **Bundle everything into one directory.** Chapter 6's storage recommendation is one directory per run — `outputs/<run_id>/` with `manifest.yaml`, `config.yaml`, `logs/`, `report.md`, and (for this exercise) `reproduction/` and `reproduction.md` as children. The full artifact is one `tar` command from being shippable.

## Acceptance criteria

You have completed the exercise when:

- `manifest.yaml` exists with all six sections populated from code, not by hand.
- `requirements.lock` (or `uv.lock`) is committed alongside the manifest.
- `report.md` exists, is one page, and hits chapter 6's six section titles verbatim.
- `outputs/reproduction/` exists with its own log directory and a completed reproduction run.
- `reproduction.md` states the tolerance in advance, compares the numbers, includes the overlay plot, and gives a PASS/FAIL verdict with a diagnosis if FAIL.
- A colleague who has never seen the run can read `report.md` and `manifest.yaml`, run `train.py`, and reproduce your numbers within the stated tolerance.
- The manifest survives the "delete the run directory, re-launch from the manifest" test.

## Stretch goals

- **Reproduce on a different GPU.** Take the manifest to a different GPU model (say, A10G → L4, or L4 → A100). Report the additional delta introduced by hardware variance. Add a `hardware_notes` section to the manifest for future readers.
- **Register the run.** Build the `runs.jsonl` (or SQLite, or your team's tracking-tool table) chapter 6 recommends: one row per run with `run_id`, `base_model`, `dataset_hash`, `eval_loss_final`, the primary benchmark delta, and a link to the report. Six months from now, "which run did we ship?" is a one-line query.
- **Bit-identical run.** Set `torch.use_deterministic_algorithms(True)`, `PYTHONHASHSEED=0`, disable `cudnn.benchmark`, force a single dataloader worker, and try to reproduce loss values *bit-identically* on the same hardware. Report the throughput cost of full determinism.
- **Sign the manifest.** Add a GPG signature or a `cosign`-style attestation to the manifest so a downstream consumer can verify the artifact came from a known signer. This is a small operational change that matters a lot for regulated deployments.
- **Manifest schema in code.** Write a `manifest_schema.py` (a `pydantic` model or a JSON Schema) and validate every manifest against it in CI. A missing field is now a build-time failure, not a "we noticed six months later" failure.
- **Reproduce a colleague's run.** Swap manifests with a peer. Their manifest + their `train.py` + their `requirements.lock` should be enough for you to reproduce their loss curve to their tolerance. If it is not, together you have a concrete improvement to make to the manifest template.
