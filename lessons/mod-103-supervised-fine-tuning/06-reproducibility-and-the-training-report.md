# Reproducibility and the Training Report

**Estimated read:** 35 minutes.
**You should be able to, by the end:** produce a manifest that lets a colleague reproduce your SFT run to within noise, and a one-page training report that a governance or safety reviewer can read in five minutes and know what they need to know.

## Why reproducibility is a first-class deliverable

An SFT run is not a one-off ceremony. Every checkpoint eventually gets: revisited (a downstream stage regressed and you need to reproduce the source of the regression), reproduced (a new hire needs to build the same thing from scratch), compared against (a new run's numbers only mean something relative to the previous one), or explained to a stakeholder who was not in the room. A run that cannot be reproduced from its manifest cannot be defended, and a run that cannot be defended cannot be shipped.

Reproducibility in fine-tuning is imperfect — CUDA nondeterminism, driver versions, and hardware differences will always introduce small variance. The goal is not bit-identical. The goal is:

- **Reproduce the loss curve within tolerance** (typically within a few percent, and same qualitative shape).
- **Reproduce the eval numbers within tolerance** (typically within ~1 percentage point on aggregate benchmarks; within the standard error on small ones).
- **Rerun cleanly** — the manifest is enough to spin up a fresh environment and start the run, without hunting through Slack for what version of TRL you had installed.

## The run manifest: what has to be in it

A manifest is a single file (`manifest.yaml`, `run.yaml`, whatever your team's convention is) that fully specifies the run. Six sections, none of them optional.

### 1. Base model and tokenizer, pinned by revision

Names are aliases; revisions are what actually resolves.

```yaml
base_model:
  repo: meta-llama/Llama-3.2-1B
  revision: 9213176726f574b556790deb65791e0c5aa438b6   # pin the actual commit hash
  precision: bfloat16
  attn_implementation: flash_attention_2

tokenizer:
  repo: meta-llama/Llama-3.2-1B
  revision: 9213176726f574b556790deb65791e0c5aa438b6
  chat_template_source: "tokenizer_config.json"        # or a path to a template file you shipped
  chat_template_hash: sha256:<hash of the template string>
```

The revision hash matters because open model repos are mutable. Meta or Alibaba can (and do) push tokenizer_config.json updates that change chat-template behavior without changing the model name. A run pinned only to `meta-llama/Llama-3.2-1B` is not reproducible next week.

### 2. Dataset, pinned by content hash

The dataset is a specific file with a specific content hash. Not "the same HF dataset."

```yaml
dataset:
  name: my-org/customer-support-sft-v3
  source_files:
    - path: data/train.jsonl
      sha256: 8f2e4d7a1b3c9e5f6a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f
      rows: 18432
    - path: data/eval.jsonl
      sha256: 3c7d1e6f5a2b4c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d
      rows: 250
  build_pipeline_ref: "https://git.internal/data/sft-mix/commit/abc123def456"
  filters_applied: ["length", "language", "toxicity", "dedup:minhash-0.75", "decontam:mmlu"]
```

Both the file hashes and a reference back to the mod-102-owned build pipeline. When your run regresses six months from now, "which version of the dataset was this?" has an exact answer.

### 3. Training config, the exact `SFTConfig` used

```yaml
training:
  num_train_epochs: 1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-5
  lr_scheduler_type: cosine
  warmup_steps: 25
  weight_decay: 0.0
  max_length: 2048
  packing: true
  packing_strategy: ffd
  gradient_checkpointing: false
  bf16: true
  seed: 42
  data_seed: 42
  optim: adamw_torch
```

Persist this as YAML *and* as the frozen output of `trainer.args.to_dict()`. TRL and `transformers.Trainer` can add or rename fields between versions; the frozen dict is the source of truth for what your run actually saw.

### 4. Environment: hardware and software

```yaml
env:
  python: "3.11.9"
  cuda: "12.4"
  cuda_driver: "550.90.07"
  gpu:
    model: "NVIDIA A10G"
    count: 1
    memory_gb: 24
  packages:
    torch: "2.4.1+cu124"
    transformers: "4.45.2"
    trl: "0.11.4"
    accelerate: "0.34.2"
    peft: "0.12.0"
    bitsandbytes: "0.43.3"
    flash_attn: "2.6.3"
    datasets: "3.0.1"
```

Populate `packages` from `pip freeze` (or `uv pip freeze`) — do not type versions by hand. Ship the full `requirements.lock` or `uv.lock` alongside the manifest.

### 5. Determinism knobs

```yaml
determinism:
  seed: 42                          # torch, numpy, python random
  data_seed: 42                     # dataset shuffling / packing order
  cuda_deterministic: false         # true forces deterministic algorithms, slower
  cudnn_benchmark: true             # false when cuda_deterministic is true
  torch_compile: false              # compile can introduce non-determinism vs. eager
```

The honest position is: `cuda_deterministic: false` for most SFT runs. Full determinism is expensive (up to 2× slower) and the practical reproducibility target — same-shape loss curve, same-tolerance eval numbers — does not require it. Set `cuda_deterministic: true` only for the run you plan to actually re-run from scratch to prove reproducibility.

### 6. Output artifacts, hashed after the fact

```yaml
outputs:
  final_checkpoint:
    path: outputs/final
    sha256: <hash of model weights (safetensors)>
    tokenizer_sha256: <hash of the exported tokenizer.json>
  loss_curves:
    train_loss_jsonl: outputs/logs/train.jsonl
    eval_loss_jsonl: outputs/logs/eval.jsonl
  training_report_md: outputs/report.md
```

Filled in by a post-run script, not by hand. Any of these hashes changing without a corresponding change in inputs is evidence of a nondeterminism source you have not accounted for.

## Producing the manifest without pain

Do not write the manifest at end of training as a documentation task. Emit it *from* the training script:

```python
import json, hashlib, subprocess, yaml
from datetime import datetime, timezone

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def build_manifest(cfg, model, tokenizer, dataset_paths):
    return {
        "run_id":     f"sft-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "base_model": {
            "repo": model.config._name_or_path,
            "revision": model.config._commit_hash,
            "precision": str(model.dtype),
            "attn_implementation": model.config._attn_implementation,
        },
        "dataset":  {p: sha256(p) for p in dataset_paths},
        "training": cfg.to_dict(),
        "env": {
            "python": subprocess.check_output(["python", "--version"]).decode().strip(),
            "pip_freeze": subprocess.check_output(["pip", "freeze"]).decode().splitlines(),
        },
    }

manifest = build_manifest(cfg, model, tokenizer, ["data/train.jsonl", "data/eval.jsonl"])
with open("manifest.yaml", "w") as f:
    yaml.safe_dump(manifest, f, sort_keys=False)
```

Emit it *before* `trainer.train()` starts. If the run fails, you still have a manifest describing what you attempted, and post-mortem is one file shorter.

## Seeds: a small honest section

Set four seeds:

```python
import random, numpy as np, torch
from transformers import set_seed

SEED = 42
set_seed(SEED)                   # sets python random, numpy, torch (CPU + CUDA)
# TRL / Trainer will also set the seed on the DataLoader / dataset shuffling if you pass seed / data_seed
```

Note the honesty: seeding gets you close to reproducible, not bit-identical. Multi-GPU DataLoader shuffling, FA2 kernel non-determinism, and cuDNN algorithm selection all introduce small variance. Publish the seed anyway — the delta between "same seed" and "different seed" runs is the noise floor you calibrate your interpretation of eval-number deltas against.

## The training report: one page, six sections

At end of run, produce `report.md` — a page a colleague reads in five minutes. Six sections:

### 1. Run identity and provenance

```markdown
# SFT run report — sft-20260802T140000Z

- **Base model:** meta-llama/Llama-3.2-1B @ 9213176
- **Dataset:** my-org/customer-support-sft-v3 (18432 train, 250 eval, sha256:8f2e…4f)
- **Trainer:** TRL SFTTrainer 0.11.4 / transformers 4.45.2 / torch 2.4.1
- **Hardware:** 1× A10G 24GB, CUDA 12.4
- **Run duration:** 47 minutes, 625 optimizer steps.
- **Manifest:** [manifest.yaml](./manifest.yaml)
```

### 2. Objective and dataset summary

One paragraph: what behavior you were trying to induce, dataset row count and category breakdown, and a link to the mod-102 dataset card if separate.

### 3. Training-loop numbers

The two-plot view (train + eval loss) from chapter 4, and a small table:

```markdown
|                | value |
|----------------|-------|
| loss @ step 0  | 3.42  |
| loss @ final   | 0.83  |
| eval loss @ step 100 | 1.24 |
| eval loss @ step 625 | 0.94 |
| best eval loss (step) | 0.91 (500) |
| final gradient norm | 0.42 |
| final LR       | 2.0e-6 (cosine, floor) |
| effective batch (tokens) | 32768 |
```

Plus a one-sentence diagnosis: "Loss curve healthy per chapter-4 shape guide; no divergence, small train/eval gap."

### 4. Held-out benchmark deltas (base → SFT)

Never ship a training report without this table.

```markdown
| benchmark            | base  | SFT   | delta |
|----------------------|-------|-------|-------|
| MMLU (subset 500)    | 45.2  | 44.7  | -0.5  |
| GSM8K (100 problems) | 12.0  | 11.0  | -1.0  |
| IFEval (loose)       | 21.0  | 44.5  | +23.5 |
| ARC-easy (200)       | 71.5  | 71.0  | -0.5  |
```

Interpretation in a sentence: "IFEval up substantially as expected from instruction data; small MMLU/GSM8K decline within noise for this subset size; no catastrophic forgetting flagged." If the delta is worrying, say so and link to the chapter-5 mitigation you plan to try next.

### 5. Template smoke test

The 20-generation smoke test from chapter 5, section 3. A count plus one or two flagged examples if any.

```markdown
- 20 prompts at temperature 0, `max_new_tokens=200`.
- 20/20 responses ended with the expected `<|eot_id|>` terminator.
- 0/20 responses contained stray template tokens in the body.
- 3/20 responses truncated at `max_new_tokens`; those prompts were "write a 500-word essay…" which is expected.
```

### 6. Known issues and next steps

The honest section. What is not yet done and what you would do next.

```markdown
- No MMLU-full run (only 500-row subset); a full-set run is queued.
- Safety evaluation not run in this cycle; scheduled for the next.
- LR sweep pending: exercise 03 candidates are 1e-5, 2e-5, 5e-5.
```

That is the whole report. Six sections, one page. Reviewers read it in five minutes. It is the *primary artifact* you hand to any downstream stakeholder — not the checkpoint, not the loss curve.

## Storing runs so future-you can find them

- One directory per run, named after `run_id`. Contains manifest, training config, log files, checkpoint pointers, and report. Never mix runs into one directory.
- Registry: a single index (`runs.jsonl`, a small SQLite, a table in your team's tracking tool) with `run_id`, `base_model`, `dataset_hash`, `eval_loss_final`, `mmlu_delta`, and a link to the report. Six months from now, "which run did we ship?" is a one-line query.
- Weights & Biases / MLflow / Aim etc. integrations are fine but do not replace the on-disk manifest. Cloud tracking services go away, get migrated, or lose data. The on-disk manifest is the source of truth.

## Handoff to mod-105 and mod-109

The mod-103 SFT checkpoint is the input to two downstream modules:

- **mod-105 (preference optimization)** will re-load your checkpoint as the reference model in a DPO run. It needs the tokenizer, the exact chat template, and the manifest's model revision hash to reproduce your training-time context.
- **mod-109 (quantization / export)** will merge or quantize your checkpoint into a serving artifact. It needs the tokenizer, and the manifest's precision + attn implementation so the quantized artifact behaves the same way.

Both downstream stages will accept a checkpoint that does not carry its manifest, and both will occasionally regress mysteriously when the training-time context is not preserved. Ship the manifest with the checkpoint; treat them as one artifact.

## Exercise pointer

`exercises/exercise-05-reproducibility-and-training-report.md` has you take a run you already did in an earlier exercise, produce the full manifest, produce the one-page training report, and demonstrate the run reproduces (loss curve within tolerance, eval numbers within tolerance) from the manifest alone.

## Summary

- A reproducible SFT run means loss curves within tolerance and eval numbers within a percentage point — not bit-identical. Manifests exist to make this achievable.
- Six-section manifest: base model + tokenizer (pinned by revision), dataset (pinned by content hash), training config (as `to_dict()`), env (from `pip freeze`), determinism knobs, and post-run output hashes.
- Emit the manifest from the training script, before `trainer.train()`, not as a documentation task at the end.
- Seed everything (`transformers.set_seed`); accept that this gets you close to reproducible, not bit-identical.
- The training report is one page, six sections: identity, objective, training numbers (two-plot + table), held-out benchmark deltas, template smoke test, known issues. This is the primary artifact you hand downstream, not the checkpoint.
- Store one run per directory, register runs in an index, and treat the checkpoint + manifest + report as one indivisible artifact for handoff to mod-105 and mod-109.
