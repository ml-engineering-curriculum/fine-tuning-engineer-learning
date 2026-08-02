# mod-103-supervised-fine-tuning: Supervised Fine-Tuning End-to-End

**Estimated effort:** 18 hours
**Position in the track:** third module. Builds on mod-101's transformer / tokenizer vocabulary and consumes the training-ready mix from mod-102. Feeds into mod-104 (PEFT), mod-105 (preference optimization), and mod-108 (evaluation).

Mod-102 taught you what to put in an SFT set. This module puts that set through a training loop and hands off a checkpoint that a downstream module — DPO, quantization, serving — can actually use. The through-line is that an SFT run is a small, disciplined engineering project: the config, the seeds, the loss curves, the eval regressions, and the environment together *are* the artifact. A checkpoint without them is not shippable.

## Learning objectives

- Run an SFT job on a 1B–8B base model with TRL `SFTTrainer` (or equivalent in Axolotl / Unsloth)
- Implement prompt/response masking, sequence packing, and gradient accumulation correctly
- Pick learning-rate schedule, warmup, weight decay, and checkpoint cadence by reasoning about loss curves
- Diagnose overfitting, catastrophic forgetting, and template-leak failure modes
- Produce a reproducible run (seeds, config, env, dataset hash) and a 1-page training report

## Lecture chapters

1. [`01-sft-training-loop-anatomy.md`](01-sft-training-loop-anatomy.md) — The SFT Training Loop, End to End. What TRL's `SFTTrainer` actually does under the hood, the minimum viable config for a 1B run, the model / tokenizer / collator / trainer contract, and where Axolotl and Unsloth sit in the same picture.
2. [`02-masking-packing-and-effective-batch-size.md`](02-masking-packing-and-effective-batch-size.md) — Masking, Packing, and Effective Batch Size. Trainer-side loss masking (`DataCollatorForCompletionOnlyLM`, `response_template`), packing with per-example attention masks, gradient accumulation math, and the "why is my throughput awful?" checklist.
3. [`03-learning-rate-warmup-and-checkpoint-cadence.md`](03-learning-rate-warmup-and-checkpoint-cadence.md) — Learning-Rate Schedule, Warmup, Weight Decay, and Checkpoint Cadence. Sane defaults for full-parameter vs LoRA SFT, cosine / linear / constant with warmup, weight-decay scope, and how to choose `save_steps` / `eval_steps` so you can roll back to the checkpoint you actually want.
4. [`04-reading-loss-curves.md`](04-reading-loss-curves.md) — Reading Loss Curves. Train vs eval loss, gradient-norm plots, throughput as a diagnostic, when the loss is lying to you, and the two-plot rule for calling a run good or bad before you run the full eval suite.
5. [`05-overfitting-forgetting-template-leak.md`](05-overfitting-forgetting-template-leak.md) — Overfitting, Catastrophic Forgetting, and Template Leak. The three failure modes that specifically hit SFT runs, how to detect each one from loss + eval + hand-inspection signals, and mitigation tactics that are worth trying in what order.
6. [`06-reproducibility-and-the-training-report.md`](06-reproducibility-and-the-training-report.md) — Reproducibility and the Training Report. Seeds, deterministic algorithms, config-as-artifact, env pinning, dataset and checkpoint hashes, base-model revision pins, and the one-page training report that ships with every run.

## Exercises

Each exercise is a hands-on companion to one or more chapters. They live in [`exercises/`](exercises/).

- [`exercise-01-first-sft-run-on-a-1b-model.md`](exercises/exercise-01-first-sft-run-on-a-1b-model.md) — do a real end-to-end SFT run on a 1B-class base with TRL `SFTTrainer`, on a small slice of an open dataset, and ship a checkpoint + the config that produced it.
- [`exercise-02-masking-and-packing-correctness.md`](exercises/exercise-02-masking-and-packing-correctness.md) — prove your trainer only computes loss on assistant tokens and that packing does not leak across examples, using assertions and a print-out of the actual `labels` tensor.
- [`exercise-03-lr-schedule-and-loss-diagnosis.md`](exercises/exercise-03-lr-schedule-and-loss-diagnosis.md) — sweep three learning rates × two schedules on a fixed dataset and produce loss-curve plots + a one-page written diagnosis of what each curve is telling you.
- [`exercise-04-catastrophic-forgetting-investigation.md`](exercises/exercise-04-catastrophic-forgetting-investigation.md) — measure MMLU (or an equivalent held-out benchmark) before and after an SFT run, quantify the delta, then re-run with one mitigation and show whether it helped.
- [`exercise-05-reproducibility-and-training-report.md`](exercises/exercise-05-reproducibility-and-training-report.md) — take a run you already did in an earlier exercise, produce the full reproducibility manifest and one-page training report, and demonstrate the run reproduces (loss curve within tolerance) from the manifest alone.

## Labs and quizzes

Long-form labs land in [`labs/`](labs/) and the knowledge-check quiz in [`quizzes/`](quizzes/) on subsequent authoring cycles.

## Resources

See [`resources.md`](resources.md) for primary papers, official documentation, and tooling references cited across the chapters.

## What comes next

- **mod-104 — PEFT: LoRA, QLoRA, and Friends** swaps full-parameter SFT for parameter-efficient fine-tuning against the same training loop you build here. Most of what you learn in this module carries over unchanged.
- **mod-105 — Preference Optimization** starts from the SFT checkpoint this module produces; its DPO / ORPO / KTO recipes are literally trainer subclasses of the loop you build here.
- **mod-107 — Distributed Fine-Tuning** takes this single-GPU / single-node loop and scales it under DDP / FSDP / DeepSpeed / accelerate.
- **mod-108 — Evaluating Fine-Tuned Models** runs the eval suites you learn to protect against catastrophic forgetting in chapter 5.
