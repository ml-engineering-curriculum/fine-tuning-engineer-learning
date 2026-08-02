# mod-104-peft-lora-qlora: Parameter-Efficient Fine-Tuning — LoRA, QLoRA, and Friends

**Estimated effort:** 16 hours
**Position in the track:** fourth module. Builds directly on mod-103's SFT training loop — the trainer, the collator, the loss are the same. The dataset engineering from mod-102 still applies. Feeds into mod-105 (preference optimization with LoRA), mod-107 (distributed LoRA / QLoRA), mod-109 (quantized inference of merged and unmerged adapters), and every module downstream that assumes you can fine-tune a 7B–13B base without a rack of GPUs.

Mod-103 handed you a full-parameter fine-tuning loop and its failure catalog. This module keeps every load-bearing detail of that loop — the mask, the collator, the LR schedule, the loss curve — and swaps the update rule: instead of training every weight, you train a small parameter-efficient addition to a frozen base. The through-line is that a PEFT run is a small run: the memory footprint drops by an order of magnitude, so the questions change. You spend less time provisioning GPUs and more time reasoning about rank, alpha, target modules, and which of the five PEFT methods actually fits the task.

## Learning objectives

- Implement LoRA from scratch (low-rank decomposition, scaling) and explain rank / alpha / target-module trade-offs
- Run QLoRA on a 7B–13B model on a single 24-GB GPU using bitsandbytes 4-bit NF4 base weights
- Merge adapters back into base weights, manage multi-adapter routing, and serve LoRA-aware
- Compare LoRA, QLoRA, IA³, prefix tuning, and prompt tuning and pick the right one per use case
- Diagnose adapter-specific failure modes: rank collapse, target-module mis-selection, alpha mis-scaling

## Lecture chapters

1. [`01-lora-math-and-implementation.md`](01-lora-math-and-implementation.md) — LoRA: the Math, the Layer, and the Trade-offs. The low-rank decomposition `ΔW = (alpha/r) · B · A`, a working `LoRALinear` in under 40 lines of PyTorch, and how `r`, `alpha`, and the target-module list control capacity / effective LR / which weights actually adapt.
2. [`02-qlora-and-4bit-base-weights.md`](02-qlora-and-4bit-base-weights.md) — QLoRA: 4-bit Base Weights on a Single GPU. NF4, double quantization, paged optimizers, the `bitsandbytes` + `transformers` + `peft` load recipe, and the memory budget worked out for a 7B on 24 GB.
3. [`03-merging-and-multi-adapter-serving.md`](03-merging-and-multi-adapter-serving.md) — Merging Adapters and Multi-Adapter Serving. `merge_and_unload()`, when to merge and when not to, named-adapter routing (`set_adapter`, `add_weighted_adapter`), and what vLLM / TGI / LoRAX do when they advertise "LoRA-aware serving."
4. [`04-peft-methods-in-context.md`](04-peft-methods-in-context.md) — PEFT Methods in Context. IA³, prefix tuning, prompt tuning (and a note on Houlsby adapters), followed by a decision matrix over base size × data volume × serving constraint × task shape.
5. [`05-adapter-failure-modes.md`](05-adapter-failure-modes.md) — Adapter Failure Modes. Rank collapse, target-module mis-selection, alpha mis-scaling; their signatures in loss curves and singular values; the one-shot LoRA health check.

## Exercises

Each exercise is a hands-on companion to one or more chapters. They live in [`exercises/`](exercises/).

- [`exercise-01-lora-from-scratch-and-against-peft.md`](exercises/exercise-01-lora-from-scratch-and-against-peft.md) — implement `LoRALinear` in PyTorch, train it on a small SFT task, and prove numeric equivalence (or explain the differences) with a `peft.LoraConfig`-driven run at the same settings.
- [`exercise-02-qlora-on-a-7b-base-single-gpu.md`](exercises/exercise-02-qlora-on-a-7b-base-single-gpu.md) — fit a 7B-class base on a 24-GB GPU with NF4 + double quantization + paged AdamW + gradient checkpointing; defend every flag against the chapter-2 memory table; ship a trained adapter, health-check numbers, and a report.
- [`exercise-03-adapter-merging-and-multi-adapter-routing.md`](exercises/exercise-03-adapter-merging-and-multi-adapter-routing.md) — train two adapters on the same base, load them side by side, route requests between them, produce a merged variant of one, and defend the "merge vs. attach" decision for a specified serving scenario.
- [`exercise-04-peft-methods-comparison-matrix.md`](exercises/exercise-04-peft-methods-comparison-matrix.md) — run three PEFT methods (LoRA, QLoRA, and one other — IA³ / prefix / prompt) on the same base and task, produce your own comparison matrix (params, VRAM, wall-clock, final eval), and defend a recommendation.

## Labs and quizzes

Long-form labs land in [`labs/`](labs/) and the knowledge-check quiz in [`quizzes/`](quizzes/) on subsequent authoring cycles.

## Resources

See [`resources.md`](resources.md) for primary papers (LoRA, QLoRA, IA³, prefix / prompt tuning), official documentation (`peft`, `bitsandbytes`, `transformers`, TRL, vLLM), and the tooling references cited across the chapters.

## What comes next

- **mod-105 — Preference Optimization** trains DPO / ORPO / KTO adapters on top of the SFT checkpoint from mod-103 or the merged LoRA output from this module. Every technique here composes with it.
- **mod-107 — Distributed Fine-Tuning** scales LoRA and QLoRA across multiple GPUs / nodes and covers the FSDP / DeepSpeed footguns chapter 2 flags for QLoRA.
- **mod-109 — Quantization, Export, and Serving** picks up where chapter 3 leaves off — quantized inference, AWQ / GPTQ, adapter formats for production serving stacks.
- **mod-111 — Post-Training Systems Design** treats multi-adapter routing as one component of a larger post-training platform.
