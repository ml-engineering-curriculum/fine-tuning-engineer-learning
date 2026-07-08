# mod-107-distributed-fine-tuning: Distributed Fine-Tuning at the Operator Altitude

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 14 hours

## Learning objectives

- Launch multi-GPU / multi-node fine-tuning runs with Accelerate, FSDP, and DeepSpeed ZeRO-2/3
- Apply gradient checkpointing, mixed precision (BF16, FP8 where available), and CPU/NVMe offload
- Profile throughput and MFU; diagnose OOM, NCCL, and load-imbalance failures
- Map a 7B → 70B fine-tuning job to a concrete cluster shape and budget
- Hand off cluster-platform concerns to `training-pipeline-engineer` rather than re-implementing them

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
