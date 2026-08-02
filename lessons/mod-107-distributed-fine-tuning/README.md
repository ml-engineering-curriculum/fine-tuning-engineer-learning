# mod-107-distributed-fine-tuning: Distributed Fine-Tuning at the Operator Altitude

**Estimated effort:** 14 hours
**Position in the track:** seventh module. Consumes the single-GPU fine-tuning loops from mod-103 (full-parameter SFT) and mod-104 (LoRA / QLoRA) and scales them across 2–8 GPUs on one node and, later, across nodes. Preference-optimization trainers from mod-105 (DPO / KTO / ORPO) and the safety-preference stages from mod-106 reuse the same distributed machinery — a DPO run is a forward + reference-forward + loss step and it shards the same way SFT does. Downstream, mod-108 (evaluation), mod-109 (quantization / serving), and mod-111 (systems design) all assume you can produce a fine-tuned checkpoint at whichever scale the task requires.

Mods 103 and 104 taught you how to fine-tune on a single GPU or a modest single-node setup. This module is about what happens when your model or your throughput requirement grows past what one card can carry: the base weights no longer fit, the optimizer state no longer fits, or the run takes so long on one card that iteration stops working. The through-line is that distributed fine-tuning is 80% memory accounting and configuration and 20% correctness — the trainers you already know (`SFTTrainer`, `DPOTrainer`, `Trainer`) are unchanged; what changes is the launcher, the wrapper that shards parameters and gradients, and the failure modes when the wrapper misfires.

This module deliberately stays at the **operator altitude**: how a fine-tuning engineer configures Accelerate / FSDP / DeepSpeed, reads a throughput / MFU report, and diagnoses OOM and NCCL failures well enough to unblock a run. The **platform altitude** — building the multi-tenant cluster, running the scheduler, tuning the fabric, and owning the reliability story — belongs to the `training-pipeline-engineer` role. Chapter 7 is the interface contract between the two.

## Learning objectives

- Launch multi-GPU / multi-node fine-tuning runs with Accelerate, FSDP, and DeepSpeed ZeRO-2/3
- Apply gradient checkpointing, mixed precision (BF16, FP8 where available), and CPU/NVMe offload
- Profile throughput and MFU; diagnose OOM, NCCL, and load-imbalance failures
- Map a 7B → 70B fine-tuning job to a concrete cluster shape and budget
- Hand off cluster-platform concerns to `training-pipeline-engineer` rather than re-implementing them

## Lecture chapters

1. [`01-parallelism-landscape.md`](01-parallelism-landscape.md) — The Parallelism Landscape and Where Fine-Tuning Lives. Data / tensor / pipeline / expert / sequence parallelism at a glance, why DDP is not enough past ~1.5B, how ZeRO decomposes optimizer / gradient / parameter state, and the FSDP-vs-DeepSpeed choice at a first pass.
2. [`02-accelerate-and-fsdp.md`](02-accelerate-and-fsdp.md) — Accelerate and PyTorch FSDP. The `accelerate config` / `accelerate launch` flow, FSDP's `ShardingStrategy` (FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, NO_SHARD), auto-wrap policies, the mixed-precision `MixedPrecision` block, `FSDP2` and `fully_shard`, and a working `fsdp_config.yaml` for a 2- and 8-GPU run.
3. [`03-deepspeed-zero-and-offload.md`](03-deepspeed-zero-and-offload.md) — DeepSpeed ZeRO-2/3 and CPU/NVMe Offload. What ZeRO-1/2/3 remove from each rank, when ZeRO-3 buys more than FSDP FULL_SHARD, the CPU-offload and NVMe-offload knobs (ZeRO-Infinity), what `zero_stage`, `stage3_prefetch_bucket_size`, `gradient_accumulation_steps`, and the `ds_config.json` blocks actually do, and the compatibility matrix with Accelerate / TRL.
4. [`04-memory-mixed-precision-and-checkpointing.md`](04-memory-mixed-precision-and-checkpointing.md) — Gradient Checkpointing, BF16 / FP8, and the Memory Budget. Activation checkpointing (Chen et al.'s idea, HF's `gradient_checkpointing_enable`, FSDP's `apply_activation_checkpointing`); the BF16 / FP16 / FP8 (Transformer Engine on Hopper) choices; FlashAttention-2 and 3; a per-component memory table you can defend a run against.
5. [`05-throughput-mfu-and-profiling.md`](05-throughput-mfu-and-profiling.md) — Throughput, MFU, and Profiling. Tokens / sec / GPU as the primary number; the MFU formula (PaLM-style: 6 · N · tokens / (peak FLOPS · GPUs)) and the HFU variant; the roofline model in one page; using the PyTorch profiler, DeepSpeed's FLOPS profiler, and `nvidia-smi dmon` / Nsight Systems to find compute vs. communication vs. bubbles; overlap heuristics.
6. [`06-oom-nccl-and-load-imbalance.md`](06-oom-nccl-and-load-imbalance.md) — Debugging OOM, NCCL, and Load-Imbalance Failures. A structured OOM playbook (which of activation / gradient / optimizer / fragmentation blew up), NCCL diagnostics (`NCCL_DEBUG=INFO`, timeouts, `TORCH_NCCL_ASYNC_ERROR_HANDLING`, P2P / IB / GDRDMA sanity checks, `nccl-tests`), load imbalance under variable sequence lengths, and the "run hangs at step N" family.
7. [`07-cluster-sizing-and-handoff.md`](07-cluster-sizing-and-handoff.md) — Cluster Sizing 7B → 70B and Handoff to `training-pipeline-engineer`. Rules of thumb per model size, worked configurations for 7B / 13B / 34B / 70B on A100 / H100 boxes, a first-pass budget calculation, and the interface contract that says what the fine-tuning engineer owns and what belongs to the platform / infra team.

## Exercises

Each exercise is a hands-on companion to one or more chapters. They live in [`exercises/`](exercises/).

- [`exercise-01-accelerate-fsdp-launch-on-2-to-8-gpus.md`](exercises/exercise-01-accelerate-fsdp-launch-on-2-to-8-gpus.md) — take a working single-GPU SFT script (mod-103 exercise 01) and scale it to 2 → 4 → 8 GPUs with Accelerate + FSDP. Produce two `fsdp_config.yaml` variants, verify per-rank memory and tokens/sec/GPU, and defend the sharding-strategy / auto-wrap / mixed-precision choices.
- [`exercise-02-deepspeed-zero-3-with-offload.md`](exercises/exercise-02-deepspeed-zero-3-with-offload.md) — run the same SFT job under DeepSpeed ZeRO-3 with and without CPU offload; compare peak memory, throughput, and end-loss against the FSDP variant from exercise 01; produce a written recommendation.
- [`exercise-03-throughput-and-mfu-profiling.md`](exercises/exercise-03-throughput-and-mfu-profiling.md) — instrument a distributed run, capture tokens/sec/GPU + step time + a PyTorch profiler trace, compute MFU from the formula, and identify the top bottleneck (compute-bound / communication-bound / bubble-limited). Produce a one-page profiling report.
- [`exercise-04-oom-and-nccl-debugging-drills.md`](exercises/exercise-04-oom-and-nccl-debugging-drills.md) — a set of injected failures (OOM at step 0, OOM at step 200, NCCL timeout, hang mid-run, load imbalance under packed sequences); for each, produce a diagnosis, a minimal reproducer, and a fix, with the log evidence that supports the diagnosis.

## Labs and quizzes

Long-form labs land in [`labs/`](labs/) and the knowledge-check quiz in [`quizzes/`](quizzes/) on subsequent authoring cycles.

## Resources

See [`resources.md`](resources.md) for primary papers (Rajbhandari et al. — ZeRO / ZeRO-Offload / ZeRO-Infinity; Zhao et al. — PyTorch FSDP; Chen et al. — gradient checkpointing; Dao et al. — FlashAttention-2 / 3; Chowdhery et al. — PaLM / MFU), the official docs (`torch.distributed.fsdp`, Hugging Face Accelerate, DeepSpeed, NCCL, NVIDIA Transformer Engine, TRL distributed guides), and the profiling / debugging tooling (PyTorch Profiler, Nsight Systems, `nccl-tests`, DeepSpeed FLOPS profiler).

## What comes next

- **mod-108 — Evaluating Fine-Tuned Models** consumes the checkpoints this module produces. Running eval across multiple checkpoints is itself embarrassingly parallel; the distributed patterns here transfer.
- **mod-109 — Quantization, Export, and Serving** takes the trained checkpoint into the deployment pipeline. Checkpoint format, sharded-save / consolidated-save decisions, and safetensors conventions from chapter 2 flow directly into it.
- **mod-110 — Managed Fine-Tuning APIs** is the "someone else runs the cluster" alternative; the cost / control trade-off it discusses is exactly the axis chapter 7's handoff table describes.
- **mod-111 — Post-Training Systems Design** places distributed fine-tuning inside a full post-training platform (schedulers, artifact stores, experiment trackers, cost accounting). Chapter 7's handoff table is where mod-111 picks up.
