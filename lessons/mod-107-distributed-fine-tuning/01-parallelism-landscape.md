# The Parallelism Landscape and Where Fine-Tuning Lives

**Estimated read:** 35 minutes.
**You should be able to, by the end:** name the five parallelism dimensions (data / tensor / pipeline / expert / sequence), state precisely which ones a fine-tuning engineer normally configures and which are reserved for the pretraining and platform teams, describe what DDP, ZeRO-1/2/3, and FSDP shard and why each was invented, and place the FSDP-vs-DeepSpeed choice as a first-pass decision rather than a religious commitment.

## Why single-GPU fine-tuning stops working

Mod-103 walked you through a full-parameter SFT loop on a single card, and mod-104 stretched that to 7B–13B bases with QLoRA on a 24 GB consumer GPU. Both work up to a point. The point they stop working is one of three:

1. **The model no longer fits.** Full-parameter fine-tuning a 13B model in BF16 needs ~26 GB for weights, ~52 GB for AdamW optimizer state (two moments in FP32), and ~26 GB for gradients — before activations. A single 80 GB A100 is already tight; a 70B is impossible.
2. **The optimizer state does not fit even when the model does.** A 7B in BF16 fits in 14 GB; AdamW in FP32 balloons that to ~84 GB. LoRA / QLoRA sidestep this by not training the base weights; full-parameter fine-tuning of 7B–70B cannot.
3. **Wall clock stops making iteration possible.** A 3-hour run per experiment on one GPU is workable. A 30-hour run per experiment on one GPU is not — you cannot debug data, tune LR, or catch regressions on that cadence. Distributing to 4–8 GPUs turns 30 hours into ~4 hours and the loop back on.

Distributed fine-tuning exists to move past all three walls. The rest of this chapter is the map of *how* — the parallelism dimensions the field has invented, which ones matter to a fine-tuning engineer, and where the two dominant frameworks (PyTorch FSDP, DeepSpeed ZeRO) fit.

## The five parallelism dimensions

The distributed-training literature enumerates five ways to split a training job across accelerators. You will see all of them in papers; you will personally configure two, maybe three.

### Data parallelism (DP)

Every rank holds a full copy of the model. Each rank processes a different micro-batch. After the backward pass, gradients are all-reduced across ranks so every rank ends the step with the same optimizer update. This is what `torch.nn.parallel.DistributedDataParallel` (DDP) implements.

- **Communication cost:** one all-reduce of the parameter gradient tensor per step (in the standard formulation, fused into buckets and overlapped with the backward pass).
- **Memory cost:** the full model, full optimizer state, and full gradients on *every* rank. No savings.
- **Use it when:** the whole training state fits on one rank and you only need throughput. DDP is the simplest and fastest distributed setup when it fits.

DDP is the ceiling on how simple distributed training can be. It is also the floor on how much memory you save — zero.

### Sharded data parallelism (ZeRO / FSDP)

The idea that made LLM fine-tuning above ~1.5B tractable on commodity hardware: instead of every rank holding the full state, *shard* the parameters, gradients, and optimizer state across ranks and gather them only when needed. Rajbhandari et al. (2020, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models") introduced the three progressive stages and PyTorch's FSDP (Zhao et al., 2023) is the same idea implemented natively in PyTorch.

- **ZeRO-1 / stage 1:** shard the optimizer state (the biggest single memory consumer with AdamW). Save ~4× vs DDP on optimizer memory.
- **ZeRO-2 / stage 2:** shard optimizer state *and* gradients. Save ~8× on optimizer + gradient memory.
- **ZeRO-3 / stage 3:** shard optimizer state, gradients, *and* parameters themselves. Every rank holds only its shard; parameters for a given layer are all-gathered on the fly during forward, all-gathered again for backward, then released. This is what FSDP's `FULL_SHARD` mode does.

Communication cost grows with stage: ZeRO-1 has the same all-reduce as DDP; ZeRO-3 adds an all-gather per forward pass and per backward pass in addition. The trade is memory for bandwidth, and it is worth it whenever memory is the binding constraint. For fine-tuning most 7B+ bases, ZeRO-3 / FULL_SHARD is the default because otherwise the model does not fit.

Sharded data parallelism *is* the workhorse of fine-tuning at scale. Chapters 2 (FSDP) and 3 (DeepSpeed ZeRO) are entirely about it.

### Tensor parallelism (TP)

Split individual weight matrices across ranks along a chosen axis. A `[hidden, hidden]` linear layer becomes `[hidden, hidden/TP]` on each of TP ranks; the matmul is computed in pieces and the results are all-reduced (Megatron-LM's column-then-row-parallel scheme, Shoeybi et al., 2019).

- **Communication cost:** an all-reduce inside *every* forward and backward layer. Bandwidth-heavy. Requires NVLink or NVSwitch — over Ethernet or slow interconnect, TP falls off a cliff.
- **Memory cost:** cuts the per-rank weight footprint by TP.
- **Typical use:** inside a single node's high-bandwidth domain (an 8× H100 NVSwitch box). Used to fit models whose *single layer* is too large for one GPU, or when a layer's compute is faster split than replicated.

Tensor parallelism is standard in pretraining stacks (Megatron-LM, NeMo, Colossal-AI). In fine-tuning it appears when you push into 70B+ full-parameter FT territory and need to combine sharded data parallelism with intra-layer sharding. FSDP2 and DeepSpeed both support TP-like sharding as a composition on top of the sharded-DP base, but the *typical* fine-tuning-engineer job is to hand the "we need TP" decision to the training-platform team and pick from their menu of shapes. Chapter 7 goes into the handoff.

### Pipeline parallelism (PP)

Split the model layer-wise across ranks: rank 0 has layers 1–8, rank 1 has layers 9–16, etc. Micro-batches are pipelined so rank 1 is doing layer 9 on batch N while rank 0 is doing layer 1 on batch N+1 (GPipe / PipeDream, Huang et al. 2019, Narayanan et al. 2019).

- **Communication cost:** point-to-point sends between adjacent stages — much lighter than TP.
- **Memory cost:** cuts per-rank weight footprint by PP.
- **Pain point:** the "pipeline bubble" — some ranks are idle at pipeline fill and drain. Interleaved schedules (1F1B, Chimera, Megatron's virtual pipeline stages) reduce but do not eliminate it.

Pipeline parallelism is a pretraining technique. In fine-tuning it appears only in the largest-model, longest-run cases (e.g., full-parameter fine-tuning of Llama-3.1-405B), and even then usually via the training-platform team's blessed configuration rather than something a fine-tuning engineer stands up from scratch. Named here so you can identify it in a config file and hand it off; not configured directly in the exercises.

### Expert parallelism (EP)

For Mixture-of-Experts models (Fedus et al. 2022, Shazeer et al. 2017), route different experts to different ranks so that on any given token only a subset of ranks does work. Communication cost: an all-to-all per MoE layer to route tokens to their assigned experts and gather results back.

- **Use it when:** the model is genuinely MoE (Mixtral, DeepSeek-MoE, Grok, Snowflake Arctic). Then you have no choice.
- **Fine-tuning of MoE bases** typically inherits the pretraining EP topology; you configure the *degree* of expert parallelism to match your node topology (all-to-all is fast within NVLink, slow across nodes). DeepSpeed-MoE and Megatron-Core are the reference implementations.

Same status as TP and PP: fine-tuning engineers set the flags on the framework the platform team has already validated. Not covered in this module's exercises but named so you can spot it.

### Sequence / context parallelism (SP)

Split the sequence dimension across ranks. Activations at each layer are much larger than weights for long-context training, and sharding along the sequence lets you train contexts (32K, 128K, 1M) that would otherwise not fit even with FSDP + gradient checkpointing. Ring attention (Liu et al. 2023), DeepSpeed's Ulysses, and Megatron's sequence-parallelism are the current implementations.

Fine-tuning on long contexts (16K → 128K) increasingly needs this, and it composes with FSDP / ZeRO. Chapter 4 revisits it in the memory-savers pass; chapter 7 lists it as a "reach for this when your context length is the binding constraint" tool.

### Summary — the fine-tuning engineer's ownership map

| Dimension | Who configures it | When you touch it in fine-tuning |
|---|---|---|
| DDP | You | Small models that fit per-rank; LoRA / QLoRA runs |
| ZeRO / FSDP (sharded DP) | You | Almost every 7B+ full-FT and most 13B+ LoRA runs |
| Tensor parallelism | Platform / pretraining teams | 70B+ full FT, and only in blessed configs |
| Pipeline parallelism | Platform / pretraining teams | 100B+ full FT; rare in fine-tuning |
| Expert parallelism | Platform, with your input | MoE-base fine-tuning (Mixtral, DeepSeek-MoE) |
| Sequence / context parallelism | You, cautiously | Long-context (32K+) fine-tuning |

Everything to the right of "You" in the second column is a red flag if it shows up in a fine-tuning kickoff without a platform-team owner. Chapter 7 has the escalation path.

## How ZeRO actually saves memory

Because the sharded-DP family is where most of your work lives, it is worth being precise about *what* each ZeRO stage removes from each rank. The following table is the memory-accounting version; internalize it before you configure a run.

For a model with `N` parameters, `P` ranks, and AdamW optimizer, in a mixed-precision (BF16 params + FP32 optimizer) setup:

| Component | Baseline (DDP) per rank | ZeRO-1 per rank | ZeRO-2 per rank | ZeRO-3 per rank |
|---|---|---|---|---|
| Model params (BF16) | 2N | 2N | 2N | 2N / P |
| Gradients (BF16) | 2N | 2N | 2N / P | 2N / P |
| Master params (FP32) | 4N | 4N / P | 4N / P | 4N / P |
| Momentum (FP32) | 4N | 4N / P | 4N / P | 4N / P |
| Variance (FP32) | 4N | 4N / P | 4N / P | 4N / P |
| **Total** | **16N** | **~16N − 12N + 12N/P** | **~16N − 14N + 14N/P** | **16N / P** |

At `P = 8`, a 7B model at `N = 7 × 10⁹`:

- DDP: ~112 GB per rank — does not fit anywhere.
- ZeRO-1: ~29 GB per rank — fits on 40 GB A100.
- ZeRO-2: ~15.5 GB per rank — fits on A100 40 GB comfortably.
- ZeRO-3 / FSDP FULL_SHARD: ~14 GB per rank — fits on A100 40 GB with room for activations.

The numbers are approximate (they ignore master-weight-in-BF16 optimizations, activations, communication buffers, PyTorch caching allocator overhead), but the order of magnitude is exactly right. **The choice of ZeRO stage is a choice of how much per-rank memory you buy in exchange for how much communication you spend.**

## FSDP vs. DeepSpeed ZeRO: same idea, two implementations

At a first pass, both frameworks implement the same set of ideas (sharded optimizer / gradient / parameter state; mixed precision; CPU offload) with different design choices. Which one to pick depends on more than one axis; the fine-tuning-engineer answer is usually "whichever your team already runs."

Where they overlap:

- Both can do FULL_SHARD / ZeRO-3.
- Both can do partial sharding (FSDP `SHARD_GRAD_OP` ≈ ZeRO-2).
- Both offload to CPU. DeepSpeed also offloads to NVMe (ZeRO-Infinity).
- Both work with Hugging Face `Trainer` via Accelerate.
- Both support mixed precision, activation checkpointing, and gradient accumulation.

Where they differ:

- **FSDP is a first-class PyTorch citizen** — implemented inside `torch.distributed`, ships with PyTorch releases, uses the same profiler / debugger / checkpoint format PyTorch itself does. FSDP2 (introduced in PyTorch 2.4) is the actively-developed API; the older `FullyShardedDataParallel` module is stable but on maintenance.
- **DeepSpeed is a separate library** (Microsoft) with its own launcher, config file, and internal state. In exchange it has features FSDP does not: ZeRO-Infinity NVMe offload, ZeRO++ communication reductions, a mature FLOPS / throughput profiler, and MoE-specific optimizations.
- **QLoRA / bitsandbytes compatibility** has historically favored FSDP in the Hugging Face stack. The compatibility matrix (`transformers`, `accelerate`, `peft`, `bitsandbytes`, `deepspeed`) moves fast; consult the current guides before committing.
- **Checkpoint format:** FSDP defaults to sharded checkpoints (native `torch.distributed.checkpoint`); consolidation to a single-file safetensors happens as a post-processing step. DeepSpeed uses its own `zero_to_fp32.py`-style consolidator. Both are fine; you have to know which one your serving pipeline expects.
- **Debuggability:** FSDP traces cleanly through the standard PyTorch profiler and PyTorch's own error messages. DeepSpeed's engine has a heavier abstraction that occasionally produces less clear errors, but its FLOPS profiler is a real advantage.

A rough decision rule for a fine-tuning engineer picking one for a new project:

- **Pick FSDP if:** your team already uses PyTorch profiler + native checkpointing; you want fewer moving parts; you may need QLoRA distributed; your models fit with CPU offload or without.
- **Pick DeepSpeed if:** you need NVMe offload (very large models on very small clusters); you're already in a DeepSpeed ecosystem (a lot of the LLM foundation-model code out there — Megatron-DeepSpeed, DeepSpeed-Chat — is DeepSpeed-native); you want its FLOPS profiler; you're doing MoE-heavy work.

The module covers both. Chapter 2 works FSDP end-to-end; chapter 3 works DeepSpeed. Exercise 01 is FSDP, exercise 02 is DeepSpeed, and each explicitly compares against the other.

## What "operator altitude" means

This module deliberately restricts itself to the choices a fine-tuning engineer makes on their own; everything else is handed off. The scope split, once, so you know what you are and are not going to see:

**In scope for a fine-tuning engineer (this module):**

- Choose between DDP / FSDP / DeepSpeed and configure sharding stage.
- Choose mixed precision (BF16 default, FP8 where hardware supports it), activation checkpointing granularity, and offload strategy.
- Write the launch command / Accelerate config / DeepSpeed config for a specific job.
- Profile a run, compute MFU, and report a bottleneck.
- Debug OOM, NCCL, and load-imbalance failures well enough to keep the run going.
- Size a run to a cluster shape and estimate cost.
- Write a checkpoint that the serving pipeline can consume.

**Out of scope (owned by `training-pipeline-engineer` and platform teams):**

- Building or operating the cluster itself (scheduler, node health, network topology).
- Sizing the fleet and negotiating capacity with a cloud provider.
- Operating the NCCL / IB / GDRDMA stack — tuning NIC firmware, MTU, PFC, etc.
- Building the artifact store, experiment tracker, and cost accounting.
- Owning failure-domain analysis at the fleet level (spot-instance handling, checkpoint frequency policy).
- Building the tensor-parallel or pipeline-parallel harness for pretraining-scale runs.
- Standing up multi-tenant training-as-a-service infrastructure.

The mental model: you configure and run the training job; someone else keeps the cluster alive around you. When something in the second list becomes a blocker, chapter 7 has the escalation checklist.

## What "fits" actually depends on

The fine-tuning-engineer question "will this fit?" always decomposes into six components, and every configuration knob in the module moves at least one of them:

1. **Base weights** — `2 × N` bytes in BF16, `4 × N` in FP32, `N / 2` in NF4 (mod-104), etc.
2. **Optimizer state** — `~12 × N` bytes for AdamW in FP32, `~4 × N` for AdamW-8bit, ~0 for SGD.
3. **Gradients** — `2 × N` bytes in BF16 by default.
4. **Activations** — depends on batch size, sequence length, hidden size, number of layers, and whether activation checkpointing is on. Formula in chapter 4.
5. **Workspace and communication buffers** — NCCL communication buffers, FSDP all-gather buffers, cuDNN workspace. Modest but non-trivial; typically 1–4 GB per rank.
6. **Fragmentation** — PyTorch's caching allocator leaves gaps. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (PyTorch 2.1+) helps; even so, budget 5–15% headroom.

Every technique in this module moves one or more of these components:

- **DDP → ZeRO-2 → ZeRO-3 / FSDP** shrinks (1), (2), (3) by up to a factor of `P`.
- **CPU / NVMe offload** removes (2) and possibly (3) from GPU memory entirely.
- **Gradient checkpointing** shrinks (4) by roughly `sqrt(depth)`.
- **Mixed precision** shrinks (1), (3), (4) by 2× vs FP32.
- **FP8 (Transformer Engine, Hopper only)** shrinks (1), (3), (4) further, at the cost of care around numerics.
- **FlashAttention** shrinks (4)'s attention component from `O(seq²)` to `O(seq)` per layer.
- **Longer gradient accumulation, smaller micro-batch** shrinks (4) linearly in batch.

Chapters 2 through 4 walk each of these levers. Chapter 5 tells you how to measure whether pulling one helped. Chapter 6 tells you what the failure looks like when you have not pulled enough.

## When distributed fine-tuning is not the right answer

Three shortcuts to keep in mind before you spin up a multi-node job:

1. **Try QLoRA on one card first.** For 7B–13B SFT-style work, a single 24 GB GPU with QLoRA (mod-104 chapter 2) is often enough. The engineering complexity of distributed fine-tuning is not free; do not pay for it unless the model or the throughput requirement demands it.
2. **Try LoRA on 2–4 GPUs before you try full FT on 8.** A LoRA rank-64 on a 34B base with FSDP FULL_SHARD across 4× A100 80 GB is a substantially simpler run than full-parameter FT across 8× H100 on the same base. If LoRA's quality is enough for the eval you care about (mod-104 chapter 4's decision matrix), take the savings.
3. **Consider a managed API for the largest jobs.** Mod-110 covers the OpenAI / Anthropic / Vertex / Bedrock fine-tuning APIs. When the budget is small and the model is at the top end of what your capacity can carry, the API is sometimes cheaper and always simpler. The trade-off is control — you lose visibility into hyperparameters, dataset mixing, and eval fidelity.

The rest of this module assumes you have decided distributed is the right answer. If you have not, come back here and reconsider before continuing.

## Exercise pointer

Exercise 01 (`exercises/exercise-01-accelerate-fsdp-launch-on-2-to-8-gpus.md`) is the concrete first step: take a working single-GPU SFT script and scale it to 2 → 4 → 8 GPUs with Accelerate + FSDP, verify per-rank memory matches the ZeRO table above, and defend the sharding-strategy / auto-wrap / mixed-precision choices. Exercises 02–04 build on it.

## Summary

- Single-GPU fine-tuning fails past ~13B full-FT or when wall-clock breaks the debugging loop. Distributed fine-tuning is how you get past both.
- The parallelism literature has five dimensions: **data, sharded-data (ZeRO / FSDP), tensor, pipeline, expert**, plus the increasingly-relevant **sequence / context** dimension. A fine-tuning engineer configures data and sharded-data directly, sequence occasionally, and hands the other three to the platform team.
- **Sharded data parallelism (ZeRO-1/2/3 or FSDP)** is the workhorse. It trades communication bandwidth for per-rank memory: ZeRO-1 shards optimizer state, ZeRO-2 adds gradients, ZeRO-3 / FULL_SHARD adds parameters. The memory table (16N → 16N/P) is the intuition to internalize.
- **FSDP and DeepSpeed** implement the same ideas with different design choices. FSDP is the PyTorch-native path; DeepSpeed adds NVMe offload, a FLOPS profiler, and MoE-specific features. Neither is universally correct.
- The fine-tuning-engineer altitude covers the training-job configuration, profiling, and debugging; it stops at the cluster fabric. Chapter 7's handoff table is the interface contract with the `training-pipeline-engineer` role.
- Before configuring a distributed job, check that (a) QLoRA on one card, (b) LoRA on a small multi-GPU setup, or (c) a managed API is not the simpler answer for what you need.
