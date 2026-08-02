# Cluster Sizing 7B → 70B and Handoff to `training-pipeline-engineer`

**Estimated read:** 40 minutes.
**You should be able to, by the end:** sketch a defensible cluster shape and per-run cost estimate for a 7B / 13B / 34B / 70B full-parameter or LoRA fine-tuning job on the current generation of accelerators, know when the answer is "you need more or larger GPUs" vs "reconfigure the run," and know precisely which decisions are yours as a fine-tuning engineer and which belong to the training-pipeline-engineer / platform team.

## The sizing question, framed

Two people always want to know before a run starts: the fine-tuning engineer (will this fit? how long will it take?) and their boss / finance / capacity planner (how much will it cost?). Both questions are answered by the same math:

1. **Per-rank memory** must fit the largest allocation phase (chapter 4's budget).
2. **Per-run wall clock** = tokens processed / (tok/s/GPU · num_GPUs).
3. **Per-run cost** = wall clock hours · num_GPUs · $/GPU-hour.

Everything in this chapter builds on those three lines.

## Rules of thumb by model size

The starting-point configurations that "should just work" for common model sizes, on A100 80 GB and H100 80 GB. Numbers are approximate; use as a design starting point and verify with a real run.

### 1B–3B

- **Fits on a single card.** DDP across 2–8 GPUs for throughput; FSDP is overkill.
- **Full fine-tune** in BF16 on a 40 GB A100 without checkpointing (`~20 GB weights + optim`, comfortable activations).
- **When to distribute**: only if wall clock is unacceptable on one card. Even then, 2–4× DDP is usually enough.
- **Typical cost frame**: an experiment on 4× A100 for 4 hours is $50–120 at commodity spot pricing (as of 2026-08; current rates vary).

### 7B–13B

- **Sweet spot for QLoRA** on a single 24 GB card if you can accept the LoRA scope (mod-104 chapter 2).
- **Full fine-tune** in BF16 on 8× A100 80 GB with FSDP FULL_SHARD, activation checkpointing on, FA2, BF16, micro-batch 2–4, gradient accumulation to a reasonable global batch (chapter 4's worked example).
- **Multi-node full FT** rarely needed; single 8-GPU node handles it well.
- **Typical cost frame**: an SFT run on 1M tokens takes ~2–6 hours on 8× A100; that's ~$80–500 depending on cloud and spot vs on-demand.

### 34B

- **Doesn't fit full-FT on 8× 40 GB A100** — need 8× 80 GB A100 minimum, or 8× H100.
- **Full FT** with FSDP FULL_SHARD on 8× 80 GB. Consider optimizer offload (DeepSpeed) as a safety valve for larger sequences.
- **LoRA** fits more comfortably (adapter parameters ≈ 1% of base; the memory savings are on gradients + optimizer state, not on the 68 GB of frozen BF16 weights that still need to be sharded).
- **Typical cost frame**: a full-FT SFT run on 1M tokens on 8× H100 takes 3–8 hours = $150–600 spot.

### 70B

- **Requires 8× 80 GB minimum** for LoRA; consider 16× 80 GB (two nodes) for full FT with comfortable sequence lengths.
- **Full FT** on a single node — 8× H100 SXM works with FSDP FULL_SHARD, BF16, gradient checkpointing, FA2, `b=1`, gradient accumulation 8–16. Optimizer offload almost always on.
- **Multi-node full FT**: HYBRID_SHARD is the answer. Two nodes of 8× H100 with a fast fabric (400 Gbps IB) is a reasonable topology; four nodes is more comfortable.
- **LoRA fits on 8× H100** (or 8× A100 80 GB with more offload) — the sizing decision hinges on whether full-parameter is what you need.
- **Typical cost frame**: full-FT SFT on 1M tokens across 16× H100 for 8–20 hours = $1,500–$8,000+.

### 405B and beyond

- Squarely in pretraining-team territory. Fine-tuning a 405B is not a routine fine-tuning-engineer job — it requires the pretraining stack (tensor + pipeline parallelism, NVMe offload, dedicated cluster) and the sizing question is not what this chapter answers. Escalate.

## A memory-fit heuristic

A single line that decides "will this fit before optimization tricks":

    per_rank_bytes_needed ≈ 16 · N / P + activation_budget + 4 GB overhead

For AdamW mixed-precision full FT under ZeRO-3 / FSDP FULL_SHARD. The `16N/P` is weights + gradients + optimizer state combined; `activation_budget` is 10–30 GB depending on batch / seq / checkpointing / FA2; overhead is framework buffers + fragmentation.

To fit on 80 GB cards with 20 GB activation budget: `16N/P + 24 ≤ 80` → `N/P ≤ 3.5B`. For a 7B, need `P ≥ 2`. For a 13B, `P ≥ 4`. For a 34B, `P ≥ 10` (round up to 16). For a 70B, `P ≥ 20` (round to 24 or 32 to give headroom).

The heuristic overstates the requirement (it doesn't count 8-bit optimizer, offload, or sequence parallelism), but it's a good "is this even in the right zip code?" check.

## Wall-clock estimation

Given the config from the memory heuristic, wall clock is:

    total_tokens = examples · avg_seq_length · epochs
    wall_clock_hours = total_tokens / (tok_per_sec_per_gpu · num_gpus · 3600)

With chapter 5's typical `tok/s/GPU` numbers:

- **7B full FT on 8× A100, well-tuned**: ~4,000–6,000 tok/s/GPU.
- **13B full FT on 8× A100 80 GB**: ~2,500–4,000 tok/s/GPU.
- **70B full FT on 16× H100 with HYBRID_SHARD**: ~1,500–3,000 tok/s/GPU.
- **7B LoRA on 4× A100**: ~5,000–8,000 tok/s/GPU (fewer FLOPs per token because gradients only flow through adapters).

For a 7B SFT on 1M examples at 2048 avg tokens each = 2B tokens, 1 epoch, 8× A100 at 5000 tok/s/GPU: wall clock ≈ 2e9 / (5000 · 8 · 3600) ≈ 14 hours.

These numbers are estimates. Run a 30-minute sizing pass at the intended shape before committing to a multi-day job.

## Cost math

Cost per GPU-hour, order-of-magnitude (as of 2026-08 — verify current pricing before quoting):

| GPU | On-demand $/GPU-hour | Spot $/GPU-hour |
|---|---|---|
| A100 40 GB | $1.50–$3.00 | $0.60–$1.20 |
| A100 80 GB | $2.00–$4.00 | $0.80–$1.60 |
| H100 SXM | $4.00–$8.00 | $2.00–$4.00 |
| H200 | $5.00–$10.00 | check current |

Cost = `wall_clock_hours · num_gpus · $/gpu-hour`. Same 7B SFT example above: 14 hours × 8 A100 × $2/hr spot = ~$225.

Add 20–40% for pre/post work (data staging, evaluation, first-run debug pass), 5–15% for storage / egress, and 10–20% for the failed runs that never work first-try. Two rules of thumb:

1. **First runs at a new scale take 3–5× the "planned" wall clock** because of debugging and re-runs. Budget accordingly.
2. **Compute is a small fraction of the total cost** for most fine-tuning projects; engineer time (yours and the platform team's) is usually 3–10× the raw compute cost. Do not optimize compute at the expense of iteration speed.

## Worked configurations

The concrete configs you can copy and adapt. Each is a starting point — verify against your actual model, data, and hardware.

### 7B full-parameter SFT on 8× A100 80 GB

```yaml
# FSDP config
distributed_type: FSDP
num_processes: 8
mixed_precision: bf16
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_activation_checkpointing: true
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_offload_params: false
  fsdp_use_orig_params: true
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_cpu_ram_efficient_loading: true
```

```
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2e-5
optim: adamw_torch  # or adamw_bnb_8bit for extra memory savings
max_seq_length: 2048
attn_implementation: flash_attention_2
```

Expected: ~35 GB per rank, ~4,500 tok/s/GPU, ~50% MFU.

### 13B full-parameter SFT on 8× A100 80 GB

Same as 7B but:

```
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

And if OOM under sequence pressure, either drop `max_seq_length` or switch to DeepSpeed with `offload_optimizer.device: cpu`.

Expected: ~60 GB per rank, ~3,000 tok/s/GPU, ~45% MFU.

### 70B full-parameter SFT on 16× H100 SXM (two nodes)

```yaml
distributed_type: FSDP
num_machines: 2
num_processes: 16
mixed_precision: bf16
fsdp_config:
  fsdp_sharding_strategy: HYBRID_SHARD  # <-- key: intra-node FULL_SHARD, inter-node DDP
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_activation_checkpointing: true
  fsdp_backward_prefetch: BACKWARD_POST  # <-- memory-conservative
  fsdp_offload_params: false
  fsdp_use_orig_params: true
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_cpu_ram_efficient_loading: true
```

```
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
max_seq_length: 4096
optim: adamw_bnb_8bit
```

Expected: ~65 GB per rank, ~2,000 tok/s/GPU, ~40% MFU. Multi-node fabric quality (IB HDR / NDR) will materially affect throughput; ~200 Gbps or better assumed.

### 7B QLoRA on a single 24 GB card

Covered in mod-104 chapter 2. Named here for completeness of the sizing table. Distributed QLoRA (mod-104 + this module) works on 2–8 GPUs with FSDP; check compatibility matrix.

### 70B LoRA on 8× H100

```
distributed_type: FSDP
num_processes: 8
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_activation_checkpointing: true
  fsdp_use_orig_params: true  # <-- required for LoRA + FSDP
  ...
```

Plus a `LoraConfig` with rank 16–64 on the standard target modules. Weights still take `2N/P = ~17.5 GB` per rank; adapter storage / gradients are tiny by comparison. Expected: ~55 GB per rank, ~2,500 tok/s/GPU, ~35% MFU (LoRA has less compute per token so MFU is lower).

## Multi-node considerations

Anything past a single 8-GPU node introduces the network fabric as a first-class concern. The relevant knobs:

- **Choose HYBRID_SHARD over FULL_SHARD.** Intra-node all-gather stays fast (NVLink); inter-node degrades to DDP-style all-reduce, which is much cheaper on slow links.
- **Verify NCCL topology awareness.** `NCCL_TOPO_FILE`, `NCCL_TOPO_DUMP_FILE=/tmp/topo.xml` — check that NCCL sees your topology correctly. Wrong topology = wrong ring / tree = suboptimal bandwidth.
- **Fabric matters.** ~200 Gbps InfiniBand HDR is the minimum for smooth multi-node LLM training; 400 Gbps NDR is the current comfort zone. 100 Gbps Ethernet works but is materially slower.
- **Checkpoint storage** — a shared filesystem all ranks can write to (Lustre, GCSFuse, NFS, S3-via-`s3fs`). Local scratch works for FSDP's SHARDED_STATE_DICT if you consolidate off-cluster; even then, at least one shared location is needed.
- **Job orchestration** — Slurm, Kubernetes, Ray, Nemo Launcher, whatever your team runs. `torchrun` and `deepspeed --hostfile` handle raw launch; production runs use the scheduler.

If any of these are unclear on your target cluster, escalate before starting.

## The two axes of scaling: strong and weak

- **Strong scaling** (more GPUs, same model / same problem): each additional GPU should shrink wall clock roughly linearly until communication overhead dominates. For fine-tuning, strong scaling typically holds cleanly up to 8 GPUs on one node, and up to 16–32 GPUs across nodes with HYBRID_SHARD and a fast fabric. Beyond that, communication overhead flattens the curve.
- **Weak scaling** (more GPUs, proportionally more data / batch): should ideally hold near-perfectly. In practice, larger effective batch sizes require LR adjustment (linear or square-root scaling; see Goyal et al. 2017 "Accurate, Large Minibatch SGD"), and past a certain batch size, gradient noise becomes too small to make progress — the "large batch pathology."

For fine-tuning specifically: the "right" effective batch is 32–256 examples for SFT and DPO, sometimes higher for continued pretraining. Above ~2M tokens per step, convergence often suffers regardless of LR.

The interpretation: strong scaling until you hit the fabric wall or the batch-size wall, whichever comes first. Both are diagnosable with a few points on a "tokens/sec vs num_gpus" curve.

## Where fine-tuning ends and `training-pipeline-engineer` picks up

The interface contract. This is what you own vs what the platform / training-pipeline team owns. The fine-tuning-engineer job is to run and configure jobs; the training-pipeline-engineer job is to make it possible.

### Fine-tuning-engineer owns:

- Choosing the parallelism strategy (FSDP / DeepSpeed / DDP) and stage.
- Writing the framework config (Accelerate YAML, DeepSpeed JSON).
- Choosing micro-batch, gradient accumulation, sequence length, mixed precision.
- Turning on activation checkpointing, FA2, 8-bit optimizer.
- Running the training script.
- Profiling and reporting throughput / MFU.
- Diagnosing OOM and NCCL errors well enough to unblock the run.
- Estimating wall clock and cost for a proposed configuration.
- Producing the final checkpoint in the format the downstream pipeline expects.
- Reporting eval results and known failure modes.

### `training-pipeline-engineer` (platform / infra team) owns:

- Provisioning the cluster (capacity, spot vs reserved, region).
- Configuring the scheduler (Slurm, Kubernetes, Ray, cloud-managed).
- Configuring the fabric (InfiniBand tuning, NCCL topology files, driver versions, MTU / PFC on Ethernet).
- Owning host OS, CUDA, PyTorch, NCCL version matrix homogeneously across the fleet.
- Providing a working `torchrun` / `accelerate launch` / `deepspeed` command template that "just works" on the cluster.
- Providing shared storage for checkpoints and datasets, sized and permission'd correctly.
- Monitoring hardware health, replacing failed nodes, handling spot preemption.
- Managing multi-tenant capacity, quotas, and prioritization.
- Providing the observability stack (Grafana dashboards for GPU util, network, job queues).
- Owning experiment tracking infrastructure (W&B / MLflow / etc. server side).

### Shared / boundary items:

- **Checkpointing cadence and format**. You decide *when* and *what*; the platform provides the storage that can absorb the write volume.
- **Container image / dependency versions**. Platform provides the base image; you extend it with your specific `requirements.txt`. Version conflicts get escalated.
- **Retry / preemption behavior**. Platform provides the primitives (checkpoint on preemption); you write the code that responds correctly (auto-resume from latest checkpoint).
- **First-run validation**. Platform verifies fabric health; you verify your training script works on a small test job before scaling up.

## Escalations that are red flags, not asks

If you find yourself doing any of the following as a fine-tuning engineer, stop and escalate:

- **Tuning NCCL environment variables to make the fabric work.** The platform team should provide a working NCCL configuration; if you're fighting with it, that's a platform gap.
- **Writing systemd units, custom container startup scripts, or reverse-engineering Slurm prolog scripts.** Not your layer.
- **Diagnosing hardware faults** (GPU falling off the bus, ECC errors, NIC issues). Escalate.
- **Negotiating cluster capacity.** The training-pipeline-engineer / platform team plans capacity; you request it. If capacity isn't there, negotiate the run size, don't build workarounds.
- **Building the eval harness for cluster-level throughput.** MFU per job is yours; fleet-level utilization is platform.

The reason these are red flags: doing them means either (a) the platform team is under-invested and needs to be told so, or (b) you're accidentally becoming a training-pipeline-engineer and duplicating work.

## The kickoff conversation

Before starting a distributed fine-tuning run at a scale that's new to you, have this conversation with the platform team:

1. **Cluster shape**: how many nodes, what GPU SKU, what fabric?
2. **Container image**: base image, PyTorch version, CUDA version, NCCL version, and which of `accelerate`/`transformers`/`trl`/`peft`/`deepspeed` are pinned to which version?
3. **Storage**: where do checkpoints go? What's the write bandwidth? Where does the dataset live?
4. **Scheduler**: `torchrun` template? `accelerate launch` template? Slurm sbatch example? K8s manifest example?
5. **Monitoring**: how do I see GPU utilization / network / job status?
6. **Preemption**: is this spot capacity? What's the auto-resume story?
7. **Multi-node fabric health check**: has `nccl-tests` been run on this cluster? What's the expected `busbw`?

If the answer to any of these is "figure it out," the platform is under-resourced and that's the first thing to fix, not your training script.

## Exercise pointer

Exercise 04 asks you to size a proposed run (dataset X, model Y, budget Z) to a specific cluster shape and defend the choice. It is the concrete practice of the sizing math above.

## Summary

- **Sizing math** in three lines: per-rank memory ≤ card size, wall clock = total tokens / (tok/s/GPU · num_GPUs), cost = wall clock · num_GPUs · $/GPU-hour.
- **The `16N/P + activation + overhead` heuristic** gets you 80% of "does it fit?" for AdamW mixed-precision full FT under ZeRO-3.
- **Model-size rules of thumb**: 1–3B on a single card, 7–13B on 8× 40–80 GB (single node), 34B on 8× 80 GB, 70B on 8–16× 80 GB, 100B+ escalate to pretraining team.
- **Worked configs** for 7B, 13B, and 70B FSDP runs are the starting points; verify with a small sizing pass before committing.
- **Multi-node** requires HYBRID_SHARD, ~200 Gbps+ fabric, shared checkpoint storage, and NCCL topology awareness. Fabric quality dominates throughput.
- **Strong scaling** typically holds to ~16–32 GPUs; beyond that, communication overhead flattens the curve. **Weak scaling** requires LR adjustment and hits an effective-batch ceiling around ~2M tokens/step for most fine-tuning.
- **The interface contract**: fine-tuning engineer owns the training script, config, throughput reporting, and OOM/NCCL diagnosis. Platform / `training-pipeline-engineer` owns the cluster, scheduler, fabric, storage, orchestration, and multi-tenant infrastructure.
- **Escalation red flags**: tuning NCCL variables to make things work at all, chasing hardware faults, negotiating capacity. Escalate — don't work around.
- **The kickoff conversation** has seven questions. If the platform team can't answer them cleanly, the platform is under-invested; that's the problem to fix, not your training loop.
