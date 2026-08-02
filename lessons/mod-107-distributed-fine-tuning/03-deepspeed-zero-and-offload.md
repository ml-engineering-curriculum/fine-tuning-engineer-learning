# DeepSpeed ZeRO-2/3 and CPU/NVMe Offload

**Estimated read:** 50 minutes.
**You should be able to, by the end:** launch a fine-tuning run under DeepSpeed via Accelerate, write a defensible `ds_config.json` for ZeRO-2 or ZeRO-3, decide whether to add CPU or NVMe offload and where each helps vs hurts, and know which situations call for DeepSpeed over FSDP and vice versa.

## Where DeepSpeed sits

DeepSpeed (Microsoft) is a training-optimization library that pre-dates PyTorch's own FSDP. Its central abstraction is the **DeepSpeed engine** (`deepspeed.initialize(...)` returns a `DeepSpeedEngine` that wraps your model, optimizer, and LR scheduler and runs the whole step under its own control flow). The engine implements ZeRO-1/2/3 (Rajbhandari et al. 2020, 2021), ZeRO-Offload (Ren et al. 2021), and ZeRO-Infinity (Rajbhandari et al. 2021 — NVMe offload for trillion-parameter models), plus a large family of adjacent features (curriculum learning, communication scheduling, quantization, MoE support).

For a fine-tuning engineer the shape of DeepSpeed's world is:

- **A launcher** — either DeepSpeed's own (`deepspeed --num_gpus 8 train.py --deepspeed ds_config.json ...`) or Accelerate's (`accelerate launch --config_file ds_accelerate.yaml train.py`, with a DeepSpeed-typed Accelerate config).
- **A JSON config** — `ds_config.json`, or an equivalent block embedded in an Accelerate config. This is the whole surface area of what DeepSpeed does per run.
- **A `deepspeed.initialize` call** somewhere — usually hidden inside Accelerate / Trainer, occasionally invoked directly for DeepSpeed-native scripts.

The DeepSpeed config file is the interesting artifact; nearly every knob is there. Chapter 2 was about the FSDP path; this chapter is the DeepSpeed path.

## ZeRO-1/2/3, revisited from DeepSpeed's angle

The three stages are the same ones described in chapter 1, but DeepSpeed exposes each with a specific `zero_stage` value:

- **`zero_stage: 0`** — no ZeRO. Equivalent to DDP.
- **`zero_stage: 1`** — shard optimizer state only. Communication cost identical to DDP (still just gradient all-reduce). Memory savings on optimizer moments.
- **`zero_stage: 2`** — shard optimizer state and gradients. Gradient all-reduce is replaced by reduce-scatter (each rank ends with only its shard of the gradient). Larger savings; same net communication volume.
- **`zero_stage: 3`** — shard parameters as well. All-gather per layer during forward and backward. Highest memory savings; highest communication cost.

The choice of stage is the same decision as in FSDP: pick the smallest stage that fits. If ZeRO-2 fits, use ZeRO-2 — it's cheaper. If ZeRO-2 does not fit, escalate to ZeRO-3 (and reach for offload if ZeRO-3 also does not fit).

## A working `ds_config.json` — ZeRO-3, no offload

The base config that most 7B–13B fine-tuning runs start from:

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 50,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

Each field, briefly:

- **`bf16.enabled: true`** — mixed precision. Same trade-off as in chapter 2; BF16 is the default for fine-tuning on Ampere+. Do not enable both `fp16` and `bf16`. FP16 needs `"loss_scale": 0` (dynamic loss scale) and Turing-era hardware; skip for modern GPUs.
- **`zero_optimization.stage: 3`** — the ZeRO stage.
- **`overlap_comm: true`** — overlap gradient communication with backward pass. Universally the right choice; costs a small amount of memory for double buffers.
- **`contiguous_gradients: true`** — store gradients contiguously so reduce-scatter operates on contiguous memory. Reduces fragmentation; standard on.
- **`reduce_bucket_size`** — the size in elements of the bucket used for gradient reduce operations. `"auto"` lets Accelerate / DeepSpeed pick; typical value 5e8 (~1 GB in FP16/BF16). Larger buckets → fewer, bigger reductions, better bandwidth utilization but higher memory. Tune only if profiling shows communication as the bottleneck.
- **`stage3_prefetch_bucket_size`** — size of the parameter-prefetch bucket for ZeRO-3. `"auto"` computes a good default from the hidden size.
- **`stage3_param_persistence_threshold`** — parameters smaller than this threshold stay resident on every rank (avoids the all-gather overhead for small tensors like LayerNorm). `"auto"` picks a sensible value.
- **`stage3_max_live_parameters`** — cap on how many parameters can be materialized (all-gathered but not yet reshared) simultaneously. Higher → more memory, more prefetch overlap. Tune down if OOMing during a specific layer.
- **`stage3_max_reuse_distance`** — how far ahead ZeRO looks when deciding whether to keep a parameter resident. Higher → more memory usage; less re-gather.
- **`stage3_gather_16bit_weights_on_model_save: true`** — write consolidated 16-bit weights on `save_pretrained()`. Without it you get sharded state that requires `zero_to_fp32.py` to consolidate. Turn on unless you're doing custom checkpoint handling.
- **`gradient_accumulation_steps: "auto"`** and **`train_micro_batch_size_per_gpu: "auto"`** — inherit from `TrainingArguments` when using Trainer / Accelerate. Setting to explicit numbers overrides.
- **`gradient_clipping: "auto"`** — DeepSpeed handles gradient clipping natively. When using with Accelerate / Trainer, "auto" reads from the TrainingArguments.
- **`train_batch_size: "auto"`** — computed as `train_micro_batch_size_per_gpu * gradient_accumulation_steps * num_gpus`. Do not set both this and the micro-batch-size independently; it will error.
- **`wall_clock_breakdown: false`** — set to `true` to have DeepSpeed print per-phase timing (forward, backward, comm, step) periodically. Very useful for profiling; noisy in production. Chapter 5 covers this.

To use it via Accelerate, an Accelerate config file references it:

```yaml
# configs/deepspeed_z3.yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_machines: 1
num_processes: 8
machine_rank: 0
main_training_function: main
mixed_precision: bf16

deepspeed_config:
  deepspeed_config_file: configs/ds_config_z3.json
  zero3_init_flag: true
  zero3_save_16bit_model: true
```

Launch with `accelerate launch --config_file configs/deepspeed_z3.yaml train.py ...`. The training script is unchanged from the FSDP version — Accelerate abstracts the difference.

`zero3_init_flag: true` enables ZeRO-3-aware model init inside `from_pretrained` — the model is constructed with parameter shards already partitioned rather than materialized-then-sharded. Turn on for anything past ~7B or you will OOM host RAM on load.

## Adding CPU offload — `ZeRO-Offload`

When ZeRO-3 alone doesn't fit — either because activations are too large or because you have fewer GPUs than expected — the next lever is offloading optimizer state and (optionally) parameters to CPU pinned memory:

```json
"zero_optimization": {
  "stage": 3,
  "offload_optimizer": {
    "device": "cpu",
    "pin_memory": true
  },
  "offload_param": {
    "device": "cpu",
    "pin_memory": true
  },
  ...
}
```

What each block does:

- **`offload_optimizer.device: cpu`** — the FP32 master weights, momentum, and variance move to CPU pinned memory. Optimizer step runs on CPU (DeepSpeed uses a highly optimized DeepSpeedCPUAdam kernel). Frees ~12N bytes of GPU memory. This is what ZeRO-Offload originally added (ZeRO-Offload paper).
- **`offload_param.device: cpu`** — sharded parameters move to CPU pinned memory. Frees ~2N/P bytes of GPU memory per rank. Requires ZeRO-3.
- **`pin_memory: true`** — use pinned host memory for faster GPU↔CPU transfers. Always on unless host memory is contested.

Trade-offs:

- **Optimizer offload alone (`offload_optimizer` only, not `offload_param`)** is cheap: 10–30% throughput hit for a substantial memory saving. Almost always worth it on a memory-limited multi-GPU box.
- **Optimizer + parameter offload** is expensive: 2–5× throughput hit vs no-offload. This is the survival mode for fitting a 70B on 8× 40 GB A100. If throughput is what you need, get more GPUs instead.
- **Host RAM requirement** grows: full optimizer state (48 GB per 7B in FP32 AdamW) plus optionally full parameter state (~14 GB per 7B in BF16) need to fit in system memory. On a 512 GB host, this is fine for one job; on a 128 GB host, be careful.
- **CPU-side compute is now on your critical path.** A slow CPU (or a heavily loaded one — data loading, other processes) drags the whole step. Pin data loaders to specific cores if you're offloading.

Rule of thumb: try `offload_optimizer` first; only add `offload_param` if it doesn't fit; consider whether you can just get more GPUs before either.

## NVMe offload — `ZeRO-Infinity`

For models that don't fit even in host RAM (400B+ scale on a single-node cluster), DeepSpeed adds a third tier: NVMe offload. Optimizer state and/or parameters spill to NVMe SSDs via direct I/O. This is what enabled the "trillion parameter model on a single DGX-2" demonstrations in the ZeRO-Infinity paper.

```json
"zero_optimization": {
  "stage": 3,
  "offload_optimizer": {
    "device": "nvme",
    "nvme_path": "/mnt/nvme_scratch",
    "pin_memory": true,
    "buffer_count": 4,
    "fast_init": false
  },
  "offload_param": {
    "device": "nvme",
    "nvme_path": "/mnt/nvme_scratch",
    "buffer_count": 5,
    "buffer_size": 1e8,
    "max_in_cpu": 1e9
  },
  ...
}
```

Fine-tuning-engineer relevance:

- **You reach for NVMe offload when the model simply cannot be trained any other way with the hardware you have.** It is much slower than CPU offload (which is much slower than no offload). Expect a 5–20× throughput hit vs no-offload for full param + optimizer NVMe.
- **NVMe path performance matters a lot.** A single NVMe SSD at ~3 GB/s reads is a bottleneck; DeepSpeed can use multiple devices via `aio` blocks (`"aio": { ... }` config). This is a systems-engineering concern that shades into `training-pipeline-engineer` territory — chapter 7 flags it.
- **Not the same as CPU offload.** If the model fits in CPU RAM, stay there. NVMe is the escape hatch when it doesn't.
- **Almost never the right tool for a 7B–70B fine-tuning job on a modern cluster.** If you find yourself reaching for it, question whether the cluster shape is wrong (see chapter 7).

## DeepSpeed's compatibility with Trainer, TRL, and Accelerate

The Hugging Face `Trainer` accepts a `deepspeed=` argument that is either a path to a `ds_config.json` or a dict:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="outputs/llama-3.1-8b-ds",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=1,
    bf16=True,
    deepspeed="configs/ds_config_z3.json",  # <-- the config file
    ...
)

trainer = Trainer(model=model, args=training_args, train_dataset=ds, ...)
trainer.train()
```

Launch with `deepspeed --num_gpus 8 train.py` or `accelerate launch --config_file configs/deepspeed_z3.yaml train.py`. Both work; Accelerate is the more portable path if your team uses it elsewhere.

TRL's `SFTTrainer`, `DPOTrainer`, and `KTOTrainer` inherit from `Trainer` and pass the same `deepspeed` argument through. `DPOTrainer`'s reference-model requirement adds one quirk: the reference model has to be initialized *after* the policy model when using ZeRO-3 (otherwise `zero3_init_flag` tries to shard the reference and interacts badly with the trainer's expectations). TRL's docs cover the exact pattern for the current version; check them before starting a DPO run.

**Version compatibility is real and moves fast.** DeepSpeed pins to specific `torch` versions; `accelerate`, `transformers`, `trl`, and `peft` each have their own matrix. When something breaks between versions, the fix is often "the compatibility matrix changed; upgrade / pin accordingly." Chapter 6 covers debugging strategy; the practical answer is to pin every dependency in a `requirements.txt` for a working run and update deliberately.

## FSDP vs DeepSpeed for a specific decision

The trade-off from chapter 1, restated with more specificity:

| Scenario | Prefer FSDP | Prefer DeepSpeed |
|---|---|---|
| Single-node, 7B–13B, model fits comfortably | ✔ (simpler stack) | — |
| Single-node, 34B–70B, needs sharding + activation checkpointing | ✔ (well-supported) | ✔ (either works) |
| Any run that needs optimizer offload but not param offload | — | ✔ (surgical offload of optimizer only; cheaper hit) |
| Any run that needs NVMe offload (ZeRO-Infinity) | — | ✔ (FSDP doesn't offload to NVMe) |
| QLoRA + distributed | ✔ (current best-supported path) | Partial (works but with more version friction) |
| MoE base fine-tuning (Mixtral, DeepSeek-MoE) | Partial | ✔ (DeepSpeed-MoE ecosystem) |
| Multi-node with slow inter-node link | ✔ (`HYBRID_SHARD` is the specific answer) | Partial (ZeRO-3 hurts on slow inter-node; workarounds exist) |
| Need PyTorch-native profiling / debugging | ✔ (first-class citizen) | Partial (DeepSpeed adds its own layer) |
| You want the FLOPS profiler out of the box | — | ✔ (DeepSpeed FLOPS profiler is very good) |
| Your team already runs one of them | Use that one | Use that one |

The last row is the most important. Switching frameworks mid-project costs a week of debugging that could have gone into data or evaluation.

## The `auto` values and how they get filled

When you pass `"auto"` in a `ds_config.json` and launch via Accelerate + Trainer, the actual value is filled from your `TrainingArguments` at initialization time:

| Config field | Filled from |
|---|---|
| `train_batch_size` | `per_device_train_batch_size * gradient_accumulation_steps * world_size` |
| `train_micro_batch_size_per_gpu` | `per_device_train_batch_size` |
| `gradient_accumulation_steps` | `gradient_accumulation_steps` |
| `gradient_clipping` | `max_grad_norm` |
| `bf16.enabled` / `fp16.enabled` | `bf16` / `fp16` |
| `optimizer` block | AdamW with LR / betas / weight_decay from TrainingArguments |
| `scheduler` block | `lr_scheduler_type` + `warmup_steps` / `warmup_ratio` |

This is convenient because you don't have to duplicate settings between two files, but the moment you hand-edit the JSON to override one thing, you own the whole field. Read the current DeepSpeed docs' "auto values" table before overriding anything.

## Checkpointing under DeepSpeed

DeepSpeed writes checkpoints in its own format:

- **`model.save_checkpoint("outputs/", tag="step_1000")`** or the Trainer's `save_state()` / `save_model()` — writes `mp_rank_00_model_states.pt`, one per rank, plus `zero_pp_rank_*_mp_rank_*_optim_states.pt` for the optimizer state.
- **Consolidating to a single-file BF16 model** — the standard script is `zero_to_fp32.py` (ships with DeepSpeed): `python zero_to_fp32.py . pytorch_model.bin`. Reads the sharded state and writes a consolidated file.
- **With `stage3_gather_16bit_weights_on_model_save: true`**, the Trainer's `save_model()` writes a `pytorch_model.bin` (or `model.safetensors`) directly at save time. Preferred for the final checkpoint.

For serving handoff (mod-109), you want the consolidated single-file version. For mid-training resume, the sharded format is fine and much faster to write.

**Checkpoint size discipline:** a 7B BF16 model is ~14 GB on disk consolidated; the sharded state is similar in total but distributed across N files. Multi-node clusters need a shared filesystem all ranks can write to (NFS, GCS, S3 via `s3fs`, EFS, Lustre). If your cluster has a distributed FS with slow write bandwidth, writing checkpoints becomes visible in the step-time trace.

## The two most-common DeepSpeed footguns

1. **Loading a Trainer-compatible model with `zero3_init_flag: true` when you didn't intend ZeRO-3.** The `from_pretrained` call then partitions parameters immediately, and if the rest of the config or the training loop assumes full parameters, things break weirdly (usually with `torch.zeros(0)`-shaped tensors). Match the flag to the stage.
2. **Mixing `deepspeed` argument to Trainer with `fsdp` argument.** The two settings are mutually exclusive but the error message is often obscure. Pick one framework per run.

Both of these fail early enough that catching them costs minutes, not hours — but they cost minutes every single time.

## When DeepSpeed hurts

Two specific scenarios where DeepSpeed is more friction than value:

- **Small models on many GPUs.** For a 1B model on 8× A100, DeepSpeed's ZeRO-3 all-gather / reduce-scatter overhead per layer is a net loss vs plain DDP. Use `zero_stage: 0` (equivalent to DDP) or FSDP `SHARD_GRAD_OP` for these — the parameter-fit isn't the problem.
- **Very short runs on a shared cluster.** DeepSpeed has non-trivial init cost (a few tens of seconds for a large model to shard on load). For a 15-minute run, that init is a meaningful fraction of the total wall clock. FSDP init is faster in most cases.

Neither is a permanent handicap — just choose the tool that fits the run.

## Exercise pointer

`exercises/exercise-02-deepspeed-zero-3-with-offload.md` runs the same SFT job under DeepSpeed ZeRO-3 with and without CPU offload, compares peak memory / throughput / end-loss against the FSDP variant from exercise 01, and asks for a written recommendation of which framework to pick for a specified follow-on scenario.

## Summary

- DeepSpeed is a training-optimization engine with its own JSON config as the primary surface. ZeRO-1/2/3 are the memory-saving stages; ZeRO-Offload adds CPU offload; ZeRO-Infinity adds NVMe.
- **The recommended workflow** is Accelerate + a DeepSpeed config file. The training script stays framework-agnostic.
- **Optimizer offload** (`offload_optimizer.device: cpu`) is cheap and often worth it; **parameter offload** (`offload_param.device: cpu`) is expensive and reserved for survival mode; **NVMe offload** is a special case for models that exceed host RAM.
- **`auto` values** in the config file are filled from `TrainingArguments` at init. Change only what you need to override.
- **Checkpoint format** — DeepSpeed writes its own sharded format; use `stage3_gather_16bit_weights_on_model_save: true` for a consolidated single-file save on the final checkpoint. Serving pipelines expect consolidated; training resume prefers sharded.
- **FSDP vs DeepSpeed** — use FSDP for QLoRA + distributed, PyTorch-native tooling, `HYBRID_SHARD` on multi-node with slow inter-node links; use DeepSpeed for surgical optimizer offload, NVMe offload, MoE, or when you want the FLOPS profiler. Otherwise, use whichever your team already runs.
- **Version compatibility** across `deepspeed`, `torch`, `transformers`, `accelerate`, `trl`, `peft` is real and moves fast. Pin dependencies for reproducibility; update deliberately.
