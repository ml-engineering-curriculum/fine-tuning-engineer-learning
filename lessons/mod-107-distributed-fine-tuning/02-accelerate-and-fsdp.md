# Accelerate and PyTorch FSDP

**Estimated read:** 50 minutes.
**You should be able to, by the end:** launch a fine-tuning run across 2 → 8 GPUs and, when your team's cluster is ready, across nodes with `accelerate launch` and PyTorch FSDP; write a working `fsdp_config.yaml` from scratch; choose the right `ShardingStrategy` for a given model / hardware combination; configure the mixed-precision policy, auto-wrap policy, and CPU offload knobs; and save + reload checkpoints in both sharded and consolidated formats.

## The Accelerate + FSDP stack in one picture

The Hugging Face `Trainer` (used by TRL's `SFTTrainer` and `DPOTrainer`) sits on top of `accelerate.Accelerator`, which sits on top of `torch.distributed` and its FSDP module. When you invoke `accelerate launch train.py`, three things happen:

1. **Accelerate reads a config file** (`~/.cache/huggingface/accelerate/default_config.yaml` by default, or one passed with `--config_file`) that tells it *which* distributed backend to use (multi-GPU DDP, FSDP, DeepSpeed, TPU, etc.) and the per-backend settings.
2. **Accelerate delegates process launch** to `torch.distributed.run` (a.k.a. `torchrun`) with the right number of workers per node and the right rendezvous config.
3. **Inside each worker**, Accelerate's `Accelerator` object wraps the model with the chosen distributed strategy — for FSDP, that means constructing a `FullyShardedDataParallel` (FSDP1) or calling `fully_shard(...)` (FSDP2) on the model — configures the optimizer / dataloader accordingly, and hands the wrapped objects back to your training script.

Your training loop is (in the ideal case) unchanged from the single-GPU version. That is the point of Accelerate: the same script runs single-GPU, DDP, FSDP, and DeepSpeed under different `accelerate config` selections. Whether it actually is that transparent depends on how carefully you avoid framework-specific escape hatches; the exercise walks you through the common landmines.

## The launcher: `accelerate config` and `accelerate launch`

The config file is written interactively (`accelerate config`) or authored by hand. The interactive path is fine for a first run; for production, commit a config file to the repository next to the training script so the launch command is `accelerate launch --config_file configs/fsdp_8gpu.yaml train.py`.

A minimal FSDP config for 8 GPUs on one node looks like:

```yaml
# configs/fsdp_8gpu.yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
num_machines: 1
num_processes: 8
machine_rank: 0
main_training_function: main
mixed_precision: bf16

fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_use_orig_params: true
  fsdp_cpu_ram_efficient_loading: true
  fsdp_sync_module_states: true
  fsdp_activation_checkpointing: true

rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
```

Every field has a story; the important ones are unpacked below. Launch it with:

```bash
accelerate launch --config_file configs/fsdp_8gpu.yaml train.py \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --dataset_name your/sft-mix \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir outputs/llama-3.1-8b-sft-fsdp
```

For multi-node, change `num_machines` to the node count, set `machine_rank` per-machine (0, 1, 2, ...), and use `rdzv_backend: c10d` with a `main_process_ip` reachable from every node. `accelerate launch` becomes `accelerate launch --multi_gpu --num_machines 2 --machine_rank $NODE_RANK ...` or the config file carries all of it. Chapter 7 covers what the platform team has to provide for this to just work.

## The FSDP APIs: FSDP1 vs FSDP2

Two generations of the API exist and you will see both in the wild.

- **FSDP1 (`torch.distributed.fsdp.FullyShardedDataParallel`)** — the original API, wraps a whole module. Introduced in PyTorch 1.11, matured through 2.x. Stable and well-documented, but has some sharp edges around parameter groups and mixed-precision policies.
- **FSDP2 (`torch.distributed.fsdp.fully_shard`)** — introduced in PyTorch 2.4, uses a per-parameter sharding model (based on `DTensor`) that is more composable with other parallelism strategies (TP, PP) and has cleaner semantics for parameter groups and gradient hooks. Actively developed; the recommended path for new code as of PyTorch 2.4+.

Accelerate abstracts most of the difference — the same YAML fields work — but a training script that reaches into FSDP internals needs to pick one. For new fine-tuning projects on PyTorch 2.4+, prefer FSDP2 (`fsdp_version: 2` in recent Accelerate versions). For 2.0–2.3 compatibility, FSDP1 is fine. Check the PyTorch release notes for the version your cluster runs.

The rest of this chapter uses the FSDP1 field names because they are what most existing configs use. Where FSDP2 differs materially, it is called out.

## `ShardingStrategy`: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, NO_SHARD

`fsdp_sharding_strategy` controls what gets sharded. Four options in FSDP1:

- **`FULL_SHARD`** — shard parameters, gradients, and optimizer state (ZeRO-3 equivalent). All-gathers full parameters per layer during forward and backward. Highest memory savings, highest communication cost. Default choice for any run where the model doesn't otherwise fit.
- **`SHARD_GRAD_OP`** — shard only gradients and optimizer state (ZeRO-2 equivalent). Parameters remain full-copy on every rank. All-reduce on gradients (like DDP); no all-gather on forward. Lower communication than FULL_SHARD; more per-rank memory. Use when parameters *do* fit per-rank but optimizer state doesn't.
- **`HYBRID_SHARD`** — FULL_SHARD within a node, DDP across nodes. On a 2-node × 8-GPU cluster, each node is a FULL_SHARD group of 8; the two groups are DDP'd across the inter-node link. Cuts inter-node all-gather traffic (which is slow over Ethernet / InfiniBand) at the cost of higher per-node memory. Named "hybrid" because it composes ZeRO-3 (fast intra-node) with DDP (bandwidth-conscious inter-node). Introduced in PyTorch 2.0; a common production choice for multi-node runs.
- **`NO_SHARD`** — equivalent to DDP (nothing is sharded). Provided so the same FSDP code path can degrade to DDP when appropriate.

The decision, roughly:

- **`FULL_SHARD`** — 7B+ full-FT on a single node; anything where the model doesn't fit per-rank.
- **`SHARD_GRAD_OP`** — 7B LoRA on multi-GPU where the model easily fits and you want to avoid the all-gather cost; smaller models where per-rank fit is not the constraint.
- **`HYBRID_SHARD`** — multi-node runs where inter-node bandwidth is the bottleneck. Almost always what you want on 2+ nodes.
- **`NO_SHARD`** — fallback / degradation path only.

FSDP2 uses `fully_shard(module, reshard_after_forward=True|False, ...)` where `reshard_after_forward=True` is FULL_SHARD-like and `False` is SHARD_GRAD_OP-like. The concept is the same; the API knob is different.

## Auto-wrap policy: which modules to shard

FSDP shards at the granularity of "wrapped modules." A too-fine wrap (every `nn.Linear`) makes too many small all-gathers and communication overhead dominates; a too-coarse wrap (the whole model as one FSDP unit) means the shard is huge and there is no way to overlap communication with computation.

The correct granularity for transformer fine-tuning is one FSDP unit per transformer block. Accelerate exposes this as `fsdp_auto_wrap_policy`:

- **`TRANSFORMER_BASED_WRAP`** — the recommended default. You supply the module class name to wrap (`LlamaDecoderLayer`, `MistralDecoderLayer`, `Qwen2DecoderLayer`, `GPT2Block`, etc.) via `fsdp_transformer_layer_cls_to_wrap`, and FSDP wraps one unit per decoder layer.
- **`SIZE_BASED_WRAP`** — wraps modules based on parameter count (default threshold 100M). Fine as a fallback when you don't know the transformer block class; less precise than TRANSFORMER_BASED_WRAP.
- **`NO_WRAP`** — wraps only the outermost module. Almost never what you want for a transformer.

If you get `fsdp_transformer_layer_cls_to_wrap` wrong (say, you specify `LlamaDecoderLayer` on a Mistral model), FSDP either silently degrades to whole-model wrapping (bad) or errors out on init. The class name is architecture-specific and you can find it by inspecting the model: `type(model.model.layers[0]).__name__` prints the right value.

For architectures without a single obvious block (some vision or multi-modal models), you can pass multiple classes as a comma-separated list. For MoE models, wrap both the decoder layer and the expert modules separately.

FSDP2's equivalent is calling `fully_shard(...)` on the layer instances explicitly in your training script. Accelerate's YAML abstracts this.

## Mixed precision: `MixedPrecision` policy

FSDP has three orthogonal dtype knobs — parameter dtype, reduce (gradient reduction) dtype, and buffer dtype — configured by the `MixedPrecision` object. Accelerate's top-level `mixed_precision: bf16` sets sensible defaults (params in BF16, reduce in BF16 or FP32 depending on version, buffers in BF16); for more control, you can override in code.

Rules of thumb:

- **`param_dtype=bf16`, `reduce_dtype=fp32`, `buffer_dtype=bf16`** — the most conservative choice, closest to full BF16 with FP32 gradient reduction. Small memory overhead for the FP32 reduce; catches most numerical issues.
- **`param_dtype=bf16`, `reduce_dtype=bf16`, `buffer_dtype=bf16`** — the fully-BF16 path. Faster and more memory-efficient; occasionally trips on numerical issues in small-magnitude gradients. Standard for most fine-tuning.
- **FP16 (not BF16):** legacy. Use only if you're on Turing (T4, V100) and truly don't have BF16 available. Requires loss scaling.
- **FP8:** Hopper-only (H100, H200), requires NVIDIA Transformer Engine. Not a Accelerate/FSDP config option — you set it up inside your model by replacing certain modules with TE equivalents. Chapter 4 covers it.

The `bf16` vs `fp32` distinction in the *reduce* is subtle but important: FP16 (which you should not be using for fine-tuning most 7B+ models anyway) with BF16 or FP16 reduce can lose small gradient magnitudes to underflow. BF16 has the same exponent range as FP32, so this is much less of a problem, and modern practice for LLM fine-tuning is BF16 all the way.

## Backward prefetch and forward prefetch

FSDP overlaps communication (the all-gather that materializes the next layer's parameters) with computation (the current layer's matmul). Two flags control the aggressiveness of that overlap:

- **`fsdp_backward_prefetch: BACKWARD_PRE`** — start the all-gather for layer N-1 before backward on layer N finishes. The default and usually the right choice. Higher throughput; slightly higher peak memory (two layers' worth of parameters resident briefly).
- **`fsdp_backward_prefetch: BACKWARD_POST`** — start the all-gather for layer N-1 after backward on layer N finishes. Lower peak memory; lower throughput.
- **`fsdp_backward_prefetch: null`** — no prefetch. Diagnostic setting; do not use in production.
- **`fsdp_forward_prefetch: true`** — start the all-gather for layer N+1 during forward on layer N. Higher throughput; higher peak memory. Off by default; turn on if profiling shows communication as a bottleneck and you have memory headroom.

If your run is OOMing right at peak, try `BACKWARD_POST` and `forward_prefetch: false`. If your run has memory headroom and you want more tokens/sec, try `BACKWARD_PRE` and `forward_prefetch: true`.

## CPU offload

`fsdp_offload_params: true` moves *sharded parameters* to CPU pinned memory, staging them back to the GPU only during their layer's forward / backward. Cuts per-rank GPU memory dramatically at the cost of PCIe traffic every step.

- Use it when the model + optimizer state + activations do not fit even with FULL_SHARD across your available GPUs and you cannot get more GPUs.
- Expect a 2–5× throughput hit relative to non-offload. It is a survival mode, not a performance mode.
- CPU offload of *optimizer state* specifically is a DeepSpeed-native feature (`offload_optimizer.device: cpu` in ZeRO config). FSDP's offload is coarser — sharded parameters, not just optimizer state — so on a like-for-like FSDP-CPU-offload vs DeepSpeed-optimizer-offload comparison, DeepSpeed often wins on throughput.

Chapter 3 covers offload in more depth; the short version is that if you find yourself reaching for offload, DeepSpeed's more surgical version is usually a better fit than FSDP's.

## Activation checkpointing under FSDP

Activation (a.k.a. gradient) checkpointing trades compute for activation memory: forward pass recomputes activations during backward instead of storing them. Under FSDP there are two ways to enable it:

1. **`fsdp_activation_checkpointing: true`** in the Accelerate config — applies `apply_activation_checkpointing` from `torch.distributed.algorithms._checkpoint.checkpoint_wrapper` to each wrapped transformer block. Preferred when using FSDP because it composes correctly with FSDP's parameter unshard/reshard flow.
2. **`model.gradient_checkpointing_enable()`** on the HF model — the transformers-side checkpointing that also works with DDP. Compatible with FSDP but occasionally has interaction quirks with `use_orig_params` and specific model architectures.

For FSDP+HF setups, **prefer the Accelerate-level flag** (`fsdp_activation_checkpointing`) and set `use_reentrant=False` under the hood (Accelerate does this by default in recent versions). Chapter 4 covers the memory math.

If you enable both, you'll likely get double checkpointing (correct but slow) or an error. Pick one path per run.

## `use_orig_params` and its consequences

`fsdp_use_orig_params: true` (default in recent versions) exposes the original parameter references through FSDP, meaning:

- Optimizer parameter groups work naturally (different LRs for different param groups).
- `named_parameters()` returns familiar names.
- Frozen parameters (LoRA setups, embedding-only tuning) work without extra ceremony.

Set it to `true` unless you have a specific reason otherwise. `false` (the legacy mode) flattens all parameters into a single tensor and makes parameter-group manipulation painful. Almost everything you'll want to do — LoRA under FSDP, different LRs for embedding vs body, freezing the LM head — is materially easier with `use_orig_params: true`.

## `sync_module_states` and `cpu_ram_efficient_loading`

Loading a 70B checkpoint takes real memory. Two flags mitigate:

- **`fsdp_sync_module_states: true`** — only rank 0 loads the weights; other ranks receive them via broadcast during FSDP init. Saves `num_gpus - 1` copies of the weights in CPU memory during load. Turn on for anything past 7B.
- **`fsdp_cpu_ram_efficient_loading: true`** — combined with the above, uses HF's `low_cpu_mem_usage=True` init to construct the model on the meta device and materialize weights per-shard. Saves RAM on the loader-rank node. Turn on for anything past ~13B or when host RAM is tight.

Together these two flags are the difference between "loads cleanly" and "OOMs on host RAM before training even starts" on multi-node runs of large models.

## Checkpoint format: sharded vs full state dict

FSDP writes checkpoints in one of three formats:

- **`SHARDED_STATE_DICT`** — every rank writes its own shard. Fast, memory-efficient, resumes cleanly to the same world size. Not directly consumable by inference / serving pipelines that expect a single file. The recommended format for training-time checkpoints (mid-run, in case of resume).
- **`FULL_STATE_DICT`** — rank 0 gathers the full weights (via all-gather) and writes a single file. Slower, requires enough rank-0 memory to hold the full model. Consumable by serving pipelines that expect `pytorch_model.bin` / `model.safetensors`. Standard for the *final* checkpoint you hand off to mod-108 / mod-109.
- **`LOCAL_STATE_DICT`** — legacy, avoid.

Common pattern: write `SHARDED_STATE_DICT` mid-training (every N steps for restart safety), and write `FULL_STATE_DICT` (or convert offline) for the shipped final checkpoint. The `torch.distributed.checkpoint` docs describe the offline consolidation flow; Accelerate's `Accelerator.save_state()` and `save_model()` handle both.

For very large models where even rank-0 cannot hold the full state, use the sharded checkpoint and run consolidation as a post-hoc distributed job (using `torch.distributed.checkpoint.load_state_dict` with a full-state-dict target).

## The training loop with Accelerate

A minimal FSDP training loop looks like this (deliberately not using the higher-level `Trainer` API so the plumbing is visible):

```python
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Accelerate reads FSDP config from the launcher's --config_file.
# No FSDP wrapping code needed here — Accelerate does it.
accelerator = Accelerator()

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B",
                                             torch_dtype="auto",
                                             low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

optimizer = AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.95), weight_decay=0.1)
train_loader = DataLoader(your_dataset, batch_size=2, collate_fn=your_collator)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                            num_training_steps=1000)

model, optimizer, train_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, scheduler
)

model.train()
for step, batch in enumerate(train_loader):
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    if step % 10 == 0 and accelerator.is_main_process:
        print(f"step {step}, loss {loss.item():.4f}")

accelerator.wait_for_everyone()
accelerator.save_state("outputs/ckpt-final")  # writes SHARDED_STATE_DICT
```

Five things to notice:

- **No FSDP import in the script.** The wrapping happens inside `accelerator.prepare(...)` based on the launcher config. Same script runs single-GPU, DDP, or DeepSpeed just by swapping the config.
- **`accelerator.accumulate(model)`** — the gradient-accumulation context manager. Under FSDP it correctly skips the all-reduce / all-gather on non-final micro-batches. Do not roll your own accumulation with a manual `optimizer.step()` skip; you'll break FSDP's gradient sync.
- **`accelerator.backward(loss)`** — call this, not `loss.backward()`. Handles mixed-precision scaling and any framework-specific backward hooks.
- **`accelerator.is_main_process`** — always gate logging / printing on this, otherwise you get 8× duplicated log lines.
- **`accelerator.save_state`** — writes both model and optimizer state in the format specified by the config. Restart with `accelerator.load_state("path/to/ckpt")`.

With `SFTTrainer` (TRL) or `Trainer` (transformers), all of this is inside the trainer; you set the same Accelerate config and pass `training_args=TrainingArguments(...)` with `fsdp="full_shard auto_wrap"` and `fsdp_config={"transformer_layer_cls_to_wrap": "LlamaDecoderLayer", ...}` as an alternative. The training-loop code disappears. Either path works.

## LoRA / PEFT under FSDP

LoRA adapters (mod-104) compose with FSDP but have a few sharp edges:

- **`use_orig_params: true` is essentially mandatory.** LoRA changes the parameter set (only adapter params require grad); with `false`, FSDP's flattening interacts badly with the mixed frozen/trainable groups.
- **Wrap the transformer block, not the adapter.** The auto-wrap policy sees the whole decoder layer as one unit; adapters live inside it. Do not add a separate wrap policy for `LoraLayer` — you'll oversplit.
- **Save the adapter, not the whole model.** After training, use `model.save_pretrained("outputs/adapter")` on the underlying PEFT model to write just the LoRA weights (~100s of MB) instead of the full sharded model. Load with `PeftModel.from_pretrained(base_model, "outputs/adapter")`.
- **QLoRA + FSDP** has historically been a compatibility minefield. As of PyTorch 2.4 + recent Accelerate / bitsandbytes / PEFT versions it works, but check the current compatibility matrix. Answer.AI published a well-known write-up ("You can now train a 70B model at home") of the working recipe; the exercise mentions it as a pointer.

## A working config, walked through

The `configs/fsdp_8gpu.yaml` at the top of the chapter, field by field:

- `compute_environment: LOCAL_MACHINE` — this is a bare-metal / VM run, not a cloud-managed environment.
- `distributed_type: FSDP` — the distributed backend choice.
- `num_machines: 1` — single node.
- `num_processes: 8` — 8 workers, one per GPU on this node.
- `machine_rank: 0` — this machine's rank in the multi-node topology (0 because single node).
- `mixed_precision: bf16` — sets the top-level dtype policy; FSDP's `MixedPrecision` block inherits.
- `fsdp_sharding_strategy: FULL_SHARD` — ZeRO-3 equivalent.
- `fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP` — one FSDP unit per transformer block.
- `fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer` — the specific block class (change per architecture).
- `fsdp_backward_prefetch: BACKWARD_PRE` — overlap the all-gather for layer N-1 with backward on layer N.
- `fsdp_forward_prefetch: false` — do not prefetch layer N+1 during forward N (memory-conscious default).
- `fsdp_offload_params: false` — parameters stay on GPU. Turn true only if OOM.
- `fsdp_state_dict_type: SHARDED_STATE_DICT` — training checkpoints are sharded (fast restart).
- `fsdp_use_orig_params: true` — expose original params (LoRA-friendly, cleaner optimizer groups).
- `fsdp_cpu_ram_efficient_loading: true` — meta-device init + per-shard materialization.
- `fsdp_sync_module_states: true` — rank 0 loads weights, others receive via broadcast.
- `fsdp_activation_checkpointing: true` — apply checkpoint wrapper to each wrapped block.
- `rdzv_backend: static` — single-node rendezvous; multi-node uses `c10d`.
- `same_network: true` — hint for the launcher.

Every one of those is a decision that was justified above. When something breaks in production, this is the file that gets reviewed first.

## Common failure at first launch

Three failures every fine-tuning engineer hits at least once:

1. **`ValueError: transformer_layer_cls_to_wrap ... not found`** — the class name in the config doesn't match the actual block class in the model. Inspect: `python -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('...'); print(type(m.model.layers[0]).__name__)"`.
2. **Silent hang at first forward pass** — usually NCCL / process-group init not reaching all ranks. Set `NCCL_DEBUG=INFO` and `TORCH_DISTRIBUTED_DEBUG=DETAIL` and re-launch; look for which rank never checked in. Chapter 6 has the debugging playbook.
3. **Loss is exactly the same on every rank at step 0 but diverges after** — usually a data pipeline bug where the same batch is served to every rank instead of split. Use Accelerate's `DataLoaderConfiguration(split_batches=True)` or a `DistributedSampler` explicitly.

## Exercise pointer

`exercises/exercise-01-accelerate-fsdp-launch-on-2-to-8-gpus.md` walks the config-and-launch path end to end: take a working single-GPU SFT script, scale it to 2 → 4 → 8 GPUs, and produce two working configs (a memory-conservative one and a throughput-optimized one), with per-rank memory and tokens/sec/GPU measurements for each.

## Summary

- Accelerate is the launcher / wrapper; FSDP is the sharded-data-parallel implementation inside PyTorch. Your training script is (mostly) the same regardless; the config picks the strategy.
- **`ShardingStrategy`** — `FULL_SHARD` (ZeRO-3), `SHARD_GRAD_OP` (ZeRO-2), `HYBRID_SHARD` (intra-node FULL_SHARD + inter-node DDP), `NO_SHARD` (DDP). `HYBRID_SHARD` is the multi-node default; `FULL_SHARD` is the single-node default for models that don't fit per-rank.
- **Auto-wrap policy** — one FSDP unit per transformer block. `TRANSFORMER_BASED_WRAP` with the correct `transformer_layer_cls_to_wrap` for your architecture is right in ~95% of cases.
- **Mixed precision** — BF16 all the way for most fine-tuning; FP32 reduce optional; FP16 only for pre-Ampere hardware; FP8 via Transformer Engine on Hopper only (chapter 4).
- **Prefetch and offload** — trade memory for throughput; back off on `BACKWARD_PRE` / `forward_prefetch` when peak memory is the constraint; reach for `offload_params` only as a survival tool.
- **Checkpoint format** — sharded during training (fast, restart-safe), full-state-dict for the final shipped artifact. `save_state` and `save_model` handle both.
- **LoRA + FSDP** — `use_orig_params: true`, wrap the transformer block (not the adapter), save the adapter separately. QLoRA + FSDP works on current versions; consult the compatibility matrix before committing.
- The chapter's example `fsdp_8gpu.yaml` is a defensible starting point for a Llama-family model on 8 GPUs. Swap the transformer-layer class per architecture; everything else transfers.
