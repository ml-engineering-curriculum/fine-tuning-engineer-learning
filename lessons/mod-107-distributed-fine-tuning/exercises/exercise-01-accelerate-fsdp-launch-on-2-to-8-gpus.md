# exercise-01: Accelerate + FSDP Launch on 2 → 8 GPUs

**Estimated effort:** 3 hours
**Companion chapters:** [`01-parallelism-landscape.md`](../01-parallelism-landscape.md), [`02-accelerate-and-fsdp.md`](../02-accelerate-and-fsdp.md), [`04-memory-mixed-precision-and-checkpointing.md`](../04-memory-mixed-precision-and-checkpointing.md)

## Objective

Take a working single-GPU SFT script (from mod-103 exercise 01, or an equivalent Trainer / `SFTTrainer` script) and scale it to 2 → 4 → 8 GPUs on a single node using **Hugging Face Accelerate + PyTorch FSDP**. Produce two working `fsdp_config.yaml` variants — a **memory-conservative** config and a **throughput-optimized** config — measure per-rank memory and tok/s/GPU on each, and write a short report defending your `ShardingStrategy`, auto-wrap, mixed-precision, and prefetch choices.

The exercise is a configuration and measurement exercise, not a modeling exercise. The training-loop code should be nearly unchanged from single-GPU; the interesting artifacts are the config files, the measurements, and the written reasoning.

## Prerequisites

- Access to at least 2 GPUs (ideally scale to 8) on a single node. The exercise is written for A100 40 GB / A100 80 GB / H100 SXM; the concepts port to any Ampere-or-later GPU (RTX 3090+, A10, L4, etc.), with the caveat that memory limits shift.
- PyTorch 2.4+ (for FSDP2 support; FSDP1 also works but the exercise references FSDP2 fields where they differ).
- Recent `accelerate`, `transformers`, `datasets`, `trl`, `peft` (pin versions in a `requirements.txt` for reproducibility).
- A single-GPU SFT script that trains cleanly. If you don't have one, the mod-103 exercise 01 script is the reference. A 1B–3B base (e.g., `meta-llama/Llama-3.2-1B` or `Qwen/Qwen2.5-1.5B`) is enough to feel every step; a 7B base is more representative but slower to iterate.
- A tokenized dataset of ~1000–10,000 SFT rows (mod-102 output). Reuse the one you already have.
- `nvidia-smi` and `py-spy` available on every node you launch on.

## Requirements

### Part A — Reproduce single-GPU baseline

Before touching any distributed code, run the single-GPU baseline and record:

- Base model, effective batch size, learning rate, sequence length, dataset size, epochs.
- Peak GPU memory (`torch.cuda.max_memory_allocated()`).
- Tokens per second (steady state; discard first 10 steps).
- Wall clock for a 100-step run.
- Final training loss.

Save the training script and a `baseline_metrics.md` file with the above. Everything downstream is measured against this baseline.

### Part B — Write the memory-conservative FSDP config

Author `configs/fsdp_conservative.yaml`. Design intent: **minimize peak memory at the cost of throughput.** Every knob choice must be defended in the writeup (Part D).

Constraints:

- `distributed_type: FSDP`
- `fsdp_sharding_strategy: FULL_SHARD`
- `fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP` with the correct `fsdp_transformer_layer_cls_to_wrap` for your model (verify by inspecting `type(model.model.layers[0]).__name__`).
- `fsdp_activation_checkpointing: true`
- `fsdp_backward_prefetch: BACKWARD_POST` (memory-conservative choice)
- `fsdp_forward_prefetch: false`
- `fsdp_offload_params: false` (add only if the config OOMs)
- `fsdp_use_orig_params: true`
- `fsdp_state_dict_type: SHARDED_STATE_DICT`
- `fsdp_sync_module_states: true`
- `fsdp_cpu_ram_efficient_loading: true`
- `mixed_precision: bf16`

Companion `TrainingArguments` (or CLI args): `per_device_train_batch_size: 1`, sequence length matching the baseline, `attn_implementation: flash_attention_2`, gradient accumulation to match the baseline's effective batch.

### Part C — Write the throughput-optimized FSDP config

Author `configs/fsdp_throughput.yaml`. Design intent: **maximize tok/s/GPU on the hardware you have.**

Compared to the conservative config, expected differences:

- `fsdp_backward_prefetch: BACKWARD_PRE`
- `fsdp_forward_prefetch: true`
- `per_device_train_batch_size` grown to the largest value that fits (you'll bisect this).
- Consider `fsdp_sharding_strategy: SHARD_GRAD_OP` (ZeRO-2 equivalent) if the model comfortably fits per-rank at your world size — trades memory for less all-gather traffic.
- Consider disabling activation checkpointing if memory allows — 30% throughput back at 4–8× activation memory.

Every deviation from the conservative config must be defensible.

### Part D — Sweep 2 → 4 → 8 GPUs, both configs

For each of `{num_processes: 2, 4, 8}` × `{conservative, throughput}` (six runs total), collect:

- **Peak per-rank memory** (`torch.cuda.max_memory_allocated()` on rank 0 and one other rank; note if they differ).
- **Steady-state tok/s total and per-GPU** (median over the last 50 steps; discard first 10).
- **Wall clock for 100 steps** (or a fixed token budget).
- **Final training loss** at the end of a fixed-length run.
- **Whether the run OOM'd, hung, or completed cleanly.** OOM at a specific step counts as a real result, not a failure — log the mitigation you would apply next.

Log format: one JSON or CSV row per run, all fields above plus config name and world size. Commit the log file to `results/exercise-01/`.

### Part E — Verify the sharding is actually happening

For at least the 8-GPU FULL_SHARD run:

1. **Print per-rank weight memory right after model load.** It should be roughly `2N/P` bytes. If it's `2N`, sharding isn't working — investigate.
2. **Print per-rank optimizer state after the first `optimizer.step()`.** Should be roughly `12N/P` bytes for FP32 AdamW.
3. **Inspect the FSDP wrapping.** With `fsdp_state_dict_type: SHARDED_STATE_DICT`, `accelerator.save_state("test_ckpt")` should produce `world_size` files. Verify.
4. **Verify activation checkpointing is on.** Turn on the PyTorch profiler for 3 steps, dump the trace, and check for `CheckpointFunctionBackward` in the backward trace.

Report the findings in `verification.md`.

### Part F — Write the report

`report.md`, structured as:

1. **Baseline recap** — from Part A.
2. **Config diff** — a table of the ~10 knobs where `fsdp_conservative.yaml` and `fsdp_throughput.yaml` differ, with a one-line justification per knob.
3. **Results table** — the six-run sweep from Part D. Compute tok/s/GPU speedup vs. baseline single-GPU (should be near-linear for well-behaved sharding; sub-linear on 8 GPUs is expected).
4. **Sharding verification** — from Part E, a paragraph confirming the per-rank memory matches expectations.
5. **Bottleneck analysis** — for the 8-GPU throughput run, what is the bottleneck? Compute? Communication? Data loading? (Use `nvidia-smi dmon` or PyTorch profiler as backup; the report can be qualitative.)
6. **Recommendation** — if a peer asked "which config should I use for a real production job?" what would you tell them and why?

### Part G — Save

Directory layout:

```
outputs/exercise-01/
├── configs/
│   ├── fsdp_conservative.yaml
│   └── fsdp_throughput.yaml
├── train.py                       # the (barely-modified) SFT script
├── requirements.txt               # pinned versions
├── baseline_metrics.md
├── results/
│   └── sweep_results.csv          # from Part D
├── verification.md
├── report.md
└── logs/                          # per-rank stdout for at least the 8-GPU runs
```

## Starter guidance

- **Do the baseline first, on the exact hardware you'll scale to.** Different GPU SKUs give different tokens/sec numbers; don't confuse yourself by using yesterday's laptop-side baseline.
- **Verify FA2 is active.** `attn_implementation="flash_attention_2"` in `from_pretrained`; the PyTorch profiler should show `flash_attn_*` kernels. Without FA2, the activation-memory numbers are much larger and the exercise's memory conclusions won't line up with what chapter 4 predicts.
- **Small model first, big model second.** Get the config working on Llama-3.2-1B or Qwen2.5-1.5B where iteration is fast. Once the config runs cleanly, swap in the 7B / 13B base and repeat the sweep. The config file itself changes little (just `fsdp_transformer_layer_cls_to_wrap` if the architecture differs).
- **Log per-rank, prefixed.** `[rank 3] step 42 loss 1.87 mem_gb 34.5` is the format. Chapter 6 covers why this matters; you'll want it immediately for this exercise.
- **`accelerate config` first, then hand-edit.** Run the interactive `accelerate config` once to see the field structure, then commit a hand-authored config file to the repo.
- **Do not fight version conflicts.** Pin a working version set in `requirements.txt` at the top. `transformers==4.NN`, `accelerate==0.MM`, `torch==2.LL`, `trl==0.KK`, and stop moving them for the duration of the exercise.

## Acceptance criteria

You have completed the exercise when:

- `configs/fsdp_conservative.yaml` and `configs/fsdp_throughput.yaml` exist and launch runs cleanly on 2, 4, and 8 GPUs (or reproducibly OOM with a documented reason).
- `results/sweep_results.csv` has all six rows filled with the fields from Part D.
- `verification.md` confirms sharded per-rank weight memory is `~2N/P`, not `2N`.
- `report.md` covers all six sections from Part F, with a defensible recommendation.
- The 8-GPU throughput run shows a tok/s speedup meaningfully greater than 4× vs baseline (roughly `world_size · 0.6` or better is a reasonable expectation for a well-tuned run; less than that means you have a bottleneck to explain).
- Sharded-checkpoint save and reload works: `accelerator.save_state()` writes shards, and re-launching with `accelerator.load_state()` on the same world size resumes cleanly.

## Stretch goals

- **Consolidate the sharded checkpoint** to a single-file `pytorch_model.bin` / `model.safetensors` with `FULL_STATE_DICT` (or `torch.distributed.checkpoint`'s offline consolidation). Verify a `AutoModelForCausalLM.from_pretrained("consolidated/")` loads and produces the same logits as the sharded version on a small eval batch.
- **Run under FSDP2 (`fully_shard`) instead of FSDP1.** Depending on your PyTorch version this may require code changes; the Accelerate abstraction usually covers it. Verify the throughput and memory numbers match FSDP1 within a few percent.
- **Try LoRA + FSDP.** Add `LoraConfig` (mod-104 exercise 01) and re-run at 8 GPUs. Confirm `use_orig_params: true` is required. Report the memory / throughput delta vs full FT.
- **Try HYBRID_SHARD.** If you have access to a two-node cluster (or can simulate with `NCCL_TESTS_DEFAULT_SIMULATE_INTERNODE=1`-style env vars), run the throughput config with `fsdp_sharding_strategy: HYBRID_SHARD` and compare inter-node bandwidth impact.
- **Add `torch.compile`.** Turn it on for the throughput config; measure the delta. Expect ~10–30% throughput on top; the first step will be slow while compilation runs.
- **QLoRA + FSDP.** If your stack supports it (check compatibility matrix), scale the mod-104 exercise 02 QLoRA run to 4 GPUs with FSDP. Compare per-rank memory to non-distributed QLoRA.
