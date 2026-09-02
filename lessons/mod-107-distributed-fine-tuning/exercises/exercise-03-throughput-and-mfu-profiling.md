# exercise-03: Throughput and MFU Profiling

**Estimated effort:** 3 hours
**Companion chapters:** [`04-memory-mixed-precision-and-checkpointing.md`](../04-memory-mixed-precision-and-checkpointing.md), [`05-throughput-mfu-and-profiling.md`](../05-throughput-mfu-and-profiling.md)

## Objective

Instrument the distributed fine-tuning run from [exercise 01](exercise-01-accelerate-fsdp-launch-on-2-to-8-gpus.md) (or [exercise 02](exercise-02-deepspeed-zero-3-with-offload.md), your choice), capture **tokens/sec/GPU**, **step-time breakdown**, and a **PyTorch profiler trace**, compute **MFU** by hand from the PaLM-style formula, identify **the single top bottleneck** (compute-bound / communication-bound / memory-bandwidth-bound / data-loader-bound / bubble-limited), and produce a one-page profiling report a peer could act on without asking you follow-up questions.

The value of this exercise is the discipline of measuring instead of guessing. "It feels slow" is worth nothing; "MFU is 32%, top kernel is `all_gather` on the critical path, forward prefetch is off — try enabling it" is worth an afternoon.

## Prerequisites

- Exercise 01 or 02 complete, with a working distributed training script that runs cleanly at 4–8 GPUs.
- One of the throughput-optimized configs from those exercises (FSDP `FULL_SHARD` with prefetch on, or DeepSpeed ZeRO-3 no-offload). This is the run you'll profile.
- `torch.profiler` (bundled with PyTorch), TensorBoard with the `torch-tb-profiler` plugin (or `chrome://tracing` for viewing the trace). If your team uses `perfetto.dev`, that also works.
- `nvidia-smi` in a second terminal.
- Optional: NVIDIA Nsight Systems (`nsys`) installed on the training node for the stretch goal. Skip if unavailable — it isn't required for the base exercise.
- Optional: DeepSpeed FLOPS profiler (only if you're profiling a DeepSpeed run; enabled via `flops_profiler` block in the JSON — you already ran this in exercise 02 Part C).

## Requirements

### Part A — Baseline the run before instrumenting

Profilers add overhead — sometimes a lot. Before turning any of them on, record clean baseline numbers so you can quantify the profiling overhead itself.

Run the throughput config for at least 100 steady-state steps (discard the first 10) and log:

- **Tokens per step**: `per_device_train_batch_size · seq_length · gradient_accumulation_steps · world_size`. Note whether the tokens include padding or only real tokens (packed sequences); document the convention.
- **Step time** (rolling median of the last 50 optimizer steps).
- **Tokens per second total** and **tokens per second per GPU**.
- **`torch.cuda.max_memory_allocated()` per rank** at the top of each epoch.
- **Model parameter count `N`** (read from `sum(p.numel() for p in model.parameters())` — record on rank 0).

Save to `results/exercise-03/baseline_throughput.md`.

### Part B — Compute MFU by hand

Using your baseline numbers:

    MFU = (6 · N · tokens_per_second_total) / (peak_FLOPS_per_GPU · num_GPUs)

Where `peak_FLOPS_per_GPU` is the **dense BF16 tensor-core** peak for your specific GPU SKU. Use the figure from NVIDIA's current datasheet — the numbers in the chapter-5 table are a starting point but verify.

Report:

- **MFU** as a percentage.
- **HFU** as well if you have activation checkpointing on: HFU uses `~8 · N` per token instead of `6 · N` (the extra `2 · N` is the recompute forward pass). Note explicitly whether you report MFU, HFU, or both, and which peak-FLOPS number you used.
- **Sanity check against chapter 5's typical ranges** — a full-FT run should land 50–60% MFU well-tuned; LoRA/QLoRA runs land lower. If yours is outside these ranges (>75% or <25%), double-check the arithmetic before doing anything else. Common errors: using the FP32 (not BF16) peak, using the sparsity-inclusive peak, dropping a factor of gradient accumulation from tokens/step, conflating total tokens/sec with per-GPU tokens/sec.

Commit `results/exercise-03/mfu_calculation.md` with the numbers, the formula, and the peak-FLOPS source (link to the specific NVIDIA datasheet page you used).

### Part C — Capture a PyTorch profiler trace

Instrument the training loop with `torch.profiler` for a short window of steady-state steps:

```python
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

prof_schedule = schedule(wait=5, warmup=2, active=3, repeat=1)
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    on_trace_ready=tensorboard_trace_handler("results/exercise-03/profiler_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
) as prof:
    for step, batch in enumerate(train_loader):
        # normal training step, unchanged
        prof.step()
        if step >= 15:
            break
```

Only rank 0 needs to write the trace (`if accelerator.is_main_process:` around the `with profile(...)` block); a full multi-rank trace multiplies file size and confuses the viewer.

Open the resulting `.pt.trace.json` in TensorBoard's profiler plugin, `chrome://tracing`, or perfetto. Zoom in on **one representative optimizer step** and answer the following in `results/exercise-03/profiler_findings.md`:

1. **Kernel top-5**: sort by total CUDA time. Report the top 5 kernels by name and total time. Expected: 3–4 gemm kernels (`ampere_bf16_...gemm`, `sm90_..._gemm`, `flash_fwd_...`, `flash_bwd_...`) and possibly one attention kernel. Any surprise entry (e.g., a large `dropout` kernel, unfused element-wise, indexing) is a lead worth naming.
2. **Is FlashAttention actually active?** Look for `flash_attn_*` or `_scaled_dot_product_attention_efficient` kernels. If you see explicit softmax + matmul kernels for attention instead, FA2 is off — investigate. This is the single most common reason for a low MFU number.
3. **Communication overlap**: look for NCCL kernels (`ncclKernel_AllReduce_...`, `ncclKernel_AllGather_...`, `ncclKernel_ReduceScatter_...`). Are they on a separate CUDA stream from compute, and do they overlap in time? Screenshot or annotate the trace.
4. **Step-time breakdown**: from the profiler's "GPU Kernel" summary, what fraction of step time is compute vs. communication vs. Python/CPU overhead? Report percentages.
5. **CUDA stream gaps**: are there visible gaps in the CUDA stream (bottom lanes of the trace) where nothing is running? These are wasted time. Common causes: host-side Python overhead, sync points, data-loader waits.

Include at least one **annotated screenshot** of the trace in the writeup, with an arrow pointing at whatever your top finding is.

### Part D — Run `nvidia-smi dmon` alongside

In a second terminal while the profiled run is executing, run:

```bash
nvidia-smi dmon -s pucm -d 1 > results/exercise-03/dmon.log
```

for at least 60 seconds of steady-state training. Interpret in `results/exercise-03/profiler_findings.md`:

- **GPU utilization steady at 95–100%** — kernels are always running; the bottleneck is finer-grained than "no kernels."
- **Utilization oscillating** — some phase of the step is not saturating (communication, data loading, or optimizer step). Correlate with the profiler trace.
- **Utilization consistently 60–80% with clock pegged at max** — memory-bandwidth-bound. Kernel fusion, FA2, or `torch.compile` are the levers.
- **Memory usage at 90–100% of card** — well-tuned; further growth risks OOM. Below 70% means you're leaving headroom for larger micro-batch or sequence.

### Part E — Diagnose the top bottleneck

From Parts B–D, pick **exactly one** top bottleneck category:

- **Compute-bound**: MFU ≥ 55% (full FT) or ≥ 40% (LoRA), GPU util pegged, kernel top-5 is dominated by gemms. Not much room to improve without a bigger micro-batch or a different framework (FP8 on Hopper, `torch.compile`).
- **Communication-bound**: MFU 30–55%, GPU util oscillates around backward, NCCL kernels visible on critical path. Levers: bigger reduce/prefetch buckets, forward prefetch on, `HYBRID_SHARD` on multi-node, larger micro-batch to amortize fixed all-gather cost.
- **Memory-bandwidth-bound**: MFU 30–50%, GPU util steady at 60–80%, top kernels are small element-wise (unfused LayerNorm, dropout, indexing), and the trace shows many short kernels. Levers: `torch.compile`, kernel fusion, verify FA is active.
- **Data-loader-bound**: MFU < 30%, visible CPU-side gaps between steps in the profiler trace, `nvidia-smi` shows GPU util dropping to 0% between steps. Levers: more `num_workers`, `pin_memory=True`, pre-tokenize on disk, move data augmentation off the critical path.
- **Bubble/optimizer-step-bound**: rarer; visible when the optimizer step is disproportionately long (usually with CPU offload). The `wall_clock_breakdown` from exercise 02 makes this obvious.

Write **one sentence** naming the bottleneck and the specific evidence. Then write **one specific next change** to try (not a list; the top one). Both go in the report.

### Part F — Write the one-page profiling report

`results/exercise-03/profiling_report.md`, exactly one page, structured as:

1. **Configuration line** — model, `N` parameter count, framework (FSDP/DeepSpeed) + config file name, sharding stage, mixed precision, micro-batch, sequence length, gradient accumulation, world size, GPU SKU and count.
2. **Headline throughput** — steady-state tok/s total and per-GPU (rolling median, N-step window), MFU %, HFU % if gradient checkpointing is on.
3. **Step-time breakdown** — a small table: forward, backward, optimizer step, communication, data loading, other. From the PyTorch profiler summary or DeepSpeed's `wall_clock_breakdown`.
4. **Kernel top-5** — kernel name and total CUDA time for the top five kernels.
5. **Communication overlap** — one sentence: "NCCL kernels overlap cleanly on a separate stream" or "reduce-scatter is on the critical path, 22% of step time not overlapped."
6. **Memory usage** — peak per-rank memory, headroom to card size, whether you could grow batch or sequence.
7. **Bottleneck diagnosis** — the one-sentence answer from Part E, with the specific evidence.
8. **Recommended next change** — one specific tuning knob to try. Not a list; the top one.

Everything else (full kernel table, full trace, per-op timings, raw `dmon` log) goes in appendices or the `results/exercise-03/` directory. The report itself should fit in about one printed page.

### Part G — Save

Directory layout:

```
outputs/exercise-03/
├── train.py                                 # instrumented training script
├── requirements.txt
├── results/
│   ├── baseline_throughput.md               # from Part A
│   ├── mfu_calculation.md                   # from Part B
│   ├── profiler_logs/                       # trace files from Part C
│   ├── profiler_findings.md                 # from Part C + D
│   ├── dmon.log                             # from Part D
│   ├── profiling_report.md                  # THE deliverable, from Part F
│   └── screenshots/                         # annotated trace screenshots
└── logs/
```

## Starter guidance

- **Profile a stable run, not the first launch.** Profiler overhead compounds with first-step compilation / autotuning. Warmup the run, then wrap `prof.step()` around 3–5 steady-state steps.
- **Only rank 0 writes the trace.** Multi-rank traces are large, unwieldy, and rarely more useful than the rank-0 view. If you truly need cross-rank timing, run one rank-0-only trace and use `nvidia-smi dmon` on the other ranks for validation.
- **Use `record_shapes=True` and `profile_memory=True`.** The extra columns pay for themselves — you get per-op input shapes and per-op memory attribution.
- **`with_stack=False` unless you need it.** Stack traces balloon file size and slow the profiler substantially.
- **The `6 · N` PaLM approximation is not exact.** Some transformer variants have MLP intermediate sizes different from `4h`, MoE has expert-sparse compute, DPO has a second reference forward. If your DeepSpeed FLOPS profiler number (exercise 02 Part C) disagrees with `6 · N` by >20%, trust the profiler-measured FLOPs.
- **HFU vs. MFU matters for the honest comparison.** With gradient checkpointing on, the hardware really does more FLOPs per token; HFU is higher than MFU. Pick one convention and label it.
- **Screenshot the top-5 kernels table.** It's the single most compact piece of evidence for "is FA2 on and are gemms dominating?"
- **Don't over-instrument.** More than 10 profiled steps balloons the trace file and rarely tells you more than 3 steps do. Use `schedule(wait=5, warmup=2, active=3)`.

## Acceptance criteria

You have completed the exercise when:

- `results/baseline_throughput.md` records tok/s/GPU with the tokens-per-step formula spelled out.
- `results/mfu_calculation.md` shows the MFU (and HFU if applicable) with the peak-FLOPS source cited.
- MFU is inside the "sanity" range for your model size and framework, or your report explains why not.
- A PyTorch profiler trace exists in `results/profiler_logs/` and opens in TensorBoard / chrome tracing.
- `results/profiler_findings.md` answers all five questions from Part C and interprets the `dmon` log.
- `results/profiling_report.md` is roughly one printed page and covers all eight sections from Part F.
- The bottleneck diagnosis names one specific bottleneck category, one specific piece of evidence, and one specific next change to try.
- At least one annotated screenshot of the profiler trace lives in `results/screenshots/`.

## Stretch goals

- **Actually apply the recommended change** and re-measure. Did tok/s/GPU move as predicted? If not, what does the follow-up trace show? Write this up as a "second pass" section in the report.
- **Compare MFU across sharding strategies.** Rerun the same job under `SHARD_GRAD_OP` (ZeRO-2 equivalent) instead of `FULL_SHARD` and recompute MFU. The delta answers "how much did FULL_SHARD's all-gather cost me?"
- **Turn on `torch.compile`.** Add `model = torch.compile(model)` after `accelerator.prepare(...)`. Measure the tok/s and MFU delta. Expect 10–30% throughput; the first step or two will be slow (compilation). If it breaks, fall back to eager — document what broke.
- **Nsight Systems trace.** If Nsight is available on the node, run `nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas --sample=cpu --output=my_run accelerate launch ...`. Open the `.nsys-rep` in the Nsight GUI and cross-check the top-5 kernels against the PyTorch profiler. Note any discrepancies.
- **Multi-node throughput point.** If a second node is available, run the same config with `HYBRID_SHARD` (FSDP) or `zero_stage: 3` (DeepSpeed) across two nodes. Report the strong-scaling factor: `tok/s at 2 nodes / tok/s at 1 node`. Anything below ~1.6× on a 2× scale-out means inter-node communication is dominating; explain what's happening from the trace.
- **`nccl-tests` baseline.** Run `all_gather_perf` and `all_reduce_perf` from `github.com/NVIDIA/nccl-tests` on your GPU set at message sizes 1 MB, 100 MB, 1 GB. Compare the achieved `busbw` against the vendor spec for your interconnect. Note whether the measured bandwidth explains any communication-bound diagnosis from Part E.
- **Data-loader stress test.** Turn `num_workers` to 0 in your `DataLoader` and re-profile. This turns any data-loader bottleneck loud and obvious — good calibration for the "how much of my current wait is data loading?" question.
