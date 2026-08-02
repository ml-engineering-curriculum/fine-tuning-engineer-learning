# Throughput, MFU, and Profiling

**Estimated read:** 40 minutes.
**You should be able to, by the end:** compute tokens per second per GPU from a running job, apply the PaLM-style MFU formula to that number and interpret the result, use the roofline model to argue whether a given run is compute-bound or communication-bound, drive the PyTorch Profiler + DeepSpeed's FLOPS profiler + `nvidia-smi dmon` well enough to identify a bottleneck, and write a one-page profiling report a peer can act on.

## The two numbers you always report

Every fine-tuning run has two headline throughput numbers that fit on the same line:

1. **Tokens per second per GPU (tok/s/GPU)** — the raw throughput. Product of micro-batch size, sequence length, gradient-accumulation, and the number of steps per unit time, divided by the number of GPUs.
2. **Model FLOPS Utilization (MFU)** — the fraction of theoretical peak FLOPS your job is actually using, expressed as a percentage.

Tok/s/GPU tells you *how fast the run is going.* MFU tells you *how efficiently the hardware is being used.* You want both because they answer different questions: tok/s/GPU is the number you compare against your budget; MFU is the number you compare against "is this run leaving performance on the table."

## Computing tokens per second per GPU

The mechanics are boring but easy to get wrong. Once per training loop:

```python
import time

step_start = time.time()
outputs = model(**batch)
loss = outputs.loss
accelerator.backward(loss)
optimizer.step()
scheduler.step()
optimizer.zero_grad()
step_time = time.time() - step_start

# tokens = micro_batch * sequence_length * gradient_accumulation_steps * world_size
tokens_per_step = per_device_train_batch_size * seq_length * grad_accum_steps * world_size
tokens_per_sec = tokens_per_step / step_time
tokens_per_sec_per_gpu = tokens_per_sec / world_size
```

Three gotchas:

- **Include gradient accumulation.** The "step" in this reporting is the *optimizer step*, which is `grad_accum_steps` forward+backward passes. If you report per-forward tokens instead, your number will be smaller by a factor of `grad_accum_steps` and comparisons will be nonsense.
- **Include padding tokens or not — decide and document.** For packed sequences (mod-103 chapter 2), all tokens are real. For left-padded batches, some tokens are padding; reporting the whole tensor overstates throughput. Both conventions exist in the field. Say which one you're using.
- **Warmup and steady-state.** Compilation, autotuning, and first-step allocations skew the first few steps. Discard the first 5–10 steps from your throughput calculation; report on a rolling window (last 50–100 steps).

For a Llama-3.1-8B on 8× A100 80 GB with FSDP + BF16 + FA2 at `b=4, s=2048, grad_accum=8`:

- Tokens per step: `4 * 2048 * 8 * 8 = 524,288` per optimizer step.
- Typical step time in the neighborhood of 12–20 seconds depending on config.
- Tok/s ≈ 26k–44k total; per GPU ≈ 3.3k–5.5k.

Numbers vary widely with sequence length, model size, and cluster; use these as order-of-magnitude sanity checks, not targets.

## The MFU formula

Chowdhery et al. (2022, "PaLM: Scaling Language Modeling with Pathways") introduced Model FLOPS Utilization as the field's efficiency metric. The formula is:

    MFU = (6 · N · tokens_per_second) / (peak_FLOPS_per_second · num_GPUs)

Where:

- **`N`** is the model's parameter count (in units of individual parameters, not billions).
- **`6 · N`** is the approximate FLOPs *per token* for a standard dense transformer forward + backward pass (2 · N for forward matmuls, 4 · N for backward — hence the "6·N approximation"). Kaplan et al. (2020) and the PaLM appendix are the origin.
- **`tokens_per_second`** is total across all GPUs (not per-GPU).
- **`peak_FLOPS_per_second`** is the manufacturer's advertised peak for your dtype and hardware.

Peak FLOPS for common accelerators (BF16 / FP16 tensor-core throughput; not TF32, not FP32):

| GPU | Peak BF16 / FP16 (TFLOPS, dense) | Peak FP8 (H100+) |
|---|---|---|
| A100 40 GB / 80 GB | 312 | — |
| H100 SXM | 989 | 1979 |
| H100 PCIe | ~756 | ~1513 |
| H200 SXM | 989 | 1979 |
| B100 / B200 | check current NVIDIA spec | check current NVIDIA spec |
| RTX 3090 | 71 | — |
| RTX 4090 | ~165 (with sparsity)  | — |

Consult NVIDIA's current datasheet for your specific SKU — the numbers move with each generation and are sometimes reported with sparsity assumed. **Use the dense number** for MFU calculations unless your model actually uses structured sparsity.

Worked example: Llama-3.1-8B (`N = 8e9`), 8× A100, 40k tok/s total:

    MFU = (6 · 8e9 · 40,000) / (312e12 · 8)
        = 1.92e15 / 2.5e15
        = 0.77 → 77%

That would be a very well-tuned run. Typical fine-tuning MFU on modern hardware:

- **50–60%** for a well-configured full-parameter run on standard SFT.
- **35–45%** for LoRA / QLoRA runs (less compute per token; more sensitive to communication).
- **25–35%** for runs with heavy CPU offload or with FSDP FULL_SHARD across a slow multi-node link.
- **<20%** is a red flag — profiling is warranted.

## MFU vs HFU: which do you report?

Two subtle definitions in the literature:

- **MFU (Model FLOPS Utilization)** — uses `6 · N` as the per-token FLOPs, i.e., the *theoretical* FLOPs for the model's forward + backward.
- **HFU (Hardware FLOPS Utilization)** — uses the actual FLOPs the hardware performed, including the extra FLOPs from activation checkpointing (which does one extra forward pass, adding ~2 · N per token → total 8 · N per token).

Concretely:

- With activation checkpointing on, HFU > MFU (the hardware really is doing more FLOPs).
- The "correct" number to compare against your peers depends on the convention they used. PaLM's paper is MFU (`6 · N`); many follow-ups mix.
- **Report both if space allows**, or pick MFU and label it. HFU alone can mislead: a 60% HFU with gradient checkpointing represents the same "algorithmic throughput" as a 45% MFU without.

## The roofline model in one page

Roofline (Williams et al. 2009) is a way of thinking about whether your kernel is limited by compute or by memory bandwidth. The picture is a plot of achievable FLOPS/s against operational intensity (FLOPs per byte of memory access):

- **Below a diagonal** (memory-bandwidth-limited): the kernel is spending time waiting on memory; more FLOPS/s is only possible with more arithmetic per memory read.
- **Above / on a horizontal ceiling** (compute-bound): the kernel is running as fast as the tensor cores allow; no amount of extra bandwidth will help.

For LLM training kernels:

- **Matmuls (dense layers, attention Q·Kᵀ and score·V)** — large-arithmetic-intensity, compute-bound. This is where you want the runtime to be spent.
- **Element-wise ops (LayerNorm, GELU, activation stats, dropout)** — low-arithmetic-intensity, memory-bound. Individually cheap but can accumulate.
- **Attention softmax and log-sum-exp** — memory-bound if computed naively; FlashAttention re-fuses them into a compute-bound block.
- **All-reduce / all-gather** — pure communication, no local compute. Cost is set by NCCL bandwidth and topology.

Rules of thumb for a distributed fine-tuning run:

- **Compute-bound (MFU > ~55%)**: the run is well-tuned; further speedups require better kernels (Flash, TE FP8), a bigger micro-batch, or more efficient sharding.
- **Communication-bound (MFU 30–50%, `nvidia-smi` shows low GPU utilization during backward)**: shrink communication (bigger buckets, `HYBRID_SHARD` instead of `FULL_SHARD` on multi-node, larger micro-batch to amortize the fixed all-gather cost, forward-prefetch on).
- **Memory-bound in small ops (MFU <30%, GPU utilization high but SM occupancy low, many small kernel launches in the trace)**: consider `torch.compile`, kernel fusion, or checking that FlashAttention is actually being used.
- **Pipeline / bubble-bound (rare in fine-tuning, applies with PP)**: bigger pipeline schedule, more micro-batches per accumulation.

Chapter 6 revisits some of these when the failure is total (OOM, hang) rather than just slow.

## The tools

Four tools cover 95% of profiling needs.

### `nvidia-smi dmon` and `nvitop`

The 30-second first-pass tool. Run in a second terminal while a training step is running:

```
nvidia-smi dmon -s pucm -d 1
```

- **`pu`** — power usage and GPU utilization (percentage of time SMs were active).
- **`c`** — clock speed (SM and memory).
- **`m`** — memory usage (used / free / total).

Interpretation:

- **GPU utilization consistently at 95–100%** — kernels are running; whatever bottleneck exists is at a finer grain than "no kernels running." Move to profiler.
- **GPU utilization oscillating between 30% and 100%** — some phase of the step is not saturating; this is communication or data loading. Look at profiler traces.
- **GPU utilization consistently at 60–80% with clock at max** — memory-bandwidth-bound. Kernel fusion / FA2 / `torch.compile` are the levers.
- **Memory usage at 95%+ of card** — expected on a well-tuned run; anything less is unused headroom for larger micro-batch or sequence.

`nvitop` (a third-party TUI) is the same information plus per-process breakdown. Use whichever your team prefers.

### PyTorch Profiler

The workhorse. Instrument a few steps of your training loop:

```python
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

prof_schedule = schedule(wait=5, warmup=2, active=3, repeat=1)
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    on_trace_ready=tensorboard_trace_handler("./profiler_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
) as prof:
    for step, batch in enumerate(train_loader):
        # normal training step
        prof.step()
        if step >= 15:
            break
```

Then view the trace in the Chrome trace viewer (`chrome://tracing`) or TensorBoard's profiler plugin. What you're looking for:

- **The "step" timeline** — one long block per training step. Zoom in on one representative step.
- **CUDA stream utilization** — gaps in the CUDA stream (bottom lanes of the trace) are wasted time. Common causes: sync points, data loading, host-side Python overhead.
- **Kernel breakdown** — sort by total CUDA time. The top kernels should be matmuls (`ampere_bf16_...gemm`, `sm90_..._gemm`, or `flash_fwd_...` / `flash_bwd_...`); if the top is a small element-wise kernel, you have a fusion problem.
- **Communication kernels** — look for `ncclKernel_AllReduce_...`, `ncclKernel_AllGather_...`. Ideally these overlap with compute kernels on other streams. If they're on the same stream as compute, you're not overlapping and communication is on the critical path.

The trace answers three specific questions:

1. **Is FA2 actually being used?** — if you see `_scaled_dot_product_attention_efficient` or `flash_attn_*` kernels, yes. If you see explicit softmax + matmul kernels for attention, no.
2. **Is communication overlapping?** — if the NCCL kernels and compute kernels are on different streams and overlap in time, yes.
3. **Where is the time going?** — the "GPU kernels" summary table tells you compute vs. communication vs. Python-overhead in aggregate.

### DeepSpeed FLOPS Profiler

DeepSpeed-only, but excellent when you're using DeepSpeed:

```json
"flops_profiler": {
  "enabled": true,
  "profile_step": 10,
  "module_depth": -1,
  "top_modules": 3,
  "detailed": true,
  "output_file": "profile.txt"
}
```

Runs a profile pass on the specified step and dumps a per-module FLOPs breakdown, MACs, parameter count, and inferred throughput. Answers "which layer / module type is dominant?" without requiring a full trace-viewer session. Complements PyTorch profiler; use for a first-pass on a DeepSpeed run.

### Nsight Systems (`nsys`)

NVIDIA's system-wide profiler; the deepest tool. Run:

```bash
nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas \
    --sample=cpu \
    --output=my_run \
    accelerate launch --config_file configs/fsdp_8gpu.yaml train.py --max_steps 30
```

The output `.nsys-rep` opens in Nsight Systems (GUI). Nsight is what you use when the PyTorch profiler is not giving you enough — e.g., you need to see NIC / PCIe traffic, or you need per-warp SM utilization. Overkill for most fine-tuning debugging; standard for platform-team work.

## What a good profiling report looks like

A one-page profiling report for a fine-tuning run should contain:

1. **Configuration line** — model, base weights hash, framework (FSDP / DeepSpeed), sharding stage, mixed precision, micro-batch, seq length, gradient accumulation, world size, hardware.
2. **Headline throughput** — steady-state tok/s total and per-GPU (last 100 steps rolling median), MFU %, HFU % if activation checkpointing is on.
3. **Step-time breakdown** — from the PyTorch profiler or DeepSpeed's `wall_clock_breakdown`, per phase: forward, backward, optimizer step, communication, data loading, other.
4. **Kernel top-5** — the five most-CUDA-time-consuming kernels. Expected: 3–4 gemm kernels and 1 attention kernel. Any surprise entry (dropout, LayerNorm, indexing) is a lead.
5. **Communication overlap** — a note on whether NCCL kernels are on the critical path (bad) or overlapped (good). PyTorch profiler trace makes this visible.
6. **Memory usage** — `torch.cuda.max_memory_allocated()` per rank; free headroom for larger batch / sequence.
7. **The bottleneck diagnosis** — one sentence: "compute-bound at 62% MFU," or "communication-bound; NCCL all-gather is 22% of step time and on critical path," or "data-loader bound; 400 ms per step of Python-side collate."
8. **The recommended next change** — one specific tuning knob to try. Not a list; the top one.

Everything else (kernel counts, full trace, per-op timing) goes in appendices or logs.

## What a bad number usually means

A quick pattern-match table for MFU numbers that came in below expectations:

| MFU | Likely cause |
|---|---|
| <15% | FlashAttention off; `torch.compile` off; data loading is the bottleneck; or heavy CPU offload. Check the profiler trace before doing anything else. |
| 15–25% | Communication-bound on a slow interconnect (multi-node without NVLink / IB); or activation checkpointing turned off is filling every micro-batch with small kernels; or optimizer offload with a slow CPU. |
| 25–40% | Under-batched (`b=1` with lots of accumulation on a small model); short sequences amortizing kernel launch cost poorly; QLoRA (expected). |
| 40–55% | Normal for LoRA / QLoRA; low end of "well-tuned" for full-FT. Try growing micro-batch and enabling forward prefetch. |
| 55–65% | Well-tuned full-FT on Ampere; nothing else to do without changing framework or model. |
| 65–75% | Excellent. Verify the MFU calculation (are you using the right peak-FLOPS number?). |
| >75% | Verify twice. Either you found a very well-tuned config or you're miscounting somewhere. |

The last two rows are common footguns: MFU is easy to over-report if you use the sparsity-inclusive peak-FLOPS number for a run that doesn't use sparsity, or if you conflate MFU with HFU.

## Communication as a first-class concern

For multi-GPU and especially multi-node runs, communication is often where wall-clock hides. The relevant NCCL operations, in fine-tuning:

- **All-reduce** — DDP gradient reduction; ZeRO-1 gradient reduction. `2 · gradient_bytes` transferred per rank per step (roughly).
- **Reduce-scatter** — ZeRO-2 / ZeRO-3 gradient reduction. Same volume as all-reduce but ends with each rank holding only its shard.
- **All-gather** — ZeRO-3 / FSDP FULL_SHARD parameter unshard. `2 · param_bytes` per rank per layer per forward, and again per backward.
- **All-to-all** — MoE routing. Cost depends on token distribution.

Bandwidth numbers (order-of-magnitude, for context):

- **NVLink 4.0 (H100 SXM)**: ~900 GB/s per direction between two GPUs in the same node.
- **NVSwitch 4.0**: full-mesh at ~900 GB/s inside an 8× H100 box.
- **NVLink 3.0 (A100 SXM)**: ~600 GB/s.
- **PCIe Gen4 x16** (A100 PCIe cards, no NVLink): ~64 GB/s.
- **InfiniBand HDR (200 Gb/s)**: ~25 GB/s per link, per direction.
- **InfiniBand NDR (400 Gb/s)**: ~50 GB/s per link.
- **100 Gb Ethernet with RoCE**: ~12 GB/s per link.

The comparison: an intra-node all-gather over NVSwitch is ~40× faster than an inter-node all-gather over 100 Gb Ethernet. This is exactly why `HYBRID_SHARD` exists.

`nccl-tests` (`all_gather_perf`, `all_reduce_perf`, `alltoall_perf`) is the standard microbenchmark for verifying your cluster's effective bandwidth. Chapter 6 covers using them for diagnosis.

## Compiling: `torch.compile`

`torch.compile` (PyTorch 2.0+) can add 10–30% throughput on transformer training by fusing kernels and reducing Python overhead. Under FSDP and DeepSpeed:

- **FSDP2 + `torch.compile`** — better supported than FSDP1 + compile. Recent PyTorch versions (2.4+) improve on this.
- **DeepSpeed + `torch.compile`** — supported on recent DeepSpeed / PyTorch versions; check the version matrix.
- **Compile time cost** — the first step or two are slow (10s to minutes of compilation). Amortize by long runs.
- **Fallback**: if compile breaks (odd shapes, dynamic control flow), fall back to eager. It is optional.

For a first-pass fine-tuning run, leave `torch.compile` off until you have a working baseline. Turn on later for the throughput win.

## Exercise pointer

`exercises/exercise-03-throughput-and-mfu-profiling.md` walks the full profiling pass: instrument the training loop, capture a PyTorch profiler trace, compute MFU by hand, identify the bottleneck, and write the one-page report described above.

## Summary

- Report **two numbers**: tokens per second per GPU (throughput) and MFU (efficiency). Both matter and answer different questions.
- **MFU formula**: `(6 · N · tokens/sec) / (peak_FLOPS · num_GPUs)`. Uses the dense-BF16 peak FLOPS for your hardware. Distinguish from HFU (which counts extra FLOPs from activation checkpointing).
- **Typical fine-tuning MFU**: 50–60% for well-tuned full-FT, 35–45% for LoRA / QLoRA, 25–35% with heavy offload, <20% is a red flag.
- **Roofline** frames the bottleneck: compute-bound (fine, more FLOPS/s only via better kernels / bigger batch), memory-bandwidth-bound (kernel fusion, FA, compile), communication-bound (bigger buckets, HYBRID_SHARD, forward prefetch), or bubble/data-loader-bound.
- **Tools**: `nvidia-smi dmon` for 30-second sanity; PyTorch Profiler for kernel and communication breakdown; DeepSpeed FLOPS Profiler if you're on DeepSpeed; Nsight Systems for deeper investigation.
- **A profiling report** is one page: config, headline throughput + MFU, step-time breakdown, kernel top-5, comm overlap note, memory usage, one-sentence diagnosis, one recommended next change.
- **Communication bandwidth** matters: NVLink / NVSwitch is ~10–40× faster than InfiniBand or Ethernet. `HYBRID_SHARD` and larger reduce buckets are the levers when inter-node comms dominates.
- **`torch.compile`** is a ~10–30% win when it works; leave off until the baseline is stable, then turn on for the extra throughput.
