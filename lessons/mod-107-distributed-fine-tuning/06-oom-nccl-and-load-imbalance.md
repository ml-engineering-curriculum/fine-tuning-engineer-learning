# Debugging OOM, NCCL, and Load-Imbalance Failures

**Estimated read:** 45 minutes.
**You should be able to, by the end:** apply a structured playbook to any OOM failure that classifies the root cause (activation, gradient, optimizer, fragmentation) and points to the specific mitigation; read NCCL logs well enough to distinguish "peer unreachable" from "collective timeout" from "silent hang," and know the environment variables that surface each; recognize load-imbalance patterns under variable-length sequences and pick the right fix; and stop escalating once the failure is a platform problem rather than a training-loop problem.

## The three failure families

Distributed fine-tuning failures fall into three groups with distinct diagnostic playbooks:

1. **Out-of-memory** — the run crashes at (or shortly after) allocation.
2. **NCCL failures** — the run crashes or hangs during a collective communication. Includes "silent hang" (nothing crashes, no progress).
3. **Load imbalance** — the run completes but some ranks are chronically slow, dragging step time.

Each has its own tell-tales and its own fix menu. Getting them confused wastes hours.

## OOM: structured triage

An OOM in a distributed run has more possible causes than in a single-GPU run. The step-by-step playbook:

### Step 1 — read the error message carefully

The PyTorch OOM message tells you three useful things:

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.53 GiB.
GPU 0 has a total capacity of 79.15 GiB of which 892.00 MiB is free.
Process 12345 has 78.28 GiB memory in use. Of the allocated memory,
72.34 GiB is allocated by PyTorch, and 5.94 GiB is reserved by PyTorch
but unallocated.
```

- **The size of the failing allocation** — `12.53 GiB` here — hints at what phase blew up. Tens of GB → optimizer step or activation with `s²` (no FA); single-digit GB → typical activation; sub-GB → buffer or fragmentation.
- **Free memory at time of failure** — how tight things are. `892 MiB` free with a `12.53 GiB` request means you need to free at least 12 GB; fewer if fragmentation is inflating the required contiguous block.
- **Reserved-but-unallocated** — `5.94 GiB` here is fragmentation. If this number is large relative to your total, `expandable_segments:True` may reclaim most of it.

### Step 2 — check *when* the OOM happens

- **Step 0 (first forward)** — model doesn't fit at all. Under-sharded (wrong FSDP / ZeRO stage), missing gradient checkpointing, sequence too long, or `zero3_init_flag: false` when it should be true.
- **Step 0 (during optimizer step)** — model fits but optimizer state doesn't. Bump ZeRO stage from 1 → 2 → 3, or add `offload_optimizer` (DeepSpeed).
- **Step N > 0, growing over time** — memory leak. Activation retention bug (usually a stray tensor held by Python), optimizer accumulator drift (rare), or fragmentation compounding. Enable memory snapshot (`torch.cuda.memory._record_memory_history`) to bisect.
- **Step N > 0, one-off spike** — data-driven. A long sequence in the batch, an unusual attention pattern (all tokens attend to all), or a validation pass with different batch shape. Log per-step memory alongside batch metadata to find it.

The `torch.cuda.max_memory_allocated()` reset-and-report pattern is the workhorse here:

```python
torch.cuda.reset_peak_memory_stats()
# ... run one step ...
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"step {step} peak {peak_gb:.2f} GB")
```

### Step 3 — bisect the memory categories

If step 0 OOMs, work through the memory components (chapter 4) in order:

1. **Base weights.** Under FSDP FULL_SHARD / ZeRO-3, per-rank weight memory should be `~2N/P`. Check with `torch.cuda.memory_allocated()` right after model load. If it's `2N` instead of `2N/P`, sharding isn't happening — check `distributed_type`, `zero_stage`, `fsdp_sharding_strategy`.
2. **Activations.** Compute the expected value from the chapter-4 formula. If it's much higher, either you don't have gradient checkpointing on, FlashAttention isn't active (check for `flash_attn_*` kernels in the profiler), or sequence length is unexpectedly large.
3. **Optimizer state.** After the first `optimizer.step()`, memory should grow by `~12N/P` (FP32 AdamW) or `~4N/P` (AdamW-8bit). If it grows by `12N`, optimizer state isn't sharded.
4. **Fragmentation.** Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Rerun. If the difference is dramatic, fragmentation was the killer.
5. **Framework buffers.** NCCL communication buffers grow with `NCCL_BUFFSIZE` (default 8 MB) — usually not the cause of OOM but can add up on many small ops.

If step N > 0 OOMs, drop into the memory snapshot flow:

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
# ... run ~10 steps ...
torch.cuda.memory._dump_snapshot("mem_snapshot.pickle")
```

Open in the PyTorch memory viewer (`pytorch.org/memory_viz`) or process locally. Time-slicing shows which allocations grew step-over-step — leaks and growth are visually obvious.

### Step 4 — apply the tiered mitigation

In order of throughput cost (cheapest first):

1. Reduce `per_device_train_batch_size` to 1 (if not already).
2. Reduce `max_seq_length` (if application allows).
3. Turn on activation checkpointing (should already be on for 7B+).
4. Turn on FlashAttention-2 (should already be on if hardware supports it).
5. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
6. Switch from AdamW to AdamW-8bit (`optim="paged_adamw_8bit"`).
7. Add optimizer offload (DeepSpeed `offload_optimizer.device: cpu`).
8. Move FSDP `backward_prefetch` to `BACKWARD_POST` and turn off `forward_prefetch`.
9. Add parameter offload (DeepSpeed `offload_param.device: cpu`, or FSDP `offload_params: true`).
10. Reconsider the cluster shape — you may need more or larger GPUs (chapter 7).

Report the tried mitigations in your run's log even when they don't help. When a peer looks at the run six weeks later they need the history.

## NCCL: the diagnostic environment variables

NCCL logs are the single best debugging tool when things go wrong in the fabric. Set at least these three before any multi-GPU / multi-node debug session:

```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1  # or TORCH_NCCL_ASYNC_ERROR_HANDLING in newer torch
```

- **`NCCL_DEBUG=INFO`** — verbose NCCL init and per-collective logging. First useful lines: which transport is selected (`NET/IB`, `NET/Socket`, `P2P`), which interfaces are being used, and any warnings about missing peers.
- **`TORCH_DISTRIBUTED_DEBUG=DETAIL`** — PyTorch-side logging of distributed operations. Prints when each rank enters each collective; invaluable for finding which rank is hung.
- **`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`** — makes NCCL errors on any rank cause a fast, non-silent failure on every rank rather than hanging. Modern PyTorch defaults this to on for the ProcessGroupNCCL; explicit is safer.

Additional variables for specific investigations:

- **`NCCL_DEBUG_SUBSYS=INIT,NET,ENV`** — narrow the debug output to specific subsystems. Use when `NCCL_DEBUG=INFO` is too noisy.
- **`NCCL_IB_HCA=<name>`** — force a specific InfiniBand HCA. Set when auto-detection picks the wrong one on multi-NIC nodes.
- **`NCCL_SOCKET_IFNAME=<iface>`** — force a specific network interface for socket transport. Necessary in bridged / Docker network setups where NCCL picks the wrong iface.
- **`NCCL_P2P_DISABLE=1`** — disable P2P (NVLink / PCIe P2P) transport. Diagnostic — if the run works with P2P disabled and fails with it enabled, you have a P2P setup problem.
- **`NCCL_IB_DISABLE=1`** — disable InfiniBand. Same idea for IB setup problems.
- **`NCCL_TIMEOUT=<seconds>`** or **`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=<seconds>`** — how long a collective can hang before NCCL / PyTorch calls it a timeout and aborts. Default is quite long (~30 min). Shorten during debugging to fail fast; lengthen if very large collectives legitimately take long.

## NCCL failure modes and their signatures

### "Peer unreachable" at init

```
NCCL WARN Connect to 10.0.0.3:24632 failed
```

- **Cause**: rank on one node can't reach rank on another. Network config problem, wrong `MASTER_ADDR`, firewall.
- **Fix**: verify inter-node connectivity from the shell (`nc -zv <peer_ip> <port>`); check `MASTER_ADDR` matches a reachable IP; check firewall / security-group rules. Escalate to platform team if the cluster is supposed to have this connectivity.

### "Transport type X not supported"

```
NCCL WARN NET/IB : No device found.
```

- **Cause**: NCCL tried to use InfiniBand but no IB device is present. Might fall back to slower socket transport (a warning, not an error) or fail outright.
- **Fix**: verify IB devices with `ibv_devices` on each node. If IB is intended but missing, escalate. If IB is not intended, set `NCCL_IB_DISABLE=1` to silence.

### Collective timeout

```
[rank 3] Watchdog caught collective operation timeout: WorkNCCL(...)
ran for 1801000 milliseconds before timing out.
```

- **Cause**: one or more ranks did not join the collective. Most commonly: another rank crashed (its process is gone but the survivors are still waiting), one rank is stuck in Python (data loading blocked, deadlock in your training loop), or a very slow collective on a saturated link.
- **Fix**: check which ranks logged the timeout and which didn't (the not-logging rank is the culprit). Look at that rank's log for crash. If no crash, attach `py-spy dump --pid <pid>` on the hung rank to see where Python is stuck. `torch.distributed.barrier(timeout=...)` calls report clearly which ranks are missing.

### Silent hang

The run appears to be alive (Python process running, some memory allocated) but no forward progress. No error, no log spam. The worst.

- **Cause**: either NCCL is hung waiting for a peer that will never respond (see "collective timeout" above but before the timeout fires), or your training loop is stuck in Python (dataloader deadlock, tokenizer that never returns, a `wait_for_everyone()` call where one rank took a different code path).
- **Fix**: attach `py-spy dump --pid <pid>` on every rank. Compare stacks — if all ranks are in the same NCCL wait, you have a genuine collective hang (some rank crashed silently, see above). If ranks are in different places, you have a code-path divergence (one rank hit an early `return` in a conditional; another rank is still in the loop).
- Setting `NCCL_TIMEOUT` / `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` to a smaller value (say 300 seconds) upgrades silent hangs to explicit timeouts.

### P2P access failure

```
NCCL WARN [Rank 0] Failed to CUDA IPC handle: peer access is not enabled
```

- **Cause**: GPUs in the same node aren't peer-accessible. Sometimes a driver / config issue, sometimes hardware (mixed-generation GPUs), sometimes container isolation.
- **Fix**: `nvidia-smi topo -m` shows the peer-access matrix. If neighboring GPUs show `NODE` or `SYS` instead of `NV#` / `PIX`, you're not going over NVLink and P2P is degraded. In a container, ensure the container has `--ipc=host` (Docker) or the equivalent. Escalate to platform if the topology is unexpected.

## `nccl-tests`: the microbenchmark

When you suspect the cluster fabric is slow or misconfigured, `nccl-tests` (github.com/NVIDIA/nccl-tests) is the canonical microbenchmark. Common invocations:

```bash
# All-reduce on 8 GPUs, sizes from 8 B to 8 GB
mpirun -np 8 --allow-run-as-root ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```

The output includes achieved bandwidth (`busbw`) at each message size. Interpretation:

- **Peak `busbw` well below expected** — see the bandwidth table in chapter 5. E.g., NVSwitch H100 should peak at ~700+ GB/s on large messages; if you see 100 GB/s, something is misconfigured.
- **`busbw` degrades sharply at some message size** — a topology or transport bottleneck; possibly a P2P vs. socket switch, possibly hitting IB message-size limits.

Running `nccl-tests` on your cluster before starting a big fine-tuning run catches "the fabric is slower than the platform team says" problems cheaply. This is very close to platform-team work; if the numbers are wrong, escalate rather than debug the fabric yourself.

## Load imbalance: the "some ranks are slower" pattern

A run that completes but has poor throughput often has load imbalance: some ranks finish a step's compute faster and wait for others at the sync point. Diagnosis:

- **PyTorch profiler shows some ranks with long "wait" phases at reduce-scatter / all-reduce.** These ranks are the fast ones; the slow ones are elsewhere in compute.
- **`nvidia-smi` shows uneven per-GPU utilization.**
- **Step time is dominated by the slowest rank.**

Common causes and fixes:

### Variable sequence length in the batch

If batches have raggedness (some sequences 512 tokens, others 8192), the rank with the long sequence takes longer per step. Fixes:

- **Length-bucketed sampling** — batches contain sequences of similar length. Standard for SFT; every fine-tuning framework supports it (see `datasets.sort` or a `LengthGroupedSampler` in HF).
- **Sequence packing** — concatenate short sequences into fixed-length packed examples. Every batch is the same size. Standard for large-scale SFT; TRL's `SFTTrainer` supports it with `packing=True`. Requires attention masking to prevent cross-sequence attention (chapter 2 of mod-103).
- **Pad-to-multiple** — pad every sequence to a fixed length. Simple but wastes compute on padding. Fine for small runs.

### Uneven expert routing (MoE)

Different ranks receive different numbers of tokens if expert routing is imbalanced. Fixes are MoE-specific: loss balancing (auxiliary load-balancing loss), expert capacity limits, or shuffled routing. DeepSpeed-MoE and Megatron-Core have knobs for this. Usually a platform-team concern.

### Straggler ranks (hardware asymmetry)

One GPU is slower than the others (thermal throttling, ECC errors, aging silicon). `nvidia-smi -q -d TEMPERATURE,PERFORMANCE` shows per-GPU clocks and thermal state. If one GPU is clocking down, escalate — the platform team either replaces it or the node.

### Uneven data sharding

If the `DistributedSampler` gives some ranks more samples per epoch than others, they run more steps and the run stalls at the barrier. Fix: pass `drop_last=True` to the sampler / dataloader, or use `Accelerator.split_between_processes(...)` correctly.

## The "hangs at step N" family

A specific pattern worth naming: the run completes some steps successfully, then hangs at a specific step (say, step 200). Almost always one of:

- **Evaluation callback** — a `TrainingArguments(eval_steps=200)` triggers eval on step 200; the eval loop has different data-loading behavior (larger batches, longer sequences) and one rank OOMs silently. Then the rest of the ranks wait forever at the next collective.
- **Checkpoint save on a shared FS** — `save_steps=200` triggers a checkpoint write to an FS one rank can't reach. Rank 0 waits forever on rank 3 to acknowledge.
- **Data anomaly at row N** — a specific row has a pathological length, an encoding issue, or triggers a bug in the collator. Row N is served on step `N / batch_size / world_size`. Bisect by logging the batch indices per step and checking what changed.
- **Warmup transition** — some schedulers change behavior at the end of warmup (step 100 with warmup 100). Diagnostic: does the hang correlate with a known transition?

Set `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300` (5 min) during debugging so silent hangs turn into loud errors with rank-attributed stack traces.

## `py-spy dump`: the missing debugger

When Python is stuck and you don't know where, `py-spy dump --pid <PID>` prints the current Python stack of a running process without stopping it. Install with `pip install py-spy`. Absolutely essential for debugging hangs across ranks:

```bash
# For each rank / PID
for pid in $(pgrep -f "train.py"); do
  echo "=== PID $pid ==="
  py-spy dump --pid $pid
done
```

If all ranks are in `_wait_for_completion` inside `torch/distributed/distributed_c10d.py`, you have a genuine NCCL hang (one rank silently died, or a rank is missing from the collective). If ranks are in different places, one is in an unexpected code path — logging around control-flow divergences finds it.

`py-spy record --pid <PID> --output trace.svg --duration 30` produces a flame graph for a 30-second window — for progress-but-slow debugging rather than hang debugging.

## Structured logging discipline

Two habits that pay for themselves an order of magnitude the first time you need them:

- **Log per-step**: step number, wall-clock, loss, tokens/step, tokens/sec, GPU memory, LR. One line per step, structured (JSON or a stable CSV). Ingesting this later into pandas turns hours of "when did it go wrong?" into two minutes.
- **Log with rank prefix**: `[rank 3] step 200 loss 1.42 mem_gb 43.2`. Without this, the multi-rank log stream is unreadable. Set up in the logger config once; every logger.info call gets the prefix automatically.

Weights & Biases, TensorBoard, and MLflow all handle multi-rank runs reasonably (usually via "only rank 0 logs"). That works for loss / accuracy but hides per-rank memory / throughput. Log a compact per-rank stream to stderr in addition to whatever central tracker you use.

## When to stop debugging and escalate

Some failures are not the fine-tuning engineer's to fix. Escalate to the `training-pipeline-engineer` / platform team when you see:

- **`nccl-tests` bandwidth is materially below the vendor spec.** The fabric is misconfigured or degraded. Not your layer.
- **Inter-node connectivity works from the shell but NCCL init hangs.** Either NCCL environment is wrong for this cluster (a platform-provided config file should exist) or there's a routing / firewall / IB fabric issue. Escalate.
- **The cluster keeps losing GPUs** (`Xid errors` in `dmesg`, GPUs disappearing from `nvidia-smi` mid-run). Hardware failure. Escalate.
- **Spot-instance preemption is killing runs.** Checkpoint frequency and preemption-tolerant orchestration is platform territory; if the platform doesn't provide it, escalate as a platform gap.
- **Driver / CUDA / NCCL version mismatch across nodes.** Escalate — this should be homogeneous.

Chapter 7 has the interface contract; use it to structure the escalation.

## Exercise pointer

`exercises/exercise-04-oom-and-nccl-debugging-drills.md` is a set of injected failures — OOM at step 0, OOM at step 200, NCCL timeout, hang mid-run, load imbalance under packed sequences. For each you produce a diagnosis, a minimal reproducer, the log evidence, and a fix. It's designed to build the diagnostic reflex before you need it under time pressure.

## Summary

- **Three failure families** — OOM, NCCL, load imbalance — each with its own signatures and playbook. Don't confuse them.
- **OOM**: read the error size and phase, check `torch.cuda.max_memory_allocated()`, bisect through weights / activations / optimizer / fragmentation, apply the tiered mitigation menu (batch → seq → checkpoint → offload → smaller shard step → bigger cluster).
- **NCCL environment variables** every distributed run needs: `NCCL_DEBUG=INFO`, `TORCH_DISTRIBUTED_DEBUG=DETAIL`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`. Set `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` short during debugging.
- **NCCL failure signatures**: peer unreachable (network / MASTER_ADDR); no transport (missing IB); collective timeout (a rank crashed or hung); silent hang (either the above pre-timeout or a code-path divergence); P2P failure (topology).
- **`nccl-tests`** is the fabric microbenchmark. If numbers are wrong, escalate — fabric tuning is platform work.
- **Load imbalance** comes from variable sequence length (fix: length-bucketed sampling or packing), uneven MoE routing, straggler GPUs, or uneven data sharding.
- **`py-spy dump`** is the missing debugger for hangs. Compare stacks across ranks; if they all point to NCCL wait, one rank is missing; if they diverge, code path is off.
- **Structured per-rank logging** turns hours of forensics into minutes. Log every step, prefix with rank.
- **Escalate** to the platform team when the failure is in the fabric, the hardware, the drivers, or the orchestration — not the training loop.
