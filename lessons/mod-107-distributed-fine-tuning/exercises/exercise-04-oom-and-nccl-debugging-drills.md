# exercise-04: OOM and NCCL Debugging Drills

**Estimated effort:** 3 hours
**Companion chapters:** [`04-memory-mixed-precision-and-checkpointing.md`](../04-memory-mixed-precision-and-checkpointing.md), [`06-oom-nccl-and-load-imbalance.md`](../06-oom-nccl-and-load-imbalance.md), [`07-cluster-sizing-and-handoff.md`](../07-cluster-sizing-and-handoff.md)

## Objective

Work through **five injected failure scenarios** on top of the distributed training script from [exercise 01](exercise-01-accelerate-fsdp-launch-on-2-to-8-gpus.md). For each scenario, you will (a) reproduce the failure locally, (b) diagnose the root cause from log evidence, (c) write a minimal reproducer, and (d) apply a fix — then document all four in a per-scenario debugging report.

The point of the exercise is to build the diagnostic reflex before you need it under time pressure on a real run. When a 16-GPU run OOMs at step 400 on Friday afternoon, the difference between "diagnosed and fixed in 30 minutes" and "still bisecting on Monday" is entirely the practice you did on exercises like this one.

## Prerequisites

- Exercise 01 complete, with a working `train.py` that runs on 2 → 8 GPUs under FSDP `FULL_SHARD`. Some scenarios also work with the DeepSpeed setup from exercise 02; use whichever you're more comfortable with.
- A single node with at least 2 GPUs. 4+ GPUs makes some scenarios more realistic; 2 is enough to hit every failure family.
- `py-spy` installed on the training host (`pip install py-spy`). This is the single most useful tool for the hang scenarios and is not optional for the exercise.
- The environment variables from chapter 6 available to set at launch time: `NCCL_DEBUG`, `TORCH_DISTRIBUTED_DEBUG`, `TORCH_NCCL_ASYNC_ERROR_HANDLING`, `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`. Practice setting them in the launch command — that muscle memory matters.
- `nvidia-smi` and `free -h` accessible while jobs run.
- A tolerance for jobs that fail loudly and by design. This exercise is about the failure, not the fix.

## Requirements

Each of the five scenarios below is worked through the same four-step protocol:

- **Reproduce** — apply the injection to `train.py` (or your launch command / config), then run the job and capture the log until it fails.
- **Diagnose** — from the log evidence, name the failure family and the specific root cause. Cite the log line(s) that proved it.
- **Minimal reproducer** — write a 20–50 line standalone script that exhibits the same failure without the whole SFT training machinery. This is the artifact you'd send to a peer or file an issue with.
- **Fix** — apply the correct mitigation, verify the job now runs (or fails for a documented, different reason), and record the change.

Each scenario gets its own directory under `outputs/exercise-04/scenario-XX/` with the log, the reproducer, and the fix diff.

### Scenario 1 — OOM at step 0

**Injection:** in your Accelerate FSDP config, set `fsdp_activation_checkpointing: false` and set `per_device_train_batch_size` to a value 4× larger than what worked in exercise 01. Alternatively, remove `attn_implementation="flash_attention_2"` from your `from_pretrained` call and re-run with the exercise 01 batch size at sequence length 4096.

Expected symptom: `torch.cuda.OutOfMemoryError` during the first forward pass, with a large activation allocation named in the message.

**Deliverables** in `outputs/exercise-04/scenario-01/`:

- `log.txt` — the full failure log.
- `diagnosis.md` — which of the six memory components from chapter 4 blew up, evidence from the OOM message (the size of the failing allocation, the "free" and "reserved" numbers), and why (checkpointing off, or `s²` attention scores without FA2, or a batch that grew activations).
- `reproducer.py` — a minimal script that constructs the model, runs one forward on a batch of the offending shape, and reproduces the OOM without any FSDP / trainer wrapping.
- `fix.md` — the specific config / code change that resolves it, and the peak per-rank memory measured after the fix.

### Scenario 2 — OOM at step 200 (drifting memory)

**Injection:** in your training loop, add a small "logging" bug that retains a reference to `outputs.hidden_states` (or another large intermediate tensor) on rank 0 every step, e.g.:

```python
# BUG: keeps the entire history of hidden states resident forever.
_debug_history = []
...
outputs = model(**batch, output_hidden_states=True)
_debug_history.append(outputs.hidden_states)   # <-- injected
loss = outputs.loss
```

Run for ~250 steps or until rank 0 OOMs. (Depending on model size, you may need to shorten `output_hidden_states` to `outputs.hidden_states[-1]` if the append is too aggressive to reach step 200.)

Expected symptom: memory grows monotonically over the first 50–250 steps, then OOM on rank 0 (only). Other ranks continue then time out at a collective.

**Deliverables** in `outputs/exercise-04/scenario-02/`:

- `log.txt` — the failure log plus a per-step memory line (`[rank 0] step N mem_gb X.X`) showing the drift.
- `diagnosis.md` — the "memory grows over time" signature, evidence from the per-step memory line, and how you localized the leak. Include a screenshot or export from `torch.cuda.memory._record_memory_history` / the PyTorch memory viewer (`pytorch.org/memory_viz`) that shows the growing allocations.
- `reproducer.py` — the leaking loop in isolation (fake model, real leak).
- `fix.md` — the fix (drop the reference; or if you legitimately need the intermediate, `.detach().cpu()` it and stream to disk) plus the flat post-fix per-step memory line.

### Scenario 3 — NCCL collective timeout

**Injection:** in the training loop, add a conditional early-exit that only rank 3 (or the highest rank in your world size) takes at step 5:

```python
if step == 5 and accelerator.process_index == accelerator.num_processes - 1:
    print(f"[rank {accelerator.process_index}] intentionally exiting early")
    import sys; sys.exit(0)
```

Also set `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=180` at launch so the timeout fires in ~3 minutes rather than the ~30 min default.

Expected symptom: at step 6 (the first collective after the missing rank), the surviving ranks block on all-reduce / reduce-scatter for 3 minutes, then the watchdog reports a collective timeout and the job crashes.

**Deliverables** in `outputs/exercise-04/scenario-03/`:

- `log.txt` — the launch log (with `NCCL_DEBUG=INFO` and `TORCH_DISTRIBUTED_DEBUG=DETAIL` set) showing the intentional exit on the highest rank and the timeout on the surviving ranks.
- `diagnosis.md` — the "peer went away silently" signature. Which rank logged the timeout (all except the missing one), which collective was the survivor stuck in, and how you would identify this in the wild (compare rank logs; whichever rank stopped logging is the culprit).
- `reproducer.py` — a two-rank `torch.distributed`-only script that demonstrates the same shape: both ranks enter an all-reduce, one exits partway through.
- `fix.md` — the fix in this case is "the exiting rank must not exit"; document the general fix pattern (any code path that can early-return needs a `dist.barrier()` before the return, or must not be taken on a subset of ranks; better yet, replace early-return with a per-rank flag that all ranks aggregate via all-reduce).

### Scenario 4 — Silent hang mid-run

**Injection:** in the training loop, add a `wait_for_everyone` (or explicit `dist.barrier()`) that only ranks 0 and 1 hit:

```python
if step == 50 and accelerator.process_index < 2:
    accelerator.wait_for_everyone()   # <-- only 2 of N ranks call this
```

Do **not** set the short `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` for this scenario. Let the job hang.

Expected symptom: the job stops printing after step 50. `nvidia-smi` shows the processes still resident and memory allocated, but GPU utilization drops to 0. No error, no progress.

**Deliverables** in `outputs/exercise-04/scenario-04/`:

- `log.txt` — the training log ending mid-run at step 50 with no error message, and a `nvidia-smi` snapshot from ~1 minute after the hang started.
- `pyspy_dump.txt` — `py-spy dump --pid <pid>` output for **every rank**. Include the shell wrapper you used to collect them (a `for pid in $(pgrep -f train.py); do ...` loop is fine).
- `diagnosis.md` — the "silent hang" playbook applied. Which ranks are in the barrier / `wait_for_everyone` and which are elsewhere? Explicitly compare stacks. What is the general failure family this represents (code-path divergence — some ranks called a collective, others did not).
- `reproducer.py` — a two-rank `torch.distributed` script where rank 0 calls `dist.barrier()` and rank 1 sleeps. The hang should be immediate.
- `fix.md` — remove the conditional collective; the general pattern is "collectives are contracts every rank enters, or none does."

### Scenario 5 — Load imbalance under variable-length sequences

**Injection:** disable sequence packing / length-bucketed sampling in your data pipeline. Use a raw `DataLoader` over a dataset that has both short (~256 token) and long (~4096 token) sequences, with a `collate_fn` that just pads each batch to its longest member. Run for at least 200 steps.

Expected symptom: the run completes but step times are highly variable (some batches finish in 4 s, others in 12 s), and `nvidia-smi dmon` shows uneven per-GPU utilization (some ranks idle waiting at collectives).

**Deliverables** in `outputs/exercise-04/scenario-05/`:

- `log.txt` — per-step wall time for at least 200 steps, showing the variance.
- `nvidia-smi-dmon.log` — 60+ seconds of `dmon` output showing uneven per-GPU utilization.
- `diagnosis.md` — the load-imbalance signature. Correlate step time with batch's max sequence length (log both per step). Confirm the correlation. Discuss why sharded-DP (FSDP / ZeRO-3) does not help this — activations don't shard across DP ranks, so the rank with the longest sequence in the batch takes the longest.
- `reproducer.py` — a minimal script that generates a raggedly-sized fake dataset and measures step time vs. max seq length. Should show the linear (or quadratic, without FA2) relationship clearly.
- `fix.md` — enable one of: (a) length-bucketed sampling (`LengthGroupedSampler`), (b) sequence packing (`SFTTrainer(packing=True)` with an appropriate attention mask), or (c) fixed-length pad-to-multiple. Re-run and show the post-fix step-time variance shrinks materially. Note the trade-offs of the chosen fix.

### Part F — Cross-scenario summary

`outputs/exercise-04/summary.md`, one page total:

- **A table** with columns: scenario, failure family (OOM / NCCL / load imbalance), how quickly you'd diagnose it in the wild ("under 5 min from the log" / "5–30 min with `py-spy`" / "hour+ if unfamiliar"), the single environment variable or tool that made the diagnosis fastest, and the specific fix category.
- **One paragraph on the diagnostic reflex you built.** Which of the five was easiest to diagnose? Which was hardest? Which one felt like it should have been easy but tripped you up? This is the honest self-assessment that will pay dividends on the next real run.
- **Escalation notes.** For each of the five, would you fix it yourself, or would it warrant an escalation to the `training-pipeline-engineer` / platform team (chapter 7's handoff)? If escalation, what evidence would you gather first?

### Part G — Save

Directory layout:

```
outputs/exercise-04/
├── train.py                                 # baseline script (unmodified)
├── requirements.txt
├── scenario-01/
│   ├── injection.diff                       # the change that reproduces the failure
│   ├── log.txt
│   ├── diagnosis.md
│   ├── reproducer.py
│   └── fix.md
├── scenario-02/
│   ├── injection.diff
│   ├── log.txt
│   ├── memory_viz_screenshot.png
│   ├── diagnosis.md
│   ├── reproducer.py
│   └── fix.md
├── scenario-03/
│   ├── injection.diff
│   ├── log.txt
│   ├── diagnosis.md
│   ├── reproducer.py
│   └── fix.md
├── scenario-04/
│   ├── injection.diff
│   ├── log.txt
│   ├── nvidia-smi.txt
│   ├── pyspy_dump.txt
│   ├── diagnosis.md
│   ├── reproducer.py
│   └── fix.md
├── scenario-05/
│   ├── injection.diff
│   ├── log.txt
│   ├── nvidia-smi-dmon.log
│   ├── diagnosis.md
│   ├── reproducer.py
│   └── fix.md
└── summary.md                               # from Part F
```

## Starter guidance

- **Always launch with the diagnostic env vars set**, at least for this exercise:

  ```bash
  NCCL_DEBUG=INFO \
  TORCH_DISTRIBUTED_DEBUG=DETAIL \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  accelerate launch --config_file ... train.py ...
  ```

  Practice reading NCCL and torch-distributed debug output. It's the single highest-signal source when things break.

- **Log per-rank, prefixed, per-step.** `[rank 3] step 200 loss 1.42 mem_gb 43.2 max_seq_len 3200` is the format. Every scenario benefits from this; without it, scenarios 2 and 5 are very hard to diagnose.

- **`py-spy dump --pid <pid>` on every rank at the same wall-clock moment.** Compare stacks; that's how you find code-path divergence hangs (scenario 4). One-liner:

  ```bash
  for pid in $(pgrep -f "train.py"); do echo "=== PID $pid ==="; py-spy dump --pid $pid; done
  ```

- **Reproducers are the deliverable, not an afterthought.** A good reproducer is 20–50 lines, uses `torch.distributed` directly (not through `accelerate` or `Trainer`), and exhibits the failure in under 10 seconds. If your reproducer is 500 lines, you haven't reproduced the failure — you've re-run the training script.

- **Some scenarios have overlapping symptoms.** Scenario 2 (memory leak) and scenario 3 (rank crash then survivor timeout) can both manifest as "OOM followed by hang" if you don't have `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` set. That's exactly why the exercise sets it — you learn to disambiguate by evidence, not by guessing.

- **You do not have to inject the exact bug written.** If your environment makes a specific injection awkward (e.g., you can't easily control per-rank behavior in your launcher), pick an equivalent injection that produces the same failure family. Document what you changed and why.

- **Don't cheat the diagnosis.** The exercise is worth roughly zero if you know the injection and pattern-match to the fix immediately. Set the injection, close the tab, run the job, and diagnose only from the log. Your reflex on real (unknown) failures is the whole point.

## Acceptance criteria

You have completed the exercise when:

- All five scenarios have a directory in `outputs/exercise-04/` with `injection.diff`, `log.txt`, `diagnosis.md`, `reproducer.py`, and `fix.md`.
- Each `diagnosis.md` explicitly cites the log line(s) that proved the root cause. "It's OOM" is not enough; "the OOM message shows a 12.5 GB allocation during forward on the attention scores tensor, which means FlashAttention is off and we're materializing the `[b, a, s, s]` matrix" is what you're aiming for.
- Each `reproducer.py` is standalone (only imports from `torch`, `torch.distributed`, and stdlib) and reproduces the same failure family in under 100 lines.
- Each `fix.md` shows the specific change and evidence that the job now runs (log excerpt, memory number, step-time histogram, whatever proves it).
- Scenario 4's `pyspy_dump.txt` includes stacks from every rank at the same wall-clock moment.
- Scenario 5's `nvidia-smi-dmon.log` shows the uneven-utilization signature before the fix.
- `summary.md` has the cross-scenario table, the honest self-assessment paragraph, and the escalation notes.

## Stretch goals

- **Add a sixth self-designed scenario.** Pick a failure you've heard of but haven't hit yourself — a NCCL "peer unreachable" from a wrong `MASTER_ADDR`, a `Xid` GPU fault, a hang caused by an eval callback with a different batch shape, a DataLoader worker deadlock, a `save_steps` collision with a slow shared filesystem. Follow the same four-step protocol.
- **Turn the OOM tiered mitigation menu (chapter 6) into a script.** Automates the ordered fallbacks: reduce micro-batch, reduce seq length, turn on gradient checkpointing, `expandable_segments:True`, 8-bit optimizer, optimizer offload. On OOM in the training script, applies the next mitigation and re-launches. This is the recovery-orchestration piece that borders on `training-pipeline-engineer` work — implementing it also teaches you what the boundary feels like.
- **Chaos test.** Once your training loop passes all five scenarios, randomly kill one rank mid-run (`kill -9 <pid>` on a random rank at a random step) and verify the surviving ranks fail loudly within your configured heartbeat timeout. This is the "how does my run behave under spot preemption?" question, minus the platform-team plumbing.
- **`nccl-tests` on your GPU set.** Run `all_reduce_perf` and `all_gather_perf` at message sizes 1 MB, 100 MB, 1 GB. Compare the achieved `busbw` against the vendor spec for your interconnect. If the numbers are materially below spec, escalate to the platform team — this is the specific evidence that unblocks fabric-tuning work.
- **Memory snapshot deep-dive.** For scenario 2, use `torch.cuda.memory._record_memory_history()` + `_dump_snapshot()` and load the resulting pickle into the PyTorch memory viewer (`pytorch.org/memory_viz`). Time-slice the trace and screenshot the specific step where the leak starts to visibly grow. Discuss what the visualization made obvious that the per-step log did not.
- **Compare `torch.compile` behavior under injection.** Some of the scenarios interact interestingly with `torch.compile` (autograd graph invalidation on shape drift, silent recompiles under variable-length inputs). Turn `torch.compile` on and re-run scenario 5; note any additional failure modes or diagnostic clues.
