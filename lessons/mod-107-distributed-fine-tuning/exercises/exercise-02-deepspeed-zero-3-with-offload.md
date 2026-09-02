# exercise-02: DeepSpeed ZeRO-3 with CPU Offload

**Estimated effort:** 3 hours
**Companion chapters:** [`01-parallelism-landscape.md`](../01-parallelism-landscape.md), [`03-deepspeed-zero-and-offload.md`](../03-deepspeed-zero-and-offload.md), [`04-memory-mixed-precision-and-checkpointing.md`](../04-memory-mixed-precision-and-checkpointing.md)

## Objective

Take the same SFT script you scaled to 8 GPUs with FSDP in [exercise 01](exercise-01-accelerate-fsdp-launch-on-2-to-8-gpus.md) and run it under **DeepSpeed ZeRO-3** in three offload configurations: **(a)** no offload, **(b)** optimizer-state offload to CPU, **(c)** optimizer + parameter offload to CPU. Compare peak per-rank memory, tok/s/GPU, and end-of-run training loss across those three DeepSpeed runs and against the FSDP `FULL_SHARD` baseline from exercise 01. Then write a one-page recommendation that answers: **for a specific follow-on scenario, which framework and offload level would you pick and why?**

The exercise is not about which framework "wins" in the abstract — it is about learning how to think through the framework / offload / hardware trade-off on a specific job.

## Prerequisites

- Exercise 01 complete, with the FSDP `FULL_SHARD` results in `results/sweep_results.csv` and a working `train.py` that runs on 2 → 8 GPUs.
- A single node with 4–8 GPUs (A100 40 GB / A100 80 GB / H100 SXM). If you only have 2 GPUs, the exercise still runs; the memory savings from offload will just be less dramatic.
- Host RAM headroom: full FP32 AdamW state for the base you're training must fit in CPU RAM when you enable optimizer offload. For a 7B base that is ~48 GB; verify with `free -h` before starting. If your host is tight (e.g., ≤128 GB), stay on a 1B–3B base for this exercise.
- Recent `deepspeed`, matched to the `torch` version you used in exercise 01. Pin in `requirements.txt`; version mismatch between `deepspeed` and `torch` is the most common install failure. Consult the current DeepSpeed compatibility matrix before pinning.
- The same tokenized SFT dataset used in exercise 01 (reproducibility matters here — different data changes throughput and loss numbers).
- `nvidia-smi` and `free -h` available while the job runs so you can watch GPU and host RAM in a second terminal.

## Requirements

### Part A — Author three DeepSpeed configs

Write three `ds_config_*.json` files. Every knob choice must be defensible in the writeup (Part D).

#### `configs/ds_config_z3_noffload.json` — ZeRO-3, no offload

Baseline DeepSpeed run. Compare directly against the FSDP `FULL_SHARD` throughput number.

- `bf16.enabled: true`
- `zero_optimization.stage: 3`
- `zero_optimization.overlap_comm: true`
- `zero_optimization.contiguous_gradients: true`
- `zero_optimization.reduce_bucket_size: "auto"`
- `zero_optimization.stage3_prefetch_bucket_size: "auto"`
- `zero_optimization.stage3_param_persistence_threshold: "auto"`
- `zero_optimization.stage3_gather_16bit_weights_on_model_save: true`
- `gradient_accumulation_steps: "auto"`, `train_batch_size: "auto"`, `train_micro_batch_size_per_gpu: "auto"`, `gradient_clipping: "auto"`
- `wall_clock_breakdown: false` (turn to `true` for Part C profiling)

#### `configs/ds_config_z3_optoffload.json` — ZeRO-3 + optimizer offload

Adds:

- `zero_optimization.offload_optimizer.device: "cpu"`
- `zero_optimization.offload_optimizer.pin_memory: true`

Everything else identical to the no-offload config.

#### `configs/ds_config_z3_full_offload.json` — ZeRO-3 + optimizer + parameter offload

Adds the block above **and**:

- `zero_optimization.offload_param.device: "cpu"`
- `zero_optimization.offload_param.pin_memory: true`

This is the survival-mode config. Expect a materially slower step.

Companion Accelerate config `configs/deepspeed_accelerate.yaml`:

- `distributed_type: DEEPSPEED`
- `deepspeed_config.deepspeed_config_file` pointing at the specific JSON you're launching with (or make three Accelerate configs, one per JSON)
- `deepspeed_config.zero3_init_flag: true`
- `deepspeed_config.zero3_save_16bit_model: true`
- `mixed_precision: bf16`
- `num_processes` matching your GPU count

Launch with `accelerate launch --config_file configs/deepspeed_accelerate.yaml train.py ...`. You can swap the JSON per run without editing the Accelerate yaml if you templatize the path via an env var.

### Part B — Sweep the three DeepSpeed configs

For each of the three configs, on the same GPU count you used for the FSDP 8-GPU run in exercise 01 (fall back to 4 GPUs if 8 is not available), collect:

- **Peak per-rank GPU memory** (`torch.cuda.max_memory_allocated()`).
- **Peak host RAM** during a steady-state step (`free -h` sampled every 5 s in a second terminal, or a small Python thread that logs `psutil.virtual_memory().used`).
- **Steady-state tok/s total and per-GPU** (median of last 50 steps; discard first 10).
- **Wall clock for 100 steps** (or a fixed token budget matching exercise 01).
- **Final training loss** at the end of a fixed-length run (should be within noise of the FSDP number for the same data + hyperparameters).
- **Whether the run OOM'd (GPU or host) or completed cleanly.** OOM at a specific step counts as a real result — document the mitigation you would try next.

Log format: one JSON or CSV row per run. Commit to `results/exercise-02/deepspeed_sweep.csv`.

### Part C — Turn on the DeepSpeed FLOPS profiler and `wall_clock_breakdown`

For the **no-offload** run only, enable in the JSON:

```json
"flops_profiler": {
  "enabled": true,
  "profile_step": 10,
  "module_depth": -1,
  "top_modules": 3,
  "detailed": true,
  "output_file": "results/exercise-02/flops_profile_noffload.txt"
},
"wall_clock_breakdown": true
```

Capture the output. The FLOPS profiler prints per-module FLOPs, MACs, and inferred throughput; `wall_clock_breakdown` prints per-phase timing (fwd / bwd / step / comm) periodically.

Extract from the profile output:

- **Per-token FLOPs** DeepSpeed reports vs. the `6 · N` PaLM approximation you used in exercise 01. Compute the ratio; report whether the two agree within ~20%.
- **Fraction of step time in communication** vs. compute vs. optimizer step. This is your bottleneck-shape signal for Part D.

Save the raw profile output alongside the CSV.

### Part D — Cross-framework comparison

Build a single comparison table (`results/exercise-02/framework_comparison.md`) with **five rows** and the following columns: peak per-rank GPU memory, peak host RAM, tok/s/GPU, tok/s speedup vs. exercise 01 single-GPU baseline, final loss, notes.

Rows:

1. **FSDP `FULL_SHARD`, throughput config** — reuse the exercise 01 number for the same GPU count.
2. **DeepSpeed ZeRO-3, no offload** — from Part B.
3. **DeepSpeed ZeRO-3, optimizer offload** — from Part B.
4. **DeepSpeed ZeRO-3, optimizer + parameter offload** — from Part B.
5. **DeepSpeed ZeRO-2, no offload** — one extra run to see what dropping to ZeRO-2 buys you (change `zero_optimization.stage: 2` in the no-offload config; keep everything else). Only include this row if the model + optimizer state actually fits at ZeRO-2 per rank; if it OOMs, log "OOM as expected" and move on.

Loss should be within ~1% of the FSDP baseline on all rows that completed — if it isn't, something in the training recipe changed and you need to reconcile before comparing throughput.

### Part E — Write the recommendation report

`report.md`, structured as:

1. **Setup recap** — model, dataset, hardware, world size, mixed precision, sequence length, effective batch, effective LR schedule. Confirm parity with exercise 01.
2. **The comparison table** from Part D.
3. **Memory story** — for the ZeRO-3 no-offload vs. FSDP `FULL_SHARD` rows, do the per-rank memory numbers agree with the `16N/P + activations + overhead` heuristic from chapter 7? Where they disagree, why?
4. **Offload cost curve** — from ZeRO-3 no-offload → optimizer offload → full offload, what's the % throughput cost of each step? Does it match the "10–30% for optimizer, 2–5× hit for full" heuristic from chapter 3?
5. **Communication vs. compute breakdown** — from the Part C FLOPS profile + `wall_clock_breakdown`. One sentence: is the no-offload run compute-bound, communication-bound, or optimizer-step-bound?
6. **Recommendation** — the exercise's headline output. For **each of the three follow-on scenarios below**, recommend one specific configuration (FSDP vs. DeepSpeed, ZeRO stage, offload y/n) and justify in 3–5 sentences:
   - **Scenario 1:** Same 7B–13B base, same cluster (8× A100 80 GB single node), 5M-example SFT run. Iteration speed is the priority.
   - **Scenario 2:** 34B base full-parameter SFT, 8× A100 80 GB single node, 500k examples. Fitting the model is the priority; throughput is secondary.
   - **Scenario 3:** 70B base full-parameter SFT, only 4× A100 80 GB available. The team can wait a week. Feasibility is the priority; throughput is not.

### Part F — Save

Directory layout:

```
outputs/exercise-02/
├── configs/
│   ├── deepspeed_accelerate.yaml
│   ├── ds_config_z3_noffload.json
│   ├── ds_config_z3_optoffload.json
│   ├── ds_config_z3_full_offload.json
│   └── ds_config_z2_noffload.json         # only if you ran the ZeRO-2 row
├── train.py                               # the exercise-01 SFT script, unchanged if possible
├── requirements.txt                       # pinned versions
├── results/
│   ├── deepspeed_sweep.csv                # from Part B
│   ├── flops_profile_noffload.txt         # from Part C
│   ├── wall_clock_breakdown_noffload.txt  # from Part C
│   └── framework_comparison.md            # from Part D
├── report.md                              # from Part E
└── logs/                                  # per-rank stdout for at least one run per config
```

## Starter guidance

- **Match the training recipe to exercise 01 exactly.** Same base model, tokenizer, dataset, effective batch, LR schedule, sequence length, epochs. Framework should be the only free variable; if loss numbers diverge, look at the recipe first.
- **Use `"auto"` for the batch and clipping fields in the DeepSpeed JSON.** Let Accelerate + Trainer fill them from `TrainingArguments`. Overriding is the fastest way to end up with a `train_batch_size != micro * accum * world` error.
- **`zero3_init_flag: true` is not optional** for anything past ~7B. Without it, `from_pretrained` materializes the full model in host RAM on every rank before sharding, and multi-rank OOMs the host.
- **Watch `free -h` from a second terminal** during the optimizer-offload run. Optimizer offload frees GPU memory by moving state to pinned host memory; if the host is contended (data loader on the same cores, other tenants), the optimizer step will slow down and drag every step behind it.
- **Numbers, not vibes.** "It felt slower" is not an acceptable comparison. Report tokens/sec/GPU and let the ratios speak.
- **The FLOPS profiler output is verbose.** Read the top block (per-model summary) first, then dig into `top_modules` for the per-layer breakdown. `detailed: true` prints a lot; you probably want to save it to a file and grep, not stare at the terminal.
- **Don't try to make offload win on throughput.** It won't. The value proposition of offload is that it fits, not that it's fast. Frame the recommendation accordingly.

## Acceptance criteria

You have completed the exercise when:

- Three DeepSpeed configs (`ds_config_z3_noffload.json`, `ds_config_z3_optoffload.json`, `ds_config_z3_full_offload.json`) launch cleanly on your GPU count.
- `results/deepspeed_sweep.csv` has one row per DeepSpeed run with all Part B fields populated.
- `results/framework_comparison.md` shows the five-row cross-framework table.
- `results/flops_profile_noffload.txt` contains the DeepSpeed FLOPS profiler output for the no-offload run, and you have compared its per-token FLOPs against the `6 · N` approximation.
- Final training loss is within ~1% across the completed runs (framework and offload should not change training math beyond mixed-precision noise). If yours diverges, `report.md` explains why.
- Throughput cost of optimizer offload is documented (typically 10–30%); throughput cost of full parameter offload is documented (typically 2–5× vs. no-offload).
- `report.md` gives a recommendation for each of the three scenarios in Part E, each 3–5 sentences and defended with reference to a specific measurement in your comparison table.

## Stretch goals

- **Consolidate a DeepSpeed sharded checkpoint** to a single-file `pytorch_model.bin` / `model.safetensors` using `zero_to_fp32.py`. Confirm that loading via `AutoModelForCausalLM.from_pretrained("consolidated/")` produces identical logits (to numerical precision) as loading the sharded state via DeepSpeed. This is the handoff path to mod-109 / serving.
- **DPO under DeepSpeed ZeRO-3.** Swap `SFTTrainer` for `DPOTrainer` (mod-105) and re-run the no-offload config. The reference model doubles memory footprint — verify per-rank memory grows roughly as predicted, and note any TRL-specific quirks (reference-model init order under `zero3_init_flag: true` is the classic one; check the current TRL docs).
- **NVMe offload smoke test.** If you have local NVMe on the node (`/mnt/nvme_scratch` or equivalent), add a fourth DeepSpeed config with `offload_optimizer.device: "nvme"` and `nvme_path: /mnt/nvme_scratch`. Measure the throughput drop vs. CPU offload. This is deliberately outside the "should you use this?" scope from chapter 3 — you're doing it to feel the cost.
- **8-bit AdamW under DeepSpeed.** Add `optim="adamw_bnb_8bit"` to the TrainingArguments and re-run the no-offload config. Report the per-rank memory delta and whether loss shifted materially.
- **Small-model regression test.** Run the no-offload ZeRO-3 config on a 1B base and confirm that DeepSpeed is *worse* than DDP / FSDP `SHARD_GRAD_OP` at that scale (chapter 3's "when DeepSpeed hurts" claim). Report the numbers.
- **Version pinning discipline.** Produce a `requirements.txt` with every package pinned to an exact version, then bump `deepspeed` by one minor version and re-run the no-offload config. Report whether the numbers or behavior changed. This is a reproducibility test, not a performance test.
