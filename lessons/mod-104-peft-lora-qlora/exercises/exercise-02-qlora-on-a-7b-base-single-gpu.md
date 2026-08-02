# exercise-02: QLoRA on a 7B Base, Single GPU

**Estimated effort:** 5 hours (mostly wall-clock training)
**Companion chapters:** [`02-qlora-and-4bit-base-weights.md`](../02-qlora-and-4bit-base-weights.md), [`05-adapter-failure-modes.md`](../05-adapter-failure-modes.md)

## Objective

Fine-tune a 7B-class base model on a real SFT slice using QLoRA — NF4 base weights, double quantization, paged AdamW, gradient checkpointing — on a *single* 24-GB GPU. Ship a trained adapter, a memory-and-throughput report against chapter 2's memory table, and a health-check that confirms the adapter learned something useful in every layer you wrapped.

## Prerequisites

- A single GPU with ≥ 24 GB VRAM. RTX 3090, RTX 4090, A5000, A6000, L4, A10, A100 all work. RTX 3080 (12 GB) does not fit a 7B QLoRA cleanly — drop down to a 3B base and note it in the report.
- Python 3.10+ with `torch`, `transformers`, `trl`, `datasets`, `accelerate`, `peft`, `bitsandbytes` all at recent versions. Pin them in `requirements.txt`.
- Hugging Face Hub access with any gated-model licenses accepted (Llama 3.1 requires a click-through).
- Optional but strongly recommended: `wandb` or `tensorboard`, `flash_attn` for a throughput boost.

## Requirements

### Part A — Pick the base and dataset

Choose one base and one dataset. Do not switch mid-exercise.

Suggested bases (7B–13B, publicly available):

- `meta-llama/Llama-3.1-8B` — the reference 8B base.
- `mistralai/Mistral-7B-v0.3` — long-standing 7B base with broad ecosystem support.
- `Qwen/Qwen2.5-7B` — ChatML template baked in.

Suggested datasets:

- `HuggingFaceH4/no_robots` (~10k rows, small enough to train in an hour on a 24 GB GPU).
- `HuggingFaceH4/ultrachat_200k` — take a ~20k-row slice; larger and higher variance.
- A curated slice from mod-102 exercise 01, if you have it.

Use the **base** (non-Instruct) checkpoint. The whole point is to create instruction-following capability with QLoRA.

### Part B — Write the training script

Author `train_qlora.py`. Minimum requirements:

1. Load the base with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`.
2. Call `prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)`.
3. Configure `LoraConfig` with:
   - `target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` (all-linear inside the block; **exclude** LM head / embeddings).
   - `r=16`, `lora_alpha=32`, `lora_dropout=0.05`.
   - `task_type="CAUSAL_LM"`, `bias="none"`.
4. Wrap with `get_peft_model` and log `print_trainable_parameters()` to your run log.
5. Use `PagedAdamW32bit` (or `PagedAdamW8bit` if VRAM is tight) via the trainer's `optim="paged_adamw_32bit"` argument.
6. `SFTConfig`:
   - `per_device_train_batch_size` and `gradient_accumulation_steps` derived from your VRAM. Target an effective batch of ~128k–256k tokens/step (chapter-2 formula from mod-103).
   - `learning_rate=2e-4` (the QLoRA-paper starting point) with `lr_scheduler_type="cosine"` and `warmup_ratio=0.03`.
   - `bf16=True`, `max_length=2048`, `packing=True`.
   - Sane `save_steps`, `eval_steps`, `logging_steps`. `save_total_limit=2` to keep disk under control.
7. Wire your logs to at least one persistent backend (WandB / TensorBoard / JSONL).

### Part C — Verify the memory budget

At the top of `trainer.train()`, capture:

- `torch.cuda.reset_peak_memory_stats()` before training.
- The output of `torch.cuda.memory_allocated()` after model construction (base + adapter loaded, before any optimizer step). Call this `mem_at_load`.
- After the first optimizer step, `torch.cuda.max_memory_allocated()`. Call this `mem_peak`.

In your run log, print:

```
mem_at_load = X GB     # should match chapter 2's "base weights + adapters" row
mem_peak    = Y GB     # should be below your card's total VRAM by at least a few GB
```

If `mem_peak > 0.95 · total VRAM`, your run is one bad prompt away from OOM. Lower `per_device_train_batch_size` or `max_length` before you commit to the full run.

### Part D — Run the sanity checks (from chapter 1 of mod-103)

Before the full run:

1. Print one collated batch and confirm `labels` masking (mod-103 chapter 2 pattern).
2. Run one forward pass — confirm loss is finite and in the 2–5 range.
3. Run `max_steps=10` and confirm loss decreases.

Only after all three pass do you commit to the full training run.

### Part E — Ship the artifacts

At end of run, `outputs/` contains:

- `adapter/` — the saved LoRA adapter (safetensors + `adapter_config.json`).
- `logs/train.jsonl` and `logs/eval.jsonl`.
- `config.yaml` — full `SFTConfig` and `LoraConfig` dumps, plus base model + dataset choice.
- `plots/loss.png` — the two-curve (train + eval) plot.
- `memory.md` — the numbers from Part C alongside the chapter-2 memory table. Confirm you are within a couple GB of the expected values in every row; if not, explain what pushed you off.

### Part F — Adapter health check

Run the chapter-5 `lora_health` script (or equivalent SVD-based analysis) against your trained adapter. Save results as `outputs/health.json` and produce `outputs/health.md` (one page) with:

1. A per-layer table (or a plot) of effective rank vs. layer depth.
2. Any layer whose effective rank is below `r / 3` — call these out explicitly.
3. Any layer whose `max_sv / min_sv` ratio exceeds 1e6 — call these out too.
4. One paragraph interpreting the results. Did every layer contribute? If some layers underused their rank, was it because the task did not need to change them (fine), or because your `alpha/r` was too low (config bug)?

### Part G — Diagnosis

Write `outputs/diagnosis.md` (one paragraph, ~200 words):

1. Which mod-103 chapter-4 loss-curve shape does your curve match?
2. Was quality "close to full-precision LoRA territory" (loss under ~1.5 for a well-behaved SFT slice) or "something is off"?
3. If you were to redo this run, what would you change first and why?

## Starter guidance

- **Read the QLoRA paper's Guanaco recipe** before you start. Learning rate `2e-4`, cosine schedule, `warmup_ratio=0.03`, `weight_decay=0.0` for the LoRA path is a good starting point. You can improve on it later; do not start by fighting the paper.
- **The three-item sanity check is not optional.** A silently-broken NF4 load path is one of the more common QLoRA bugs (usually a `bnb_4bit_compute_dtype` mismatch that produces a coherent-but-degraded run). The sanity checks catch it in five minutes.
- **If `mem_peak` is 2× your prediction from chapter 2**, the most likely culprit is missing `prepare_model_for_kbit_training` (specifically, gradient checkpointing did not turn on properly). Confirm via `model.print_trainable_parameters()` and `torch.cuda.memory_summary()`.
- **If training loss looks great but the health check shows every layer at effective rank 1–2**, you either under-trained (short run + gentle LR) or your `alpha/r` was too low. Chapter 5 has the fix ranking.
- **Use `nvitop`, `nvidia-smi dmon`, or `wandb`'s system panel** to watch VRAM and GPU utilization live. A GPU that dips to 0% every few seconds is dataloader-starved; fix `dataloader_num_workers` before you touch anything else.
- **Do not merge the adapter in this exercise.** Chapter 3 and exercise 03 are the right place to reason about merging QLoRA outputs; the merge cannot happen against the 4-bit base cleanly anyway.

## Acceptance criteria

You have completed the exercise when:

- `train_qlora.py`, `config.yaml`, and `outputs/` (adapter, logs, plots, memory.md, health.json, health.md, diagnosis.md) all exist and are self-consistent.
- `mem_peak` is verifiably below your card's total VRAM by ≥ 2 GB, and the memory numbers reconcile against chapter 2's table.
- The three sanity checks are visible in your run log.
- The health check identifies at least one insight per layer group (attention vs. MLP), not just "all layers healthy."
- `diagnosis.md` answers the three questions above with numbers, not vibes.
- Someone else could re-run the script and land within 5% of your loss curve.

## Stretch goals

- **Sweep `r ∈ {8, 16, 32}` on the same base and dataset.** Produce a three-line comparison of trainable params, VRAM, wall-clock, and final eval loss. Compare to chapter 1's "rank ranges seen in the literature" claim.
- **Sweep `target_modules ∈ {attention-only, all-linear}`.** Same three-line table. Note which win where.
- **Rerun with `PagedAdamW8bit` instead of `PagedAdamW32bit`** and report the memory delta and any effect on loss. The paper's finding was "no measurable quality regression at the 7B scale" — confirm it (or don't) on your slice.
- **Rerun without `bnb_4bit_use_double_quant`.** Compare `mem_at_load` and defend whether the ~0.37 bits/param claim is holding up on your model. Numbers from the paper are approximate; your architecture might be different.
- **Compare to a plain-LoRA baseline.** If you have access to a 40 GB card (A100 40 GB, A6000 48 GB), redo the same run in BF16 without `load_in_4bit`. Diff the loss curves. This is the direct test of chapter 4's "QLoRA vs LoRA quality" claim.
- **Try Llama-3.1-8B and Mistral-7B-v0.3 at the same settings.** Which base tolerates QLoRA better on your task? Report and speculate on why.
