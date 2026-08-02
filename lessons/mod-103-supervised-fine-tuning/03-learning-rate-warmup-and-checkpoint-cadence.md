# Learning-Rate Schedule, Warmup, Weight Decay, and Checkpoint Cadence

**Estimated read:** 40 minutes.
**You should be able to, by the end:** pick a starting learning rate, schedule, warmup fraction, and weight decay for a full-parameter or LoRA SFT run and defend the choice; and set `save_steps` / `eval_steps` so the checkpoint you would want to roll back to actually exists on disk.

## The hyperparameter that decides most of the outcome

Learning rate is not one of many knobs. It is *the* knob. Loss curves, overfitting behavior, catastrophic forgetting, whether a run recovers from warmup — all of it moves more with the learning rate than with any other hyperparameter you can turn.

Two anchor numbers, from published recipes, worth memorizing before you make anything up:

- **Full-parameter SFT on a modern base model (1B–13B):** starting LR in the range **1e-5 to 5e-5**. Llama 2 SFT used 2e-5 (Touvron et al., 2023b); modern open recipes (Tülu 2/3, OpenHermes, Zephyr) cluster in 1e-5 to 2e-5. Higher than 5e-5 on full-parameter SFT of a well-trained base commonly produces catastrophic forgetting (chapter 5).
- **LoRA / QLoRA SFT on the same size range:** starting LR in the range **1e-4 to 3e-4**. LoRA (Hu et al., 2021) reports that the small trainable adapter can — and should — take a much larger step than the full parameter count would tolerate.

These are not "correct." They are *good starting points*. Sweep locally around them (exercise 03) before deciding on the run-of-record.

## Why full-param and LoRA sit at different orders of magnitude

Every parameter in a full-parameter run gets an SGD-scale update every step. A large LR moves *every weight* of a delicately pretrained model an equally large amount — over-committing to the SFT distribution and forgetting the pretraining distribution. Small LR + slightly-over-one epoch on well-curated data is the reliable recipe.

LoRA adapts a **low-rank decomposition** on top of frozen weights. The trainable parameter count is a fraction of a percent of the base. The optimizer step affects only that fraction; the pretrained weights are literally untouched. A larger LR here does not risk forgetting in the same way — the base model is still exactly where it was when you started.

Chapter 4 will make this concrete via loss curves: a LoRA run at 2e-5 barely moves the loss; a full-parameter run at 2e-4 blows up.

## Learning-rate schedule: cosine, linear, or constant

The three schedules you will use in SFT are:

- **Cosine with warmup.** LR ramps from 0 to peak over `warmup_steps`, then decays as a cosine to (usually) 10% of peak by end of training. Default in most open recipes; smooths the tail of training and hedges against a slightly-too-high peak.
- **Linear with warmup.** LR ramps up then linearly decays to 0 over the remaining steps. Marginally worse than cosine at the tail in most benchmarks, marginally cheaper mentally to reason about.
- **Constant with warmup.** LR ramps up and stays there. Useful for very short runs (a few hundred steps) and for continual-training scenarios where you plan to resume. Not recommended for single-shot SFT — you lose the natural "cool down" that lets the loss settle.

Pick cosine by default; the case for the other two is narrow.

## Warmup: enough to prevent the early-training blowup, no more

At step 0 of a full-parameter SFT run, the model has never seen your dataset distribution before. The first few batches are a shock. Without warmup, the optimizer takes a full-magnitude step in a direction that reflects that shock, and you either see a giant loss spike or divergence within a hundred steps.

Warmup ramps the LR from 0 (or a very small value) up to peak over the first few percent of training. Reasonable defaults:

- **Full-parameter SFT:** `warmup_ratio=0.03` (3% of steps). If your run is very short (say, 500 optimizer steps), pin `warmup_steps=50` as a floor.
- **LoRA / QLoRA:** `warmup_ratio=0.03` — same reasoning; short is fine.
- **Very short runs (<200 optimizer steps):** consider `warmup_ratio=0.1` to give warmup enough room to actually help.

There is a specific way warmup can lie to you. If `warmup_steps` (in *optimizer* steps) is much longer than the number of `logging_steps` intervals inside warmup, the loss curve during warmup will look flat or noisy and give you no information. Log frequently during the warmup — see chapter 4.

## Weight decay: less than you think

Weight decay applied through AdamW (technically decoupled weight decay; Loshchilov & Hutter, 2019) pulls weights toward zero every step, independently of the gradient. In pretraining it is important for regularization; in SFT, where the base model is already extensively pretrained and you are moving weights only a small amount, the effect is minor and can even be counterproductive.

Sane defaults:

- **Full-parameter SFT:** `weight_decay=0.0` to `0.1`. Llama 2 SFT used 0.1; many open recipes use 0. Both are defensible; pick one and stick with it.
- **LoRA / QLoRA:** `weight_decay=0.0` almost always. The LoRA parameters are close to zero already; decaying them further can prevent the adapter from moving where it needs to.

If you use non-zero weight decay, exclude bias and normalization parameters — apply it only to matmul weights. `transformers.Trainer` does this by default; if you write a custom optimizer setup, replicate the exclusion.

## Optimizer choice

The default optimizer for SFT is **AdamW** (`torch.optim.AdamW` or `transformers.optimization.AdamW`). It is what every published recipe you will read uses; the analytical apparatus in chapter 4 assumes it.

Two alternates you may see:

- **8-bit AdamW** (`bitsandbytes.optim.AdamW8bit`) — halves optimizer-state memory at negligible quality cost. Standard for QLoRA (Dettmers et al., 2023). Turn on with `optim="adamw_bnb_8bit"` in `TrainingArguments`.
- **Lion** (Chen et al., 2023, `lion-pytorch`) — smaller optimizer state, momentum only. Some published results claim comparable quality at a much smaller memory footprint. Less battle-tested in SFT; do not use for run-of-record without verifying on your dataset.

Do not chase optimizer novelty in SFT. AdamW is well-understood; every other lever in this module gives you more mileage.

## Number of epochs

The SFT literature has settled on a narrow range:

- **1 epoch** — the modern default for large curated SFT mixes (Tülu 3, OpenHermes-2.5). One high-quality pass through the data is preferred to multiple passes through anything with lower quality.
- **2–3 epochs** — reasonable for small, high-signal datasets (LIMA-style; a couple thousand rows). Watch eval loss closely for over-training past epoch 2 (chapter 4).
- **>3 epochs** — almost never justified for SFT of a modern base model. If you find yourself here, you probably have too little data — go back to mod-102 and grow the dataset.

Rule of thumb: **one high-quality epoch on a large mix, or two–three epochs on a small mix**, and never both at once.

## Checkpoint cadence: `save_steps`, `save_total_limit`, `save_strategy`

`save_steps=500` and `save_total_limit=3` is a fine default and hides a design decision: **you are keeping the last three checkpoints, not the best three**. When the training report (chapter 6) says "we shipped the last checkpoint," verify that assertion. Common patterns:

- **`save_strategy="steps"` + `save_steps` and `save_total_limit`.** Time-based rotation; the *last N* checkpoints survive. Default and reasonable for most runs.
- **`save_strategy="epoch"`.** Save at every epoch boundary. Cleaner for small runs; not enough resolution to roll back for large ones.
- **`load_best_model_at_end=True` + `metric_for_best_model="eval_loss"` (or a custom metric via a callback).** Save every `eval_steps`, and at end-of-training reload the checkpoint with the best eval metric. This is the "best" checkpoint semantics and is what you want for run-of-record.

An `eval_steps` that is smaller than `save_steps` and does not divide it evenly means the "best" checkpoint may be an unsaved one. Set `save_steps == eval_steps` or a small multiple of it. This is the single most common mistake in checkpoint policy.

### Retention policy across runs

For a real project you want a **retention policy separate from the trainer's rotation**:

- Keep the final checkpoint of every run for at least 30 days.
- Keep the "released" checkpoint indefinitely.
- Delete intermediate checkpoints from experimental sweeps immediately after selection.

Trainer rotation is fine for the intra-run rolling window; it does not answer the "which run's checkpoint do we still have?" question. That is a project-level artifact-management concern.

## Evaluation cadence: `eval_steps` and what to evaluate

Two evals are worth running during training:

1. **Eval loss on a held-out slice of the SFT dataset.** Cheap. Runs every `eval_steps`. This is what chapter 4 teaches you to read.
2. **A "quality proxy" eval on a small held-out benchmark slice** — a 100-row MMLU subset, a 20-item held-out task set, a small MT-Bench slice run through a fast judge. More expensive; run less often. Catches catastrophic forgetting before end-of-training.

Sanity numbers for eval cadence on a small SFT job:

- `eval_steps=200` on a run with 5,000 optimizer steps: 25 eval readings — enough resolution.
- `eval_steps=50` on a run with 500 optimizer steps: 10 readings — the minimum.
- Never `eval_steps > save_steps` — you will save checkpoints you have no eval signal for.

For custom evaluations (quality proxy, safety spot-check, format-compliance), use `transformers.Trainer`'s `TrainerCallback` API to hook into `on_evaluate` and log to the same metrics stream as the built-in loss. The training report (chapter 6) references those metrics; a metric that was never logged does not appear in the report.

## Two worked examples

### Full-parameter SFT of `meta-llama/Llama-3.2-1B` on a 20k-row mix

Reasoning:

- Effective batch: 2 (per_device) × 8 (grad_accum) × 1 (GPU) = 16 sequences × 2048 tokens = ~32k tokens per step.
- Dataset: 20k rows, ~1000 tokens each on average, so ~20M training tokens per epoch.
- Steps: 20M / 32k ≈ 625 steps per epoch. One epoch: 625 optimizer steps.
- Warmup: 3% × 625 = ~19 steps. Round to 25.
- LR: 2e-5, cosine.
- Weight decay: 0.
- Eval / save: 100 / 100.

```python
SFTConfig(
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_steps=25,
    weight_decay=0.0,
    bf16=True,
    max_length=2048,
    packing=True,
    eval_steps=100,
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=42,
)
```

### QLoRA SFT of `Qwen/Qwen2.5-7B` on the same 20k rows

Reasoning:

- Effective batch: 4 (per_device on 4-bit weights + LoRA) × 4 (grad_accum) × 1 (GPU) = 16 sequences × 2048 tokens = ~32k tokens per step. Same as above by design.
- Steps: same 625 per epoch.
- LR: 2e-4 (10× the full-param number).
- Weight decay: 0.
- Warmup: 3% × 625 ≈ 20.

The rest of the config is identical. This is the point: at the trainer level, LoRA vs. full-parameter is one line — the LR — plus the `peft` model wrapper (mod-104). Everything else you learned in this chapter carries over unchanged.

## What NOT to change from the defaults, on a first SFT run

You will be tempted to touch these. Do not, on a first run:

- `adam_epsilon`, `adam_beta1`, `adam_beta2` — trainer defaults are fine.
- `max_grad_norm` — the default of 1.0 is fine; only lower if you see gradient-norm spikes in chapter 4.
- Any per-layer learning-rate group. Reserve for after you have measured a specific problem.
- `dropout` on model layers — the base model already ships with the dropout its architects picked; SFT rarely changes it.

The chapter's point is that a first SFT run should differ from the recipe above by *at most* the base model and the dataset. Every other change is future work, not first-run work.

## Exercise pointer

`exercises/exercise-03-lr-schedule-and-loss-diagnosis.md` has you sweep three learning rates × two schedules on a fixed dataset and produce loss-curve plots plus a one-page written diagnosis. This chapter's defaults are the middle of your sweep, not the goal — the goal is that you learn to *read* the curves.

## Summary

- Full-parameter SFT LR: 1e-5 to 5e-5. LoRA / QLoRA SFT LR: 1e-4 to 3e-4. LR is the knob that decides most of the outcome; other knobs move less.
- Cosine with warmup is the default schedule. Warmup 3% of steps for most runs, more for very short ones.
- Weight decay 0.0 (or up to 0.1 on full-param) is a fine default; do not bother with anything exotic. Exclude bias and norm parameters when non-zero.
- AdamW is the SFT optimizer. 8-bit AdamW for QLoRA memory savings. Do not chase Lion or novel optimizers on run-of-record.
- 1 epoch on a large curated mix; 2–3 epochs on a small high-signal mix; never more than that on a modern base.
- Checkpoint policy: set `save_steps == eval_steps`, keep the last few, and use `load_best_model_at_end` for run-of-record. Keep the final checkpoint of every run under a project-level retention policy separate from trainer rotation.
- Evaluate loss frequently, evaluate a quality proxy less frequently. The training report references only metrics you actually logged.
