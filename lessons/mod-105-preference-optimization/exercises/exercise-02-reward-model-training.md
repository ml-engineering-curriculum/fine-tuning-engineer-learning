# exercise-02: Reward Model Training

**Estimated effort:** 3 hours (mostly wall-clock training)
**Companion chapter:** [`03-reward-modeling.md`](../03-reward-modeling.md)

## Objective

Train a reward model on the preference dataset from exercise 01, evaluate it against pairwise accuracy, calibration, and length-conditional slices, and produce a one-page RM report. The RM you ship here plugs directly into exercise 03's PPO loop; a low-quality RM makes exercise 03 look like a failure. Getting the RM right is why this exercise exists.

## Prerequisites

- The `outputs/preference-set-v1/` artifact from exercise 01.
- `torch`, `transformers`, `trl` (≥ 0.9), `peft`, `bitsandbytes` (if using QLoRA), `datasets`, `accelerate`.
- A single GPU with ≥ 16 GB VRAM for a 1B-class RM; ≥ 24 GB (QLoRA) or ≥ 40 GB (BF16 LoRA) for a 7B RM.
- Familiarity with the mod-104 LoRA / QLoRA recipe — this exercise reuses it.

## Requirements

### Part A — Pick a base for the RM

Two reasonable starting points:

- **Same base as your SFT model** (e.g., the base of `meta-llama/Llama-3.2-1B` or `Qwen/Qwen2.5-1.5B`). Simpler and traditional.
- **A slightly stronger open model of similar family** (e.g., an already-instruction-tuned variant). Often gives a stronger RM.

Pin the model ID and revision hash. Note that using an *identical* base for RM and policy (upcoming PPO run) can help — reward-model / policy interference is a real concern at scale (Coste et al., 2023) but not visible at exercise scale.

### Part B — Configure `RewardTrainer`

Wire up TRL's `RewardTrainer` following the chapter-3 recipe. Include:

- `AutoModelForSequenceClassification.from_pretrained(BASE_ID, num_labels=1, torch_dtype="bfloat16")`.
- Set `model.config.pad_token_id = tokenizer.pad_token_id`.
- `RewardConfig` with `learning_rate=5e-6` (adjust for LoRA / QLoRA — go to 1e-4 range), `num_train_epochs=1`, `bf16=True`, `max_length=1024`, `center_rewards_coefficient=0.01`.
- If LoRA / QLoRA: `LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, target_modules=[...attn+mlp...], modules_to_save=["score"])`. The `modules_to_save=["score"]` is critical — the classifier head must be fully trainable. Verify the actual head module name for your architecture; some are `classifier`, some are `v_head`.

Load train and eval splits from `outputs/preference-set-v1/` into the trainer as `train_dataset` and `eval_dataset`.

### Part C — Train and log

Run one epoch with `eval_strategy="steps"`, `eval_steps=200`, `logging_steps=25`. Capture:

- `train/loss` and `eval/loss` per step.
- Peak VRAM (`torch.cuda.max_memory_allocated()`).
- Wall-clock training time.
- Final adapter (if LoRA) or full model (if full-parameter) at `outputs/reward-model/`.

If your run explodes (loss → NaN, or `eval/loss` immediately plateaus), the standard first-things-to-try are: (1) reduce LR by 5×, (2) enable gradient checkpointing, (3) check that `pad_token_id` is set on the model config.

### Part D — Evaluate on `eval_benchmark`

The `eval_benchmark` split from exercise 01 is what you use here — the split neither the trainer nor exercise 03 will see.

Write an eval script `eval_rm.py` that loads the trained RM and computes:

1. **Pairwise accuracy.** For each row: score `chosen` and `rejected`; the RM is correct iff `chosen_reward > rejected_reward`. Report the aggregate.
2. **Calibration.** Bin pairs by `chosen_reward − rejected_reward` (5 bins by quantile). Compute per-bin accuracy. Report the table and a small `matplotlib` plot.
3. **Length-conditional accuracy.** Compute pairwise accuracy separately for pairs where `len(chosen) > len(rejected)` vs. `len(chosen) < len(rejected)`. Drop ties.
4. **Reward distribution.** Histogram of `chosen_reward` and `rejected_reward` on the benchmark split, overlaid. Report shape (well-separated / overlapping / bimodal).
5. **Best-of-N sanity check** (small): pick 20 canned prompts from the benchmark, sample 4 responses from your SFT model at temperature 0.9 each, score them with the RM, and pick the argmax. Read the outputs. Are the highest-scored responses visibly better?

Save all outputs (numbers, tables, plots, best-of-4 transcripts) to `outputs/reward-model/eval/`.

### Part E — Write the RM report

`outputs/reward-model/RM_REPORT.md`, ~1 page (~500–700 words). Include:

- **Base + config.** Base model ID + revision, LoRA config or full-parameter, LR / batch size / epochs / seed.
- **Dataset.** Reference to your `outputs/preference-set-v1/` version.
- **Training curves.** A single PNG (or Markdown-embedded) plot of train + eval loss over steps.
- **Final numbers.** Pairwise accuracy on `eval_benchmark`, overall and length-conditional (chosen-longer / chosen-shorter). Calibration table (5 bins). Reward histogram description.
- **Best-of-4 transcript sample.** 2–3 example prompts and the RM's ranking of the four responses.
- **Failure-mode audit.** Score three canned "sycophancy" prompts (leading questions with false premises), three "refusal" prompts (things a model should not refuse), and three "format" prompts (same content in prose vs bullet form). Report whether the RM shows any bias.
- **Compute + wall-clock + peak VRAM.**
- **Known limitations.** Where you think the RM is likely to fail; what data would improve it.

## Starter guidance

- **The `modules_to_save=["score"]` line is load-bearing.** Forget it and your classifier head is a randomly-initialized 4096→1 linear that never trains. You will get 50% pairwise accuracy and blame the base model.
- **RM overfits fast.** Watch the eval-loss curve; if eval starts climbing while train drops, stop the run. One epoch is typically optimal.
- **Length bias is your top risk.** If your length-conditional accuracy shows a >10-point gap between chosen-longer and chosen-shorter, you have a length-biased RM. Note this in the report — it will bite in exercise 03.
- **Do not skip the failure-mode audit.** Three sycophancy / refusal / format prompts is 15 minutes of work and it catches things the aggregate numbers hide.
- **RM training loss values are less interpretable than SFT loss.** The Bradley-Terry loss on random guessing is `log(2) ≈ 0.693`. A well-trained RM has train loss in the 0.3–0.5 range. Below 0.1 on train usually means overfit.
- **If your base is small (< 1B), pairwise accuracy in the 60–70% range is expected**; do not chase 90%. Larger bases (7B) can reach 75–80% on UltraFeedback with a good LoRA config.

## Acceptance criteria

You have completed the exercise when:

- `outputs/reward-model/` contains the trained RM (adapter or full weights), `eval/` with all evaluation artifacts, and `RM_REPORT.md`.
- The RM's pairwise accuracy on `eval_benchmark` is reported and >55% (below that, retrain).
- Length-conditional accuracy is reported *separately* for chosen-longer and chosen-shorter pairs. The gap between them is documented (even if it is large — documenting a length-biased RM is a pass; not documenting is a fail).
- Calibration table exists and has 5 non-empty bins.
- The failure-mode audit is filled in with actual scores on the 9 canned prompts.
- The RM loads cleanly from disk in a fresh session and can score a prompt/response pair via `AutoModelForSequenceClassification.from_pretrained(...)`.

## Stretch goals

- **Full-parameter RM comparison.** Retrain the RM at full-parameter (no LoRA) with everything else identical; compare pairwise accuracy, length bias, and wall-clock. Report the delta and speculate on the cause.
- **RewardBench evaluation.** Run your RM through the `RewardBench` benchmark (see `resources.md`). Report your model's per-category scores. This gives a public-comparable number.
- **Reward-model ensemble.** Train two RMs with different seeds; ensemble by averaging their scores; measure accuracy against a single RM. Coste et al., 2023 argues this reduces reward-hacking; you can measure whether the ensemble's length-bias gap is smaller than the single RM's.
- **Length-penalty variant.** Add a length penalty to the reward loss: `L = L_BT + α · (r_chosen - r_rejected) · sign(len(chosen) - len(rejected))`. Train; compare length-conditional accuracy. This is a directly-testable mitigation for exercise 07's length-bias section.
- **Iterative refresh preview.** Sample 500 responses from your SFT model on unseen prompts, hand-label 50 of them (or use a stronger LLM as judge), add to your training set, retrain the RM. Compare the OOD reward distribution before and after.
