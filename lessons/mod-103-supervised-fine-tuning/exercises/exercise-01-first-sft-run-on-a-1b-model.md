# exercise-01: First SFT Run on a 1B Model

**Estimated effort:** 4 hours
**Companion chapters:** [`01-sft-training-loop-anatomy.md`](../01-sft-training-loop-anatomy.md), [`03-learning-rate-warmup-and-checkpoint-cadence.md`](../03-learning-rate-warmup-and-checkpoint-cadence.md)

## Objective

Run a real, end-to-end supervised fine-tuning job on a 1B-class base model using TRL's `SFTTrainer` (or the equivalent Axolotl YAML config), on a small open dataset, and produce (a) a trained checkpoint, (b) the config that produced it, and (c) a "was this run good?" answer defended in one paragraph against the two-plot rule from chapter 4.

The goal is not the model. The goal is the loop. By the end you should be able to point at every line of the config and say what it does and why you chose that value.

## Prerequisites

- Access to a single GPU with ≥ 16 GB VRAM (a T4, A10, RTX 3090 / 4090, or L4 is more than enough). If you are on CPU only, follow the "no-GPU fallback" note in *Starter guidance*.
- Python 3.10+ with `torch`, `transformers`, `trl`, `accelerate`, `datasets`. Use recent versions; pin them for reproducibility (exercise 05 makes this a hard requirement).
- Hugging Face Hub access with any gated-model licenses accepted (Llama 3.x requires a click-through).
- Optional: `wandb` or `tensorboard` for logging; `flash_attn` for a throughput boost on Ampere+ GPUs.

## Requirements

### Part A — Choose your base and dataset

Pick ONE base and ONE dataset. Do not switch mid-exercise.

Suggested bases (all 1B-class, all publicly downloadable):

- `meta-llama/Llama-3.2-1B` (base, not Instruct)
- `Qwen/Qwen2.5-1.5B`
- `HuggingFaceTB/SmolLM2-1.7B`

Suggested datasets (small enough to train in an hour on a single GPU):

- `HuggingFaceH4/no_robots` — 10k human-authored pairs, `train_sft` / `test_sft` splits already exist.
- `GAIR/lima` — 1k pairs, single-turn.
- A ~5k-row slice of `OpenAssistant/oasst1` filtered to English assistant-terminating threads.
- The `curated.jsonl` output of exercise 01 in mod-102, if you have it.

Whichever you pick, use the **base** checkpoint, not the Instruct variant. The point of this exercise is to *create* instruction-following capability, not to modify it.

### Part B — Write the training script

Author `train.py` following the chapter-01 minimum viable script as a starting point. Requirements:

1. Load the base model at BF16 with `attn_implementation="flash_attention_2"` (or `"sdpa"` if FA2 is not installed).
2. Load and, if needed, set the tokenizer's chat template. Assert `tokenizer.chat_template is not None` before training.
3. Prepare train and eval splits from the chosen dataset. Eval must be disjoint from train — verify with a hash-set check, not trust.
4. Configure `SFTConfig` with the following values *derived* from the reasoning in chapters 2 and 3 (not copy-pasted from the chapter examples). You must be able to defend each in the report:
   - `num_train_epochs`
   - `per_device_train_batch_size` × `gradient_accumulation_steps` giving your target effective tokens/step (chapter 2 formula)
   - `learning_rate`, `lr_scheduler_type`, `warmup_ratio` or `warmup_steps`
   - `weight_decay`
   - `max_length`, `packing`
   - `bf16`
   - `save_steps`, `eval_steps`, `save_total_limit`, `load_best_model_at_end`, `metric_for_best_model`, `greater_is_better`
   - `logging_steps`
   - `seed`, `data_seed`
5. Wire up the trainer, log to at least one persistent backend (`report_to=["wandb"]` or a `JSONLogger` callback writing to `outputs/logs/train.jsonl`).
6. Run the three sanity checks from chapter 1 *before* the full training:
   - Print one collated batch and confirm `labels` is `-100` on non-assistant tokens.
   - Run one forward pass and confirm loss is finite and in the 2–5 range.
   - Run `max_steps=10` and confirm loss decreases.

Only after the three sanity checks pass do you run the full `trainer.train()`.

### Part C — Ship the artifacts

At end of run, your `outputs/` directory should contain, at minimum:

- `final/` — the checkpoint (weights + tokenizer).
- `logs/train.jsonl` and `logs/eval.jsonl` (or your logger's equivalent).
- `config.yaml` — the exact `SFTConfig.to_dict()` output, plus your own base-model and dataset choices.
- `plots/loss.png` — the two-curve (train + eval) plot from chapter 4.

Do not delete intermediate checkpoints yet; you will use them in later exercises. Exercise 05 introduces the full manifest.

### Part D — Write the one-paragraph diagnosis

At the very end of the run, look at `plots/loss.png` and write a single paragraph (5–8 sentences) in `outputs/diagnosis.md` answering:

1. Which chapter-4 shape (healthy / one of the seven deviations) does the curve match?
2. What are the numeric guideposts (step-0 loss, final loss, best eval loss and its step)?
3. Do train and eval agree? Any sign of overfitting?
4. Is this a "call it done, move on" run or a "something is off, re-run" run?
5. If re-run — what is the *one* config change you would make first?

This is the deliverable that matters most. A trained checkpoint you cannot diagnose is not a completed exercise.

## Starter guidance

- **No-GPU fallback.** You can run this exercise on CPU with `Qwen/Qwen2.5-0.5B` (a smaller base than the 1B target) at `max_length=512` and `per_device_train_batch_size=1` on a 100-row slice — expect ~30 minutes for a 200-step run. The diagnosis question is answerable; the loss curves are noisier. Say so in your report.
- **Do not skip the sanity checks.** Every failed real run I have seen started with someone skipping the batch-decode step to save five minutes.
- **Use `pad_token = eos_token` only after verifying padding is masked.** For most modern tokenizers, `tokenizer.pad_token = tokenizer.eos_token` is the standard fix if `pad_token` is unset. Verify by inspecting one right-padded batch — pad positions must be `-100` in `labels`.
- **If training is slow.** Walk the seven-item checklist in chapter 2 before assuming the model or dataset is the problem.
- **Do not use LoRA.** LoRA is mod-104. This exercise is deliberately about the plain full-parameter loop so you see the numeric ranges from chapter 3 firsthand.

## Acceptance criteria

You have completed the exercise when:

- `train.py`, `config.yaml`, and `outputs/` (checkpoint, logs, plots, diagnosis) all exist and are self-consistent.
- The three sanity checks (batch decode, one forward pass, 10-step run) are visible in your run log — you should be able to point at the specific commit or notebook cell where each ran.
- The two-plot view exists and shows both curves; the plot script is committed alongside `train.py`.
- `diagnosis.md` classifies the run against the chapter-4 shape guide with numeric guideposts and a defensible next-step recommendation.
- A colleague could re-run the same script and hit within ~5% of your loss curve.
- The final checkpoint loads via `AutoModelForCausalLM.from_pretrained("outputs/final")` and produces a coherent generation for a canned prompt.

## Stretch goals

- **Compare against the base.** Generate 5 responses from the base and 5 from your fine-tuned model on the same prompts. Which are visibly better? Which are subtly worse? Write one paragraph.
- **Redo the run with Axolotl.** Write the equivalent YAML config and run it. Diff the loss curves against your TRL run. Bit-identical is not the goal; understanding the diff is.
- **Redo the run with Unsloth.** If your base is on Unsloth's supported list, swap `AutoModelForCausalLM` for `FastLanguageModel`, keep everything else the same, and report the throughput delta.
- **Do the LR ablation.** Bump the LR by 5× and re-run. Confirm chapter 3's "much higher LR degrades runs of a well-trained base" claim with your own loss curves. This is a natural setup for exercise 03.
- **Log per-example loss** on the eval set at the last eval step, join back to the dataset metadata (source, category), and produce a small table of eval loss by category. This is the analysis that pays off six months later when you are debugging a downstream regression.
