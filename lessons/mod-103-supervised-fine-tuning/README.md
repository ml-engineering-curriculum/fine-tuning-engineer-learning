# mod-103-supervised-fine-tuning: Supervised Fine-Tuning End-to-End

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 18 hours

## Learning objectives

- Run an SFT job on a 1B–8B base model with TRL `SFTTrainer` (or equivalent in Axolotl / Unsloth)
- Implement prompt/response masking, sequence packing, and gradient accumulation correctly
- Pick learning-rate schedule, warmup, weight decay, and checkpoint cadence by reasoning about loss curves
- Diagnose overfitting, catastrophic forgetting, and template-leak failure modes
- Produce a reproducible run (seeds, config, env, dataset hash) and a 1-page training report

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
