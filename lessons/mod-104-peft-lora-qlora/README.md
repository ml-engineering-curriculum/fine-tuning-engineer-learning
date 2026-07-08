# mod-104-peft-lora-qlora: Parameter-Efficient Fine-Tuning: LoRA, QLoRA, and Friends

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 16 hours

## Learning objectives

- Implement LoRA from scratch (low-rank decomposition, scaling) and explain rank / alpha / target-module trade-offs
- Run QLoRA on a 7B–13B model on a single 24-GB GPU using bitsandbytes 4-bit NF4 base weights
- Merge adapters back into base weights, manage multi-adapter routing, and serve LoRA-aware
- Compare LoRA, QLoRA, IA3, prefix tuning, and prompt tuning and pick the right one per use case
- Diagnose adapter-specific failure modes: rank collapse, target-module mis-selection, alpha mis-scaling

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
