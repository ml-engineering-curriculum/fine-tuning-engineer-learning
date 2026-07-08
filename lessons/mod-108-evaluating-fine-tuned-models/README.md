# mod-108-evaluating-fine-tuned-models: Evaluating Fine-Tuned Models

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 15 hours

## Learning objectives

- Run the EleutherAI `lm-evaluation-harness` against a fine-tuned model for academic benchmarks (MMLU, ARC, GSM8K, HumanEval, TruthfulQA, IFEval)
- Build task-specific held-out evals tied to the production use case
- Design an LLM-as-judge protocol with MT-Bench-style rubrics, pairwise comparison, and position-bias controls
- Detect benchmark contamination and prevent it via decontamination at dataset-engineering time
- Run a release-gate eval suite and write a regression report against the previous adapter

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
