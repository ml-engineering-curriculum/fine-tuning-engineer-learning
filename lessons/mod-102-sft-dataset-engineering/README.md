# mod-102-sft-dataset-engineering: SFT Dataset Engineering: From Raw Conversations to Training-Ready Mixes

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 16 hours

## Learning objectives

- Curate, deduplicate, and decontaminate an SFT dataset against eval suites (MMLU, HumanEval, GSM8K, MT-Bench)
- Design chat templates and multi-turn conversation construction for a target base model family
- Generate synthetic instruction data with Self-Instruct / Evol-Instruct style pipelines, with provenance and quality filters
- Mix datasets across capabilities (reasoning, code, safety, persona) with documented mixing ratios
- Track dataset licensing and provenance well enough to ship a defensible model card

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
