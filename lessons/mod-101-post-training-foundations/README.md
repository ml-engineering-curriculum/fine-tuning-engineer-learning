# mod-101-post-training-foundations: Post-Training Foundations: Transformer Internals for Fine-Tuners

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 12 hours

## Learning objectives

- Trace a forward pass through a decoder-only transformer and identify the layers that PEFT methods target
- Reason about tokenization (BPE, SentencePiece, tiktoken), chat templates, and special-token budgets
- Compare positional encodings (absolute, RoPE, ALiBi) and explain why they matter for context-length fine-tuning
- Map the pretrain → SFT → preference-optimization → quantization → serving pipeline end-to-end

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
