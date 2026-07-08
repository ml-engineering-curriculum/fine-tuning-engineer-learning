# mod-109-quantization-export-serving: Quantization, Export, and Serving Adapted Models

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 12 hours

## Learning objectives

- Apply post-training quantization (GPTQ, AWQ, bitsandbytes 8-bit / 4-bit) and measure quality / latency / VRAM impact
- Choose between adapter-merging vs. LoRA-aware serving (vLLM multi-LoRA) per use case
- Deploy a fine-tuned model behind vLLM and Text Generation Inference (TGI) with a structured serving config
- Export to GGUF for llama.cpp / CPU / edge inference and verify equivalence against the source weights
- Reason about latency / throughput / cost trade-offs and write a serving runbook

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
