# mod-101-post-training-foundations: Post-Training Foundations: Transformer Internals for Fine-Tuners

**Estimated effort:** 12 hours
**Position in the track:** first module. Sets the mental model every subsequent module builds on.

Before you fine-tune anything, you need to be able to point at a specific tensor in a specific layer of a specific base model and say "this is what LoRA changes, this is what SFT rewrites, this is what my chat template touches." This module gives you that vocabulary.

## Learning objectives

- Trace a forward pass through a decoder-only transformer and identify the layers that PEFT methods target
- Reason about tokenization (BPE, SentencePiece, tiktoken), chat templates, and special-token budgets
- Compare positional encodings (absolute, RoPE, ALiBi) and explain why they matter for context-length fine-tuning
- Map the pretrain → SFT → preference-optimization → quantization → serving pipeline end-to-end

## Lecture chapters

1. [`01-decoder-only-forward-pass.md`](01-decoder-only-forward-pass.md) — The Decoder-Only Forward Pass (and Where PEFT Hooks In). Walks the forward pass through a Llama-style decoder-only stack, names every `nn.Linear` per block, and labels which layers each PEFT family (LoRA, QLoRA, IA3, prefix / prompt tuning, BitFit, adapter modules) touches.
2. [`02-tokenization-and-chat-templates.md`](02-tokenization-and-chat-templates.md) — Tokenization, Chat Templates, and the Special-Token Budget. Compares BPE, SentencePiece, and tiktoken; walks the Hugging Face `apply_chat_template` API; enumerates the three tokenization footguns that silently break SFT runs.
3. [`03-positional-encodings.md`](03-positional-encodings.md) — Positional Encodings: Absolute, RoPE, and ALiBi. Explains why attention needs a position channel, contrasts the three families in wide use, and gives the exact levers (`rope_theta`, `rope_scaling`, PI / NTK / YaRN) a fine-tuner uses to extend context length.
4. [`04-post-training-pipeline-map.md`](04-post-training-pipeline-map.md) — The Post-Training Pipeline, End to End. Names every stage between pretrained base and served adapter, describes the artifact contract between stages, and maps each stage to the module in this track that owns it.

## Exercises

Each exercise is a hands-on companion to a chapter. They live in [`exercises/`](exercises/).

- [`exercise-01-decoder-only-forward-pass-walkthrough.md`](exercises/exercise-01-decoder-only-forward-pass-walkthrough.md) — instrument a real 1B-class Llama-family model, produce a labeled shape trace of every intermediate tensor, and identify by name every `nn.Linear` a default QLoRA config would target.
- [`exercise-02-tokenizer-and-chat-template-audit.md`](exercises/exercise-02-tokenizer-and-chat-template-audit.md) — audit and diff the chat templates of three open base models; find at least one real-world tokenization footgun in a public tokenizer config.
- [`exercise-03-rope-vs-alibi-context-length-experiment.md`](exercises/exercise-03-rope-vs-alibi-context-length-experiment.md) — run a RoPE base past its trained context length with and without scaling; repeat on an ALiBi base; write up what you observed and why.
- [`exercise-04-post-training-pipeline-map.md`](exercises/exercise-04-post-training-pipeline-map.md) — produce a one-page pipeline map for a realistic post-training project, with go/no-go criteria at each gate.

## Labs and quizzes

Long-form labs land in [`labs/`](labs/) and the knowledge-check quiz in [`quizzes/`](quizzes/) on subsequent authoring cycles.

## Resources

See [`resources.md`](resources.md) for primary papers, official documentation, and reading lists.

## What comes next

- **mod-102 — SFT Dataset Engineering** takes the tokenization and chat-template understanding from this module and turns it into a dataset-building discipline.
- **mod-103 — Supervised Fine-Tuning End-to-End** puts the forward pass to work under a training loop.
- **mod-104 — PEFT: LoRA, QLoRA, and Friends** attaches to the same `nn.Linear`s you learn to name here.
- **mod-107 — Distributed Fine-Tuning** uses the positional-encoding levers from chapter 3 to extend context length under real cluster constraints.
