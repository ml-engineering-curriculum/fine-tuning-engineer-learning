# mod-102-sft-dataset-engineering: SFT Dataset Engineering: From Raw Conversations to Training-Ready Mixes

**Estimated effort:** 16 hours
**Position in the track:** second module. Builds on mod-101's tokenizer / chat-template understanding, and feeds directly into mod-103's SFT training loop.

Supervised fine-tuning is a data problem wearing a training-loop mask. The trainer is a straight next-token cross-entropy over a specific selection of tokens; almost every quality win or loss traces back to the dataset that went in. This module makes you the person on the team who can say — with receipts — what is in your SFT mix, where it came from, what was removed, why the ratios are what they are, and which evals it is legal to run afterwards.

## Learning objectives

- Curate, deduplicate, and decontaminate an SFT dataset against eval suites (MMLU, HumanEval, GSM8K, MT-Bench)
- Design chat templates and multi-turn conversation construction for a target base model family
- Generate synthetic instruction data with Self-Instruct / Evol-Instruct style pipelines, with provenance and quality filters
- Mix datasets across capabilities (reasoning, code, safety, persona) with documented mixing ratios
- Track dataset licensing and provenance well enough to ship a defensible model card

## Lecture chapters

1. [`01-curation-and-quality-filtering.md`](01-curation-and-quality-filtering.md) — Curation and Quality Filtering for SFT. What makes one instruction–response pair worth training on and the next worth dropping; the per-example metadata schema you should carry from ingest to model card; the small set of filters (length, language, PII, toxicity, refusal, response-quality) that most SFT mixes actually run.
2. [`02-deduplication.md`](02-deduplication.md) — Deduplication: Exact, Near, and Semantic. Why duplicates matter more for SFT than for pretraining; exact hashing; MinHash + LSH for Jaccard near-duplicates; embedding-based semantic dedup; the cross-source dedup trap.
3. [`03-eval-decontamination.md`](03-eval-decontamination.md) — Decontamination Against the Big-Four Evals. What contamination is, why it is trivially easy to leak MMLU / HumanEval / GSM8K / MT-Bench prompts through synthetic pipelines, and the n-gram-overlap recipe (Brown et al., 2020; Llama 2 / 3 model cards) you should run before every SFT release.
4. [`04-chat-templates-and-multi-turn.md`](04-chat-templates-and-multi-turn.md) — Chat Templates and Multi-Turn Conversation Construction. Loss masking on assistant tokens only, prompt-response splitting, multi-turn packing, tool-call and multi-message assistant turns, and the exact `apply_chat_template` invocations for training vs. inference.
5. [`05-synthetic-data-self-instruct-evol-instruct.md`](05-synthetic-data-self-instruct-evol-instruct.md) — Synthetic Instruction Data: Self-Instruct and Evol-Instruct. The Self-Instruct bootstrapping loop (Wang et al., 2023), Evol-Instruct's depth / breadth evolutions (Xu et al., 2023), the quality-filter and provenance stack that keeps a synthetic pipeline honest, and the licensing question you must answer *before* you generate from a proprietary teacher.
6. [`06-dataset-mixing.md`](06-dataset-mixing.md) — Mixing Datasets Across Capabilities. Per-source weights, temperature sampling across sources of very different sizes, capability categories (reasoning / code / safety / general chat / persona), the "safety data catastrophic forgetting" tradeoff, and how to actually ablate a mix.
7. [`07-licensing-provenance-model-card.md`](07-licensing-provenance-model-card.md) — Licensing, Provenance, and the Model Card. Reading a dataset license as an engineer (not a lawyer), the common license buckets you will meet on the Hugging Face Hub, tracking per-example provenance so a downstream model card is auditable, and the sections of a defensible SFT model card.

## Exercises

Each exercise is a hands-on companion to one or more chapters. They live in [`exercises/`](exercises/).

- [`exercise-01-sft-data-quality-audit.md`](exercises/exercise-01-sft-data-quality-audit.md) — audit a real open SFT corpus (OpenAssistant, No Robots, LIMA, or similar) against a written per-example quality rubric, compute the drop rate for each filter, and produce a before/after quality report.
- [`exercise-02-eval-set-decontamination.md`](exercises/exercise-02-eval-set-decontamination.md) — implement an n-gram-overlap decontamination scan of an SFT dataset against MMLU, HumanEval, GSM8K, and MT-Bench; report which rows would be removed and what the aggregate leak rate is.
- [`exercise-03-chat-template-and-multi-turn-formatting.md`](exercises/exercise-03-chat-template-and-multi-turn-formatting.md) — take a raw multi-turn conversation and produce a byte-identical training example (prompt tokens masked to `-100`, assistant tokens contributing to loss) for a Llama-family and a ChatML base.
- [`exercise-04-synthetic-data-with-self-instruct.md`](exercises/exercise-04-synthetic-data-with-self-instruct.md) — implement a small Self-Instruct / Evol-Instruct pipeline from a small seed set, wire in three quality filters, and write the provenance record for every generated example.
- [`exercise-05-dataset-mix-card-and-licensing-trace.md`](exercises/exercise-05-dataset-mix-card-and-licensing-trace.md) — design and document a 100k-example SFT mix across at least four capability categories; produce the dataset-mix card and per-source licensing trace that would ship with the model.

## Labs and quizzes

Long-form labs land in [`labs/`](labs/) and the knowledge-check quiz in [`quizzes/`](quizzes/) on subsequent authoring cycles.

## Resources

See [`resources.md`](resources.md) for primary papers, dataset cards, official tooling docs, and the license references cited across the chapters.

## What comes next

- **mod-103 — Supervised Fine-Tuning End-to-End** consumes the training-ready mix you produce here. The chat-template masking rules in chapter 4 are exactly what its trainer expects.
- **mod-105 — Preference Optimization** builds preference pairs on top of the SFT model produced from a mix you designed here.
- **mod-106 — Safety Alignment Tuning** revisits the "safety data catastrophic forgetting" tradeoff of chapter 6 in more depth.
- **mod-108 — Evaluating Fine-Tuned Models** runs the very eval suites you decontaminate against in chapter 3. Contamination discovered late is a release blocker.
