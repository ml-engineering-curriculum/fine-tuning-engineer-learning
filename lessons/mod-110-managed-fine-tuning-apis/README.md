# mod-110-managed-fine-tuning-apis: Managed Fine-Tuning APIs

**Estimated effort:** 9 hours
**Position in the track:** tenth module. Sibling to mod-103 (SFT), mod-104 (LoRA / QLoRA), mod-105 (preference optimization), and mod-109 (quantization / serving) — same product question ("get a fine-tune into production"), different answer. The training loop is someone else's; the dataset engineering (mod-102), the eval harness (mod-108), and the platform gates (mod-111) are still yours. This module is the bridge between "we have SFT data and an eval gate" and "the fine-tune lives behind OpenAI's / Anthropic's / Bedrock's endpoint instead of our own vLLM."

You already know how to fine-tune a model end-to-end on your own hardware. The question this module answers is: **when should you not?** Sometimes the answer is capability — no open base you can afford to fine-tune matches GPT-4.1 or Claude Haiku 4.5 on your task, and the productive move is to fine-tune the closed model through the vendor's API. Sometimes it is speed — a managed job runs in an hour, requires no GPU procurement, and gets you a shipped model by end-of-day. Sometimes it is the reverse: cost, IP ownership, data residency, and portability push you back to self-hosted every time. This module gives you the concrete mechanics for the first case and the reasoning framework for choosing between them.

The through-line: **a managed fine-tune is still a fine-tune.** The dataset engineering from mod-102 is unchanged. The eval harness from mod-108 is unchanged (though the backend flips from `hf` to `openai-chat-completions`). The safety review from mod-106 is unchanged. What changes is (a) who owns the training loop, (b) what artifact you get at the end — a model ID pointing at a hosted endpoint, not a weight file on disk — and (c) which trade-offs move from "engineering decision" to "vendor contract." This module makes those changes explicit and shows how to keep the rest of your pipeline agnostic to which side of the decision you land on.

## Learning objectives

- Run a managed fine-tuning job on OpenAI (and at least one of Anthropic, Gemini, Mistral, Cohere, Bedrock, Azure AI Foundry) — file upload, job creation, hyperparameter selection, monitoring, and inference against the resulting model ID.
- Convert an SFT dataset between Hugging Face and managed-API JSONL formats correctly — messages vs. prompt/completion, system prompts, tool schemas, chat-template resolution, per-vendor field quirks.
- Reason about cost, data residency, IP, and privacy implications of managed vs. self-hosted fine-tuning — training vs. inference cost decomposition, retention windows, regional pinning, ownership of the fine-tuned weights, HIPAA / GDPR / SOC2 posture, and the DPA / opt-out flags that matter.
- Decide between fine-tuning a closed-weight frontier model and fine-tuning an open base model — the capability-ceiling / cost / portability / lock-in axes, when each answer is correct, and how to write down the decision so it survives the next model release.
- Hand off eval to the same eval harness used for self-hosted adapters so comparisons are fair — running the mod-108 release gate against a hosted endpoint, matching sampling and chat-template settings, the log-likelihood-task gap, and cost-controlled sampling strategies.

## Lecture chapters

1. [`01-managed-ft-landscape.md`](01-managed-ft-landscape.md) — The Managed Fine-Tuning Landscape. Why managed exists, the seven vendors that matter in 2026, the shared shape of every managed FT API (upload → validate → job → checkpoint → inference), and the decision axis that this module sharpens across every subsequent chapter.
2. [`02-openai-end-to-end.md`](02-openai-end-to-end.md) — Running a Managed Fine-Tune on OpenAI End-to-End. The Files / Fine-Tunes API, JSONL schema for chat models, hyperparameter surface (`n_epochs`, `learning_rate_multiplier`, `batch_size`, `suffix`, method selection), monitoring via events, integration validation, cost math, and the operational patterns for shipping the resulting `ft:...` model ID.
3. [`03-vendor-tour.md`](03-vendor-tour.md) — A Tour of the Other Managed Vendors. Anthropic (Claude fine-tuning via Bedrock and the Claude API), Google Vertex AI Gemini tuning, Mistral La Plateforme, Cohere Command, AWS Bedrock Custom Models, Azure AI Foundry — what each supports, what method (SFT / DPO / adapter), what data format, what artifact ships, and the ways they differ from OpenAI.
4. [`04-dataset-format-conversion.md`](04-dataset-format-conversion.md) — Converting Datasets Between Hugging Face and Managed JSONL. From a `datasets.Dataset` in messages format to per-vendor JSONL and back, tool schemas, system prompts, multi-turn conventions, image / multimodal rows, per-vendor field quirks and validators, and the round-trip test that catches the ten most common conversion bugs.
5. [`05-cost-privacy-ip-residency.md`](05-cost-privacy-ip-residency.md) — Cost, Data Residency, IP, and Privacy Trade-offs. The four-part cost model (training compute, hosting rent, per-token inference, egress), retention and training-on-your-data flags per vendor, regional pinning and the residency map, ownership of the fine-tuned weights, and the compliance surface (HIPAA / GDPR / SOC2 / EU AI Act) that the platform team needs disclosed.
6. [`06-closed-frontier-vs-open-base.md`](06-closed-frontier-vs-open-base.md) — Fine-Tuning a Closed Frontier vs. Fine-Tuning an Open Base. The five-axis decision (capability ceiling, cost, portability, method flexibility, dependability), the concrete regimes where each answer is correct, the escape hatches that let you switch later without rebuilding, and the ADR template that records the decision so the next model release does not restart the argument.
7. [`07-eval-handoff-across-backends.md`](07-eval-handoff-across-backends.md) — Feeding Managed Fine-Tunes Through the Same Eval Harness. Running mod-108's release gate against a hosted endpoint via `lm-evaluation-harness`'s `openai-chat-completions` / `local-chat-completions` / `anthropic-chat-completions` backends, the log-likelihood-vs-chat-only gap, matching sampling and chat-template settings across backends, cost-controlled sampling, and the released-artifact hash equivalent for a hosted model ID.

## Exercises

Each exercise is a hands-on companion to one or more chapters. They live in [`exercises/`](exercises/).

- [`exercise-01-openai-fine-tuning-end-to-end.md`](exercises/exercise-01-openai-fine-tuning-end-to-end.md) — take a converted SFT dataset all the way through the OpenAI fine-tuning API, from file upload to a running `ft:...` model ID that you can call from `openai` and evaluate with the mod-108 gate. The tour-of-the-mechanics exercise.
- [`exercise-02-managed-vs-self-hosted-cost-and-privacy-matrix.md`](exercises/exercise-02-managed-vs-self-hosted-cost-and-privacy-matrix.md) — for one concrete deployment scenario, build a defended cost / privacy / IP / residency matrix comparing your own self-hosted stack from mod-109 against two or more managed vendors, and write the decision memo. The reasoning exercise.
- [`exercise-03-shared-eval-harness-across-vendors.md`](exercises/exercise-03-shared-eval-harness-across-vendors.md) — configure the mod-108 release gate to run against a self-hosted vLLM endpoint AND a hosted OpenAI / Anthropic / Vertex fine-tune, reconcile the settings so the numbers are actually comparable, and produce a like-for-like leaderboard. The fair-comparison exercise.

## Labs and quizzes

Long-form labs land in [`labs/`](labs/) and the knowledge-check quiz in [`quizzes/`](quizzes/) on subsequent authoring cycles.

## Resources

See [`resources.md`](resources.md) for the primary docs (OpenAI Fine-tuning guide, Anthropic Claude fine-tuning docs, Vertex AI tuning guide, Mistral La Plateforme fine-tuning, Cohere fine-tuning, AWS Bedrock Custom Models, Azure AI Foundry fine-tuning), the JSONL schema references per vendor, the `lm-evaluation-harness` API-model backends, and the governance references (DPAs, residency maps, HIPAA BAAs, EU AI Act guidance) cited across the chapters.

## What comes before

- **mod-102 — SFT Dataset Engineering** owns the dataset. This module consumes a `messages` dataset from mod-102 chapter 3 and converts it to per-vendor JSONL in chapter 4. Every dataset-quality guarantee (deduplication, decontamination, licensing trace, dataset card) carries forward regardless of who runs the training loop.
- **mod-108 — Evaluating Fine-Tuned Models** owns the release gate. Chapter 7 of this module is entirely about running that gate against a hosted endpoint instead of a local checkpoint, so managed and self-hosted candidates land on the same leaderboard.
- **mod-109 — Quantization, Export, and Serving** owns the self-hosted side of the "how does the fine-tune reach production?" question. The cost-per-token and latency numbers from mod-109 chapter 7 are the baseline this module's chapter 5 compares managed vendors against.

## What comes next

- **mod-111 — Post-Training Systems Design** is where the managed-vs-self-hosted decision stops being a per-project ADR and becomes a platform property: a model registry that indexes both `ft:...` IDs and internal artifact hashes, a routing layer that abstracts over the backend, a cost dashboard that ties per-request dollars to a tenant, and the governance workflow for approving a new vendor. This module produces the individual decisions; mod-111 consumes them at scale.
