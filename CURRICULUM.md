# Fine-Tuning Engineer Curriculum

**Role level:** 30 (deep specialist — peer to Senior ML Engineer on the ladder)
**Status:** planned — modules and projects below are the planned scope authored from [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json). Lessons and projects will be drafted by subsequent autonomous content cycles.

## Overview

This track teaches the post-training stack end-to-end for a deep-specialist Fine-Tuning Engineer: dataset engineering → supervised fine-tuning (SFT) → parameter-efficient methods (LoRA / QLoRA / adapters) → preference optimization (RLHF / DPO / ORPO / KTO / GRPO) → safety tuning → distributed fine-tuning at the operator altitude → evaluation of fine-tuned models → quantization, export, and serving → managed fine-tuning APIs → release-level systems design.

It is a **specialist track at level 30** — it inherits ML / PyTorch / packaging fundamentals from the build-altitude [`ml-engineer-learning`](https://github.com/ml-engineering-curriculum/ml-engineer-learning) track and operates — but does not build — the distributed-training platforms owned by [`training-pipeline-engineer-learning`](https://github.com/ml-engineering-curriculum/training-pipeline-engineer-learning).

Total planned commitment: **157 hours** across 11 modules + **135 hours** across 3 projects = **~292 hours**.

## Ownership rule

Following the project-wide ownership rule, this curriculum:

- **Owns** the post-training stack end-to-end at depth — SFT, PEFT, preference optimization, safety tuning, eval, quantization, serving of adapted models, and managed fine-tuning workflow.
- **Defers down** to [`ml-engineer-learning`](https://github.com/ml-engineering-curriculum/ml-engineer-learning) (level 20) for the build-altitude practitioner foundation (Python / PyTorch / FastAPI / Docker / MLflow / experiment tracking / classical eval) and to [`ai-infra-junior-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning) for engineering-craft prerequisites.
- **Defers sideways** to [`training-pipeline-engineer-learning`](https://github.com/ml-engineering-curriculum/training-pipeline-engineer-learning) for distributed-training PLATFORM engineering (multi-tenant schedulers, NCCL / fabric tuning, throughput at cluster scale); to [`model-evaluation-engineer-learning`](https://github.com/ml-engineering-curriculum/model-evaluation-engineer-learning) / [`ai-eval-engineer-learning`](https://github.com/ml-engineering-curriculum/ai-eval-engineer-learning) for eval platform engineering depth; to [`nlp-engineer-learning`](https://github.com/ml-engineering-curriculum/nlp-engineer-learning) / [`rag-engineer-learning`](https://github.com/ml-engineering-curriculum/rag-engineer-learning) / [`llm-application-developer-learning`](https://github.com/ml-engineering-curriculum/llm-application-developer-learning) for their respective specialist domains; to [`ai-risk-engineer-learning`](https://github.com/ml-engineering-curriculum/ai-risk-engineer-learning) for alignment-risk and red-team methodology depth.
- **Defers up** to [`staff-ml-engineer-learning`](https://github.com/ml-engineering-curriculum/staff-ml-engineer-learning) and [`principal-ml-engineer-learning`](https://github.com/ml-engineering-curriculum/principal-ml-engineer-learning) for architectural and leadership scope; to [`ai-infra-ml-platform-learning`](https://github.com/ai-infra-curriculum/ai-infra-ml-platform-learning) for self-service ML platform engineering; to [`ai-infra-security-learning`](https://github.com/ai-infra-curriculum/ai-infra-security-learning) for deep ML/AI security.
- **Links out** to [`ai-governance-analyst-learning`](https://github.com/ml-engineering-curriculum/ai-governance-analyst-learning) for governance / compliance / model-card review depth.

See [`JOB_REQUIREMENTS.md`](JOB_REQUIREMENTS.md) for the requirements-to-coverage map and the cited public references the catalog is grounded in.

## Module Plan

| Module | Title | Hours | Status |
|---|---|---|---|
| mod-101-post-training-foundations | Post-Training Foundations: Transformer Internals for Fine-Tuners | 12 | planned |
| mod-102-sft-dataset-engineering | SFT Dataset Engineering: From Raw Conversations to Training-Ready Mixes | 16 | planned |
| mod-103-supervised-fine-tuning | Supervised Fine-Tuning End-to-End | 18 | planned |
| mod-104-peft-lora-qlora | Parameter-Efficient Fine-Tuning: LoRA, QLoRA, and Friends | 16 | planned |
| mod-105-preference-optimization | Preference Optimization: RLHF, DPO, ORPO, KTO, and the Modern Family | 18 | planned |
| mod-106-safety-alignment-tuning | Safety and Alignment Tuning | 13 | planned |
| mod-107-distributed-fine-tuning | Distributed Fine-Tuning at the Operator Altitude | 14 | planned |
| mod-108-evaluating-fine-tuned-models | Evaluating Fine-Tuned Models | 15 | planned |
| mod-109-quantization-export-serving | Quantization, Export, and Serving Adapted Models | 12 | planned |
| mod-110-managed-fine-tuning-apis | Managed Fine-Tuning APIs | 9 | planned |
| mod-111-post-training-systems-design | Post-Training Systems Design | 14 | planned |

## Project Plan

| Project | Title | Hours | Status |
|---|---|---|---|
| project-101-domain-sft-adapter | Domain-Specialised SFT Adapter (LoRA on an Open Base) | 35 | planned |
| project-102-preference-aligned-chat-model | Preference-Aligned Chat Model: SFT → DPO Recipe | 40 | planned |
| project-103-post-training-capstone | Post-Training Capstone: Multi-Stage Adapter with Safety Tuning, Eval Gates, and Release Review | 60 | planned |

## Module summaries

### mod-101 — Post-Training Foundations
Decoder-only transformer internals at the depth a fine-tuner needs: attention, KV cache, positional encodings (RoPE / ALiBi), tokenization, chat templates, and the layer structure that PEFT methods target. Ends with a map of the pretrain → SFT → preference-optimization → quantization → serving pipeline.

### mod-102 — SFT Dataset Engineering
Curate, deduplicate, and decontaminate an SFT mix against eval suites; design chat templates and multi-turn conversation structure; generate synthetic instruction data with provenance; mix datasets across reasoning / code / safety / persona; produce a model-card-ready data trace with licensing.

### mod-103 — Supervised Fine-Tuning End-to-End
Run an SFT job on a 1B–8B base with TRL `SFTTrainer` (or Axolotl / Unsloth); implement prompt/response masking, sequence packing, gradient accumulation; pick LR schedule and checkpoint cadence by reading loss curves; diagnose overfitting, catastrophic forgetting, and template-leak failure modes.

### mod-104 — PEFT: LoRA, QLoRA, and Friends
Implement LoRA from scratch and against `peft`; run QLoRA on a 7B–13B model on a single 24-GB GPU with bitsandbytes 4-bit NF4; manage adapter merging and multi-adapter routing; compare LoRA / QLoRA / IA3 / prefix / prompt tuning and pick correctly.

### mod-105 — Preference Optimization
Construct preference-pair datasets; train a reward model; run PPO-RLHF via TRL `PPOTrainer`; run DPO end-to-end and compare to PPO; apply the post-2024 family (ORPO / KTO / IPO / SimPO / GRPO) and reason about when each fits; detect reward hacking, length bias, KL collapse.

### mod-106 — Safety and Alignment Tuning
Refusal and harm-reduction data, Constitutional AI / RLAIF self-critique at small scale, red-team data ingestion, measurement of refusal rate / over-refusal / jailbreak resistance / policy adherence, and the hand-off interface to the security, governance, and risk tracks.

### mod-107 — Distributed Fine-Tuning at the Operator Altitude
Multi-GPU / multi-node fine-tuning with Accelerate, FSDP, and DeepSpeed ZeRO-2/3; gradient checkpointing, BF16 / FP8, CPU / NVMe offload; throughput / MFU profiling; OOM and NCCL debugging; 7B → 70B feasibility studies. Cluster-platform concerns are explicitly handed to `training-pipeline-engineer`.

### mod-108 — Evaluating Fine-Tuned Models
`lm-evaluation-harness` for academic benchmarks (MMLU, ARC, GSM8K, HumanEval, TruthfulQA, IFEval); task-specific held-out evals tied to the production use case; LLM-as-judge with MT-Bench-style protocols; contamination detection; release-gate regression reports.

### mod-109 — Quantization, Export, and Serving
Post-training quantization (GPTQ / AWQ / bitsandbytes 8-bit / 4-bit) with quality / latency / VRAM measurement; adapter merging vs. LoRA-aware serving (vLLM multi-LoRA); TGI deployment; GGUF export for llama.cpp; a serving runbook for an adapter release.

### mod-110 — Managed Fine-Tuning APIs
Managed fine-tuning on OpenAI and at least one of Anthropic / Gemini / Mistral / Cohere / Bedrock / Azure AI Foundry; JSONL format conversion; cost / data residency / IP / privacy reasoning; managed-vs-self-hosted decision; shared eval harness so comparisons are fair.

### mod-111 — Post-Training Systems Design
Product goal → base-model + dataset-budget + objective-stack proposal; dataset budget allocation across SFT / DPO / RLHF / safety; eval gates, release-candidate selection, adapter registry, rollback; fine-tune-vs-RAG-vs-prompt-engineering decision; model card and release review for governance / risk / security partners.

## Assessment

Each module ships **1 quiz** plus the exercises and a lab listed in [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json). Each project ships a portfolio-grade README, a reproducibility bundle (config + dataset hashes + seeds), and an explicit rubric covering data engineering, training correctness, evaluation, safety, deployment, and documentation.

## Where to go after this curriculum

- **`senior-ml-engineer-learning`** — peer at the same level (generalist counterpart on the ladder).
- **`staff-ml-engineer-learning` / `principal-ml-engineer-learning`** — next steps on the ML engineering ladder for architectural / leadership scope.
- **`training-pipeline-engineer-learning`** — peer specialist for distributed-training platform engineering depth.
- **`model-evaluation-engineer-learning` / `ai-eval-engineer-learning`** — peer specialists for eval engineering depth.
- **`ai-infra-ml-platform-learning`** — for self-service ML platform engineering.
- **`ai-infra-security-learning`** — for ML/AI security depth, including adapter supply-chain and training-data exfiltration.
- **`ai-risk-engineer-learning` / `ai-governance-analyst-learning`** — for alignment-risk methodology and governance / compliance depth.

<!-- needs-research: backfill industry-frequency evidence into JOB_REQUIREMENTS.md once the autonomous research loop runs with WebSearch / WebFetch exercised against live job boards; demote any module or exercise whose underlying requirement does not show up in ≥3 in-window postings. -->
