# ML Engineering · Fine-Tuning Engineer — Learning Repository

<!-- aicg:site-banner -->
> 🎓 Part of the free, open-source **AI Career Curriculum** ecosystem — [Infrastructure](https://github.com/ai-infra-curriculum) · [ML Engineering](https://github.com/ml-engineering-curriculum) · [AI Engineering](https://github.com/ai-engineering-curriculum) · [Governance](https://github.com/ai-governance-curriculum) · [Startup](https://github.com/ai-startup-curriculum) · [Workforce](https://github.com/ai-workforce-curriculum).
<!-- /aicg:site-banner -->

<!-- aicg:sponsor -->
> 💜 **[Sponsor this curriculum](https://github.com/sponsors/ml-engineering-curriculum)** — sponsorships keep the whole open-source AI Career Curriculum free and moving.
<!-- /aicg:sponsor -->

Adapt models: SFT, LoRA / QLoRA, full fine-tuning, and preference optimization (DPO / RLHF / ORPO / KTO / GRPO).

> **Status**: curriculum plan authored (2026-06-25). Lessons and project READMEs will be drafted by subsequent autonomous content cycles. See [`CURRICULUM.md`](CURRICULUM.md) for the planned scope and [`JOB_REQUIREMENTS.md`](JOB_REQUIREMENTS.md) for the requirements-to-coverage map (postings list deferred to the next research cycle — see `JOB_REQUIREMENTS.md` Status section).

**Level:** 30 (deep specialist — peer to Senior ML Engineer on the ladder).
**Planned scope:** 11 modules (157h) + 3 projects (135h) = ~292 hours.

## What this track owns

The post-training stack end-to-end at depth:

- Dataset engineering for SFT (curation, decontamination, chat templates, synthetic data with provenance)
- Supervised fine-tuning with TRL / Axolotl / Unsloth
- PEFT — LoRA, QLoRA, IA3, prefix / prompt tuning, adapter merging, multi-adapter routing
- Preference optimization — reward modeling, PPO-RLHF, DPO, ORPO, KTO, IPO, SimPO, GRPO
- Safety / alignment tuning — refusal training, Constitutional AI / RLAIF, red-team data ingestion
- Distributed fine-tuning at the operator altitude — Accelerate, FSDP, DeepSpeed ZeRO
- Evaluation of fine-tuned models — lm-evaluation-harness, MT-Bench-style LLM-as-judge, contamination detection, release gates
- Quantization, export, and serving — GPTQ / AWQ / bitsandbytes; vLLM multi-LoRA; TGI; GGUF
- Managed fine-tuning APIs — OpenAI, Anthropic, Gemini, Mistral, Cohere, Bedrock, Azure AI Foundry
- Post-training systems design — base-model selection, dataset budget, eval gates, adapter registry, model cards

## What this track defers

- Build-altitude ML / PyTorch / packaging fundamentals → [`ml-engineer-learning`](https://github.com/ml-engineering-curriculum/ml-engineer-learning) (level 20).
- Distributed-training PLATFORM engineering → [`training-pipeline-engineer-learning`](https://github.com/ml-engineering-curriculum/training-pipeline-engineer-learning).
- Eval platform engineering → [`model-evaluation-engineer-learning`](https://github.com/ml-engineering-curriculum/model-evaluation-engineer-learning) / [`ai-eval-engineer-learning`](https://github.com/ml-engineering-curriculum/ai-eval-engineer-learning).
- LLM application / RAG / NLP depth → peer specialist tracks linked from `CURRICULUM.md`.
- ML/AI security and governance / risk depth → [`ai-infra-security-learning`](https://github.com/ai-infra-curriculum/ai-infra-security-learning), [`ai-governance-analyst-learning`](https://github.com/ml-engineering-curriculum/ai-governance-analyst-learning), [`ai-risk-engineer-learning`](https://github.com/ml-engineering-curriculum/ai-risk-engineer-learning).

## Layout

```
fine-tuning-engineer-learning/
├── .aicg/                    curriculum-plan.json and job-requirements.json (machine-readable catalog)
├── lessons/mod-XXX-*/        modules with lectures, exercises, labs, quizzes (to be drafted)
├── projects/project-XXX-*/   multi-module capstones (to be drafted)
├── CURRICULUM.md             role-level coverage map
├── JOB_REQUIREMENTS.md       requirements catalog with citations and ownership map
├── PREREQUISITES.md          assumed entry skills
├── VERSIONS.md               release history
└── README.md                 this file
```

## Paired Solutions Repo

[`fine-tuning-engineer-solutions`](https://github.com/ml-engineering-curriculum/fine-tuning-engineer-solutions) carries the reference implementations.

---

<!-- aicg:maintained-by -->
Maintained by [VeriSwarm.ai](https://veriswarm.ai)
