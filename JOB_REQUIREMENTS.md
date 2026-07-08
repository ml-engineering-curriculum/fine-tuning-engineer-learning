# Job Requirements — Fine-Tuning Engineer

**Role level:** 30 (deep specialist — peer to Senior ML Engineer on the ladder)
**Track:** `fine-tuning-engineer-learning`
**Research window:** 2026-03-27 → 2026-06-25 (last 90 days)
**Today:** 2026-06-25

This file documents the requirements catalog used to seed the Fine-Tuning Engineer curriculum. Raw normalized data lives in [`.aicg/job-requirements.json`](.aicg/job-requirements.json); the planned curriculum lives in [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json).

## Status — bootstrap session, postings deferred

<!-- needs-research: collect ≥25 distinct in-window postings titled "Fine-Tuning Engineer" / "LLM Fine-Tuning Engineer" / "Foundation Model Fine-Tuning Engineer" / "Model Customization Engineer" / "Post-Training Engineer" (NOT Senior / Staff / Principal modifiers — those will inherit this packet — and NOT generic ML/NLP/LLM-Application/Research-Scientist titles, which are owned by peer tracks). Re-validate every requirement against the live evidence and demote any whose evidence stays empty. -->

This packet was authored in a bootstrap session **without an exercised WebSearch / WebFetch pass against live job boards**. Per the project rules (*"Do not invent facts, incidents, or salary figures. Cite sources."*), the `postings` array in `.aicg/job-requirements.json` is intentionally empty — the curriculum will not claim to have analysed 25 live postings when none were fetched.

The autonomous research loop is expected to fill that gap on its next cycle. To keep the loop deterministic, this document grounds the requirements catalog in **authoritative public references** that publish what the role is hired against: the official documentation of the dominant open-source post-training stack (Hugging Face TRL, PEFT, Accelerate, transformers; DeepSpeed; PyTorch FSDP; vLLM; TGI; bitsandbytes; lm-evaluation-harness; Axolotl; Unsloth), the canonical post-training research papers (LoRA, QLoRA, InstructGPT, DPO, ORPO, LIMA, Self-Instruct, Constitutional AI, Llama 2, Zephyr, TULU 3), and the canonical managed-fine-tuning vendor docs (OpenAI, Anthropic). Every requirement below cites at least one such reference, and every requirement is shaped so that posting-frequency evidence can be added underneath it without restructure.

## Methodology

1. Sourced the canonical task domains for a deep-specialist post-training engineer from public references — see `authoritative_references` in `.aicg/job-requirements.json`:
   - Hugging Face TRL, PEFT, Accelerate, transformers documentation
   - DeepSpeed and PyTorch FSDP documentation
   - vLLM and Text Generation Inference (TGI) documentation
   - bitsandbytes and llama.cpp / GGUF documentation
   - lm-evaluation-harness (EleutherAI)
   - Axolotl and Unsloth documentation
   - Foundational papers: LoRA (Hu 2021), QLoRA (Dettmers 2023), DPO (Rafailov 2023), ORPO (Hong 2024), InstructGPT (Ouyang 2022), Llama 2 (Touvron 2023), Zephyr (Tunstall 2023), LIMA (Zhou 2023), Self-Instruct (Wang 2022), Constitutional AI (Bai 2022), TULU 3 (Lambert 2024), MT-Bench / LLM-as-judge (Zheng 2023)
   - OpenAI Fine-tuning Guide; Anthropic Claude Fine-tuning docs
2. Mapped each task domain to (a) the role on our level ladder that should own it primarily and (b) the curriculum module that covers it.
3. Applied the **ownership rule**: assign coverage to the lowest-level role that genuinely requires the skill, with higher levels linking back rather than duplicating fundamentals. Defer down to `ml-engineer` (level 20) for ML / PyTorch / packaging fundamentals; defer sideways to `training-pipeline-engineer` (distributed-training INFRASTRUCTURE), `model-evaluation-engineer` / `ai-eval-engineer` (eval platform engineering), `nlp-engineer` / `rag-engineer` / `llm-application-developer` (peer specialist domains), `ai-infra-security-learning` / `ai-governance-analyst` / `ai-risk-engineer` (security / governance / risk depth).
4. Flagged everything that has not yet been validated against in-window postings so the next cycle can demote any requirement whose evidence stays empty.

## Requirement themes → curriculum ownership

The table below lists each requirement theme, its planned owner per the level hierarchy, and the curriculum coverage path. **Freq** is intentionally blank for this cycle — it will be backfilled by the next research pass.

| # | Theme | Freq | Owner role | Coverage |
|---|---|---|---|---|
| 1 | Transformer internals (attention, KV cache, position encodings, tokenization, decoder vs. encoder-decoder) | <!-- needs-research --> | `fine-tuning-engineer` (this) | [`mod-101-post-training-foundations`](lessons/mod-101-post-training-foundations) |
| 2 | SFT dataset engineering (curation, dedup, decontamination, chat templates, multi-turn, synthetic data, provenance) | <!-- needs-research --> | `fine-tuning-engineer` | [`mod-102-sft-dataset-engineering`](lessons/mod-102-sft-dataset-engineering) |
| 3 | Supervised fine-tuning (SFT) end-to-end (TRL SFTTrainer, masking, packing, schedule, checkpointing) | <!-- needs-research --> | `fine-tuning-engineer` | [`mod-103-supervised-fine-tuning`](lessons/mod-103-supervised-fine-tuning) |
| 4 | PEFT: LoRA, QLoRA, adapter merging, multi-adapter routing, IA3 / prefix / prompt tuning | <!-- needs-research --> | `fine-tuning-engineer` | [`mod-104-peft-lora-qlora`](lessons/mod-104-peft-lora-qlora) |
| 5 | Preference optimization (RLHF/PPO, DPO, IPO, KTO, ORPO, SimPO, GRPO, reward modeling) | <!-- needs-research --> | `fine-tuning-engineer` | [`mod-105-preference-optimization`](lessons/mod-105-preference-optimization) |
| 6 | Safety / alignment tuning (refusal training, harm reduction, Constitutional AI / RLAIF, red-team data ingestion) | <!-- needs-research --> | `fine-tuning-engineer` | [`mod-106-safety-alignment-tuning`](lessons/mod-106-safety-alignment-tuning) |
| 7 | Distributed fine-tuning at the operator altitude (Accelerate, FSDP, DeepSpeed ZeRO, gradient checkpointing, BF16/FP8, OOM debugging) | <!-- needs-research --> | `fine-tuning-engineer` | [`mod-107-distributed-fine-tuning`](lessons/mod-107-distributed-fine-tuning) |
| 8 | Evaluation of fine-tuned models (lm-eval-harness, MT-Bench-style LLM-as-judge, task-specific evals, regression suites, contamination) | <!-- needs-research --> | `fine-tuning-engineer` | [`mod-108-evaluating-fine-tuned-models`](lessons/mod-108-evaluating-fine-tuned-models) |
| 9 | Quantization, export, serving (GPTQ/AWQ/bitsandbytes; vLLM multi-LoRA; TGI; GGUF) | <!-- needs-research --> | `fine-tuning-engineer` | [`mod-109-quantization-export-serving`](lessons/mod-109-quantization-export-serving) |
| 10 | Managed fine-tuning APIs (OpenAI / Anthropic / Gemini / Mistral / Cohere / Bedrock / Azure AI Foundry) | <!-- needs-research --> | `fine-tuning-engineer` | [`mod-110-managed-fine-tuning-apis`](lessons/mod-110-managed-fine-tuning-apis) |
| 11 | Post-training systems design (base-model selection, dataset budget, eval gates, adapter registry, rollback, build-vs-buy) | <!-- needs-research --> | `fine-tuning-engineer` | [`mod-111-post-training-systems-design`](lessons/mod-111-post-training-systems-design) + [`project-103-post-training-capstone`](projects/project-103-post-training-capstone) |
| 12 | PyTorch / classical ML / FastAPI / Docker / MLflow / experiment tracking fundamentals | n/a — prerequisite | `ml-engineer` (level 20) | Listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught |
| 13 | Distributed-training PLATFORM engineering (multi-tenant GPU schedulers, NCCL/fabric tuning) | n/a — peer track | `training-pipeline-engineer` (level 25) | Module 107 covers the operator altitude; deep platform engineering linked out |
| 14 | Eval platform engineering (eval-as-a-product, statistical methodology depth, LLM-as-judge platform internals) | n/a — peer track | `model-evaluation-engineer` / `ai-eval-engineer` (level 25) | Module 108 covers the eval workflow; platform depth linked out |
| 15 | LLM application development depth (prompting, agents, tool use, product integration) | n/a — peer track | `llm-application-developer` (level 25) | Surfaced only where it intersects (mod-111 build-vs-buy); depth linked out |
| 16 | RAG depth (chunking, embeddings, vector stores, hybrid retrieval, rerankers) | n/a — peer track | `rag-engineer` (level 25) | Surfaced only in the fine-tune-vs-RAG discussion in mod-111; depth linked out |
| 17 | Deep ML/AI security (data poisoning, model extraction, training-data exfiltration, adapter supply chain) | n/a — higher level | `ai-infra-security-learning` (level 35) | Surfaced as awareness in mod-106 / mod-111; depth owned upstream |
| 18 | Governance / compliance / model cards / dataset licensing review | n/a — peer track | `ai-governance-analyst` / `ai-risk-engineer` | Surfaced as awareness in mod-102 / mod-111; depth owned upstream |

## Posting evidence

<!-- needs-research: populate with the ≥25 in-window postings sampled next cycle. Use the same table shape as `ai-infra-agentic-ai-engineer-learning/JOB_REQUIREMENTS.md`. -->

No postings were sampled this cycle. See the **Status** section above for the reason. The next autonomous research cycle should fan out across:

- `job-boards.greenhouse.io`, `jobs.lever.co`, `jobs.ashbyhq.com`, `boards.greenhouse.io`
- Frontier-lab employer ATS roots: OpenAI, Anthropic, Google DeepMind, Mistral, xAI, Cohere, Reka, Adept, Inflection, Character.AI, AI21
- Open-weight / model-host employers: Hugging Face, Together, Databricks (Mosaic), Snowflake (Cortex), NVIDIA, Predibase, Anyscale, Replicate, Fireworks, OctoAI, Modal, RunPod, Lambda, CoreWeave, Crusoe
- Tooling / data employers: Scale AI, Surge, Mercor, Argilla, Weights & Biases
- Inference platform employers paired with fine-tuning roles: vLLM project, TGI / Hugging Face Inference
- Enterprise post-training roles at: Salesforce AI Research, Apple AI/ML, Meta GenAI, Amazon AGI, Microsoft AI, ServiceNow AI, IBM Research, Adobe Sensei, Bloomberg AI, JPMorgan AI Research

For each posting, capture employer, exact title, URL, date_observed, date_posted (or `estimated:2026-MM`), location, 5–10 verbatim requirement bullets, 2–6 preferred-qualification bullets, salary range when published, and one short representative quote. Filter out Senior / Staff / Principal modifiers (those will inherit this packet), generic ML / NLP / LLM-application / Data Scientist / Research Scientist titles (owned by peer tracks), and pure platform / infra titles (owned by the training-pipeline / ML platform tracks).

## Ownership map — quick reference for next cycle

When backfilling postings, use this ownership decision to keep the curriculum from drifting into peer territory:

- **Fine-Tuning Engineer (this track, level 30)** owns the post-training stack end-to-end: dataset engineering → SFT → PEFT → preference optimization → safety tuning → distributed fine-tuning at the operator altitude → evaluation → quantization / export / serving → managed fine-tuning APIs → systems-level release management.
- **ML Engineer** (level 20) owns the build-altitude practitioner workflow that this curriculum assumes (Python/PyTorch fluency, FastAPI/Docker packaging, MLflow tracking, classical eval).
- **Senior ML Engineer** (level 30, peer ladder) is the *generalist* peer at the same level — overlaps in fine-tuning vocabulary but does not own the depth.
- **Staff ML Engineer** (level 40) and higher inherit / link to this packet for fine-tuning depth, but add architectural and cross-team scope.
- **Training Pipeline Engineer** (level 25) owns distributed-training PLATFORM engineering — schedulers, NCCL tuning, throughput at cluster scale. This curriculum operates those tools but does not build them.
- **Model Evaluation Engineer / AI Evaluation Engineer** (level 25) own eval platform engineering. This curriculum runs the eval workflow at release time.
- **NLP / RAG / LLM Application / Applied AI Engineers** own their respective specializations; this curriculum links out where the boundaries touch.
- **AI Infrastructure Security Engineer** (level 35) owns deep ML/AI security.
- **AI Governance Analyst / AI Risk Engineer** own governance and alignment-risk depth.

## Conclusion

<!-- needs-research: re-run on the next autonomous cycle with web tools exercised against live job boards, populate `postings` in `.aicg/job-requirements.json`, backfill the Freq column above, and demote any requirement whose evidence_post_ids stays empty. -->

The curriculum plan in [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json) is structured so that requirement frequencies can be added underneath each module without restructure. The themes themselves are grounded in cited public references — the dominant open-source post-training stack, foundational papers, and the two largest managed-fine-tuning vendor docs — not invented; they are the canonical task domains the role is hired against in the open-source community and the frontier-lab / model-host industry.
