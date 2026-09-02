# Resources for mod-111-post-training-systems-design

Curated primary references. **Governance frameworks and vendor policies move** — check the retrieval date on any specific clause, requirement, or price you cite in a real deliverable. The links below are official entry points; where a chapter cites a specific number or clause, verify against the current page before quoting.

<!-- needs-research: on next authoring cycle, spot-check every URL below (governance-body sites and vendor policy portals are reorganized frequently — NIST AI RMF has evolving profiles; ISO/IEC 42001 has profile updates; the EU AI Act has ongoing secondary legislation) and add commit-SHA / snapshot dates where the reference has a version. -->

## Chapter 1 — From product goal to a fine-tuning proposal

### Framing / decision intake

- Michael Nygard, "Documenting Architecture Decisions" — the ADR template pattern the proposal / ADR docs in this module derive from — https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
- Google Cloud "Architecture decision records overview" — the same pattern, adapted for cloud-scale engineering — https://cloud.google.com/architecture/architecture-decision-records
- ThoughtWorks / adr-tools — the CLI-shaped ADR workflow — https://github.com/npryce/adr-tools
- Google SRE workbook — "Documenting knowledge" chapter (runbook / postmortem discipline that the proposal's risk register descends from) — https://sre.google/workbook/documenting-knowledge/

### The success-metric section (hand-off to mod-108 vocabulary)

- EleutherAI `lm-evaluation-harness` — the fast-lane harness the proposal names — https://github.com/EleutherAI/lm-evaluation-harness
- Hugging Face `evaluate` — a lighter-weight metric library often used alongside — https://huggingface.co/docs/evaluate/
- OpenAI Evals — reference for how a first-party team writes an eval — https://github.com/openai/evals

### Cost / timeline modeling

- MLPerf Training benchmarks — reference for realistic training-throughput numbers when the proposal estimates cost — https://mlcommons.org/benchmarks/training/
- Cloud provider GPU pricing pages (cited for cost per GPU-hour in section 5):
  - AWS EC2 pricing — https://aws.amazon.com/ec2/pricing/on-demand/
  - Google Cloud pricing — https://cloud.google.com/compute/all-pricing
  - Azure NDm A100 / H100 pricing — https://azure.microsoft.com/pricing/details/machine-learning/

## Chapter 2 — Selecting the base under real constraints

### Model catalog and revision pinning

- Hugging Face Hub — model card guide (the canonical carrier of model metadata the rubric consumes) — https://huggingface.co/docs/hub/model-cards
- Hugging Face Hub — repository revisions and shas (how to pin a base) — https://huggingface.co/docs/huggingface_hub/guides/manage-cache
- Hugging Face `transformers` — chat templates (`tokenizer.apply_chat_template`) — https://huggingface.co/docs/transformers/chat_templating

### Common open-weight bases (for candidate scoring)

- Meta — Llama family — https://www.llama.com/
- Meta — Llama 3 Community License — https://github.com/meta-llama/llama-models/blob/main/models/llama3/LICENSE
- Meta — Llama Responsible Use Guide — https://ai.meta.com/llama/responsible-use-guide/
- Alibaba — Qwen (Hugging Face org) — https://huggingface.co/Qwen
- Alibaba — Qwen2.5 technical report — https://arxiv.org/abs/2412.15115
- Mistral AI — model catalog — https://mistral.ai/technology/
- Google DeepMind — Gemma family — https://ai.google.dev/gemma
- Google — Gemma Terms of Use — https://ai.google.dev/gemma/terms
- Microsoft — Phi family — https://azure.microsoft.com/products/phi/
- DeepSeek — model catalog — https://www.deepseek.com/

### License references

- SPDX license list (canonical identifiers cited on rubric axis 1) — https://spdx.org/licenses/
- Apache 2.0 — https://www.apache.org/licenses/LICENSE-2.0
- MIT — https://opensource.org/license/mit

### Method-compatibility matrices

- Hugging Face PEFT — supported models — https://huggingface.co/docs/peft/
- Hugging Face TRL — SFTTrainer / DPOTrainer / KTOTrainer / ORPOTrainer / PPOTrainer — https://huggingface.co/docs/trl/
- vLLM — supported models — https://docs.vllm.ai/en/latest/models/supported_models.html
- HuggingFace TGI — supported models — https://huggingface.co/docs/text-generation-inference/supported_models
- SGLang — supported models — https://github.com/sgl-project/sglang
- llama.cpp — supported models — https://github.com/ggerganov/llama.cpp/blob/master/README.md

### Recipe / community references (axis 7)

- Unsloth — https://github.com/unslothai/unsloth
- Axolotl — https://github.com/axolotl-ai-cloud/axolotl
- LLaMA-Factory — https://github.com/hiyouga/LLaMA-Factory
- Meta Llama Recipes — https://github.com/meta-llama/llama-recipes

## Chapter 3 — Dataset budget allocation across the objective stack

### Objective references (handoff to mods 103, 104, 105, 106)

- Ouyang et al., 2022 — "Training language models to follow instructions with human feedback" (InstructGPT — the reference paper for SFT + RLHF) — https://arxiv.org/abs/2203.02155
- Rafailov et al., 2023 — "Direct Preference Optimization" — https://arxiv.org/abs/2305.18290
- Ethayarajh et al., 2024 — "KTO: Model Alignment as Prospect Theoretic Optimization" — https://arxiv.org/abs/2402.01306
- Hong et al., 2024 — "ORPO: Monolithic Preference Optimization without Reference Model" — https://arxiv.org/abs/2403.07691
- Meng et al., 2024 — "SimPO: Simple Preference Optimization with a Reference-Free Reward" — https://arxiv.org/abs/2405.14734
- Schulman et al., 2017 — "Proximal Policy Optimization" (PPO — the RL algorithm behind classical RLHF) — https://arxiv.org/abs/1707.06347
- Shao et al., 2024 — "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (GRPO — the group-relative-preference variant chapter 3 references) — https://arxiv.org/abs/2402.03300
- Bai et al., 2022 — "Constitutional AI: Harmlessness from AI Feedback" (safety-tuning reference) — https://arxiv.org/abs/2212.08073

### Dataset-mix / capability-preserving references

- Ivison et al., 2023 — "Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2" — https://arxiv.org/abs/2311.10702
- Lambert et al., 2024 — "Tulu 3: Pushing Frontiers in Open Language Model Post-Training" — https://arxiv.org/abs/2411.15124
- HuggingFace H4 — `no_robots` (capability-preserving mix reference) — https://huggingface.co/datasets/HuggingFaceH4/no_robots
- Databricks — `databricks-dolly-15k` — https://huggingface.co/datasets/databricks/databricks-dolly-15k

### Ablation-plan / diagnostic references

- Hugging Face TRL — SFTTrainer / DPO trainer implementations (where training loss curves are read) — https://huggingface.co/docs/trl/sft_trainer
- Weights & Biases — experiment tracking (where ablation results live) — https://docs.wandb.ai/
- MLflow tracking — https://mlflow.org/docs/latest/tracking.html

## Chapter 4 — The build-vs-buy and fine-tune-vs-alternative decision

### ADR patterns

- Michael Nygard, "Documenting Architecture Decisions" — https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
- Google Cloud "Architecture decision records overview" — https://cloud.google.com/architecture/architecture-decision-records
- ADR template collection (MADR) — https://adr.github.io/madr/
- Michael Keeling & the ADR community — the "Sustainable Architectural Design Decisions" pattern — https://adr.github.io/

### RAG references

- Lewis et al., 2020 — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (the original RAG paper) — https://arxiv.org/abs/2005.11401
- LangChain — RAG concepts and reference implementations — https://python.langchain.com/docs/concepts/rag/
- LlamaIndex — retrieval / indexing docs — https://docs.llamaindex.ai/
- Facebook FAISS — https://github.com/facebookresearch/faiss
- Chroma — https://docs.trychroma.com/
- LanceDB — https://lancedb.github.io/lancedb/

### Function-calling / tool-use references

- OpenAI function calling — https://platform.openai.com/docs/guides/function-calling
- Anthropic tool use — https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- Google Vertex — function calling — https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling

### Prompt engineering references

- OpenAI prompt engineering guide — https://platform.openai.com/docs/guides/prompt-engineering
- Anthropic prompt engineering overview — https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
- Google Vertex prompt design — https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/introduction-prompt-design

## Chapter 5 — Release gates, candidate selection, adapter registry, and rollback

### Registry / MLOps references

- MLflow Model Registry — https://mlflow.org/docs/latest/model-registry.html
- Weights & Biases Model Registry — https://docs.wandb.ai/guides/registry
- Databricks Unity Catalog — https://docs.databricks.com/aws/en/data-governance/unity-catalog/index
- Hugging Face model versioning — https://huggingface.co/docs/hub/models-the-hub
- SigStore / cosign (artifact-signing pattern for supply-chain integrity) — https://docs.sigstore.dev/

### Deployment / rollback references

- Google SRE book — "Release Engineering" chapter — https://sre.google/sre-book/release-engineering/
- Google SRE book — "Managing Incidents" chapter — https://sre.google/sre-book/managing-incidents/
- Google SRE workbook — "Managing Load" and "Canarying Releases" — https://sre.google/workbook/canarying-releases/
- Kubernetes — Deployments and rollout strategies — https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
- Kubernetes — Rolling back a Deployment — https://kubernetes.io/docs/concepts/workloads/controllers/deployment/#rolling-back-a-deployment
- Argo Rollouts (progressive-delivery pattern) — https://argoproj.github.io/argo-rollouts/
- Flagger (progressive-delivery operator) — https://docs.flagger.app/

### Multi-adapter serving (handoff to mod-104 / mod-109)

- vLLM — multi-LoRA serving — https://docs.vllm.ai/en/latest/models/lora.html
- HuggingFace `peft` — LoRA multi-adapter — https://huggingface.co/docs/peft/developer_guides/lora
- Predibase LoRAX — https://github.com/predibase/lorax

## Chapter 6 — Authoring the model card and the release review

### Model-card literature

- Mitchell et al., 2019 — "Model Cards for Model Reporting" (the paper) — https://arxiv.org/abs/1810.03993
- Google Model Cards — the tool and template — https://modelcards.withgoogle.com/about
- Hugging Face — model cards guide — https://huggingface.co/docs/hub/model-cards
- Hugging Face — model card annotated template — https://github.com/huggingface/hub-docs/blob/main/modelcard.md
- Meta Llama 3 Model Card (a widely-cited real example) — https://github.com/meta-llama/llama-models/blob/main/models/llama3/MODEL_CARD.md
- Anthropic Claude system cards (system-card variant) — https://www-cdn.anthropic.com/ (see the model releases; verify the current URL)
- OpenAI system cards index — https://openai.com/safety/

### Dataset-card literature (handoff to mod-102)

- Gebru et al., 2018 — "Datasheets for Datasets" — https://arxiv.org/abs/1803.09010
- Hugging Face — dataset cards guide — https://huggingface.co/docs/hub/datasets-cards
- Bender & Friedman, 2018 — "Data Statements for NLP" (a complementary standard often cited alongside Datasheets) — https://aclanthology.org/Q18-1041/

### Governance / risk frameworks

- NIST AI Risk Management Framework 1.0 — https://www.nist.gov/itl/ai-risk-management-framework
- NIST AI RMF Playbook — https://airc.nist.gov/AI_RMF_Knowledge_Base/Playbook
- NIST Generative AI Profile (companion to the AI RMF) — https://airc.nist.gov/AI_RMF_Knowledge_Base/GenAI
- ISO/IEC 42001:2023 — Information technology — Artificial intelligence — Management system — https://www.iso.org/standard/81230.html
- ISO/IEC 23894:2023 — Guidance on risk management (AI-specific) — https://www.iso.org/standard/77304.html
- ISO/IEC 5259 series — Data quality for analytics and machine learning — https://www.iso.org/standard/81088.html
- EU AI Act — Official regulation text — https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- EU AI Act — General-purpose AI (GPAI) provider obligations (article 55 area) — https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- EU AI Act — GPAI Code of Practice (secondary legislation) — https://digital-strategy.ec.europa.eu/en/policies/ai-code-practice
- OECD AI Principles — https://oecd.ai/en/ai-principles

### Environmental footprint (section 8)

- Strubell et al., 2019 — "Energy and Policy Considerations for Deep Learning in NLP" — https://arxiv.org/abs/1906.02243
- Lacoste et al., 2019 — "Quantifying the Carbon Emissions of Machine Learning" — https://arxiv.org/abs/1910.09700
- MLCO2 Impact — https://mlco2.github.io/impact/
- CodeCarbon — https://codecarbon.io/

## Chapter 7 — Post-training as a platform property

### Post-deployment monitoring

- Google SRE book — "Monitoring Distributed Systems" — https://sre.google/sre-book/monitoring-distributed-systems/
- Google SRE workbook — "SLOs" — https://sre.google/workbook/implementing-slos/
- OpenTelemetry (for standardized traces / metrics on serving stacks) — https://opentelemetry.io/docs/
- Prometheus (metrics collection) — https://prometheus.io/docs/introduction/overview/
- Grafana (dashboards for SLO tracking) — https://grafana.com/docs/grafana/latest/

### Drift and eval-set maintenance

- Evidently AI — data / model / prompt drift reports — https://docs.evidentlyai.com/
- WhyLabs — model-monitoring platform (reference for the shape of production monitoring) — https://docs.whylabs.ai/
- LangSmith (LangChain's LLM observability product) — https://docs.smith.langchain.com/

### Incident response and postmortems

- Google SRE book — "Postmortem Culture: Learning from Failure" — https://sre.google/sre-book/postmortem-culture/
- PagerDuty Incident Response docs — https://response.pagerduty.com/
- Learning from Incidents — the community-driven blameless-postmortem site — https://www.learningfromincidents.io/

### Model retirement / deprecation

- OpenAI model deprecation policy (mod-110 handoff) — https://platform.openai.com/docs/deprecations
- Anthropic model deprecation notes — https://docs.anthropic.com/en/docs/about-claude/models
- Google Vertex AI — model lifecycle notes — https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions
- AWS Bedrock — model lifecycle — https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html

## Cross-track / meta

### Post-training foundations (across the whole track)

- Ouyang et al., 2022 — InstructGPT — https://arxiv.org/abs/2203.02155
- Christiano et al., 2017 — "Deep reinforcement learning from human preferences" — https://arxiv.org/abs/1706.03741
- Bai et al., 2022 — "Training a Helpful and Harmless Assistant" — https://arxiv.org/abs/2204.05862
- Lambert et al., 2024 — Tulu 3 (state-of-the-art open post-training pipeline) — https://arxiv.org/abs/2411.15124
- OpenAI — Model Spec (a reference "principles document" that a fine-tune's intended-use section can cite) — https://cdn.openai.com/spec/model-spec-2024-05-08.html

### Reproducibility / provenance (for the model-card and release-verdict provenance chains)

- Hugging Face Hub — model / dataset repository model — https://huggingface.co/docs/hub/
- Git LFS (for large-artifact versioning) — https://git-lfs.com/
- DVC — data / model versioning — https://dvc.org/doc

### Vendor governance / trust portals (for the model card §9 license and DPA references)

- OpenAI Trust Portal — https://trust.openai.com/
- Anthropic Trust Portal — https://trust.anthropic.com/
- Google Cloud compliance offerings — https://cloud.google.com/security/compliance/offerings
- AWS Artifact (SOC / ISO reports, DPAs) — https://aws.amazon.com/artifact/
- Microsoft Service Trust Portal — https://servicetrust.microsoft.com/

## Suggested reading order for this module

1. Read chapter 1 with the Michael Nygard ADR post open. The proposal shape and the ADR shape rhyme; noticing that they rhyme is what makes chapters 1, 2, and 4 easier to keep straight.
2. Read chapter 2 with the SPDX license list, one open-weight base's model card (Llama 3 or Qwen 2.5), and the Hugging Face chat-templating guide open. Rubric axes 1, 4, and 5 are more concrete when you have looked at the artifacts they consume.
3. Read chapter 3 alongside the InstructGPT paper's section 3 (training pipeline) and the Tulu 3 paper's section 3 (their allocation across SFT, DPO, and RLVR). Chapter 3's four default stacks are visible in both.
4. Read chapter 4 with the LangChain RAG concepts page open and a small managed-API sandbox available. Chapter 4's tests are cheaper to run than to reason about; do them, even at 30 prompts.
5. Read chapter 5 with the Google SRE book's release-engineering and managing-incidents chapters open. The release-verdict shape and the rollback runbook shape are the SRE runbook pattern applied to model artifacts.
6. Read chapter 6 with Mitchell et al. 2019 open, plus one real published model card (Meta's Llama 3 model card is convenient). The nine sections are easier to internalize once you have seen a real card of that shape.
7. Read chapter 7 with the Google SRE book's monitoring-distributed-systems and postmortem-culture chapters open. Chapter 7's four-rung maturity ladder maps almost directly onto SRE maturity.

<!-- needs-research: at next authoring cycle, add snapshot / archive URLs for standards bodies whose live URLs are prone to reorganization (NIST AI RMF pages, EU AI Act secondary-legislation pages, ISO/IEC 42001 profile updates), so readers can reproduce the exact framework version this module was authored against. -->
