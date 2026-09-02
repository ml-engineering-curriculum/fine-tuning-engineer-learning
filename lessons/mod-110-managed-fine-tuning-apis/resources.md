# Resources for mod-110-managed-fine-tuning-apis

Curated primary references. **Vendor docs move quarterly** — check the retrieval date on any specific field, schema, price, or quota you cite in a real deliverable. The links below are the official entry points; where a chapter cites a specific number (a price, a retention window, a model list) verify against the current page before quoting.

<!-- needs-research: on next authoring cycle, spot-check every URL below (vendor docs are reorganized frequently, and some vendors have moved from `docs.<vendor>.com` to `<vendor>.com/docs` or renamed the fine-tuning surface entirely — e.g., "Azure OpenAI Service" → "Azure AI Foundry" in 2024–25). -->

## Chapter 1 — The managed fine-tuning landscape

### Framing / meta

- OpenAI Model Distillation guide (the "distill via Stored Completions then fine-tune" workflow chapter 1 disambiguates from managed fine-tuning) — https://platform.openai.com/docs/guides/distillation
- Hugging Face `datasets` — the canonical intermediate `messages` format the module converts *from* — https://huggingface.co/docs/datasets/
- Hugging Face TRL SFTTrainer docs (the self-hosted analog the module contrasts against) — https://huggingface.co/docs/trl/sft_trainer

### The seven vendors

Each vendor's fine-tuning entry point (the top-level doc used in chapters 2 and 3):

- OpenAI Fine-tuning — https://platform.openai.com/docs/guides/fine-tuning
- Anthropic docs (fine-tuning surface, direct and via Bedrock) — https://docs.anthropic.com/
- Google Vertex AI — Tune models — https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-models
- Mistral La Plateforme — Fine-tuning — https://docs.mistral.ai/capabilities/finetuning/
- Cohere Fine-tuning — https://docs.cohere.com/docs/fine-tuning
- AWS Bedrock — Custom models — https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html
- Azure AI Foundry — Fine-tuning — https://learn.microsoft.com/azure/ai-foundry/how-to/fine-tune-serverless

### Hybrid / serverless-LoRA-on-open-weights (the "third path" chapter 6 discusses)

- Fireworks AI Fine-tuning — https://docs.fireworks.ai/fine-tuning/fine-tuning-models
- Together AI Fine-tuning — https://docs.together.ai/docs/fine-tuning-overview
- Replicate Training — https://replicate.com/docs/get-started/fine-tune-with-flux
- Anyscale Fine-tuning (product URL, verify current status) — https://www.anyscale.com/

## Chapter 2 — Running a managed fine-tune on OpenAI end-to-end

### Official docs

- OpenAI Fine-tuning guide — https://platform.openai.com/docs/guides/fine-tuning
- OpenAI Fine-tuning API reference (`fine_tuning.jobs.create` / `.retrieve` / `.list_events`) — https://platform.openai.com/docs/api-reference/fine-tuning
- OpenAI Files API reference (`files.create(purpose="fine-tune")`) — https://platform.openai.com/docs/api-reference/files
- OpenAI Models API reference (`models.list` / `models.delete`) — https://platform.openai.com/docs/api-reference/models
- OpenAI Uploads API (multipart, resumable) — https://platform.openai.com/docs/api-reference/uploads
- OpenAI DPO fine-tuning docs — https://platform.openai.com/docs/guides/fine-tuning#preference
- OpenAI Reinforcement Fine-Tuning docs — https://platform.openai.com/docs/guides/reinforcement-fine-tuning
- OpenAI pricing (fine-tuning tier prices — verify retrieval date on every number) — https://openai.com/api/pricing/

### JSONL schema and tokenization

- OpenAI chat-completion messages schema (the target format of chapter 4's OpenAI converter) — https://platform.openai.com/docs/api-reference/chat
- `tiktoken` (the per-model tokenizer used for local token counting) — https://github.com/openai/tiktoken
- OpenAI Cookbook — Fine-tuning examples (validator, format check, cost estimation) — https://cookbook.openai.com/examples/how_to_finetune_chat_models

### Integrations and observability

- OpenAI W&B integration for fine-tuning — https://docs.wandb.ai/guides/integrations/openai-fine-tuning/
- OpenAI event / job monitoring (`list_events`) — https://platform.openai.com/docs/api-reference/fine-tuning/list-events

### SDKs

- `openai` Python SDK — https://github.com/openai/openai-python
- `openai` Node.js SDK — https://github.com/openai/openai-node

## Chapter 3 — A tour of the other managed vendors

Read each vendor's fine-tuning doc as primary; the linked pages have the current model catalog, JSONL schema, supported methods, and hyperparameter surface.

### Anthropic (direct + via Bedrock)

- Anthropic API docs home — https://docs.anthropic.com/
- Anthropic Messages API reference (target inference API after fine-tune) — https://docs.anthropic.com/en/api/messages
- Anthropic on AWS Bedrock — https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
- Bedrock model customization for Claude — https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html
- Anthropic model overview (currently supported models — for verifying tuneable list) — https://docs.anthropic.com/en/docs/about-claude/models

### Google Vertex AI (Gemini)

- Vertex AI — Tune Gemini models — https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-models
- Vertex AI — Supervised tuning for Gemini — https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-gemini
- Vertex AI — Supervised tuning dataset format — https://cloud.google.com/vertex-ai/generative-ai/docs/models/tuning-datasets
- Vertex AI Python SDK — https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk
- Google `generateContent` API reference — https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
- Gemma (open-weights, separate self-hosted path) — https://ai.google.dev/gemma

### Mistral (La Plateforme)

- Mistral fine-tuning docs — https://docs.mistral.ai/capabilities/finetuning/
- Mistral fine-tuning API reference — https://docs.mistral.ai/api/#tag/fine-tuning
- Mistral Python client — https://github.com/mistralai/client-python
- Mistral open weights (Hugging Face org — the escape hatch to self-hosted) — https://huggingface.co/mistralai
- Mistral DPO fine-tuning (where supported) — https://docs.mistral.ai/capabilities/finetuning/#preference-fine-tuning

### Cohere

- Cohere Fine-tuning docs — https://docs.cohere.com/docs/fine-tuning
- Cohere fine-tuning for Chat (Command family) — https://docs.cohere.com/docs/chat-fine-tuning
- Cohere fine-tuning for Rerank — https://docs.cohere.com/docs/rerank-fine-tuning
- Cohere fine-tuning for Embed — https://docs.cohere.com/docs/embed-fine-tuning
- Cohere API reference — https://docs.cohere.com/reference/about

### AWS Bedrock (multi-provider)

- Bedrock Custom Models overview — https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html
- Bedrock model customization job API (`CreateModelCustomizationJob`) — https://docs.aws.amazon.com/bedrock/latest/APIReference/API_CreateModelCustomizationJob.html
- Bedrock Continued Pre-training — https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-continued-pre-training.html
- Bedrock Provisioned Throughput — https://docs.aws.amazon.com/bedrock/latest/userguide/prov-throughput.html
- Bedrock Converse API — https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
- Bedrock VPC endpoints — https://docs.aws.amazon.com/bedrock/latest/userguide/vpc-interface-endpoints.html
- Bedrock supported foundation models — https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html

### Azure AI Foundry

- Azure AI Foundry — Fine-tune models — https://learn.microsoft.com/azure/ai-foundry/how-to/fine-tune-serverless
- Azure OpenAI Fine-tuning — https://learn.microsoft.com/azure/ai-services/openai/how-to/fine-tuning
- Azure OpenAI DPO — https://learn.microsoft.com/azure/ai-services/openai/how-to/fine-tuning-direct-preference-optimization
- Azure OpenAI Reinforcement Fine-Tuning — https://learn.microsoft.com/azure/ai-services/openai/how-to/reinforcement-fine-tuning
- Azure AI Foundry deployments — https://learn.microsoft.com/azure/ai-foundry/how-to/deploy-models-serverless
- Azure private endpoints (residency / networking) — https://learn.microsoft.com/azure/ai-services/openai/how-to/use-your-data-securely

## Chapter 4 — Converting datasets between Hugging Face and managed JSONL

### The canonical intermediate

- Hugging Face `datasets` docs — https://huggingface.co/docs/datasets/
- Hugging Face chat-template docs (the `messages` format canonical shape) — https://huggingface.co/docs/transformers/chat_templating
- TRL SFTTrainer messages support — https://huggingface.co/docs/trl/sft_trainer#dataset-format-support

### Per-vendor JSONL schema references

- OpenAI chat fine-tune JSONL — https://platform.openai.com/docs/guides/fine-tuning#example-format
- OpenAI tool-calling fine-tune JSONL — https://platform.openai.com/docs/guides/fine-tuning#tools
- OpenAI vision (image) fine-tune JSONL — https://platform.openai.com/docs/guides/fine-tuning#vision
- Anthropic training data format (via Bedrock or direct) — https://docs.anthropic.com/en/docs/build-with-claude/fine-tuning (verify current URL)
- Vertex Gemini tuning dataset format — https://cloud.google.com/vertex-ai/generative-ai/docs/models/tuning-datasets
- Mistral fine-tuning data format — https://docs.mistral.ai/capabilities/finetuning/#dataset-format
- Cohere fine-tuning data format — https://docs.cohere.com/docs/chat-preparing-the-data
- Bedrock model customization data format (per base) — https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-prepare.html

### JSONL / JSON Lines tooling

- JSON Lines spec — https://jsonlines.org/
- `jsonlines` Python library — https://jsonlines.readthedocs.io/
- `jq` (command-line JSON validator, used for eyeball checks) — https://jqlang.github.io/jq/

### Validators and cookbooks

- OpenAI Cookbook — Data preparation and analysis for chat model fine-tuning — https://cookbook.openai.com/examples/chat_finetuning_data_prep
- OpenAI moderation endpoint (pre-upload safety scan) — https://platform.openai.com/docs/guides/moderation

## Chapter 5 — Cost, data residency, IP, and privacy trade-offs

### Vendor pricing pages (verify retrieval date on every quoted number)

- OpenAI API pricing — https://openai.com/api/pricing/
- Anthropic API pricing — https://www.anthropic.com/pricing
- Google Vertex AI pricing — https://cloud.google.com/vertex-ai/pricing
- Mistral La Plateforme pricing — https://mistral.ai/technology/#pricing
- Cohere pricing — https://cohere.com/pricing
- AWS Bedrock pricing — https://aws.amazon.com/bedrock/pricing/
- Azure OpenAI pricing — https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/

### Data-processing and enterprise terms (the DPA-clause citations chapter 5 requires)

- OpenAI Business Terms — https://openai.com/policies/business-terms/
- OpenAI Data Processing Addendum — https://openai.com/policies/data-processing-addendum/
- OpenAI Enterprise privacy — https://openai.com/enterprise-privacy/
- Anthropic Commercial Terms of Service — https://www.anthropic.com/legal/commercial-terms
- Anthropic Data Processing Addendum — https://www.anthropic.com/legal/dpa
- Google Cloud Data Processing Addendum — https://cloud.google.com/terms/data-processing-addendum
- AWS Data Processing Addendum — https://aws.amazon.com/service-terms/
- AWS Service Terms — https://aws.amazon.com/service-terms/
- Azure OpenAI data, privacy, and security — https://learn.microsoft.com/legal/cognitive-services/openai/data-privacy
- Mistral Terms of Service — https://mistral.ai/terms/
- Cohere Terms of Use — https://cohere.com/terms-of-use

### Batch APIs (cost lever from chapter 5 lane 4 and chapter 7 lever 2)

- OpenAI Batch API — https://platform.openai.com/docs/guides/batch
- Anthropic Message Batches — https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
- Google Vertex AI batch prediction — https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini
- AWS Bedrock batch inference — https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html

### Regional availability and residency

- OpenAI regional data residency — https://help.openai.com/en/articles/7947663-chatgpt-enterprise-privacy
- Anthropic regional availability — https://docs.anthropic.com/en/api/getting-started
- Google Vertex AI locations — https://cloud.google.com/vertex-ai/docs/general/locations
- AWS Bedrock regions and quotas — https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html
- Azure OpenAI regional availability — https://learn.microsoft.com/azure/ai-services/openai/concepts/models#model-summary-table-and-region-availability

### Compliance references

- HIPAA — HHS Business Associate Agreement (BAA) overview — https://www.hhs.gov/hipaa/for-professionals/covered-entities/sample-business-associate-agreement-provisions/index.html
- OpenAI HIPAA compliance — https://help.openai.com/en/articles/8660679-how-do-i-get-a-business-associate-agreement-baa-with-openai
- Anthropic HIPAA offering — https://www.anthropic.com/legal/baa (verify current URL)
- AWS HIPAA eligible services (Bedrock inclusion) — https://aws.amazon.com/compliance/hipaa-eligible-services-reference/
- Google Cloud HIPAA compliance — https://cloud.google.com/security/compliance/hipaa
- Azure HIPAA / HITECH compliance — https://learn.microsoft.com/compliance/regulatory/offering-hipaa-hitech
- GDPR — Official EU regulation text — https://eur-lex.europa.eu/eli/reg/2016/679/oj
- EU AI Act — Official regulation text — https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- EU AI Act — General-Purpose AI (GPAI) provider obligations — https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- SOC 2 — AICPA overview — https://www.aicpa-cima.com/topic/audit-assurance/audit-and-assurance-greater-than-soc-2
- ISO/IEC 27001 (information security management) — https://www.iso.org/standard/27001

### Vendor trust and compliance portals

- OpenAI Trust portal — https://trust.openai.com/
- Anthropic Trust portal — https://trust.anthropic.com/
- Google Cloud compliance offerings — https://cloud.google.com/security/compliance/offerings
- AWS Artifact (SOC / ISO reports, DPAs) — https://aws.amazon.com/artifact/
- Microsoft Service Trust Portal — https://servicetrust.microsoft.com/

## Chapter 6 — Fine-tuning a closed frontier vs. fine-tuning an open base

### Framing / decision-record templates

- Michael Nygard, "Documenting Architecture Decisions" (the original ADR pattern) — https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
- Google Cloud "ADR process" write-up — https://cloud.google.com/architecture/architecture-decision-records
- ThoughtWorks ADR template (adr-tools) — https://github.com/npryce/adr-tools

### The self-hosted counter-references (mods 103–109 handoff)

- vLLM docs — https://docs.vllm.ai/
- Hugging Face TGI docs — https://huggingface.co/docs/text-generation-inference/
- SGLang docs — https://github.com/sgl-project/sglang
- llama.cpp (edge / CPU inference) — https://github.com/ggerganov/llama.cpp
- Hugging Face PEFT (LoRA / QLoRA library) — https://huggingface.co/docs/peft/
- Hugging Face TRL (DPO / KTO / ORPO / SFT trainers) — https://huggingface.co/docs/trl/

### Model-card / model-registry conventions (referenced when writing the ADR)

- Model Cards for Model Reporting (Mitchell et al., 2018) — https://arxiv.org/abs/1810.03993
- Hugging Face Hub model card guide — https://huggingface.co/docs/hub/model-cards
- MLflow Model Registry — https://mlflow.org/docs/latest/model-registry.html
- Weights & Biases Model Registry — https://docs.wandb.ai/guides/registry

## Chapter 7 — Feeding managed fine-tunes through the same eval harness

### The eval harness itself

- EleutherAI `lm-evaluation-harness` — https://github.com/EleutherAI/lm-evaluation-harness
- `lm-evaluation-harness` interface guide (backends, chat-template flags, `--gen_kwargs`) — https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
- `lm-evaluation-harness` task catalog — https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
- BigCode Evaluation Harness (HumanEval / MBPP / MultiPL-E) — https://github.com/bigcode-project/bigcode-evaluation-harness

### Cross-vendor gateway

- LiteLLM — https://docs.litellm.ai/
- LiteLLM proxy server (OpenAI-shape in, any vendor out) — https://docs.litellm.ai/docs/simple_proxy
- OpenRouter (managed OpenAI-compat gateway across vendors) — https://openrouter.ai/docs

### Backend-specific notes

- OpenAI Chat Completions API (target for `openai-chat-completions` backend) — https://platform.openai.com/docs/api-reference/chat
- OpenAI logprobs on chat completions — https://platform.openai.com/docs/api-reference/chat/create#chat-create-logprobs
- Anthropic Messages API (target for `anthropic-chat-completions` backend) — https://docs.anthropic.com/en/api/messages
- Vertex AI `generateContent` — https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
- vLLM OpenAI-compatible server (`vllm serve`) — https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- TGI OpenAI-compatible API — https://huggingface.co/docs/text-generation-inference/en/messages_api

### Log-likelihood vs. generation eval variants

- MMLU (Hendrycks et al., 2021) — https://arxiv.org/abs/2009.03300
- MMLU-Pro (Wang et al., 2024) — https://arxiv.org/abs/2406.01574
- GSM8K (Cobbe et al., 2021) — https://arxiv.org/abs/2110.14168
- IFEval (Zhou et al., 2023) — https://arxiv.org/abs/2311.07911
- HumanEval (Chen et al., 2021) — https://arxiv.org/abs/2107.03374
- BBH — Challenging BIG-Bench Tasks (Suzgun et al., 2022) — https://arxiv.org/abs/2210.09261

### Chat-template and sampling reconciliation

- Hugging Face `apply_chat_template` reference — https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.apply_chat_template
- vLLM sampling parameters — https://docs.vllm.ai/en/latest/dev/sampling_params.html
- OpenAI seed / determinism note (chat completions) — https://platform.openai.com/docs/api-reference/chat/create#chat-create-seed

### Contamination (handoff to mod-108 chapter 5)

- Oren et al., 2023 — "Proving Test Set Contamination in Black-Box Language Models." — https://arxiv.org/abs/2310.17623
- Golchin & Surdeanu, 2023 — "Time Travel in LLMs" — https://arxiv.org/abs/2308.08493
- Sainz et al., 2023 — "NLP Evaluation in Trouble" — https://arxiv.org/abs/2310.18018

### Statistical treatment (paired tests, bootstrap CIs)

- SciPy `bootstrap` — https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
- SciPy McNemar test — https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.mcnemar.html
- SciPy Wilcoxon signed-rank — https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
- Dror et al., 2018 — "The Hitchhiker's Guide to Testing Statistical Significance in NLP." — https://aclanthology.org/P18-1128/

## Reference model / API version pages (for pinning the base at manifest time)

- OpenAI models list — https://platform.openai.com/docs/models
- Anthropic model overview — https://docs.anthropic.com/en/docs/about-claude/models
- Google Gemini models — https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models
- Mistral models — https://docs.mistral.ai/getting-started/models/
- Cohere models — https://docs.cohere.com/docs/models
- AWS Bedrock supported foundation models — https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
- Azure OpenAI models — https://learn.microsoft.com/azure/ai-services/openai/concepts/models

## Suggested reading order for this module

1. Read chapter 1 alongside two vendors' fine-tuning entry pages (OpenAI's guide plus one other you might use). Chapter 1's "seven-step pipeline" is easier to internalize once you have seen it in two shapes.
2. Read chapter 2 with the OpenAI fine-tuning guide, the API reference, and the OpenAI Cookbook data-prep notebook open. Run exercise 01 before finishing the chapter — the mechanics stick faster from doing than from reading.
3. Skim chapter 3, then bookmark the specific vendor you actually use and read its doc end-to-end. Chapter 3 is a tour; the depth belongs to the vendor's own docs.
4. Read chapter 4 with the per-vendor JSONL schema pages open in tabs. Every conversion bug the chapter catalogs is a specific field in a specific vendor's schema — the docs are the source of truth.
5. Read chapter 5 with each vendor's pricing page, DPA, and residency page open. Chapter 5 gives you the shape of the questions; the vendors' pages give you the answers *this quarter*.
6. Read chapter 6 with the ADR template from Michael Nygard's blog post open. Every decision made without a written record becomes an argument to re-run at the next model release.
7. Read chapter 7 with the `lm-evaluation-harness` interface guide and mod-108 chapter 6 open. Exercise 03 wires the two together; the reconciliation checklist in Part E of that exercise is the payoff.

<!-- needs-research: at next authoring cycle, add commit-SHA / version pins for the tools whose behavior changes across releases (`lm-evaluation-harness`, `openai` SDK, `anthropic` SDK, `vllm`, `litellm`) and dated snapshots of every pricing / DPA URL, so readers can reproduce the exact context this module was authored against. -->
