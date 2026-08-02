# A Tour of the Other Managed Vendors

**Estimated read:** 40 minutes.
**You should be able to, by the end:** for each of Anthropic (via Bedrock and Claude API), Google Vertex AI (Gemini), Mistral La Plateforme, Cohere, AWS Bedrock (as a multi-provider platform), and Azure AI Foundry, name the currently-tuneable models, the JSONL / dataset format, the supported method(s), the artifact you receive, and the two or three ways each vendor differs meaningfully from the OpenAI reference implementation in chapter 2. You should also be able to pick one non-OpenAI vendor and walk through the seven-step pipeline on it.

Read this chapter as a comparative tour. The mental model is: **chapter 2's seven-step pipeline holds; each vendor changes the fields, the model catalog, and one or two operational quirks.** If you find yourself thinking "this is completely different," go back and re-anchor on the pipeline — you're probably reading around a naming change.

<!-- needs-research: every vendor's model catalog, supported methods, JSONL schema fields, and pricing move every quarter. Verify each table and code snippet against the vendor's live docs before shipping — the URLs in resources.md are the primary sources of truth. -->

## Anthropic — Claude fine-tuning

Anthropic's fine-tuning story in 2026 is bifurcated across two surfaces you need to know:

- **AWS Bedrock.** The mature path. Broader Claude model coverage, tighter enterprise procurement fit, uses Bedrock's Custom Models control plane (covered separately below).
- **Direct Claude API.** In-preview / narrower model coverage but a more OpenAI-shaped API surface for teams already on the Claude SDK.

### What is tuneable

Via Bedrock (as of Aug 2026): Claude Haiku family variants; broader coverage under preview. Via the direct Claude API: check the [Anthropic fine-tuning docs](https://docs.anthropic.com/) for the current tuneable model list.

### The JSONL format

Anthropic's SFT format is a JSONL where each row wraps the same `messages` shape you use at inference — one `system` prompt at the top, alternating `user` / `assistant` turns, one row per training example:

```jsonl
{"system": "You are a support triage agent.", "messages": [{"role": "user", "content": "My charge failed."}, {"role": "assistant", "content": "I can help. Which card were you using?"}]}
{"system": "You are a support triage agent.", "messages": [{"role": "user", "content": "Refund on order #40219"}, {"role": "assistant", "content": "Locating order #40219 now."}]}
```

Two schema differences from OpenAI worth noting:

1. **`system` is a top-level field** on each row, not the first message in `messages`. This matches Claude's inference API shape.
2. **No `weight` per turn.** All assistant turns contribute to loss; if you need to mask, rewrite or drop the turn.

Multimodal training (image inputs on Claude) uses the same content-block shape as inference — `{"type": "image", "source": {...}}` inside a user message.

### Method and hyperparameters

SFT is the currently supported method. Hyperparameters mirror OpenAI's shape: number of epochs, learning-rate multiplier, batch size, and a random seed for reproducibility.

### The seven-step pipeline on Anthropic

Via Bedrock: use Bedrock Custom Models (below); the JSONL sits in S3, the job is a Bedrock CreateModelCustomizationJob, the resulting artifact is a Provisioned Throughput or on-demand custom model. Via the direct Claude API: file upload → job create → poll job status → receive model ID → call via the standard `messages.create` API with the fine-tuned model ID.

### The two things that surprise OpenAI-natives

- The system prompt lives at the top-level, not in `messages`. Convert accordingly.
- The Bedrock path is the enterprise default; if your data-governance conversation has anything to do with AWS accounts, VPCs, or IAM, you want Bedrock, not the direct API.

## Google Vertex AI — Gemini supervised tuning

Google's managed fine-tuning for Gemini runs through **Vertex AI**. It is deeply integrated with the rest of GCP — GCS for datasets, Vertex Pipelines for orchestration, Vertex Model Registry for artifacts, IAM for access — which is a large win if your data already lives there and a source of friction if you're an outsider clicking through the Google Cloud console for the first time.

### What is tuneable

Gemini 2.5 Flash and Gemini 2.5 Pro (tuneable variants); check the [Vertex AI tuning docs](https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-models) for the current list. Gemma (open-weights) has a separate self-hosted path — that is not what this section is about; Gemma tuning lives closer to mods 103–104.

### The JSONL format

Vertex's SFT format is a JSONL where each row has `contents` (the multi-turn conversation), and optionally `systemInstruction`:

```jsonl
{"systemInstruction": {"parts": [{"text": "You are a support triage agent."}]}, "contents": [{"role": "user", "parts": [{"text": "My charge failed."}]}, {"role": "model", "parts": [{"text": "I can help. Which card were you using?"}]}]}
```

Two schema differences from OpenAI:

1. **Roles are `user` and `model`** (not `user` and `assistant`). Same semantic; different word.
2. **Content is a list of `parts`** (each a `{"text": "..."}` or an image / other modality), not a top-level `content` string. Multimodal is native to the format.

### Data location

The JSONL lives in a **GCS bucket** that the Vertex service account can read. Not uploaded via an API call — pointed at via `gs://` URI. This is the biggest procedural difference from OpenAI and the source of most first-time-user friction: if the service account cannot list the bucket or read the file, the job fails opaquely.

### Method

Vertex's SFT uses a LoRA-style adapter under the hood — the model returned is a "tuned model" that you invoke with a specific model resource name. Preference-based tuning has been shipped on some Gemini tiers; check the current docs for method availability.

### The seven-step pipeline on Vertex

Upload JSONL to `gs://` → create a `SupervisedTuningJob` (via the Python SDK or REST) pointing at the JSONL URI, the base model, and the hyperparameters → poll `job.state` → receive a `tunedModel` resource name → call it via the Vertex `generateContent` API.

Concrete SDK call (illustrative — verify against current SDK):

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

job = aiplatform.PreviewGenAiModelTuning.tune_supervised(
    source_model="gemini-2.5-flash-preview-04-17",
    train_dataset="gs://my-bucket/support/train.jsonl",
    validation_dataset="gs://my-bucket/support/val.jsonl",
    tuned_model_display_name="support-triage-v3",
    epochs=3,
)

# job.tuned_model_name → "projects/.../locations/.../models/...@1"
```

### The two things that surprise OpenAI-natives

- Everything is a GCP resource. IAM permissions on the training bucket, on the Vertex Model Registry, and on the endpoint are three separate things — a job that "works" for the developer often fails for the service account that runs eval.
- The `role` field says `"model"`, not `"assistant"`. Trivial to fix in the conversion script; catches first-time engineers.

## Mistral — La Plateforme fine-tuning

Mistral's managed fine-tuning through **La Plateforme** is notable for one reason above all: it is the only major managed vendor whose base models are **also downloadable open weights**. You can fine-tune Mistral Small on La Plateforme *or* download the base and fine-tune it yourself with mods 103–104. This flexibility is the whole reason to choose Mistral for a project where the escape hatch matters.

### What is tuneable

Ministral 3B / 8B, Mistral Small, Mistral Large, Codestral — check the [Mistral La Plateforme fine-tuning docs](https://docs.mistral.ai/capabilities/finetuning/) for the currently-tuneable list, as it shifts per model release.

### The JSONL format

Mistral's SFT format is chat-conversation JSONL that closely matches OpenAI's — a top-level `messages` array with `role` / `content`:

```jsonl
{"messages": [{"role": "system", "content": "You are a support triage agent."}, {"role": "user", "content": "My charge failed."}, {"role": "assistant", "content": "I can help. Which card were you using?"}]}
```

Very close to OpenAI's schema; conversion from one to the other is often a no-op.

### Method

SFT is standard; some Mistral models support **DPO** (mod-105 chapter 2 shape), with the JSONL row format including `chosen` / `rejected` completions.

### The seven-step pipeline on Mistral

Upload file via `client.files.upload(purpose="fine-tune")` → create job via `client.fine_tuning.jobs.create(...)` → poll status → receive model ID → call via `client.chat.complete(model=<ft-model-id>, ...)`.

The Python SDK's shape is deliberately close to OpenAI's; if you know chapter 2, this is 90% familiar.

### The two things that surprise OpenAI-natives

- The escape hatch: for tuneable models that are also open-weight, you can download the base at any time and re-fine-tune yourself. This changes the portability calculus in chapter 6 (managed vs. self-hosted becomes a "which one this quarter" question, not "which platform for the artifact's lifetime").
- Fine-tuned model deployment is a separate step in some tiers — the training completes but the model is not yet callable at inference until you attach it to a serving deployment. Check the deployment step in the current docs.

## Cohere — Command fine-tuning, plus Rerank and Embed

Cohere's fine-tuning story is the broadest across model *types* — they are the only major vendor with first-party fine-tuning for **rerankers** and **embedders** in addition to generative Command models. If you are building a retrieval stack, this is often the reason Cohere is on the shortlist.

### What is tuneable

- **Command family** (Command R, Command R+, Command A) — generative chat / instruct models.
- **Rerank v3** — reranker used in RAG pipelines to re-order candidate documents.
- **Embed v3 / v4** — embedding models for retrieval.

Check the [Cohere fine-tuning docs](https://docs.cohere.com/docs/fine-tuning) for the current tuneable list.

### The JSONL format

Cohere's generative fine-tune JSONL is a chat conversation shape:

```jsonl
{"messages": [{"role": "System", "content": "You are a support triage agent."}, {"role": "User", "content": "My charge failed."}, {"role": "Chatbot", "content": "I can help. Which card were you using?"}]}
```

The role names are the Cohere convention: **`System`**, **`User`**, **`Chatbot`** (capitalized, and `Chatbot` where OpenAI would say `assistant`). Convert accordingly.

For **Rerank** fine-tuning, the JSONL is different — each row is a query with a list of relevant and irrelevant documents:

```jsonl
{"query": "how do I reset my password?", "relevant_passages": ["Go to Settings > Security > Reset Password.", "Click 'Forgot password' on the login page."], "hard_negatives": ["Change your username under Profile.", "Refund policy is 30 days."]}
```

For **Embed** fine-tuning, similar shape with anchor / positive / negative triples.

### Method

SFT for generative; Rerank and Embed use specialized contrastive training. All three are exposed through the same "fine-tuning" API surface but with different `finetune_type` values.

### The seven-step pipeline on Cohere

Upload dataset → create fine-tune with `base_model`, `hyperparameters`, `dataset_id` → poll status → receive a fine-tuned model ID → call via the appropriate inference API (`co.chat` for Command, `co.rerank` for Rerank, `co.embed` for Embed).

### The two things that surprise OpenAI-natives

- Role naming: `System` / `User` / `Chatbot`, capitalized. Silent conversion bug source.
- The Rerank / Embed fine-tunes are qualitatively different products — the dataset shape, the eval metric (nDCG / MRR / recall@k, not accuracy or judged quality), and the downstream integration (into a retrieval pipeline, not a chat one) all differ. Mod-108's chapter 3 held-out framework applies, but the metrics are retrieval-specific.

## AWS Bedrock — Custom Models

Bedrock is not a model vendor — it is a **multi-provider control plane** over Anthropic Claude, Amazon Nova, Meta Llama, Mistral, Cohere, and others. Its fine-tuning surface, **Bedrock Custom Models**, gives you one API and one IAM story across many bases.

### What is tuneable

Depends on the base. As of Aug 2026: certain Anthropic Claude models (Haiku family and broader in preview), Amazon Nova family, Meta Llama family, Mistral family, Cohere Command family. Each base has different supported methods; the [Bedrock model customization docs](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html) list them.

### The JSONL format

Depends on the base. Bedrock passes the JSONL through to the underlying provider's training pipeline; the schema is whatever the underlying model expects (Anthropic-shape for Claude, Cohere-shape for Cohere, etc.). Read the specific base's Bedrock Custom Models doc for the exact schema — this is the single largest source of first-time-user confusion.

### Data location

**S3.** Same story as Vertex / GCS: your JSONL sits in S3, IAM controls whether the Bedrock service can read it, and IAM misconfiguration is the most common first-run failure.

### Method

Two big ones:

- **Fine-tuning (SFT).** The default. Same shape as everywhere else.
- **Continued Pretraining.** Only on some bases. Consumes an unlabeled text corpus (not chat data) and continues pretraining the base on your domain text. Different eval shape — perplexity on held-out domain text, downstream task quality after a subsequent SFT — different cost profile, different failure modes. Chapter 4 of mod-102 has the dataset-side view; Bedrock is currently the main managed surface for this method.

Some bases expose additional methods (e.g., DPO on specific Claude models). Verify per model.

### The seven-step pipeline on Bedrock

Upload JSONL to S3 → create a training job via `bedrock.create_model_customization_job(customModelName, roleArn, baseModelIdentifier, trainingDataConfig, hyperParameters, ...)` → poll job status → the resulting Custom Model is registered in your account → attach a **Provisioned Throughput** (a paid-per-hour dedicated capacity commitment) OR use on-demand invocation (available for a subset of bases) → call via `bedrock-runtime.invoke_model` or the Converse API with the custom model ARN.

### The two things that surprise everyone

- The custom model does not serve on-demand by default. For most bases, you must attach a **Provisioned Throughput** — a fixed-cost hourly commitment — before you can call it. This is a very different cost shape from OpenAI's per-token: high fixed cost, cheap at scale, expensive if you have low QPS. Chapter 5 covers the arithmetic.
- The JSONL schema is base-dependent. There is no "Bedrock format" — there is "the Anthropic format when the base is Claude" and "the Cohere format when the base is Command." Cross-reference the specific base's Bedrock docs, not the generic Bedrock Custom Models page.

## Azure AI Foundry — Fine-tuning

Azure's fine-tuning surface, **Azure AI Foundry** (rebranded from Azure OpenAI Service / Azure ML), gives you the OpenAI model catalog with Azure's compliance, residency, private networking, and enterprise-procurement wrappers, plus additional bases (Llama, Mistral, Phi) trained on Azure infrastructure.

### What is tuneable

- The **OpenAI model catalog** (GPT-4.1 family, GPT-4o family, o-series), through Azure — same models, tuned via the same seven-step pipeline you learned in chapter 2, but through Azure's control plane, in an Azure region you choose, with Azure billing.
- **Meta Llama, Mistral, Phi.** Fine-tuned via a Foundry-native tuning workflow, either via managed compute or serverless.

Check the [Azure AI Foundry fine-tuning docs](https://learn.microsoft.com/azure/ai-foundry/how-to/fine-tune-serverless) for the current catalog.

### The JSONL format

**OpenAI models on Azure use OpenAI's JSONL format.** This is the point of using Azure for OpenAI-catalog models — the API surface is 1:1 compatible with OpenAI direct, so you can move a fine-tune between the two with minimal code changes. The `messages` schema, `weight` masking, method surface, and hyperparameters all match chapter 2.

Non-OpenAI models on Foundry (Llama, Mistral, Phi) use per-vendor JSONL — usually a chat-messages shape close to OpenAI's.

### Deployment model

The important operational difference from OpenAI direct: **the fine-tuned model must be deployed to a Foundry deployment before you can call it.** A successful fine-tune produces a "customized model" registered in your Foundry project; you then deploy it (choosing SKU, region, capacity) and get an inference endpoint. This is more paperwork than OpenAI's "model ID appears, you can call it immediately," and more control over region / capacity / private-endpoint-vs-public.

### The seven-step pipeline on Azure

Upload file (via the Foundry SDK or REST, same shape as OpenAI's Files API) → create fine-tune job → poll status → receive a "customized model" ID → **deploy it to a Foundry deployment** → call via the standard Azure OpenAI chat completions endpoint with the deployment name.

### The two things that surprise OpenAI-natives

- The extra deployment step. On OpenAI direct, `job.status == "succeeded"` means the model is callable. On Azure, it means the model is *registered* — you still need to create a deployment before it serves traffic.
- Region and SKU choices at deployment time. This is often the reason to use Azure — you can pin the model to a specific region (e.g., West Europe for GDPR requirements) with a private endpoint, in a way that OpenAI direct does not offer.

## The comparison-at-a-glance table

The whole tour, compressed:

| Vendor | Base catalog highlight | JSONL role convention | Storage for data | Method(s) | Callable how | Notable |
|---|---|---|---|---|---|---|
| **OpenAI (direct)** | GPT-4.1, GPT-4o, o-series | `system`/`user`/`assistant` | Vendor's Files API | SFT, DPO, RFT | `client.chat.completions.create(model="ft:...")` | Reference implementation. |
| **Anthropic (direct + Bedrock)** | Claude family | Top-level `system` + `user`/`assistant` in `messages` | Vendor Files API OR S3 (Bedrock) | SFT | `messages.create(model=<ft-id>)` | Bedrock path is enterprise default. |
| **Google Vertex AI** | Gemini 2.5 Flash / Pro | `user`/`model`, `contents` with `parts` | GCS bucket (`gs://`) | Supervised (LoRA under hood), preference on some tiers | `tunedModel` resource → `generateContent` | Deep GCP integration; IAM matters. |
| **Mistral (La Plateforme)** | Ministral, Mistral Small/Large, Codestral | `system`/`user`/`assistant` | Vendor Files API | SFT, DPO on some | `client.chat.complete(model=<ft-id>)` | Open weights available as escape hatch. |
| **Cohere** | Command R/R+/A, Rerank v3, Embed v3/v4 | `System`/`User`/`Chatbot` (capitalized) | Vendor upload | SFT (generative), contrastive (Rerank / Embed) | Command via `co.chat`; Rerank/Embed via their APIs | First-party rerank / embed FT. |
| **AWS Bedrock** | Multi-provider (Claude, Nova, Llama, Mistral, Cohere) | Depends on base | S3 | SFT, Continued Pretraining, more per base | `bedrock-runtime.invoke_model` / Converse | Provisioned Throughput cost model. |
| **Azure AI Foundry** | OpenAI catalog + Llama/Mistral/Phi | OpenAI's for OpenAI models; per-vendor otherwise | Azure Files / Blob | SFT, DPO (OpenAI models), RFT (o-series) | Deployment endpoint | Region pin, private endpoint, Azure compliance. |

<!-- needs-research: verify each row against the vendor's current docs. Model catalogs, supported methods, and JSONL schemas shift per release; this table is the shape, not a commitment. -->

## The decision procedure — first-order

Given a fine-tune to run, in what order should you consider vendors?

1. **What model do you need?** If you need GPT-4.1, you go to OpenAI or Azure. If you need Claude, you go to Anthropic or Bedrock. If you need Gemini, Vertex. Model choice usually collapses the vendor set to one or two.
2. **What is the procurement path?** If your company procures cloud through AWS, Bedrock is default. Through GCP, Vertex. Through Azure, Foundry. Direct vendor SDKs bypass this — a real option for a small team, not for a regulated enterprise.
3. **Do you need on-prem-ish data guarantees?** Bedrock (VPC endpoints), Azure Foundry (private endpoints), and Vertex (VPC-SC) are the three paths where "the data never leaves my cloud account" is a claim you can make. Direct SDKs generally send data to the vendor's infrastructure.
4. **Do you want the escape hatch to self-hosted?** Then Mistral (open weights) or a Bedrock Llama fine-tune (open weights) or Cohere's downloadable options (case-by-case) — because when you decide to self-host next quarter, you can download the base and re-run the exercise on your own hardware with mod-104.

Chapter 6 formalizes this into an ADR framework.

## What is genuinely the same across all vendors

If you take one thing away from this chapter beyond the vendor-specific quirks:

- **The dataset engineering (mod-102) is unchanged.** Deduplication, decontamination, licensing trace, dataset card — every vendor gets the same pre-processed dataset.
- **The eval discipline (mod-108) is unchanged.** Task-specific held-out sets, judged evals, safety evals — all run against the resulting model regardless of vendor.
- **The safety review (mod-106) is unchanged.** Every vendor's safety filters are their own; that is orthogonal to yours.
- **The manifest discipline** is unchanged — pin the base version, hash the JSONL, record hyperparameters, log the resulting model ID, estimate the cost.

What changes: the JSONL role names, the storage location, the API method names, the model-ID string format, the deployment step, and the pricing model. All of those are wrappers around the same seven steps.

## What this chapter did not cover

- **The dataset conversion.** How to go from a Hugging Face `messages` dataset to each vendor's specific JSONL — chapter 4 owns the conversion, with per-vendor validators.
- **Cost breakdown per vendor.** Chapter 5 has the four-part cost model applied to each vendor.
- **When to actually pick which vendor.** Chapter 6 turns the tour into a decision procedure and ADR template.
- **Running eval against each vendor's endpoint.** Chapter 7 covers the `lm-evaluation-harness` backend flags for OpenAI, Anthropic, Vertex, and generic HTTP endpoints.

## Summary

- Every managed vendor implements the same seven-step pipeline; each changes the schema fields, the storage layer, and one or two operational quirks. Learn the pipeline once (chapter 2); adapt to each vendor's differences from this chapter.
- The role-naming conventions are the single most-common source of first-run bugs: OpenAI says `assistant`, Anthropic says `assistant` (with `system` at top level), Vertex says `model`, Cohere says `Chatbot`. Convert deliberately.
- Storage layer differs: OpenAI / Anthropic direct / Mistral / Cohere upload via the vendor's API; Vertex reads from GCS; Bedrock reads from S3; Azure reads from Azure Files / Blob. The associated IAM configuration is the most common infrastructure-layer failure mode.
- Method availability is broader than "SFT only": OpenAI has DPO and Reinforcement Fine-Tuning, Bedrock has Continued Pretraining, Mistral has DPO on some models, Vertex has some preference-based paths. Verify the current catalog per vendor.
- Callable-artifact shape varies: OpenAI returns a model ID directly usable via `chat.completions`; Bedrock and Azure require a subsequent deployment step (Provisioned Throughput or Foundry deployment); Vertex returns a `tunedModel` resource name; Mistral / Cohere sit between.
- What does not change across vendors: your dataset (mod-102), your eval (mod-108), your safety review (mod-106), your manifest discipline. The substrate is portable even when the training loop is not.
