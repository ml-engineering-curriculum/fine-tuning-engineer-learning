# The Managed Fine-Tuning Landscape

**Estimated read:** 25 minutes.
**You should be able to, by the end:** say what a managed fine-tuning API is and what it is not, name the seven vendors that matter in 2026 and roughly what each one supports, describe the shared shape that every managed API follows (upload → validate → job → checkpoint → inference), and place the decision "managed vs. self-hosted" on the same map as the mod-109 quantization / serving decisions rather than as a separate question.

## What "managed fine-tuning" actually means

Everything in mods 103–109 assumed you owned the training loop. Your GPUs, your `trainer.train()` call, your checkpoint directory, your vLLM process. Managed fine-tuning shifts that boundary: **the vendor owns the training loop; you own the data, the eval, and the request that consumes the resulting model.** In exchange for giving up ownership of the training loop, you get:

- No GPU procurement, no distributed-training config, no CUDA versions to reconcile.
- A fine-tune on a model you cannot download — the closed frontier ones (OpenAI's GPT-4.1 family, Anthropic's Claude 4.x family, Google's Gemini 2.5 family, Cohere's Command family, various Mistral tier) — which is often the whole reason for using a managed API in the first place.
- Someone else's inference stack answering the requests at production quality.

You give up:

- The weights. You cannot download the fine-tuned model; you can only call it by ID.
- Portability. The `ft:...` ID from OpenAI does not run on Anthropic. Switching vendors is a re-fine-tune, not a redeploy.
- Fine-grained control. `n_epochs` and `learning_rate_multiplier` are the surface; there is no `weight_decay`, no gradient clipping knob, no LR schedule choice, no chance to inspect a loss curve at step 137.
- Cost visibility. The training-compute cost is opaque (you pay per token trained, not per GPU-hour), and the ongoing hosting rent is a line item that self-hosted models do not have.

This is the whole trade in one sentence. The rest of the module makes each of these concrete.

## Not the same as calling a hosted model

**Managed fine-tuning is not RAG, prompt engineering, or few-shot on a hosted model.** All three of those change what the model sees at inference time without ever adjusting a weight. Managed fine-tuning actually trains weights — the vendor runs SFT (and, increasingly, preference optimization) on your data and produces new weights that live inside their inference stack.

The distinction matters because the failure modes and the compliance surface are completely different. RAG leaks a document into a prompt; fine-tuning bakes patterns from your data into the model. If your organization's data-governance team has cleared RAG on an API, they have not automatically cleared fine-tuning on the same API — the training-data path is a separate contract question. Chapter 5 covers this in detail.

**Managed fine-tuning is also not distillation.** OpenAI's "model distillation" workflow (Stored Completions → fine-tune the smaller model on the larger model's outputs) is a *use* of the fine-tuning API, not a separate product; it sits on top of the same JSONL upload / job / model-ID plumbing this module teaches. When you see "distillation" in a vendor doc, read it as "here is a supported way to build the SFT dataset that you then fine-tune with."

## The vendors that matter in 2026

Seven vendors run production managed fine-tuning at a scale that a fine-tuning engineer will encounter. Each is here because either the model family is not otherwise available (closed weights) or the operational stack around the fine-tune is what you are buying (Bedrock, Azure).

<!-- needs-research: verify each vendor's currently-tuneable model list and posted method (SFT / DPO / adapter / full) against their live docs before shipping the module — this landscape shifts every ~90 days as new base models roll out and old ones deprecate. -->

| Vendor | Product name | Models currently tuneable (as of Aug 2026) | Method(s) | Notable |
|---|---|---|---|---|
| **OpenAI** | Fine-tuning API | GPT-4.1, GPT-4.1-mini, GPT-4o, GPT-4o-mini, o-series (limited); Reinforcement Fine-Tuning on o-series | SFT, DPO (chat), Reinforcement FT (o-series) | Deepest tooling; canonical API shape. Chapter 2 uses it as the reference. |
| **Anthropic** | Claude fine-tuning (via AWS Bedrock and, in preview, the Claude API) | Claude Haiku family; broader Claude models in Bedrock preview | SFT | Bedrock-first; direct Claude API path exists but has narrower model coverage. |
| **Google (Vertex AI)** | Gemini supervised tuning | Gemini 2.5 Flash, Gemini 2.5 Pro (tunable variants); Gemma via a separate self-hosted path | Supervised tuning (LoRA-style adapter under the hood) | Deep integration with GCS, BigQuery, Vertex Pipelines. |
| **Mistral (La Plateforme)** | Fine-tuning API | Ministral 3B/8B, Mistral Small, Mistral Large, Codestral | SFT, some models support DPO | Open-weight models tuneable through the API OR downloadable and self-hosted — the hybrid option. |
| **Cohere** | Fine-tuning | Command R, Command R+, Command A, Rerank models, Embed models | SFT (generation), specialized workflows for Rerank and Embed | The only vendor with first-party fine-tuning for rerankers and embedders — matters for retrieval stacks. |
| **AWS Bedrock** | Custom Models | Anthropic Claude family, Amazon Nova family, Meta Llama family, Mistral family, Cohere Command family | SFT, Continued Pretraining, some model-specific methods | Multi-provider under one control plane; the enterprise-procurement default. |
| **Azure AI Foundry** | Fine-tuning | OpenAI models (same catalog as OpenAI direct), Meta Llama, Mistral, Phi family | SFT, DPO on OpenAI models, RFT on o-series | Same OpenAI models with Azure's compliance / residency / networking wrappers. |

Two secondary paths worth naming but not walked through in chapters:

- **Fireworks / Together / Replicate / Anyscale-style "serverless fine-tuning" on open weights.** These are LoRA-training-as-a-service on open bases (Llama, Qwen, DeepSeek). They belong to a category between "managed" and "self-hosted" — you don't run the training, but you often can download the LoRA. Chapter 6 discusses when this hybrid is the right pick.
- **Model-provider first-party fine-tuning on smaller labs** (Databricks Mosaic, Snowflake Arctic, IBM watsonx, Alibaba's Qwen paid tier). These follow the same API shape and are mostly relevant when the surrounding platform (Databricks, Snowflake) is where the data already lives.

## The shared shape of every managed FT API

Under the surface, every vendor's fine-tuning API follows the same seven-step pipeline. Learn the shape once and every vendor becomes a matter of "which fields, which quirks, which model IDs" — not a fresh mental model.

```
                       Your side                          Vendor side
             +-------------------------+          +---------------------------+
             |  1. Prepare dataset     |          |                           |
             |     (mod-102 → JSONL)   |          |                           |
             +-----------+-------------+          |                           |
                         |                        |                           |
                         v                        |                           |
             +-------------------------+          |                           |
             |  2. Upload as a "file"  |--------->|  Ingest, hash, store      |
             |     (multi-GB OK)       |          |                           |
             +-----------+-------------+          |                           |
                         |                        v                           |
                         |               +-----------------+                  |
                         |               | 3. Validate     |                  |
                         |               |   (schema, size,|                  |
                         |               |   safety, PII)  |                  |
                         |               +--------+--------+                  |
                         |                        |                          |
                         v                        v                           |
             +-------------------------+   +--------------+                  |
             |  4. Create fine-tune    |-->| Enqueue job  |                  |
             |     job (model, method, |   +--------------+                  |
             |     hyperparams, files) |          |                          |
             +-------------------------+          v                          |
                         |                +--------------------+             |
                         |                | 5. Train (opaque)  |             |
                         |                |    — you watch     |             |
                         |                |    events / loss   |             |
                         |                +----------+---------+             |
                         |                           |                       |
                         v                           v                       |
             +-------------------------+   +---------------------+           |
             |  6. Receive model ID    |<--| Checkpoint saved,   |           |
             |     ft:base:org:suffix  |   | job.status=succeeded|           |
             +-----------+-------------+   +---------------------+           |
                         |                                                   |
                         v                                                   |
             +-------------------------+   +---------------------+           |
             |  7. Inference against   |<--| Model deployed to   |           |
             |     the model ID via    |   | inference endpoint  |           |
             |     the normal chat API |   +---------------------+           |
             +-------------------------+                                     |
             ^                                                               |
             |                                                               |
             |----------- Eval (mod-108) runs against this --------------|
```

Every chapter after this one is a deep dive into a subset of these steps for one or more vendors:

- Chapter 2 walks through steps 2–7 for OpenAI in detail.
- Chapter 3 tours the same steps for the other six vendors, focusing on what is different.
- Chapter 4 owns step 1: the dataset conversion.
- Chapter 5 layers the cost / privacy contract onto every step.
- Chapter 6 decides *whether* to walk this pipeline for the closed-weight model or the open-weight self-hosted alternative.
- Chapter 7 focuses on step 7 and how it feeds mod-108's eval gate.

## The two axes of the trade

You will make the managed-vs-self-hosted decision at least once per project, and often several times as the project matures. Two axes drive it. Every subsequent chapter can be read as sharpening one axis or the other.

### Axis 1 — Capability ceiling

Some tasks are only served well by a closed-weight frontier model. If your production-grade quality bar cannot be met by any 7B–70B open model you can afford to fine-tune, then fine-tuning the frontier model is the productive move even if you would prefer to own the weights. This is the axis that pulls you toward managed.

Not every task is on this axis. A classification model, an entity extractor, a rewriter, a domain-specific chatbot with narrow surface — these routinely reach acceptable quality on a fine-tuned 8B open model, and the "managed frontier" premium buys nothing you need. Chapter 6 walks through the concrete regimes.

### Axis 2 — Portability and control cost

Every managed contract adds a switching cost. If you fine-tune on OpenAI and later need to move to Anthropic (customer request, pricing shift, availability incident, regulatory change), you retrain — the artifact does not port. If you fine-tune on your own Llama and later need to move from vLLM to TGI, you redeploy — the artifact does port. The switching cost is a permanent tax on the managed path.

For low-stakes, short-lived, or exploratory work, this tax is invisible. For a fine-tune that will be under your platform's SLO for years, it is a large cumulative cost. Chapter 6 turns this into a concrete calculation; chapter 5 shows how the vendor's contract terms interact with it.

## Where this module sits relative to the rest

The one-page mental model:

```
    +-----------------------------------------------------------------+
    |                    Fine-Tuning Engineer                          |
    |                                                                  |
    |    +-------------------+          +-------------------+         |
    |    |  Self-hosted path |          |  Managed path     |         |
    |    |  (mods 103-109)   |          |  (this module)    |         |
    |    +---------+---------+          +---------+---------+         |
    |              |                              |                    |
    |              v                              v                    |
    |    +---------+----------+       +-----------+-----------+       |
    |    | Local checkpoint,  |       | Hosted model ID,      |       |
    |    | vLLM/TGI/llama.cpp |       | vendor's inference    |       |
    |    +---------+----------+       +-----------+-----------+       |
    |              |                              |                    |
    |              +---------------+--------------+                    |
    |                              |                                   |
    |                              v                                   |
    |                    +---------+----------+                        |
    |                    | Same dataset       |  (mod-102)             |
    |                    | Same eval gate     |  (mod-108)             |
    |                    | Same safety review |  (mod-106)             |
    |                    | Same platform      |  (mod-111)             |
    |                    +--------------------+                        |
    +-----------------------------------------------------------------+
```

The block at the bottom is the point of this module: **the shared substrate does not change when the training loop moves off-cluster.** If your dataset build, eval gate, safety review, and platform are structured cleanly, swapping between the two paths is a config change — the same messages JSONL feeds either, the same eval JSON comes back from either, the same governance workflow signs off on either. If they are ad-hoc, every managed experiment reinvents them.

## Common misconceptions worth naming early

**"Managed is more expensive."** Often true for high-QPS production traffic, but the training-time cost of a managed fine-tune is frequently *lower* than the equivalent self-hosted job once you account for the amortized cost of your own GPU capacity, the engineer-hours to configure the training run, and the failed-run overhead. Chapter 5 does the full arithmetic; the takeaway is that neither side is universally cheaper.

**"Managed is less private."** The default configuration on major vendors' fine-tuning products is that your data is not used for provider model training and is retained only as long as needed to run and audit the job. This is not the same as "no data leaves your VPC" — Bedrock and Azure offer that; OpenAI direct does not — but "the vendor trains on your data by default" is not the current reality. Chapter 5 covers each vendor's actual contract.

**"Managed models are easier to eval."** They are *harder* to eval in one specific way (log-likelihood tasks — the vendor may not expose logprobs, or the model itself may be chat-only) and easier in another (the endpoint is always up; there is no vLLM to stand up). Chapter 7 covers the reconciliation.

**"Managed means no method choice."** Two years ago this was largely true (SFT-only across the board). It is not true now: OpenAI supports DPO on chat models and Reinforcement Fine-Tuning on o-series, Bedrock supports Continued Pretraining alongside SFT, and Mistral / Cohere / Vertex all expose beyond-SFT surfaces. The method surface is smaller than the self-hosted world (mod-104 / 105 / 106), but not empty. Chapter 3 catalogs it.

## The chapter map from here

Each chapter is designed to stand alone once you've read this one:

- **Chapter 2** teaches the mechanics using OpenAI as the reference implementation. If your team is on any other vendor, you'll still recognize 80% of what chapter 3 covers because it maps back to the same seven-step pipeline.
- **Chapter 3** is a tour: skim the whole thing once, then bookmark the vendor(s) you actually use. It is intentionally scoped to "what is different from OpenAI" rather than re-explaining the whole API for each vendor.
- **Chapter 4** solves the concrete pain point that trips up every first managed fine-tune: the dataset you have (Hugging Face `datasets` in messages format) is not the format the vendor wants, and the conversion is subtle enough to break silently.
- **Chapter 5** is what your platform team, compliance team, and finance team will read. Bring these numbers to any managed-vs-self-hosted decision meeting.
- **Chapter 6** is the ADR. It teaches the decision framework and gives you a template that records the choice so the next model release doesn't restart the argument.
- **Chapter 7** is the mod-108 handoff: the eval gate must run against whichever backend you picked, and the numbers must be like-for-like against the alternative. This chapter closes the loop.

## Summary

- Managed fine-tuning trades ownership of the training loop for the ability to fine-tune closed-weight frontier models (or to skip the GPU-procurement / distributed-training cost on an open one). You keep the dataset, the eval, and the request; you give up the weights, portability, and fine-grained control.
- The seven vendors that matter in 2026 are OpenAI, Anthropic (via the Claude API and Bedrock), Google Vertex AI (Gemini), Mistral (La Plateforme), Cohere, AWS Bedrock (multi-provider), and Azure AI Foundry. Serverless-LoRA-on-open-weights providers (Fireworks, Together, Replicate) are a hybrid category chapter 6 discusses.
- Every managed API follows the same seven-step pipeline: prepare dataset → upload as a file → validate → create job → train → receive model ID → inference. The rest of the module walks each step for each vendor.
- Two axes drive the managed-vs-self-hosted decision: **capability ceiling** (does any open model reach your quality bar for an affordable training cost?) and **portability / control cost** (how much do you pay in switching cost over the artifact's lifetime?). Chapter 6 turns these into a decision procedure.
- The shared substrate — dataset, eval gate, safety review, platform — should not change when you flip between paths. If it does, the fault is in the substrate, not in the vendor.
