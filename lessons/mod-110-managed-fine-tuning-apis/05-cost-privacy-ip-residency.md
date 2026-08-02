# Cost, Data Residency, IP, and Privacy Trade-offs

**Estimated read:** 35 minutes.
**You should be able to, by the end:** decompose the total lifetime cost of a managed fine-tune into training, hosting, inference, and egress lanes; compare that total against the equivalent self-hosted cost from mod-109 chapter 7 for a specified workload; enumerate each vendor's default posture on data retention, provider-training-on-your-data, and regional residency, and cite the specific flags / SKUs that change the defaults; describe how ownership of the fine-tuned weights differs across vendors and what that means for portability; and produce a compliance one-pager (HIPAA, GDPR, SOC 2, EU AI Act) that a platform / legal reviewer can act on.

Chapter 6 uses everything here to make the managed-vs-self-hosted decision. Read this first.

## The four-part cost model

Every managed fine-tune has costs in four lanes. Miss one and your comparison to self-hosted is off by a large factor.

### Lane 1 — Training

Priced per token trained, at the vendor's fine-tuning-tier price for the base model:

```
training_cost = (tokens_in_train_file × n_epochs) × price_per_training_token[base]
```

For a mid-tier base at a hypothetical $0.025 per 1M training tokens (verify current pricing):

| Rows × mean tokens × epochs | Training tokens | Approx. cost |
|---|---|---|
| 10k × 500 × 3 | 15M | ~$0.40 |
| 100k × 500 × 3 | 150M | ~$3.75 |
| 1M × 500 × 3 | 1.5B | ~$37.50 |
| 100k × 2000 × 3 (long-context) | 600M | ~$15 |

Training cost is usually **not** the dominant lifetime cost. It is one-shot, small compared to inference at production scale, and dwarfed by hosting rent for long-lived fine-tunes with low QPS. Do the math; do not skip it.

### Lane 2 — Hosting

The retained fine-tuned model has an ongoing cost while it is available to call. Structure varies dramatically by vendor:

- **OpenAI direct**: a small monthly hosting rent per retained fine-tune. Multiplies with the number of models you keep — 50 fine-tunes = 50× the rent.
- **Anthropic direct**: check current terms; hosting rent structure has evolved.
- **AWS Bedrock (Provisioned Throughput)**: fixed **hourly** cost for reserved model units (MU). This is the big one: PT can be several dollars per hour per unit, and the model must be deployed to PT before you can invoke it (for most base models — some support on-demand for custom models, verify). At $/hour × 24 × 30 = monthly rent that dominates most other line items.
- **Azure AI Foundry (deployment SKU)**: similar to Bedrock — the deployment has a per-unit cost, either hourly or per-token depending on SKU.
- **Vertex AI**: tuned models are billed at inference time (per-token or per-endpoint hourly, depending on serving mode). Check current billing docs.
- **Mistral, Cohere**: mostly per-token inference, with some tier-specific hosting fees.

The rule: **for low-QPS workloads, per-token inference beats provisioned hosting; for high-QPS workloads, provisioned wins on unit cost.** Chapter 5 exercise 02 walks the crossover point.

### Lane 3 — Inference

Priced per token in / per token out at the model's inference tier. Fine-tuned models are typically priced at a **premium over the base model** — often 1.5–3× the base for direct-vendor tuning, roughly at base + a customization tax for Bedrock / Azure.

Do this arithmetic for your actual expected traffic:

```
monthly_inference_cost =   monthly_requests × mean_input_tokens  × price_per_1M_input
                         + monthly_requests × mean_output_tokens × price_per_1M_output
```

For a modest production workload — 1M requests/month, 500 input + 300 output tokens each, base pricing $0.30 / $1.20 per 1M with a 2× fine-tune premium ($0.60 / $2.40):

```
= 1_000_000 × 500  × $0.60/1M + 1_000_000 × 300  × $2.40/1M
= $300 + $720
= $1,020 / month
```

<!-- needs-research: verify current published fine-tune-tier pricing per vendor and per model before publishing exact numbers. Prices shift; the arithmetic shape does not. -->

For high-traffic (100M+ requests/month), inference is by far the dominant cost.

### Lane 4 — Egress and adjacent

Often forgotten:

- **Data transfer.** Bytes leaving the vendor's region are billed at cloud egress rates (typically $0.05–$0.09 per GB out to the internet, cheaper within-region). For a heavy inference workload with large outputs, this line item can be non-trivial.
- **File storage.** The training JSONL sitting in vendor storage has a per-GB-month cost. Small for a single JSONL; adds up for teams that keep hundreds of variants.
- **Batch API discounts.** OpenAI's Batch API (and equivalents) offers ~50% discount on inference in exchange for async submission and 24h SLA. Use it for eval and backfill — chapter 7 covers.
- **Logs and observability.** Every vendor has a logging surface (CloudWatch, Vertex logs, OpenAI dashboard) with its own storage cost. Usually small but nonzero.

### The total-cost formula

Putting it together, amortized to monthly:

```
monthly_total =   training_cost / amortization_months          (Lane 1, per-fine-tune amortized)
                + hosting_rent                                  (Lane 2, per-fine-tune per-month)
                + inference_cost_per_month                      (Lane 3, traffic-dependent)
                + egress_and_adjacent                           (Lane 4, traffic-dependent)
```

### Compare against self-hosted (mod-109 chapter 7 output)

Self-hosted cost per million tokens for a fine-tuned open model, per mod-109 chapter 7's arithmetic:

```
$/M_tokens_self_hosted = (GPU_hourly_cost × hours) / M_tokens_served
```

For a fine-tuned 8B on 1× L40S at $0.50/hour serving 200k tokens/sec sustained: ~$0.70 per 1M tokens on paper. Add operational overhead (on-call, autoscaling headroom, capacity for burst traffic, redundancy) and the real number is usually 2–5× the paper number.

The comparison matrix template you should own:

|  | Managed (OpenAI FT of gpt-4.1-mini) | Managed (Bedrock FT of Claude Haiku) | Self-hosted (Llama 3.1 8B FT on L40S) |
|---|---|---|---|
| One-time training cost | $X | $Y | $Z (GPU-hours × rate) |
| Monthly hosting | small | large (PT MU/hr) | large (reserved GPU) |
| $/1M input tokens (production) | $A | $B | $C |
| $/1M output tokens (production) | $A' | $B' | $C' |
| Monthly cost at 10M requests | $M1 | $M2 | $M3 |
| Monthly cost at 100M requests | $M1×10-ish | $M2×10-ish | often $M3×2 (capacity headroom absorbs some growth) |
| Peak-load headroom | vendor problem | vendor problem | your problem |
| Cost of a wrong model choice | rerun training, no infra change | rerun training, no infra change | rerun training + possibly re-quantize / re-deploy |

The crossover: **self-hosted wins on $/token at scale; managed wins at low volume and when the model itself is unavailable open-source.** Chapter 6 formalizes.

## Data residency — where the bytes physically live

For regulated workloads (EU personal data, HIPAA-protected PHI, government / defense, some financial), the location of the training data and the fine-tuned weights during and after the job is a hard requirement, not a preference. The vendors have very different postures:

| Vendor | Where data lives during training | Regional pinning | Cross-region movement guarantees |
|---|---|---|---|
| **OpenAI direct** | OpenAI infrastructure (US, EU, Japan, and other regions as offered) | Some regional deployment options; check current API terms and Data Residency page | Data does not leave the selected region for tuning under standard T&C, but verify current terms |
| **Anthropic direct** | Anthropic infrastructure; region policy per current docs | Limited direct region control (as of late 2025 / 2026 — verify) | Check current Trust Center |
| **AWS Bedrock** | The **AWS region you invoke from** | Yes, at the AWS region granularity; data pinned to region | Full control — no cross-region movement unless you configure it |
| **Google Vertex AI** | The **GCP region of your project / bucket** | Yes; supports Data Residency Assurance in supported regions | VPC Service Controls prevent cross-boundary movement when enabled |
| **Azure AI Foundry** | The **Azure region of your resource** | Yes; supports region pinning and Private Endpoints | Full control via Azure networking constructs |
| **Mistral La Plateforme** | Mistral EU infrastructure (check current terms) | Primarily EU; verify |
| **Cohere** | Cohere US infrastructure by default, EU option for some products | Regional deployment options; verify current terms |

The pattern: **cloud-native surfaces (Bedrock, Vertex, Azure) give you fine-grained regional control; direct vendor surfaces give you less.** For regulated data, the cloud-native surfaces are almost always the right choice.

## Training-on-your-data — the default, and how to change it

The single most misunderstood aspect of managed fine-tuning. Current defaults for the major vendors:

<!-- needs-research: this is the single fastest-moving section in this chapter. Verify each row against the vendor's current Trust Center / Data Usage Policy before publishing. Provider policies changed several times in 2024-2025; the below is the shape, not a current commitment. -->

| Vendor | Default: does vendor train on your data? | How to change |
|---|---|---|
| **OpenAI (API)** | **No** for API traffic (fine-tuning and inference). Data may be retained for abuse monitoring, up to a stated retention window. | Zero Data Retention (ZDR) available on request for eligible customers. |
| **Anthropic (API)** | **No** for API traffic by default. Retention window per current policy. | Enterprise ZDR-equivalent available; verify current terms. |
| **AWS Bedrock** | **No.** Bedrock does not use your data to train or improve foundation models. | Default posture; verify in AWS Bedrock service terms. |
| **Google Vertex AI** | **No** for tuned models and inference on paid tier. | Default posture; verify current policy. |
| **Azure AI Foundry** | **No.** Azure OpenAI Service explicitly does not use your prompts or completions to train models. | Default posture; verify Azure AI Foundry data privacy docs. |
| **Mistral La Plateforme** | **No** on paid tier by default; check current terms. | Verify Mistral privacy policy. |
| **Cohere** | **No** for paid API traffic by default. | Verify Cohere Data Usage Policy. |

The key nuance: **API traffic is not the same as free-tier or logged-training features.** ChatGPT (the consumer product) has different terms; free playground modes may have different terms; certain feature previews may collect data for improvement. Read the specific product's data usage policy — do not assume "the same vendor's API terms."

### The retention window

Distinct from training-on-data. Even when a vendor promises not to train on your data, they typically **retain it for some window** for abuse detection, safety review, and auditing. Windows range from 0 (ZDR customers) to 30 days (typical API default) to several months (some enterprise plans). If your data cannot be retained *at all* by a third party, ZDR is a hard requirement, not a nice-to-have.

## Intellectual property — who owns the fine-tuned weights

The rarely-asked but load-bearing question. Two shapes:

**Shape A: you own outputs; vendor retains everything else.**
The fine-tuned weights are the vendor's IP; you have a license to call the model via API. You cannot download the weights, cannot serve them elsewhere, cannot re-train them on another provider. You own the *outputs* the model generates for you.

Most direct vendor products (OpenAI direct, Anthropic direct, some tiers of Cohere / Mistral direct) follow Shape A.

**Shape B: you own the custom model artifact.**
The fine-tuned weights are your IP (or at minimum, licensed to you exclusively). You may still not be able to download them — but the vendor cannot use them for other customers, cannot redistribute them, and (in some cases) will provide the weights on contractual exit.

AWS Bedrock Custom Models generally follow Shape B — the custom model is an asset in your AWS account, though whether you can *download the raw weights* depends on the underlying model's license (a Llama fine-tune's weights fall under Meta's Llama license; a Claude fine-tune's weights fall under Anthropic's IP). Similar shape on Azure Foundry for OpenAI models.

**Shape C: open-weight base + your weights.**
For managed fine-tunes on open-weight bases (Llama family, Mistral open-weight family), you generally have the option to download the base at any time and re-run the fine-tune on your own infrastructure — the escape hatch chapter 3 highlighted. The fine-tuned weights themselves may or may not be downloadable, but the base is portable.

**Read the current MSA / T&C.** Every claim above is subject to the vendor's current contract; assume nothing without a legal review.

## Compliance surface — HIPAA, GDPR, SOC 2, EU AI Act

The one-page compliance summary your platform / legal team will want:

| Compliance framework | What it needs from a managed FT vendor |
|---|---|
| **HIPAA** (US healthcare) | A signed **Business Associate Agreement (BAA)**. Bedrock, Azure, and Vertex all support BAAs on qualifying tiers. OpenAI, Anthropic, and others have BAA offerings on enterprise plans — verify current coverage. Without a BAA, no PHI may leave your systems into the vendor's ingest, full stop. |
| **GDPR** (EU personal data) | Data Processing Agreement (DPA), Standard Contractual Clauses (SCCs) or equivalent transfer mechanism if the vendor processes data outside the EEA, EU-hosted option for the training data, deletion / rectification workflow. All major vendors offer DPAs; EU-region hosting is stronger on Bedrock (EU regions), Vertex (EU regions), Azure (EU regions), Mistral (EU-native). |
| **SOC 2 (Type II)** | The vendor's attestation report, current within the audit window. Every major vendor publishes SOC 2 reports; request via the vendor's Trust Center or CSA STAR. |
| **ISO 27001 / 27017 / 27018** | Same — request current certificates. |
| **EU AI Act** (from 2025 onward, phased) | Vendor-side: providers of general-purpose AI models publish technical documentation and training-data summaries. Customer-side (you): deployer obligations depend on your use case's risk tier. A fine-tuned model deployed for high-risk use (employment, credit, education, biometric ID) triggers additional obligations you inherit — record-keeping, human oversight, transparency. Consult current EU AI Act guidance; the enforcement dates roll out through 2027. |
| **FedRAMP / IL 4/5 / DoD IL** | Government tiers. Bedrock and Azure have FedRAMP-authorized service offerings; specific model availability varies. Direct vendor APIs generally do not have FedRAMP authorization for their standard offerings. |

<!-- needs-research: EU AI Act obligations are still being clarified through implementing acts and Commission guidance in 2025-2026. Cite the current Commission FAQ and any relevant national-competent-authority guidance before publishing the deployer-obligations paragraph. -->

The rule: **for regulated workloads, the cloud-provider surfaces (Bedrock, Vertex, Azure) are the safe default** because they inherit the whole cloud provider's compliance apparatus (BAAs, EU regions, FedRAMP, SOC 2, ISO). Direct vendor surfaces are viable, but the compliance conversation is longer, per-vendor, and less pre-cleared.

## The compliance one-pager template

What you hand to your platform / legal reviewer to unblock a managed FT decision:

```markdown
# Managed Fine-Tune Compliance One-Pager

## Project
- Name: support-triage-v3
- Vendor: [OpenAI direct | Bedrock | Azure Foundry | Vertex | Anthropic direct | Mistral | Cohere]
- Base model: [e.g., ft-tier of gpt-4.1-mini-2025-07-14]
- Region: [e.g., us-east-1 / eu-west-1 / global]

## Data
- Training data type: [free-text / customer data / logs / synthetic / other]
- Data sensitivity: [none | PII | PHI | PCI | trade-secret | other]
- Data volume: [rows / tokens / GB]
- Data residency requirement: [none | EU-only | US-only | on-prem-only | ...]
- Retention requirement: [any window OK | <30d | ZDR only]

## Vendor posture (verified against Trust Center on YYYY-MM-DD)
- Vendor trains on our data by default: [no + policy link]
- Retention window: [days + policy link]
- ZDR available: [yes / no / on request]
- Signed BAA / DPA / SCCs: [attached]
- Region pinning available: [yes + region]
- Cross-region movement: [prevented by ... / not prevented]

## IP
- Weights ownership: [Shape A vendor / Shape B customer / Shape C open-base]
- Downloadable weights on exit: [no / yes for base / yes for adapter]
- Portability plan if vendor exits: [ ... ]

## Compliance
- Frameworks required: [ ... ]
- Frameworks satisfied by vendor tier: [ ... with attestations attached]
- Deployer obligations under [e.g., EU AI Act]: [ ... ]

## Cost
- Estimated first-year cost: [Lane 1 + 12 × Lane 2 + expected Lane 3]
- Break-even vs. self-hosted (mod-109 baseline): [request rate]

## Approvals
- Security review: [name, date]
- Legal review: [name, date]
- Platform review: [name, date]
```

Every managed fine-tune worth shipping goes through this one-pager. It is the artifact chapter 6's ADR references.

## The pattern: managed vs. self-hosted trade-off table

The rest of this chapter as one table, one row per property. Chapter 6 uses this table as the input to the decision.

| Property | Managed (typical) | Self-hosted (typical) |
|---|---|---|
| Training compute cost | Per-token, no GPU procurement | GPU-hour amortized |
| Hosting cost (low QPS) | Small per-model rent or included | Fixed GPU cost regardless of load |
| Hosting cost (high QPS) | Grows linearly with tokens | Amortized across the load |
| Inference cost per M tokens | Vendor pricing (fine-tune premium) | GPU-cost / throughput (~$1-5/M for well-utilized L40S/H100) |
| Time-to-first-model | Hours (data + JSONL + job) | Days to weeks (also data + GPU setup + training + serving) |
| Model choice | Vendor's tuneable catalog | Any open-weight model |
| Method choice | SFT (all), DPO (some), RFT (few) | SFT, DPO, KTO, ORPO, RLHF, PPO, GRPO, whatever your training loop supports |
| Weight ownership | Vendor's or vendor's + license | Yours |
| Data residency | Depends on vendor / SKU | You choose your region |
| Data movement | Vendor's infrastructure | Yours |
| Compliance surface | Vendor's certifications + your inheritance | Yours end-to-end |
| Rate limits | Vendor's TPM / RPM | Your GPU capacity |
| Peak-load headroom | Vendor absorbs (or does not) | Yours to plan |
| SLA on inference | Vendor's stated SLA | Yours to configure and meet |
| Cost of vendor exit | Retrain on new vendor (data survives) | Redeploy (weights survive) |
| Cost of model deprecation | Retrain (base gone) | Not applicable — you own the base |

Chapter 6 turns this into a decision procedure. Chapter 7 covers the eval reconciliation across both.

## Common cost-and-privacy pitfalls

**"We assumed inference was cheap because training was cheap."** Training cost is one-shot and small; inference cost is monthly and traffic-scaled. A $2 training run supporting 10M requests/month at $0.60 per 1K output tokens is $6000/month, not $2.

**"We forgot the hosting rent."** OpenAI hosting rent per fine-tune, multiplied by 20 retained fine-tunes, is a real line item. Bedrock Provisioned Throughput, at $/hour × 24 × 30, is a much larger line item. Delete unused fine-tunes; audit PT usage monthly.

**"We picked the vendor before we checked residency."** Vertex's europe-west4 satisfies most GDPR requirements; OpenAI direct may not (in some interpretations, for certain data categories). This is an "unblock the project or don't" question — resolve before the JSONL is generated.

**"We assumed the vendor would not train on our data — but we're on the free tier."** Free-tier and paid-tier data usage policies differ. Verify the specific product's data usage policy.

**"We didn't get a BAA / DPA before uploading."** The upload is the event that puts data in the vendor's infrastructure. If the BAA / DPA is not signed by then, the upload itself is the compliance violation. Contract first, upload second.

**"We planned exit via 'we'll re-fine-tune on a different vendor if we need to.'"** True, but retraining takes time, requires the JSONL to be portable (chapter 4's discipline), and requires the alternate vendor to be provisioned and evaluated ahead of time. "We'll deal with it" is not an exit plan; a pre-signed contract with an alternate vendor and a tested conversion pipeline is.

## Summary

- Managed fine-tuning has four cost lanes: **training** (per-token × epochs, one-shot), **hosting** (per-model per-month, or per-hour for provisioned SKUs), **inference** (per-token, at a premium over the base model, dominates at scale), **egress and adjacent** (bytes, storage, batch discounts, logging). Chapter 6 uses the total to compare against self-hosted from mod-109 chapter 7.
- The vendor's **data residency** and **retention** posture depends on the surface: cloud-native (Bedrock, Vertex, Azure) give fine-grained regional control; direct vendor APIs give less. Verify per SKU, not per vendor.
- All major vendors default to **not training on your data** for paid API traffic in 2026, with varying retention windows. ZDR (Zero Data Retention) is available on enterprise tiers for OpenAI and equivalents on others. Free-tier and consumer products have different terms; verify the specific product.
- **IP ownership** of the fine-tuned weights takes three shapes: Shape A (vendor owns, you have API access), Shape B (you own the custom model as an account asset, may or may not download the weights), Shape C (open base weights + your fine-tune, downloadable escape hatch). Every project needs to know which shape it is in.
- **Compliance surface** (HIPAA, GDPR, SOC 2, EU AI Act, FedRAMP) inherits from the vendor + region + SKU. Cloud-native surfaces are the pre-cleared path for regulated workloads.
- The **compliance one-pager** is the artifact you produce for every managed fine-tune. It is what unlocks legal / security / platform sign-off and what chapter 6's ADR references.
- Common pitfalls: assuming inference is cheap, forgetting hosting rent, picking vendor before checking residency, assuming BAA / DPA is in place before it is signed, treating "we'll switch vendors" as an exit plan when the alternate is not provisioned.
