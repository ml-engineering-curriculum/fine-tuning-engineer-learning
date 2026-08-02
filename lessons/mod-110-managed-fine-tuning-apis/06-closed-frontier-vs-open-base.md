# Fine-Tuning a Closed Frontier vs. Fine-Tuning an Open Base

**Estimated read:** 30 minutes.
**You should be able to, by the end:** for a specific product / use case, apply the five-axis decision framework (capability ceiling, cost, portability, method flexibility, dependability) to choose between fine-tuning a closed-weight frontier model through a managed API and fine-tuning an open-weight base model on your own hardware; name the concrete regimes where each answer is correct; identify the "hybrid" arrangements that let you avoid committing prematurely; and produce an Architecture Decision Record (ADR) that documents the choice, its assumptions, its exit triggers, and its next review date.

Chapters 1–5 gave you the pieces: the pipeline (1), the concrete APIs (2, 3), the dataset conversion (4), and the cost / privacy trade-offs (5). This chapter is the decision procedure that ties them together.

## The question, precisely

You have a task. You have a dataset (mod-102). You have an eval gate (mod-108). You need a fine-tuned model in production. Which of the two paths do you take?

- **Path A — Fine-tune a closed-weight frontier model.** You give up ownership of the weights and the training loop; you get access to a much stronger base than any open model at comparable price. Uses the managed APIs of chapters 2–3.
- **Path B — Fine-tune an open-weight base model.** You own the weights and the training loop; you have to select, tune, quantize, and serve the model yourself. Uses the pipeline of mods 103, 104, 105, 106, 107, 108, 109.

There is a small third path — **fine-tune an open-weight base via a managed serverless-LoRA vendor** (Fireworks, Together, Replicate, some Bedrock offerings) — that inherits some of both. We come back to it at the end.

## Why the choice is not obvious

If it were, this chapter would be a single line. The reason it is genuinely load-bearing:

- **Closed frontier models are capability-different, not just capability-larger.** GPT-4.1, Claude Sonnet 4.5+, Gemini 2.5 Pro often carry capabilities (long-context reasoning, instruction hierarchy, tool-use robustness, multilingual coverage) that no 8B–70B open model reaches at comparable cost. If your task lives on that capability edge, no amount of self-hosted fine-tuning gets you there.
- **Open base models are portable and controllable in ways closed models are not.** The weights are yours forever, quantization / serving / cost-per-token is under your control, and vendor risk is minimal. If your task is well-served by 8B–70B and you plan to run it for years, self-hosted is often the correct long-term posture.

Both are true simultaneously. The decision is picking which set of trade-offs your specific product can absorb.

## The five-axis decision framework

Score your project on each of five axes. Each axis pulls toward one path or the other. Where they point matters more than the individual scores.

### Axis 1 — Capability ceiling

The question: **Can any open base model you can afford to fine-tune reach your quality bar for this task?**

To answer, run a **capability probe** *before* committing to either path:

1. Take 100–300 representative task inputs from your eval set (mod-108 chapter 3's task-specific held-out set is ideal).
2. Run them against a strong open model at few-shot / no-tuning (Llama 3.3 70B Instruct, Qwen 2.5 72B, Mistral Large, whatever your open ceiling is).
3. Run them against a strong closed model at few-shot / no-tuning (GPT-4.1, Claude Sonnet 4.5+, Gemini 2.5 Pro).
4. Grade both with the mod-108 chapter 4 judge (or humans, if the judge is not yet calibrated).

Three outcomes:

- **Open model matches closed model.** The task does not need the frontier. Axis 1 → open base (Path B).
- **Open model falls short but by a bounded amount** (e.g., 5–10 points on your metric). Fine-tuning the open model may close the gap. Axis 1 → open base with a hedge (Path B, with willingness to switch if FT doesn't close it).
- **Open model falls short by a wide margin** (20+ points). Fine-tuning may reduce the gap but rarely reverses it — you are structurally under-capacity. Axis 1 → closed frontier (Path A).

The capability probe is worth its weight in gold. Skipping it means committing 4-6 weeks of engineering to a path that a $20 experiment could have told you was wrong.

### Axis 2 — Cost

For your specific traffic pattern, project both paths' total lifetime cost using chapter 5's arithmetic and mod-109 chapter 7's arithmetic.

- **Low volume (< 1M requests/month).** Managed almost always wins on total cost — the fixed cost of self-hosting (GPU + on-call + operational headroom) dominates. Axis 2 → closed frontier / managed.
- **Medium volume (1M–100M requests/month).** Depends on the specific model, hardware, and vendor pricing. Often a wash, or self-hosted wins by 1.5–3× if the workload can amortize a well-utilized GPU. Axis 2 → run the numbers.
- **High volume (100M+ requests/month) with a task that fits an open model.** Self-hosted usually wins by 5–20× on $/token — the GPU is fully utilized and the vendor's per-token premium becomes crushing. Axis 2 → open base / self-hosted.
- **High volume with a task that requires the frontier.** You pay the premium because the alternative is a quality regression. Axis 2 → closed frontier, and start planning capacity commitments.

For projects with unknown volume, budget for the worst-case scenario on each side and see which is more tolerable. Getting an unexpected 10× volume spike on managed is a bill; on self-hosted it is an outage.

### Axis 3 — Portability

The question: **How costly is switching later, in terms of engineer-hours and calendar time?**

- **Closed frontier** — switching means retraining from scratch on the new vendor. Data and eval are portable; the model is not. Switching cost: measured in weeks.
- **Open base, self-hosted** — switching hardware / runtime is redeploying the same checkpoint. Switching base model is retraining; but the retrained artifact remains yours. Switching cost: measured in days for runtime changes, weeks for base changes.
- **Open base, serverless-LoRA managed** — switching from the managed vendor to self-hosted is downloading the LoRA + base and standing up vLLM. Switching cost: days.

For projects with a >2-year expected lifetime, portability compounds. Every retraining event on closed frontier is a full-cost re-run.

### Axis 4 — Method flexibility

The question: **Which training methods do you need access to?**

- **SFT only.** All paths work. Axis 4 → wash.
- **SFT + DPO** (mod-105). OpenAI, some Bedrock bases, some Mistral, some Vertex. Most self-hosted stacks. Axis 4 → most paths viable; check specific vendor.
- **KTO, ORPO, GRPO, PPO, custom RL loops.** Almost entirely self-hosted only in 2026. Some vendors offer Reinforcement Fine-Tuning (OpenAI RFT on o-series with graders), but the general-purpose custom-RL space is self-hosted. Axis 4 → open base / self-hosted.
- **Continued pretraining.** Bedrock is the main managed surface. Otherwise self-hosted (mod-102 chapter 4's domain corpus discipline plus a training loop).
- **Multimodal training.** Available on OpenAI (image), Vertex (image / video / audio), Anthropic (image), Bedrock (per base). Also self-hosted with open multimodal bases. Axis 4 → check vendor coverage per modality.
- **Safety-specific training / red-team-driven retraining.** Largely self-hosted (mod-106); some vendors expose safety-tuning primitives but the loop is opaque.

If your product requires methods beyond SFT/DPO, self-hosted is often the only path.

### Axis 5 — Dependability and vendor risk

The question: **How much of your product's viability depends on the vendor continuing to offer this specific base at this price with this policy?**

Concrete vendor risks:

- **Deprecation.** OpenAI has deprecated bases before (davinci-002 sunset). If your fine-tune is on a base that gets deprecated, you retrain. Your migration surface: hours to weeks.
- **Price change.** Vendor unilaterally raises per-token pricing. Your budget adjusts (or you migrate).
- **Policy change.** Vendor tightens content policy in a way that impacts your use case (rare but not unprecedented). You may lose the ability to run your workload.
- **Availability incident.** Vendor has an outage or a targeted account issue. Your product is degraded.
- **Regulatory action against vendor.** A jurisdictional issue affecting the vendor cascades to your product's availability in that region.
- **Enterprise-support quality drift.** The account rep changes; response times slip.

Self-hosted vendor risk is different but not zero: your GPU vendor could raise prices, the open base's license could change (Meta's Llama license has evolved), or a specific model family's community could stagnate.

The rule: **the higher the stakes, the more the axis pulls toward self-hosted OR toward the cloud-provider surfaces (Bedrock, Vertex, Azure) which come with their own contractual bulwarks.** Direct vendor SDKs carry the most vendor risk.

## The four concrete regimes

Combining the axes yields four archetypes. Most projects land in one.

### Regime 1 — Frontier-only task, moderate volume

- Capability probe shows a wide gap between open and closed.
- Volume ~ 1–50M req/month.
- Team has no GPU infrastructure and no immediate plans to build it.
- Regulatory posture allows a managed cloud vendor.

**Answer: Closed frontier, managed API.** OpenAI direct if your ops team is comfortable with a direct vendor SDK; Bedrock / Azure / Vertex if regulatory or procurement wants cloud-native. Chapter 2 or 3.

### Regime 2 — Open-tractable task, medium/high volume

- Capability probe shows open matches or nearly matches closed.
- Volume ≥ 10M req/month.
- Team has (or is willing to build) GPU serving.
- Multi-year expected lifetime.

**Answer: Open base, self-hosted.** Choose the base (mod-104), fine-tune (mod-103–105), quantize and serve (mod-109). The cost model dominates over the lifetime.

### Regime 3 — Ambiguous case, short-term horizon

- Capability probe is uncertain or the task itself is evolving quickly.
- Volume unknown.
- Product is in early stages; the eval bar itself is still being defined.

**Answer: Managed frontier first, revisit at Milestone X.** Ship on managed, prove product-market fit, then re-run this framework at a defined trigger (X requests/month, or Y date, or Z-quality alternative reached). This is the majority regime for new products; the point is that "managed for now" is a defensible answer with an explicit exit clause, not a permanent commitment.

### Regime 4 — Regulated / air-gapped / on-prem

- Data cannot leave your infrastructure at all.
- No cloud vendor surface satisfies (or the cleared surfaces don't offer your model).
- Compliance surface is unmovable.

**Answer: Open base, self-hosted, on-prem.** Not really a choice — this is the only path that satisfies the constraints. Chapter 5's compliance one-pager will identify this regime early.

## The hybrid arrangements

Not every project fits cleanly into A or B. Three hybrids are common enough to name.

### Hybrid H1 — Serverless LoRA on open weights (Fireworks / Together / Replicate)

Fine-tune a Llama or Mistral or Qwen base through a vendor that handles the training but returns the LoRA. You get:

- The convenience of managed training (no GPUs, no distributed config).
- The escape hatch of open weights (you can download the base and the LoRA and re-serve on your own vLLM / SGLang whenever you want).
- Serverless per-token inference pricing (usually cheaper than the direct vendor's fine-tune-tier for equivalent models).

When it fits: **you want managed training but self-hosted freedom.** Common for teams that want to iterate quickly during exploration and consolidate to self-hosted for production.

### Hybrid H2 — Managed on cloud-provider surface

Bedrock, Vertex, Azure Foundry. Fine-tune a model via the cloud provider's control plane. You get:

- Fine-grained regional control, private networking, VPC endpoints.
- The compliance surface of the cloud provider.
- Multi-provider fluency (Bedrock can call Anthropic AND Meta AND Cohere from one control plane).
- Often, for open-weight bases, the ability to download the weights (subject to the base's license) — Shape B or Shape C IP from chapter 5.

When it fits: **regulatory or procurement requires a cloud-native surface**, or the team is already deep in one cloud and wants to stay within its control plane. This is the majority of enterprise fine-tuning in 2026.

### Hybrid H3 — Frontier for hard cases, open for easy cases (routing)

Deploy both an open-fine-tuned model and a managed closed-frontier fine-tune. Route requests at runtime — easy / short / common queries to the open model, hard / long / specialized queries to the frontier. Requires a router (small classifier, LLM-as-router, or heuristic).

When it fits: **your traffic has a hard-tail distribution** where 80% of queries are open-tractable and 20% genuinely need the frontier. Optimizes total cost while preserving the frontier for cases that need it.

Watch out: two models to maintain, two eval loops, two safety reviews, and the router itself is a component that needs its own eval. The complexity is real; run the numbers on whether the cost savings justify it.

## The ADR

The Architecture Decision Record that documents this decision:

```markdown
# ADR-NNNN: Fine-tune vendor / base model choice for <project>

## Context

<Two-paragraph description of the project, the task, and the volume / latency / cost budget it operates under.>

## Decision

We will fine-tune <BASE_MODEL_ID> via <PATH>.

- Path A (Closed frontier, managed) or Path B (Open base, self-hosted) or a specific hybrid.
- Vendor / surface: <OpenAI direct / Bedrock / Vertex / Mistral direct / self-hosted / etc.>.
- Method: <SFT | DPO | RFT | Continued Pretraining>.

## Axes evaluated

1. Capability ceiling — <capability probe result: open score X, closed score Y, gap justified/unjustified>.
2. Cost — <projected monthly cost on this path vs. the alternative, at expected volume; break-even volume>.
3. Portability — <switching cost estimate; alternate vendor identified: yes/no; conversion pipeline tested: yes/no>.
4. Method flexibility — <methods currently required: [...]; methods available on this path: [...]; gap: [...]>.
5. Dependability / vendor risk — <specific risks identified; mitigations>.

## Alternatives considered

- <Alternate 1>: <rejected because ...>.
- <Alternate 2>: <rejected because ...>.

## Consequences

- <One paragraph on what this decision commits us to over the next 6–24 months.>
- <One paragraph on what it precludes.>

## Compliance one-pager

<Attach or link the chapter 5 compliance one-pager. This ADR is not approved without it.>

## Exit triggers

Reconsider this ADR if any of:

- Volume exceeds <threshold> — the cost curve crosses.
- Vendor announces deprecation of <BASE_MODEL_ID> — forced migration.
- Data residency requirement changes to <new-region> — vendor may not serve it.
- Capability probe on the open alternative closes the gap to <threshold> — self-hosted becomes viable.
- <Other project-specific triggers>.

## Next review

Scheduled <YYYY-MM-DD>. Owner: <name>.
```

Every managed-vs-self-hosted decision ships one of these, gets reviewed at the specified date, and updates in-place when the calculus changes. This is what makes "we picked managed for now" a defensible answer instead of technical debt.

## Common decision-time pitfalls

**"We picked based on cost alone."** Cost is one axis of five. A choice that saves $500/month by exposing the product to vendor deprecation risk is a bad trade.

**"We picked based on capability alone."** Capability matters, but a project locked to a vendor whose price triples in year 2 is not a stable product.

**"We didn't run the capability probe."** The most expensive single decision is the wrong path chosen without evidence. The probe is cheap.

**"We assumed we could always switch later."** True in principle; expensive in practice. If you have not tested the migration path once, "we can switch" is a hope, not a plan.

**"We committed forever."** Every ADR has an exit trigger and a review date. Never make an irreversible commitment when the underlying models will be different in 18 months.

**"We over-optimized for future-proofing at the cost of shipping."** Regime 3 is defensible. Shipping on managed with an ADR that says "revisit at 5M req/month" is a better outcome than deferring the decision until the perfect answer emerges.

## What comes next

Chapter 7 closes the loop: whichever path you pick, the mod-108 eval gate must run against the result, and the numbers must be like-for-like against the alternative you did not pick. That is what makes the ADR's next review evidence-based instead of vibe-based.

## Summary

- The managed-vs-self-hosted decision is not a preference; it is a five-axis trade-off — **capability ceiling, cost, portability, method flexibility, dependability** — that varies by project and by year.
- Run the **capability probe** before committing: 100–300 task inputs, open model at few-shot, closed model at few-shot, judged. The gap tells you whether self-hosted can plausibly close it.
- Four concrete regimes cover most projects: (1) frontier-only task moderate volume → closed frontier managed; (2) open-tractable task medium/high volume → open self-hosted; (3) ambiguous / early stage → managed first, revisit at trigger; (4) regulated / air-gapped → open self-hosted on-prem, only choice.
- Three hybrids relieve pressure on the binary: (H1) serverless LoRA on open weights (managed training + open escape hatch); (H2) managed on cloud-provider surface (Bedrock / Vertex / Azure — the enterprise default); (H3) route between an open fine-tune and a frontier fine-tune (frontier for hard cases only).
- Every decision ships an **ADR** with the five axes evaluated, alternatives considered, compliance one-pager attached, exit triggers named, and a scheduled review date. This makes "managed for now" a defensible answer with an explicit expiration, not permanent technical debt.
- Common pitfalls: deciding on cost alone, deciding on capability alone, skipping the probe, assuming reversibility without testing it, committing forever, over-optimizing for portability at the cost of shipping. All of these are prevented by the ADR discipline.
