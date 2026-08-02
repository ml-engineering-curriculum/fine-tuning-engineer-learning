# Handoff to Security and Governance

**Estimated read:** 35 minutes.
**You should be able to, by the end:** name the layers of a deployed LLM system beyond the trained checkpoint, know which of those layers are your job as a fine-tuning engineer and which belong to security / governance / policy peers, describe the artifacts you must produce so the handoff is legible, and know which classes of problems you should escalate rather than try to fix in training.

## Where fine-tuning stops

Safety-alignment tuning is one layer of a defense-in-depth system. Zoomed out, a shipped LLM product usually contains, from innermost to outermost:

```
              ┌───────────────────────────────────────────────────┐
              │        External policy / legal / regulatory        │
              │  (Terms of Service, AUP, EU AI Act, NIST AI RMF,   │
              │   RSP-style commitments, industry codes)           │
              └────────────────────────┬──────────────────────────┘
                                       │
              ┌───────────────────────────────────────────────────┐
              │         Operational safety controls                │
              │  (Access control, rate limiting, abuse detection,  │
              │   incident response, takedowns, red-team ops)      │
              └────────────────────────┬──────────────────────────┘
                                       │
              ┌───────────────────────────────────────────────────┐
              │         Runtime guardrails                         │
              │  (System prompts, tool restrictions, output        │
              │   classifiers — Llama Guard / WildGuard,           │
              │   moderation APIs, prompt shields)                 │
              └────────────────────────┬──────────────────────────┘
                                       │
              ┌───────────────────────────────────────────────────┐
              │         Model                                      │
              │  Base pretraining → SFT → preference opt. →        │
              │  SAFETY-ALIGNMENT TUNING ← YOU ARE HERE            │
              └───────────────────────────────────────────────────┘
```

Fine-tuning owns the innermost box. Chapters 2–5 have been about making that box work as well as it can. The remaining three layers are owned by other teams, and getting the handoff right is what makes the whole system safe rather than just the model. This chapter is the interface contract.

Two mental models to keep in mind before we go into each layer:

- **Fine-tuning lowers the base rate of unsafe outputs; the outer layers cover the long tail.** You cannot train a model into perfect refusal on every possible attack. Runtime classifiers and operational controls are there to catch what training didn't.
- **The outer layers are more responsive than the inner ones.** A training run takes days; deploying a system-prompt change or a classifier update takes minutes. When a new attack shows up, the outer layers fix it first; the training-time fix goes into the next scheduled release.

## Layer 1 (outward): runtime guardrails

The layer immediately outside the model is a set of runtime defenses that intercept prompts before they reach the model or responses before they reach the user. Owned by whichever team runs the inference serving infrastructure — often an inference / platform team, sometimes T&S, sometimes ML infra.

### Output classifiers

The most common piece. A classifier — often Llama Guard, WildGuard, ShieldGemma, or a moderation API (OpenAI Moderation, Google Perspective, Azure AI Content Safety) — reads every prompt and / or every generated response and flags policy violations.

- The classifier's judgment can *block* the response, *rewrite* it, *log it for review*, or *return an alternative* (a canned safe response).
- Classifiers are usually smaller and faster than the generating model; they add tens of milliseconds of latency for a Llama-Guard-3-1B-scale classifier per call.
- Configuration — thresholds, per-category action policies, response templates — is *operational config*, not training data. Owned by the team running the guardrail, updated on a fast cycle.

Your job as a fine-tuning engineer at this handoff: **give the guardrail team a report card** (chapter 5's dashboard) that tells them what the classifier will see. If your model produces 5% jailbreak-compliant responses on axis 3, the guardrail needs to catch 5% of dangerous outputs; if you drove that to 1%, the guardrail's job is 5× easier. Fewer false positives, higher throughput.

### System prompts

The system prompt is the operator's channel for policy. "You are a helpful assistant. You will not discuss X, Y, Z." "If the user asks about self-harm, respond with crisis resources and do not provide instructions." "You may discuss legal topics in general but do not give specific legal advice."

- System prompts are *runtime* policy. The same base model can be deployed under multiple system prompts for multiple products; the training-time safety behavior is the *floor*, and the system prompt tightens or reshapes it per deployment.
- Wallace et al. (2024, "The Instruction Hierarchy") describes the training pattern for having models prioritize system-prompt policies over user-turn overrides.

Your handoff: the model's training-time safety behavior should *not* be conditional on any specific system prompt. Test with an empty system prompt as well as with the standard operational one. If your model only refuses when the system prompt says to, an attacker who can bypass or overwrite the system prompt bypasses safety entirely.

### Tool / capability restrictions

For agent / tool-using systems, the runtime layer decides which tools the model has access to and what payloads those tools accept. A code-execution tool that refuses to run `rm -rf /`, a browsing tool that respects a domain allowlist, a database tool that respects row-level access controls — these are runtime safety controls, not model behaviors.

Your handoff: the model should be trained to *use tools appropriately*, but the *authorization* to actually execute is the runtime's call. Do not conflate "the model chose to call this tool" with "this tool call is safe to execute."

### Prompt shields and input filters

Symmetric to output classifiers on the input side. Catch obvious prompt injection (`ignore previous instructions...`), known jailbreak strings (DAN, unicode-obfuscated attacks), and prompt-shaped content sneaked into RAG / tool outputs.

Your handoff: training-time safety should assume the input filter catches nothing (defense in depth), and the input filter should assume training-time safety catches nothing. Neither owns the other.

### Content provenance and watermarking

Whether generated content is marked as such (C2PA, SynthID, statistical watermarking). Adjacent to safety but usually a separate track; named for completeness.

## Layer 2: operational safety controls

Broader than per-request runtime. Owned by trust & safety, ops, and security teams.

### Access control

Who can call the API. Age gating (for models that can generate content restricted by age). Enterprise-vs-consumer differentiation. Regional restrictions.

- Not your call. Your model produces output; who is allowed to ask is a policy / product decision.
- Do care about: if a deployment restricts to enterprise-only, the safety-training bar is different from a consumer deployment. The report card (chapter 5) should specify the deployment context.

### Rate limiting and abuse detection

Attack strategies like GCG or PAIR require thousands of queries to optimize a working suffix. Rate limits and per-account query monitoring make the attack economically expensive. Anomaly detection on query patterns catches attempts.

- Owned by the platform / abuse team. Your job: publish which attack strategies your model is *not* robust to (chapter 5's axis 3), so the abuse team knows what patterns to watch for.

### Incident response

When a real safety failure lands in production — a user posts a screenshot; a journalist calls; a bug bounty submission arrives — the response has to be immediate. Rollback to previous model version, deploy a stopgap system-prompt change or classifier rule, comms strategy, root-cause analysis.

- The fine-tuning team supports incident response by (a) having the previous checkpoint available for fast rollback, (b) rapidly producing a diagnostic on whether the failure is training-side or runtime-side, and (c) incorporating the failure into the next red-team pass (chapter 4).
- Own the artifacts: version every checkpoint, tag every deployed model with its provenance, keep training-data snapshots per release.

### Red-team ops and bug bounties

The peer red team's cadence (chapter 4) is part of the operational safety layer. External bug-bounty programs (Anthropic's, OpenAI's, Google's various VRPs) are another source.

- Owned by security / T&S. Your job is intake and conversion (chapter 4) plus regression-testing.

### Takedowns and content moderation post-hoc

For products that store or share generated content (chat history export, community features), moderation of generated content after the fact is a separate control layer.

- Not fine-tuning. Adjacent so you know it exists.

## Layer 3: policy, legal, and regulatory

The outermost layer. What is your product *allowed* to do, and who is allowed to hold you accountable when it doesn't.

### Acceptable Use Policies and Terms of Service

The document your users agree to when using the product. Enumerates prohibited use cases, IP rules, warranty disclaimers. Anthropic's Usage Policies, OpenAI's Usage Policies, Meta's Llama AUP are public examples; every deploying org has its own.

- Your training taxonomy (chapter 1) should map to the AUP. If the AUP prohibits X, your training data should have refusal / safe-completion coverage for X. If your training data covers Y but the AUP is silent on Y, either add Y to the AUP or drop Y from training (over-restrictive training on categories that aren't policy is over-refusal by design).

### Model cards and system cards

Model cards (Mitchell et al., 2019, "Model Cards for Model Reporting") are the standard artifact for documenting what a model does and doesn't do. A safety-relevant model card includes:

- Intended use cases and out-of-scope uses.
- Training data provenance and dataset cards.
- The safety report from chapter 5.
- Known limitations (from your "Known limitations" section).
- Version, checkpoint hash, timestamp.

System cards (OpenAI's, Google's, Anthropic's) are similar but at the whole-product level, covering not just the model but the runtime layers around it.

Your handoff: publish the model card. Every training run produces one. It goes on the internal wiki (or an external one if the model is shared).

### Regulatory frameworks

Named as pointers so you know what's referenced. Details belong to the governance track and your legal team.

- **NIST AI Risk Management Framework** (NIST AI 100-1, 2023) — US framework covering govern / map / measure / manage functions. Referenced by many US procurement processes.
- **EU AI Act** — the 2024 EU regulation classifying AI systems by risk tier and imposing per-tier obligations (documentation, conformity assessment, post-market monitoring). "General-purpose AI models" carry specific obligations; safety and evaluation artifacts feed into the compliance package.
- **US Executive Order 14110** and successor guidance — US federal directive on AI safety, with reporting requirements for foundation models above compute thresholds. (Regulatory landscape may have changed since 2024; check current status.)
- **UK AI Safety Institute / AISI** — voluntary pre-deployment evaluations for frontier models.
- **Industry codes**: ISO/IEC 42001 (AI management systems), IEEE 7000-series, MLCommons AISafety benchmarks.

### Responsible-scaling / capability commitments

Frontier labs have published RSP-style documents committing to specific evaluations and gates before deploying more capable models. Anthropic's Responsible Scaling Policy, OpenAI's Preparedness Framework, DeepMind's Frontier Safety Framework are the public examples. These are voluntary but binding on the labs that publish them.

- Applicable if you're at a lab of that scale. If you're not, still worth reading — the eval categories (CBRN, cyber, autonomy, persuasion) are the categories external regulators are asking about.

## The artifacts you owe the outer layers

Concretely, what fine-tuning produces and hands off:

| Artifact | For whom | Content | Cadence |
|---|---|---|---|
| **Safety-eval report card** (chapter 5) | Runtime, T&S, governance | The four axes, per-category breakdown, deltas vs baseline, known limitations | Per release |
| **Dataset card for safety training data** (chapters 2, 4) | Governance, legal | Provenance, taxonomy, category coverage, filters, decontamination, license chain | Per data version |
| **Model card** | Everyone, sometimes public | Intended use, training summary, eval summary, known limitations, version | Per release |
| **Red-team diff report** (chapter 4) | Peer risk team | Which of their attacks now refuse, which still succeed, which are out-of-scope | Per red-team pass |
| **Attack-corpus manifest** | Security, red team | Which attack strategies / benchmarks you tested against, which you did not | Per release |
| **Checkpoint provenance record** | Ops, incident response | Base checkpoint hash, training-data hash, config, timestamp, changelog | Per checkpoint |

Every one of these has a template that lives in the team's docs. The first time you write each, it takes half a day. On subsequent releases, they're a fill-in-the-template exercise.

## What to escalate rather than train against

Some problems look like fine-tuning problems but aren't. Escalate to the right team instead of adding more training data:

### Persistent adversarial-suffix attacks

GCG / AutoDAN-family attacks are hard to fully close at training time (Zou et al., 2023 and follow-ups). Adversarial training helps but leaves residual vulnerability. Escalate to the runtime layer — an input filter for gibberish suffixes, or an output classifier that catches the compliant response — rather than iterating training-side beyond a reasonable point.

### Prompt injection in RAG / tool contexts

The model retrieves a document that contains a prompt-injection payload, and follows the injected instructions. This is a *system* problem, not a *model* problem. The retrieval layer must sanitize / mark retrieved content; the model can be trained to prefer trusted instructions (instruction hierarchy, Wallace et al., 2024) but training-side defense alone will not close the surface.

OWASP LLM Top 10 lists prompt injection as LLM01 — the top-ranked LLM app vulnerability. It is a systems-integration problem that spans model, runtime, and application code. Escalate.

### CSAM and mandatory-report content

CSAM is not a training problem. It is a mandatory-reporting legal problem. Detection is done by dedicated systems (NCMEC-standard hashing, dedicated classifiers). Any surface in your product that could accept or generate this content requires a dedicated pipeline, dedicated escalation, and legal counsel. Not fine-tuning's scope.

### Real-time misuse detection

"User is using the product to plan a specific harmful act right now." Live abuse detection is an ops problem — pattern-matching queries against known attack signatures, cross-referencing with other signals — not something training a model catches. Route to the ops / T&S team.

### High-stakes-domain accuracy

Medical advice, legal advice, financial advice. Even a perfectly-safety-tuned model can be *wrong* in these domains. The fix is not more safety training; it's (a) explicit disclaimers via system prompt or safe-completion training, (b) not deploying in the domain, or (c) building a domain-specific retrieval / verification layer. Product decision, not training decision.

### Emergent capability jumps

If you notice the fine-tuned model *gained* a capability the base didn't have — e.g., it can now solve harder chemistry problems than the base — that's a governance signal, not a training decision. Report to the governance / RSP-review process before shipping.

### Legal / policy edge cases

"Should we refuse jokes about political figures?" "Should we discuss lawful-but-controversial topics?" These are policy questions, not training questions. The policy team decides; you implement whatever policy comes back.

## The "safe to override" table

A useful mental model for what runtime layers *can* and *cannot* undo of your training.

| If runtime does... | Does it override training? |
|---|---|
| Adds a permissive system prompt ("you may discuss anything") | Partial. Model's baseline safety training still refuses hardest categories. A well-trained model does not fully unlock even with the most permissive prompt. |
| Adds a restrictive system prompt ("do not discuss X") | Yes for the additional restriction; training-time behavior on unspecified categories unchanged. |
| Bypasses the output classifier | Removes an outer defense; training-time behavior unchanged but has to catch alone. |
| Removes the input classifier | Similar; model sees raw prompts including known-bad ones. |
| Rolls back to a previous checkpoint | Reverts your training entirely for that request. |
| Deploys under a fine-tune adapter (LoRA) on top | The adapter can partially override training if the adapter was trained on data that contradicts. Adapters trained by third parties are a specific concern for open-weight models. |

The takeaway: **runtime is a superset of training in what it can express, but it cannot fully substitute for it.** Both are needed; the training establishes the floor, the runtime tightens above it.

## Where the security track picks up

The full security / adversarial-ML story lives in a separate track and is out of scope for this module. Named topics that belong there and that a fine-tuning engineer should be able to point to:

- **Model extraction attacks** — reconstructing model weights or training data via API queries.
- **Membership-inference attacks** — determining whether a specific example was in the training set.
- **Backdoor / trojan attacks** — malicious training-data insertion that produces model behavior only under specific triggers.
- **Adversarial-ML in general** — GCG, PAIR, TAP, AutoDAN attack construction (as opposed to consumption).
- **Supply-chain concerns** — trojaned base checkpoints, poisoned datasets, malicious tokenizer configs (arbitrary-code-execution paths in tokenizer libraries).
- **Model theft, weight exfiltration** — DLP for model weights.
- **Prompt injection at scale** — the systems-level defense strategy (not just per-request classifiers).

If any of these come up during a safety-tuning project, hand them to the security track and stay in your lane.

## Where the governance track picks up

Similarly, the governance / policy / legal side has its own track and its own artifacts:

- **AI-system risk assessments** — formal per-deployment risk analyses under the applicable regulatory framework.
- **Model release policy** — the decision process for deciding whether a given checkpoint is fit to release, external audit requirements, staged-rollout gating.
- **Third-party evaluations** — arrangements with external evaluators (AISI, MLCommons, academic partners, commercial red-team contractors).
- **Public reporting** — how safety findings are disclosed publicly, the timing, the redaction rules.
- **Regulatory correspondence** — responses to regulator questions, submissions under required-reporting regimes.
- **Ethics-review processes** — internal review boards for research and deployment.

## A minimum interface contract, in one page

The whole chapter in one operational summary. This is the checklist you can copy into a project kickoff.

**Fine-tuning team commits to:**

1. Publish a written harm taxonomy and map it to the deployment AUP.
2. Ship a safety-eval report card (chapter 5's four axes + capability regression) with every release.
3. Ship a dataset card for every training data version (chapter 2's coverage table + license chain).
4. Ship a model card for every deployed checkpoint.
5. Maintain a red-team-diff report for every peer red-team pass (chapter 4).
6. Maintain checkpoint provenance records for rollback.
7. Publish the list of attack strategies tested (chapter 5, axis 3); explicitly not-tested attacks are called out.
8. Escalate — not train against — the categories listed in the "escalate rather than train" section above.

**Runtime team commits to:**

1. Deploy an output classifier (Llama Guard, WildGuard, or equivalent) tuned to catch the residual failure modes from the report card.
2. Own the system-prompt policy and version it.
3. Own input-side prompt shields.
4. Not rely on training-time safety as the only defense.

**Ops / T&S team commits to:**

1. Own access control, rate limiting, and abuse detection.
2. Own incident response and rollback authority.
3. Cadence red-team ops with the fine-tuning team.
4. Own external bug bounty intake.

**Governance / legal team commits to:**

1. Own the AUP and its updates.
2. Own the deployment risk-assessment process.
3. Own regulatory correspondence and public reporting.
4. Own external audit and third-party evaluation relationships.

Nothing here is novel. This is the pattern that Anthropic, OpenAI, Meta, and DeepMind publish about their own operations. The value is having it written down for *your* organization.

## Exercise pointer

No dedicated exercise for this chapter — its output is documents (report cards, model cards, interface contracts), not code. The exercises in this module (01–04) produce the artifacts that feed the chapter-6 handoff: your dataset card (exercise 01), your CAI dataset (exercise 02), your red-team-derived preference set (exercise 03), your report card (exercise 04). Together they are the fine-tuning-team half of the interface contract above.

## Summary

- Safety in production is layered: **model → runtime guardrails → operational controls → policy / legal / regulatory**. Fine-tuning owns the model layer; every layer above is owned by a peer team and you have to hand off cleanly.
- **Runtime guardrails**: output classifiers (Llama Guard, WildGuard), system prompts, tool restrictions, input shields. Your training-time safety lowers the base rate; runtime catches the long tail.
- **Operational safety**: access control, rate limiting, abuse detection, incident response, red-team ops. Your training team supports incidents but does not own them.
- **Policy / legal / regulatory**: AUPs, model cards, system cards, NIST AI RMF, EU AI Act, RSP-style commitments. Your taxonomy should map to the AUP; your report card feeds the compliance package.
- **Artifacts to hand off**: safety-eval report card, dataset card, model card, red-team diff report, attack-corpus manifest, checkpoint provenance. Templated once, filled per release.
- **Escalate rather than train against**: persistent GCG-family attacks, RAG / tool-context prompt injection, CSAM, real-time misuse, high-stakes-domain accuracy, emergent capabilities, legal / policy edge cases. Not every problem is a training problem.
- **Runtime is a superset of training in what it can express, but not a substitute for it.** Both floors and ceilings — training establishes the floor, runtime tightens above it. Neither is optional.
