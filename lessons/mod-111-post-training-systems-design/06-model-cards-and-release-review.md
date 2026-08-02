# Authoring the Model Card and the Release Review

**Estimated read:** 35 minutes.
**You should be able to, by the end:** author a model card that satisfies governance, risk, security, and legal in a single document; run a release-review meeting that produces a signed verdict; recognize the seven card sections that trip up first-time reviewers; and defend the trade-off between disclosure completeness and confidentiality.

## Model cards are the interface

The chapters so far produced *decisions* (base, budget, ADR), *machinery* (release gate, registry, rollback), and *artifacts* (checkpoints, adapters, quantized files). The model card is the *interface* — the one document a governance reviewer, a security officer, a compliance auditor, a downstream product team, or an external partner reads to understand what this model is, what it isn't, and what to worry about.

The word "model card" is Mitchell et al. (2019)'s term of art, adopted by Google, Hugging Face, Meta, Anthropic, and OpenAI (often as "system cards" for larger systems). Every major open-model release ships one; every governance framework (NIST AI RMF, ISO/IEC 42001, the EU AI Act's general-purpose AI provisions) presupposes one exists.

This chapter is the model-card schema tuned for a *fine-tuned* model — not a base model, not a system, but a specific fine-tune shipped for a specific product. Composed with mod-102's dataset card and mod-106's safety-eval report card, the model card completes the governance handoff.

## Who reads the model card

Five audiences, sometimes overlapping. The card serves all five in one document — writing five separate cards is a mistake because they will diverge.

- **Governance / compliance / legal.** Wants: license posture, data provenance, regulatory alignment, sign-off chain.
- **Security / risk.** Wants: harm taxonomy, safety report card, red-team results, escalation paths, known attacks.
- **Downstream product teams (internal).** Wants: intended use, guaranteed behaviors, known limitations, integration guidance, SLO commitments.
- **External partners (if the model or its outputs are shared).** Wants: license, cite-ability, redistribution posture, contact.
- **Future fine-tuning engineers.** Wants: what was tried, what worked, what didn't, what to change next release.

The model card is one document; each audience finds what they need with clear section headers.

## The model card schema

Nine sections. Each has a defined owner and a defined signal for "this section is done."

```markdown
# Model card — <product> <artifact-id>

**Card version:** v1
**Model version (registry ID):** acme-support-v3.1
**Date:** YYYY-MM-DD
**Owner:** L. Ortega (fine-tuning)
**Reviewers:** J. Park (peer FT), S. Idris (safety), N. Rao (platform),
              A. Gomez (governance)

## 1. Overview
## 2. Intended use and out-of-scope use
## 3. Model details
## 4. Training data
## 5. Evaluation
## 6. Safety and known limitations
## 7. Ethical and societal considerations
## 8. Environmental and compute footprint
## 9. License, contact, and citation
```

The rest of this chapter walks each section.

### Section 1 — Overview

One paragraph. Answers: what is this model, what is it for, at what maturity level (research, beta, GA), and what changed vs. the previous release.

**Failure mode:** overview is written for marketing ("state-of-the-art support assistant"). Governance reviewers ignore marketing prose; write like a peer engineer would.

**Example:**

> Acme Support Assistant v3.1 is a fine-tune of Qwen2.5-7B-Instruct produced
> for internal use by Acme's support ops team. It targets a reduction in
> escalation rate on billing-category tickets from 12% (v2.4 baseline) to
> ≤ 8%. The v3.1 release supersedes v3.0 (released 2026-07-15) with a
> retrained DPO adapter that reduces mean response length by 15% while
> maintaining pairwise win rate against v3.0. GA for internal support ops;
> not exposed to external customers.

### Section 2 — Intended use and out-of-scope use

Explicit. What the model *is* for; what it *is not* for. This section is what determines liability and what constrains the runtime team's deployment.

- **Intended use:** the product context and the user segments served. "Answering internal support-ops questions about billing, cancellation, and account management for Acme's SMB customers, English-language only, in the Acme internal Slack workspace and CRM."
- **Out-of-scope use:** what the model was *not* validated for. "Not for medical, legal, or financial advice; not for automated ticket-close decisions without human review; not for public-facing chat surfaces; not for non-English tickets; not for enterprise-tier customer conversations."
- **Known misuse risks:** the ways a well-intentioned user could get harmed. "Users may over-trust routine billing calculations; system prompt requires the model to include amounts as ranges, not exact figures, when uncertainty is high."

The out-of-scope list is where governance reviewers spend the most time. Every deployment surface not on the intended-use list is one the runtime team is responsible for gating.

### Section 3 — Model details

Sufficient for a reader to reproduce the model, or (if the weights are not shared) to reason about its behavior.

- **Base model.** `Qwen/Qwen2.5-7B-Instruct` at revision `<sha>`, tokenizer hash `<sha>`, chat-template hash `<sha>`.
- **Fine-tuning method(s).** LoRA-based SFT + LoRA-based DPO + reused safety adapter (v3). Reference the objective-stack ADR (chapter 3 output).
- **Adapter chain and hashes.** From the registry entry.
- **Quantization.** AWQ 4-bit, calibrated on `s3://acme-cal/v2.jsonl`, hash `<sha>`. Reference the mod-109 chapter 3 output.
- **Serving stack.** vLLM 0.6.2, runbook v3.1 (mod-109 ch7 template).
- **Sizes and precision.** 7B parameters (base) + `<n>` LoRA parameters (SFT) + `<n>` LoRA parameters (DPO) + `<n>` LoRA parameters (safety); BF16 during training, AWQ 4-bit at serving.
- **Context length.** 8192 tokens (base default; not extended in this fine-tune).
- **Reproducibility pins.** Training run IDs, config hashes, git SHA of training code. Same fields as the registry entry's provenance chain.

### Section 4 — Training data

Reference out to the dataset card (mod-102 ch7) rather than reproducing it. This section summarizes:

- **Sources.** Counts per source, per capability category.
- **Licenses.** Red / yellow / green verdict per source; unresolved verdicts are blockers.
- **Synthetic disclosure.** Which subset was synthesized, by which teacher model, under what license.
- **Personally identifiable information (PII).** Scan verdict; any redactions applied.
- **Contamination.** Scan verdicts against the mod-108 eval gates.
- **Composition summary.** Mix ratios (from mod-102 ch6).

A model card whose training-data section is empty or hand-waved will not pass a governance review. If the dataset card does not exist yet, the card cannot be finalized — this is by design.

### Section 5 — Evaluation

Reference out to the release-gate report (mod-108 ch6) rather than reproducing it. This section summarizes:

- **Primary metric result.** "Escalation rate on billing held-out: 7.4% [7.0%, 7.9%] — target ≤ 8% met."
- **Capability guardrails.** MMLU / GSM8K / IFEval / MT-Bench: deltas vs. base and vs. previous release.
- **Judged evals.** Rubric-mean, pairwise win-rate vs. v3.0, length-controlled correction.
- **Method used.** Judge model + seed, harness version, evaluated with `--apply_chat_template`, statistical treatment (bootstrap CIs, paired tests).
- **Known evaluation limitations.** What the eval does not cover — the "explicitly not tested" list from mod-106 ch5.

### Section 6 — Safety and known limitations

Reference out to the safety report card (mod-106 ch5). Summarize:

- **Harm-taxonomy coverage.** Which categories were tested, refusal rates per category.
- **Harmful-completion rate.** Number on the red-team eval.
- **Capability-gain (dual-use).** Whether the fine-tune raised any capability the base didn't have on the tested dual-use axes.
- **Explicitly-not-tested attacks.** Verbatim from the report card. Includes the categories in mod-106 ch6's "escalate rather than train" list.
- **Known limitations.** Concrete, product-relevant failure modes: "hallucinates dollar amounts on ambiguous refund questions ~3% of the time"; "misclassifies enterprise-tier billing questions when the tier is not explicitly stated (7% of held-out)"; "verbosity increases 10–15% for open-ended questions vs. v3.0."
- **Runtime compensations.** What runtime guardrails (mod-106 ch6) compensate for what training-time limitations.

The list of known limitations is the section that governance reviewers most respect. A card that says "no known limitations" is a card that has not been reviewed carefully. Real fine-tunes have real limitations; naming them is what earns trust.

### Section 7 — Ethical and societal considerations

Less template-driven than the prior sections. Two questions:

- **Bias / fairness considerations.** Whose experiences are represented in the training data, whose are not, whose outputs are the eval labels drawn from. For an internal support assistant, this may be brief (single tenant, single language); for a broader system it is substantive.
- **Deployment-context risks.** Who is affected by errors, and how. For our example: incorrect billing responses could lead to customer overcharges or undercharges if the human reviewer over-trusts them; the runbook requires human review for any customer-facing action.

This section is where governance frameworks (NIST AI RMF, EU AI Act general-purpose AI provisions) look for evidence that the team considered downstream harm.

### Section 8 — Environmental and compute footprint

Not always required, but increasingly requested by both governance frameworks and sustainability reporting. Estimate:

- **Training compute.** GPU-hours × GPU class, times a public factor (kg CO₂-equivalent per GPU-hour by region). For a LoRA fine-tune this is small.
- **Inference compute per request** (from mod-109 ch7's cost model). Dollar-per-request or kg-CO₂-per-request estimate.
- **Base pretraining footprint.** Referenced (not attributed to your fine-tune, but disclosed): "Qwen2.5-7B-Instruct pretraining footprint per its own model card."

Even a rough number here is better than none. `<!-- needs-research: pull the current published GPU-hour → CO₂ conversion factor for the target region from the cloud provider's sustainability report -->`.

### Section 9 — License, contact, and citation

- **License for the fine-tuned weights.** If distributed: the SPDX identifier. If not distributed (internal only, served through an API): the license of the *outputs*, if any commitment is made.
- **Contact.** Named owner + team channel (Slack, email, ticket queue).
- **Citation.** If external partners are asked to cite the model in publications, provide a BibTeX entry or an APA-style citation string.

## The composed governance package

The model card is one document in a bundle. The full governance package for a release is:

- **Model card** (this chapter).
- **Dataset card** (mod-102 ch7).
- **Release-gate report** (mod-108 ch6).
- **Safety-eval report card** (mod-106 ch5).
- **Release verdict** (chapter 5 of this module).
- **Serving runbook** (mod-109 ch7).
- **Rollback runbook** (chapter 5 of this module).
- **ADR chain** (chapters 2 and 4 outputs — base selection, fine-tune-vs-alternative).

The model card *points at* the rest and *summarizes* it. It is the entry point; the rest is the depth.

## The release-review meeting

The release-review meeting is where the governance package is presented and signed. It is scheduled *after* the artifact is in the `staged` state (chapter 5) and *before* it promotes to `production`. Structure:

**Attendees (mandatory):**

- Fine-tuning owner (presents).
- Peer fine-tuning engineer (co-presents technical questions).
- Safety reviewer.
- Governance reviewer.
- Release captain (signs the verdict).

**Attendees (optional but recommended):**

- Platform lead (for infrastructure impact).
- Product owner (as the sponsor named in the proposal).
- Legal (for release with unresolved license notes).
- On-call representative from the runtime team (for handoff of the rollback runbook).

**Agenda (60 minutes):**

- **10 min — Overview.** Fine-tuning owner walks the model card §1 (Overview) and §2 (Intended / Out-of-scope). Reviewers align on scope.
- **15 min — Evaluation.** Fine-tuning owner walks the release-gate report. Reviewers challenge the numbers, the CI methodology, the disclosure statements. Any BLOCK-level regression is discussed here.
- **10 min — Safety.** Safety reviewer walks the safety report card. Discussion of the "explicitly not tested" list; any category added or removed since the previous release. Runtime compensations affirmed.
- **10 min — Governance / license / data.** Governance reviewer walks the dataset card. Discussion of any red-flag license verdicts or PII / contamination flags.
- **10 min — Operational readiness.** Platform lead confirms the serving runbook and rollback runbook are complete and the on-call is trained.
- **5 min — Verdict.** Release captain reads the release-verdict document and either signs APPROVE / APPROVE_WITH_DISCLOSURE (with the disclosure text agreed at the meeting) or requests changes.

**Meeting output:**

- The signed verdict document (chapter 5 template).
- Meeting notes appended to the model card §appendix (who attended, what was discussed, what changes are required — even for APPROVEs).
- The PR from `staged` → `production` in the registry.

**Anti-patterns to avoid:**

- **The pre-approved meeting.** If the outcome is decided in Slack before the meeting, do not hold the meeting. Either publish the pre-approval or run a real meeting.
- **The captain who cannot say no.** A release captain must be authorized to BLOCK. If they are not — if BLOCK requires escalation to a VP — the meeting is not producing decisions, it is theater.
- **The 90-minute meeting.** A well-prepared package is presented in 60 minutes. Meetings that run long are a signal that the package is under-prepared; ask the presenter to redo the walkthrough offline.
- **No attendee list captured.** Six months later, an audit asks "who signed this release?" — the answer is in the meeting notes.

## The disclosure trade-off

Every model card faces the trade-off: how much to disclose? Too little erodes trust and fails governance; too much may reveal sensitive information (training data provenance that gives competitors a signal, safety-eval details that enable jailbreak crafting, personnel information).

Rules of thumb:

- **Always disclose:** licenses of training data, intended and out-of-scope uses, evaluation methodology and pinned versions, known limitations, contact.
- **Usually disclose:** approximate dataset sizes and composition (bucketed rather than exact if sensitive), model architecture and size, quantization method, safety-eval headline numbers.
- **Selectively disclose:** specific dataset row counts if they reveal customer telemetry volumes; specific red-team prompts if they enable jailbreak reproduction; specific attack success rates on unshipped mitigations.
- **Do not disclose:** individual training data records, individual customer or user identifiers, unmitigated critical vulnerabilities (report through security channels instead).

The card usually has both a *public* version (for external partners, if any) and a *private* version (for internal reviewers). The public version is a subset of the private one; the private one exists in Git under access controls.

## Model card versioning

The card is versioned in Git alongside the registry entry. Bump the card version when:

- Any of §1–§9 content changes.
- The model is retrained (new registry entry with a new artifact ID → new card).
- A new limitation is discovered in production (§6 update; card version increment even if the model didn't change).

The card version is recorded in the registry entry (`release_verdict.model_card` field). An audit can walk from any point in time to the exact card version live at that time.

## Non-obvious card sections some teams add

Depending on regulatory context or product surface, you may add:

- **Data-flow diagram.** Which requests reach the model, which don't; where retrieval context comes from; which fields are logged. Useful for privacy reviews.
- **Prompt-injection posture.** If the deployment is RAG-heavy or tool-heavy, a section on how the deployment handles adversarial context. Mod-106 chapter 6 flags prompt-injection as "escalate rather than train"; the card explains how the deployment side of that escalation is implemented.
- **Model-behavior guarantees.** Explicit statements about behavior that the fine-tune commits to (e.g., "will refuse questions about competitor products," "will not disclose the system prompt"). These translate to test cases on the gate.
- **Change log for the card itself.** A running list of what changed between card versions. Distinct from the registry timeline.

## Two failure modes of the card

- **The card is stale.** A model in production whose card was last updated when v3.0 shipped and the artifact is v3.4. Fix: card version is a required field on the registry entry; a merge to `main` that changes the registry without updating the card fails CI.
- **The card is a template.** The card exists with all nine sections filled with placeholder text. This passes cursory review but not a real one; a governance reviewer who sees "no known limitations" ends the review and asks for a rewrite. Fix: peer review of the card is a required step before the release-review meeting.

## Summary

- The model card is the one document that serves governance, security, downstream product, external partners, and future engineers. Writing five separate documents leads to divergence; writing one — with clear sections for each audience — leads to a single source of truth.
- Nine sections: overview, intended / out-of-scope use, model details, training data, evaluation, safety and known limitations, ethical and societal considerations, environmental / compute, license / contact / citation.
- The card *composes* — it references the dataset card (mod-102 ch7), the release-gate report (mod-108 ch6), the safety-eval report card (mod-106 ch5), the serving runbook (mod-109 ch7), the rollback runbook (chapter 5), and the ADR chain (chapters 2 and 4). It does not reproduce them.
- Known limitations are the trust-earning section. "No known limitations" is a red flag; real fine-tunes have real limitations; name them.
- The release-review meeting is 60 minutes, five mandatory attendees (fine-tuning owner + peer, safety, governance, release captain). Output is a signed verdict document and a registry commit promoting the artifact to production.
- The disclosure trade-off has rules of thumb: always disclose license, use, methodology, limitations; usually disclose sizes and headline numbers; selectively disclose things that could enable misuse; never disclose individual records or unmitigated critical vulnerabilities.
- The card is versioned in Git; each registry entry pins a card version. Auditability is the discipline that makes governance possible.
- Two card failure modes to watch: stale (fixed by making card-version required in the registry) and template-filled (fixed by peer review before the meeting).
