# mod-111-post-training-systems-design: Post-Training Systems Design

**Estimated effort:** 14 hours
**Position in the track:** eleventh and final module. Every prior module produced *one* artifact — a dataset (mod-102), an SFT checkpoint (mod-103), an adapter (mod-104), a preference-tuned model (mod-105), a safety-aligned variant (mod-106), a distributed-training pipeline (mod-107), an eval report (mod-108), a quantized serving stack (mod-109), or a managed fine-tune (mod-110). This module is where they stop being one-off artifacts and become the substrate of a platform. Fine-tuning as a *product surface*, not as a workflow.

You have already been asked, per project, "which base?", "how much of what data?", "which method stack?", "how do we know it's good?", "how do we ship it?". The step change at platform altitude is that these questions get asked five, twenty, or fifty times a quarter — across product areas, tenants, and vendor backends — and no organization survives that if the answers are ad hoc. Every decision above is repeated per project; every artifact above is one of many; every gate is one lane of a suite; every model card is one row in a registry. Mod-111 is how you get from N fine-tunes to a fine-tuning platform without breaking under the coordination cost.

The through-line: **repeatable decisions and versioned artifacts.** A fine-tuning proposal that another engineer can pick up and defend. A dataset budget with the tradeoff written down, not inferred from a Slack thread. A release gate that composes across modalities. An adapter registry that survives its author leaving the team. A rollback that a stranger on call at 3 AM can execute. A model card that governance signs without a back-and-forth. Get those five things right and the platform holds; get any of them wrong and the platform is one bad Friday from a compliance incident.

## Learning objectives

- Walk a problem from product goal to a base-model + dataset-budget + objective-stack proposal — a reproducible design intake that a peer engineer or reviewer can defend.
- Allocate dataset budget across SFT, DPO / preference optimization, RLHF, and safety tuning with documented trade-offs — the shape of the objective stack, not just its size.
- Design eval gates, release-candidate selection, adapter registry, and rollback strategy — the platform layer that wraps mod-108's release gate and mod-104's adapters into an operable, auditable system.
- Make the build-vs-buy / fine-tune-vs-RAG / fine-tune-vs-prompt-engineering decision defensibly — the ADR that survives the next model release and the next roadmap review.
- Author a model card and release review that satisfies governance, risk, and security partners — the paperwork half of shipping, done once and repeated per release.

## Lecture chapters

1. [`01-from-product-goal-to-proposal.md`](01-from-product-goal-to-proposal.md) — From Product Goal to a Fine-Tuning Proposal. The intake pipeline that turns a product ask ("make our support agent 10 points better on billing tickets") into a written proposal a peer engineer can execute: problem definition, success metric, base-model shortlist, dataset-budget sketch, objective stack, cost / timeline estimate, risk register. The document this whole module trains you to write.
2. [`02-base-model-selection.md`](02-base-model-selection.md) — Selecting the Base Under Real Constraints. The rubric — license posture, capability floor, size / hardware ceiling, tokenizer, chat-template maturity, method compatibility, community and ecosystem, deprecation risk — turned into a per-project scorecard. When the score gets overridden by a hard constraint, and how to document the override so the next team can revisit it.
3. [`03-dataset-budget-across-objectives.md`](03-dataset-budget-across-objectives.md) — Dataset Budget Allocation Across the Objective Stack. How many rows of SFT, how many preference pairs for DPO / KTO / ORPO, whether classical RLHF is worth the engineering, and how the safety-tuning slice fits in. The trade-offs are unavoidable — the goal is to make them explicit, argued, and revisitable.
4. [`04-fine-tune-vs-rag-vs-prompt.md`](04-fine-tune-vs-rag-vs-prompt.md) — The Build-vs-Buy and Fine-Tune-vs-Alternative Decision. When *not* to fine-tune (prompt engineering, RAG, tool wiring, or a bought endpoint is the right answer), when to fine-tune, and how to write the ADR so the answer holds up at the next roadmap review and does not get relitigated every quarter.
5. [`05-release-gates-registry-rollback.md`](05-release-gates-registry-rollback.md) — Release Gates, Candidate Selection, Adapter Registry, and Rollback. Composing mod-108's release-gate manifest with mod-106's safety report card and mod-102's dataset card into a per-release verdict. The adapter / model registry schema, the release-candidate promotion rules, and the rollback runbook that on-call actually runs. The platform-layer core of the module.
6. [`06-model-cards-and-release-review.md`](06-model-cards-and-release-review.md) — Authoring the Model Card and the Release Review. The model-card schema that satisfies governance / risk / security / legal in one document; the release-review meeting itself — who attends, what artifacts are on the table, what the acceptance decision looks like, and how the meeting output gets recorded. The paperwork half of shipping.
7. [`07-post-training-as-platform-property.md`](07-post-training-as-platform-property.md) — Post-Training as a Platform Property. Post-launch monitoring, drift detection, iteration cadence, retiring a model. The maturity ladder: from "we did one fine-tune" through "we have a repeatable pipeline" to "we operate a fine-tuning platform." The exit vocabulary for the module and the track.

## Exercises

Each exercise is a documentation-and-decision-heavy hands-on companion to one or more chapters — the artifacts you produce are the paperwork a real platform team keeps versioned in Git. They live in [`exercises/`](exercises/).

- [`exercise-01-base-model-selection-rubric.md`](exercises/exercise-01-base-model-selection-rubric.md) — build a per-project base-model selection rubric, score three real candidates against it for one concrete product scenario, and produce the ADR + risk register that comes out of it. The intake-scoring exercise.
- [`exercise-02-dataset-budget-allocation-across-objectives.md`](exercises/exercise-02-dataset-budget-allocation-across-objectives.md) — for one product goal and one dataset / compute budget, allocate that budget across SFT, preference optimization, safety tuning, and (optionally) classical RLHF, defend the allocation against two named alternatives, and record the ablation plan that would revise it. The trade-off-argument exercise.
- [`exercise-03-release-gates-and-rollback-design.md`](exercises/exercise-03-release-gates-and-rollback-design.md) — extend mod-108's `release_gate.v3.yaml` into a full release-and-registry manifest for a multi-adapter product, design the adapter registry and promotion rules, and author the rollback runbook that a stranger can execute at 3 AM. The platform-layer exercise.
- [`exercise-04-fine-tune-vs-rag-vs-prompt-decision-doc.md`](exercises/exercise-04-fine-tune-vs-rag-vs-prompt-decision-doc.md) — write the decision doc that recommends fine-tuning, RAG, or prompt engineering (or an alternative) for one concrete product ask, defend it against the other two, and record the escape hatches that would revise the choice. The build-vs-buy exercise.

## Labs and quizzes

Long-form labs land in [`labs/`](labs/) and the knowledge-check quiz in [`quizzes/`](quizzes/) on subsequent authoring cycles.

## Resources

See [`resources.md`](resources.md) for the primary references: model-card literature (Mitchell et al. 2019; Google Model Cards; Hugging Face model-card guide), dataset-card literature (Datasheets for Datasets — Gebru et al.; Hugging Face dataset-card guide), governance frameworks (NIST AI RMF 1.0; ISO/IEC 42001; EU AI Act general-purpose AI provisions), decision-record patterns (ADR templates, Google SRE workbook), release-engineering references (Google SRE book chapters on release engineering and postmortems), and the vendor / open-source registry references that Chapter 5 draws from.

## What comes before

- **mod-101 — Post-Training Foundations** established the vocabulary — pretraining vs. post-training, the objective family map, the pipeline diagram. This module reads that pipeline as a *contract shape* and asks how you operate it at scale.
- **mod-102 — SFT Dataset Engineering** owns the dataset card, the mix card, and the licensing trace. Chapter 3 of this module composes those into a dataset-budget allocation across objectives; chapter 6 lifts them into the governance-facing model card.
- **mod-104 — PEFT / LoRA / QLoRA** established the adapter as a first-class artifact and the multi-adapter routing story. Chapter 5 turns that into the adapter registry.
- **mod-105 — Preference Optimization** and **mod-106 — Safety Alignment Tuning** produce the DPO, RLHF, and safety artifacts whose budget-allocation trade-offs Chapter 3 argues, and whose report cards Chapter 6 packages.
- **mod-108 — Evaluating Fine-Tuned Models** owns the release gate. Chapter 5 of this module wraps `release_gate.v3.yaml` into a platform-layer promotion contract; nothing in this module reinvents the gate.
- **mod-109 — Quantization, Export, and Serving** ships the serving runbook. Chapter 5 promotes the runbook into a deploy CRD and change log.
- **mod-110 — Managed Fine-Tuning APIs** produced the per-project managed-vs-self-hosted ADRs. Chapter 4 and Chapter 5 make those ADRs a platform-registered decision — the registry indexes both `ft:...` IDs and internal artifact hashes; the ADR template is the same shape.

## What comes next

This is the final module in the Fine-Tuning Engineer track. The next step is the capstone role — running an actual fine-tuning platform end-to-end for a real product surface — for which the artifacts you produce here (the proposal, the base-selection rubric, the budget allocation, the ADR, the release-gate manifest, the adapter registry, the model card) are the operating documents.
