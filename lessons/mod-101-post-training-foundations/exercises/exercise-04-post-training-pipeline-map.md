# exercise-04: Post-Training Pipeline Map

**Estimated effort:** 2 hours
**Companion chapter:** [`04-post-training-pipeline-map.md`](../04-post-training-pipeline-map.md)

## Objective

Produce a **one-page pipeline map** for a realistic post-training project — data sources, methods, evaluation gates, go/no-go criteria at each stage — and defend the choices in a short review. This is a design exercise, not a coding one. Its purpose is to make you walk through every stage of the pipeline in your head before you commit any compute.

Every fine-tuner ships a document that looks like this at the start of any real project. Half the value is that writing it down surfaces the questions you did not think to ask.

## Prerequisites

- Read of the companion chapter.
- No compute required.

## The scenario

Choose **one** of the following scenarios, or invent your own of comparable scope (~one paragraph):

**Scenario A — Customer-support chatbot for a mid-market SaaS company.** The current chatbot is a prompted-only wrapper around a frontier API. Product wants to migrate to a smaller open base with a company-specific fine-tune, primarily to reduce per-message cost and to keep customer data on infrastructure the company controls. Traffic: ~50k conversations/day, average 6 turns each. Policy needs: consistent brand voice, refusal on cross-customer data lookups, no personal opinions, always end with an "anything else?" callback.

**Scenario B — Legal-domain summarization assistant.** Internal tool for a legal team; consumes contract text up to 40k tokens and outputs a structured summary + a plain-English executive summary. Users are lawyers who will not tolerate hallucinated clauses. Must run on a single 24-GB GPU. Update cadence: monthly retrain as the reviewed contract corpus grows.

**Scenario C — Coding assistant tuned to a proprietary internal DSL.** Base model already codes fine in mainstream languages; company wants it to be fluent in an internal DSL used by ~200 engineers. Training data: ~40k human-authored files in the DSL plus PR descriptions. Constraint: must not regress on general-purpose Python.

**Scenario D — Custom.** Write your own scenario at similar detail. Get one sentence of feedback from a peer or your instructor before continuing.

## Requirements

Deliver a **one-page** (approximately 400–700 words) design document with these sections:

### 1. Project one-liner and success metric

Two or three sentences. What is the model for, and what specific metric moves if it succeeds? "Reduce cost per resolved conversation by 40% while holding customer-satisfaction score within 2 points" is a target. "Be a good chatbot" is not.

### 2. Base model choice

Name the base — vendor, size, tokenizer family, positional encoding, license, and one-sentence justification. Explain what you would *not* choose and why. If your choice depends on managed vs. self-hosted, note that decision here and defer the tradeoffs to section 7.

### 3. Stage-by-stage plan

For each stage below, in a short bullet or two:

- **DAPT (optional):** are you doing continued pretraining? If yes, on what data? If no, why not?
- **SFT:** dataset(s), approximate size in examples/tokens, chat template, PEFT strategy (full-parameter, LoRA, QLoRA), rank if applicable, and loss-masking policy.
- **Preference optimization:** which method (DPO, PPO, ORPO, KTO, GRPO), preference-pair source, and whether you need a reward model.
- **Safety tuning:** what refusal data, what red-team ingestion, what specific policies you are targeting.

Cite the module of this track that owns each stage.

### 4. Evaluation gates

Enumerate the evaluations you will run at each stage. For each eval, name:

- The eval — MMLU, HumanEval, MT-Bench, needle-in-a-haystack, a custom task-specific held-out set, refusal-rate battery, etc.
- The **go/no-go threshold** — what number you have to hit for the run to be released.
- Whether it is a **regression gate** (must not drop from base by more than X) or a **target gate** (must reach ≥ Y).

Include at least one custom, task-specific eval — you cannot ship a real product on academic benchmarks alone.

### 5. Serving artifact

What will the final artifact be — merged checkpoint or LoRA adapter served alongside a base? What quantization? What runtime? Estimate VRAM at your target serving batch size and context length.

### 6. Model card / release review

What will you put in the model card? Name three things a governance or risk reviewer would flag as missing if you shipped without them.

### 7. Fine-tune vs. RAG vs. prompt engineering

Two sentences. Why does this project need a fine-tune at all? What would you use RAG or prompting for instead in this scenario? (Answering "because the ticket said so" is not sufficient — you must be able to defend fine-tuning as the correct tool for the problem.)

### 8. Risks and open questions

List three risks (technical, data, or policy) that would derail this project and one open question you would want answered before starting the first SFT run.

## Format

Deliver as a single markdown file, `pipeline-map.md`, in your working repo. Optionally include an image or ASCII diagram summarizing your version of the pipeline map from chapter 4, marked up with your specific choices at each stage.

Keep it to one page rendered. If it does not fit, tighten prose, not detail — the review requires the numbers and the choices.

## Starter guidance

- The chapter's pipeline diagram is a fine skeleton to adapt. Copy it, then annotate each stage with your choices.
- The mod-102 through mod-111 summaries in [`CURRICULUM.md`](../../../CURRICULUM.md) name the tooling in scope for each stage; use them to keep your plan realistic.
- If you cannot decide between two options in a stage, write both down and note the decision you would use to break the tie (a cost number, a latency SLO, a specific eval score).
- Be honest about what you would defer. "Preference optimization deferred to phase 2; SFT-only launch in phase 1" is a legitimate stage plan, as long as you say so.

## Acceptance criteria

You have completed the exercise when:

- All eight sections are present and specific to your chosen scenario.
- Every stage in your plan names both the method / dataset and the module of this track that would own it.
- Every evaluation gate has a numeric threshold or a clearly stated procedure for deriving one.
- Section 7 defends the fine-tune-vs-alternatives choice with a real argument.
- The document fits on one printed page or one screen without scrolling more than a screen and a half.
- A peer or your instructor could read it in three minutes and know what you plan to build.

## Stretch goals

- **Write a companion "not doing" section.** Enumerate the stages or techniques you deliberately skipped and why. Explicit non-choices are as important as choices for a mid-course review.
- **Cost estimate.** Attach a rough budget: GPU-hours per stage at your chosen size, API cost estimate if you are using a managed fine-tuning API, annotator cost estimate for the preference data if you are gathering pairs. Real numbers, not hand-waves — you can look up rates.
- **Rollback plan.** In one paragraph, describe how you would roll back a bad adapter release without downtime. (This is the mod-111 territory that this exercise touches lightly; if you have already done systems-design work in another track, borrow from that.)
- **Data lineage sketch.** For every dataset in your plan, name the source, license, and the field(s) in a model card entry it would populate. Then identify one dataset in your plan whose license actually prohibits the use you are planning — if you find one, congratulate yourself for catching it here rather than in review.
