# exercise-04: Fine-Tune vs. RAG vs. Prompt Decision Doc

**Estimated effort:** 3 hours
**Companion chapters:** [`04-fine-tune-vs-rag-vs-prompt.md`](../04-fine-tune-vs-rag-vs-prompt.md), [`01-from-product-goal-to-proposal.md`](../01-from-product-goal-to-proposal.md), [`02-base-model-selection.md`](../02-base-model-selection.md)

## Objective

For **one concrete product ask** of your choice, write the ADR that recommends **fine-tuning, RAG, prompt engineering, tool wiring, or buying an endpoint** (or a defensible hybrid) and defends the recommendation against **each of the other four options**. Record the escape hatches that would revise the choice and the revision triggers that will force the next re-review.

The point is decision hygiene, not code. **You are not fine-tuning, indexing, or wiring tools in this exercise.** You will run the light-weight capability probes chapter 4 calls "Tests 1–6" — a couple of hours of managed-API budget at most — and use the numbers to defend the ADR. A reviewer six months from now should be able to reopen the ADR, replay the tests, and either agree with the pick or challenge it on a specific test number, not on interpretation.

## Prerequisites

- Chapter 4 read in full; chapter 1 skimmed for the proposal-shape context that the ADR either enables or short-circuits.
- Access to at least **two managed model tiers** from the same or different vendors — one representing "the frontier tier" (Test 1 in chapter 4's protocol) and one representing "your target-tier off-the-shelf model" (Test 2). Chapter 4's tests do not require any tuning access; a plain chat completion API is enough.
- Access to a very small **RAG POC path** — a directory of docs (a manual, a set of tickets, a wiki dump) that maps to your Part A scenario, and any mainstream embedding + vector-index tool (FAISS, Chroma, LanceDB, a hosted equivalent). Chapter 4: "if the RAG POC took longer than a day, the POC scope is wrong." Ten to a hundred docs is plenty.
- Optional but recommended: a small **function-calling probe** — one or two tools stubbed with fake responses, so you can measure Test 5. If your tool responses are ambiguous to stub, skip Test 5 and mark it `needs-research`.
- API budget of ~$20–100 for the tests. If tight, downsample per Part D below.
- No GPU required.

## Requirements

### Part A — Pick and specify the product ask

In `ask.md`, define exactly **one** product ask. The ADR only makes sense with a single, quantified subject. Pin:

- **Product ask.** One paragraph, chapter 1 section-1 vocabulary: what behavior is changing, who observes it, what the baseline is today, what the quantified target is. E.g., "in-house legal assistant that answers questions about the company's contracts with cited-answer accuracy ≥ 90% and refusal on out-of-scope questions ≥ 95%"; "customer-support assistant that reduces billing-ticket escalation rate from 12% to ≤ 8%"; "internal dev-tools chatbot that raises tool-call JSON-schema-valid rate from 82% to ≥ 98% on 100k daily calls."
- **The metric.** Named, quantified, with a held-out eval set (path or plan to build one). Chapter 4's protocol has no meaning without a metric.
- **The gap hypothesis (Part B input).** Your best guess going in: which of chapter 4's four gaps (knowledge / action / capability / behavior) do you think is causing the current miss? You will test the hypothesis in Part D; being wrong is fine.
- **Constraints.** Latency SLO, per-request cost ceiling, data residency, redistribution intent, tenancy. Any hard override that eliminates a candidate option.
- **Prior work.** Whether this ask has been evaluated before (an existing fine-tune, a shipping RAG stack, a prompt-only production endpoint). Chapter 4's ADR revision cadence assumes a prior state.

One page. Everything downstream anchors on it.

### Part B — Diagnose the gap

In `gap.md`, argue which of chapter 4's four gaps is the *primary* driver of the current miss, and which is *secondary* (there is usually more than one). Chapter 4's four:

- **Knowledge gap** — model doesn't know the facts, docs, entities. Fix: RAG.
- **Action gap** — model reasons correctly but can't call the tools. Fix: function calling.
- **Capability gap** — model can't reason at this level even with the knowledge and tools. Fix: bigger tier / different base.
- **Behavior gap** — model has knowledge + tools + capability but consistently produces the wrong *shape* of output. Fix: fine-tune.

For each gap, cite:

- **The signal.** What in the current production behavior or in a preliminary probe suggests this gap is (or is not) present. Chapter 4's per-gap signal list.
- **The falsifiable test.** Chapter 4's "test" for each gap — put the fact in the prompt (knowledge), simulate a tool call (action), run on the frontier tier (capability), add a strong system prompt (behavior).
- **Rank it.** Primary / secondary / not-applicable.

The gap diagnosis is what makes the ADR defensible. Chapter 4: the *vocabulary* of the gap anchors reviewers on the right alternative.

### Part C — Enumerate the five options

In `options.md`, list all five of chapter 4's options with one paragraph each on whether it fits your Part A ask *before running the tests*:

1. Prompt engineering.
2. RAG.
3. Tool wiring / function calling.
4. Buy an endpoint / larger tier.
5. Fine-tune.

Chapter 4's ordering (1 → 5 = increasing cost and coupling) is the discipline; you argue each option's fit in the pre-test view. Two options may be hybridized; write them out both individually and, if the pre-test view suggests it, as a hybrid.

If any option is eliminated by a hard constraint before testing (residency requires a specific vendor with no RAG offering; tenant blocks a specific base; buy-tier price already exceeds ceiling), say so plainly and skip that option in Part D.

### Part D — Run the tests

Author `tests.md`. Run chapter 4's six tests (or as many as your budget and access allow) and record numbers:

- **Test 1** — Baseline on the largest available off-the-shelf model with a plain prompt. Measures the "buy" ceiling.
- **Test 2** — Baseline on the target-tier off-the-shelf model with a plain prompt. Starting point.
- **Test 3** — Test 2 + a strong system prompt + few-shot exemplars. "Prompt engineering" ceiling.
- **Test 4** — Test 3 + RAG (retrieval context stuffed in from a small POC index). "RAG" ceiling.
- **Test 5** — Test 4 + function calling to stubbed tools. "Tool wiring" ceiling.
- **Test 6** — Optional if 1–5 do not close the gap. A small SFT slice (a few thousand rows) at the target tier — or a small managed fine-tune (mod-110 ex 01 mechanics) if self-hosted training is not accessible. Do this ONLY if the earlier tests do not close; chapter 4's protocol.

For each test:

- Score against the Part A metric — sample size, threshold pass/fail, confidence interval where feasible.
- Cost per test — API dollars, wall-clock, engineer-hours.
- Failure mode observed — chapter 4's diagnostic language ("knowledge gap remained: the model still could not cite ticket IDs it hadn't seen"; "behavior gap surfaced: model followed instruction 70% of the time, missed the required refusal 30%").

If a test is skipped, mark it `<!-- needs-research: <why skipped> -->` — chapter 4's discipline: the ADR cannot exist without at least Tests 1, 3, and one of 4 or 5.

**Downsample if budget is tight.** 30–100 held-out prompts is plenty for the ADR's purposes. Report the sample size and the 95% bootstrap CI on any test result you use for the decision.

### Part E — Author the ADR

Author `adr.md` following chapter 4's five-section template.

1. **Context.** One paragraph on the product ask, the metric, the constraints. Cite `ask.md`.
2. **Alternatives evaluated.** The five-option table with Test numbers, Cost/month estimates, and Notes. Chapter 4's example table is the shape.
3. **Decision.** Which option (or hybrid) is chosen. If fine-tuning, note that the decision closes at "fine-tune"; the *managed vs. self-hosted* sub-decision is deferred to mod-110's ADR (chapter 4's discipline).
4. **Rationale.** Two or three paragraphs. Cite the axes / tests that decided. Address the gap diagnosis (Part B) explicitly — did the tests confirm your Part A hypothesis?
5. **Escape hatches.** At least three concrete signals that would revise the decision. Chapter 4's examples: frontier-tier price drops, new base ships, retrieval infrastructure improves, metric drifts, regulatory change.
6. **Consequences.** Chapter 4's chapter-1 tie: if the decision is "fine-tune," the ADR ends and the proposal (chapter 1) begins. If it is "RAG + tools + fine-tune," the ADR names the pieces and the roadmap. If it is "prompt only," the ADR names who watches the metric so the decision gets revisited when it shifts.

Under 1200 words. Chapter 4: "the ADR is short. It refers out to the proposal, the base scorecard, the dataset-budget allocation, and the eval results. It is the *decision*, not the *design*."

### Part F — Revision triggers and open questions

Author `revisits.md`. Two sections:

- **Revision triggers.** The concrete events that would force the ADR to be revisited. Chapter 4's list: frontier tier price shift ≥ 30%; new base with meaningfully higher capability at your tier; new RAG / tools infrastructure; metric drift; regulatory / vendor / license change. For each trigger you name, specify: the owner watching the signal, the cadence of the watch (weekly / quarterly / event-driven), and the ADR version that would be produced by the revisit.
- **Open questions.** The specific things this ADR cannot resolve today. Chapter 4's Test 6 that you didn't run; a knowledge-cutoff question that RAG might address later; a tool that isn't yet wired; a base whose capability probe was inconclusive. Owner + due date per question.

Chapter 4's discipline: an ADR without a revision trigger is an ADR that will be relitigated in Slack rather than by rewrite.

### Part G — Directory layout

Final tree:

```
outputs/exercise-04/
├── ask.md
├── gap.md
├── options.md
├── tests.md
├── adr.md
├── revisits.md
└── sources.md          # every URL cited, with retrieval date
```

`sources.md` records every URL for cited vendor docs (pricing, model catalog, deprecation policy), every retrieval or embedding tool used, and every external benchmark. Chapter 4 references ADR template patterns (Michael Nygard's blog post, Google Cloud ADR write-up); if you cite either, record it here.

## Starter guidance

- **Do not skip the tests.** Chapter 4's central discipline. The ADR is defensible because Tests 1–5 exist; without them, the ADR is opinion. Even at a very small sample size and a downsampled eval, the numbers anchor the argument.
- **Diagnose the gap before the tests, then verify.** Chapter 4: writing the gap diagnosis first commits you to a hypothesis; the tests either confirm it or reveal the actual gap. Both outcomes are valuable; a document that only shows the "right" gap has skipped the reasoning.
- **RAG POC is a day, not a week.** Chapter 4: "if the RAG POC took longer than a day, the POC scope is wrong." A FAISS-over-embeddings index of 50–500 docs, plus a top-k=5 retrieval stub in the prompt, is the target scope.
- **Do not confuse hybrid with "we do a bunch of stuff."** Chapter 4: hybrids are the norm, not a cop-out. The ADR names the hybrid ("RAG + tools + light fine-tune for style") and defends the pieces. "RAG + fine-tune, TBD which is more important" is not defended.
- **Buying the frontier tier is not "cheating."** If Test 1 passes, chapter 4 is emphatic: ship prompt engineering on the bought endpoint. The ADR that concludes "buy an endpoint, no fine-tune" is a valid ADR — often the correct one at low volume.
- **Do not fine-tune the docs.** Chapter 4's three "wrong-first" patterns: teach-the-docs, live-data, internal-APIs. If your Part A ask reads like any of these, RAG or tools is the answer before fine-tune enters the conversation. State so explicitly in the ADR if it applies.
- **Do the total-cost-of-ownership math.** Chapter 4: a fine-tuned smaller model has ongoing engineering cost (release gate, model card, rollback plan, retraining every N months) that a bought endpoint does not. At low volume, buy is often cheaper TCO than fine-tune — the ADR should show the math.
- **The ADR is *not* the proposal.** Chapter 4 hands off to chapter 1: if the ADR concludes "fine-tune," the proposal is the next document, not a longer ADR. Keep the ADR focused on the decision; defer the design.
- **Do NOT commit real customer data or unmoderated retrieval corpora.** If your RAG POC pulls from a real internal corpus, redact or synthesize before publishing the exercise output.

## Acceptance criteria

You have completed the exercise when:

- `ask.md` pins the five ask fields (product ask, metric, gap hypothesis, constraints, prior work).
- `gap.md` argues primary + secondary gaps with signals and falsifiable tests, using chapter 4's four-gap vocabulary.
- `options.md` covers all five options with pre-test fit rationale; hybrids are named separately if they apply.
- `tests.md` reports at least Tests 1, 3, and one of 4 / 5 with real numbers, sample size, and CI; Test 6 is included only if 1–5 did not close.
- `adr.md` follows chapter 4's five-section template, comes in under 1200 words, names a specific decision (single option or defended hybrid), and lists at least three escape hatches.
- `revisits.md` lists at least three revision triggers with owner + cadence, plus at least two open questions with owner + due date.
- `sources.md` records every citation.

## Stretch goals

- **Add the total-cost-of-ownership sheet.** Author `tco.md` with a small table comparing the chosen option's monthly TCO to the two runners-up over a 12-month horizon. Include training amortization, engineering maintenance (release gate, model card, retraining every 6 months), inference cost, and the "fine-tune-specific" tax (drift monitoring, rollback storage, retirement). Chapter 4's TCO discussion.
- **Sensitivity-test the ADR.** Vary Part A's metric target by ±10% and re-run Parts D and E. Does the decision hold? Vary the traffic volume by 10× — does the decision hold at scale? Chapter 4: escape hatches are what make the ADR responsive to sensitivity.
- **Run the actual Test 6.** If Test 6 is deferred in the base exercise, run a small fine-tune (mod-104 QLoRA on ~2k rows, or a mod-110 managed fine-tune on ~1k rows). Compare the numbers to Tests 1–5. Chapter 4: this is the difference between "we didn't try fine-tuning" and "we tried it and it was worse than RAG."
- **Draft the hand-off to chapter 1.** If your ADR concludes "fine-tune," write the two-page proposal (chapter 1 section 1–6) that would come next. If it concludes "RAG + tools," write the RAG-and-tools design doc (which is not chapter 1's proposal — chapter 4 hands off elsewhere) and note where the artifact lands in your team's roadmap.
- **Write the counter-ADR.** Take the same ask, but flip the recommendation and write an ADR that defends the *opposite* choice. Chapter 4's discipline: a decision is only as strong as the opposing argument you can articulate and refute. Where does the counter-ADR beat yours? Where does yours beat the counter-ADR? Fold the findings back into `revisits.md`.
- **Revisit the ADR after a hypothetical trigger.** Assume six months have passed and one of your revision triggers has fired (frontier price dropped 40%; a new base ships; the metric drifted). Author `adr.v2.md` with the revised recommendation, citing chapter 4's revisit cadence. Chapter 4: version the ADR, not the argument.
