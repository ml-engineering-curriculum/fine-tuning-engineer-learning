# exercise-01: Base Model Selection Rubric

**Estimated effort:** 3 hours
**Companion chapters:** [`01-from-product-goal-to-proposal.md`](../01-from-product-goal-to-proposal.md), [`02-base-model-selection.md`](../02-base-model-selection.md), [`04-fine-tune-vs-rag-vs-prompt.md`](../04-fine-tune-vs-rag-vs-prompt.md)

## Objective

For **one concrete product scenario** of your choice, build the versioned base-model selection rubric from chapter 2, score **three real base-model candidates** against it, and land the artifact bundle a real fine-tuning team would attach to a `base-selection-<project>` decision: the scorecard spreadsheet, the ADR that names a winner, and the risk register that comes out of the exercise.

The point is the *decision hygiene*, not the answer. No training happens here. A reviewer six months from now should be able to open your ADR, walk your weights, replay your scoring, and either agree with your verdict or challenge it on a specific axis — not on taste.

## Prerequisites

- Chapter 2 read; chapter 1 skimmed for the proposal shape the rubric lives inside; chapter 4 skimmed so you can defend "why fine-tune at all" in the ADR context section.
- Access to the primary docs for every base you consider — the model card, the tokenizer config, the chat template, the license file. All three of your candidates should have public documentation you can cite.
- A spreadsheet or Markdown-table environment for the scorecard. A small local script that can `apply_chat_template` and count tokens on a representative prompt is useful for the tokenizer axis but not required — a link to the tokenizer config plus a hand-verified count on 5–10 example prompts is enough.
- No GPU required. **You are not fine-tuning.** If you want to run a zero-shot capability probe for axis 2, a managed inference endpoint (or a small local `transformers` load on CPU / MPS with quantization) suffices for a 30–100 prompt probe.
- No API budget required if you skip the capability-floor probe; ~$5–20 if you run a probe through a managed endpoint.

## Requirements

### Part A — Pick and specify the scenario

In `scenario.md`, define exactly **one** scenario. Everything downstream is anchored on it. Pin:

- **Product context.** One paragraph: what the fine-tune is for, who calls it, what quality bar it must meet. A plausible scenario is fine; if you are inventing it, say so.
- **Redistribution intent.** Internal only / SaaS behind an API / open-weights release / commercial-product with weights shipped. This is what will move the license weight in Part C.
- **Serving constraint.** GPU class and count (e.g., `1× A10G 24GB`, `2× H100 80GB`, `1× consumer 4090`) *or* a managed inference endpoint tier. This bounds the size / hardware ceiling.
- **Objective stack (planned).** SFT / SFT+DPO / joint ORPO / SFT+PPO / SFT+RFT. Chapter 3's stack A / B / C / D vocabulary. This is what drives the method-compatibility axis.
- **Data budget (rough).** Approximate SFT rows + preference pairs, from your best guess for the scenario. This drives the compute-cost axis.
- **Latency and availability constraints.** p95 TTFT target; monthly uptime target. Rough is fine.
- **Regulatory / tenancy constraints.** Any hard overrides (residency, tenant exclusion, no-vendor-X). If none, say so explicitly.

One page. This is the input to Part B.

### Part B — Pick three candidates

In `candidates.md`, name **exactly three** base-model candidates and, for each, one paragraph on **why it is a plausible choice for this scenario** (not why it will win). The composition constraint:

- At least **two open-weights bases** from mainstream families (Llama 3.x, Qwen 2.5 / 3, Mistral / Mixtral / Nemo, Gemma 2, DeepSeek, Phi — pick from families with public model cards and mainstream ecosystem support).
- At least **one candidate** that is *not the same as the other two* on some axis you care about — a different license family, a different size class, a different vendor's managed candidate. The point is contrast; three near-duplicates learn you nothing.

Pin each candidate as `<org>/<name>` at a specific revision (Hugging Face `revision`, or the vendor's dated model ID for managed candidates). Chapter 2's "everything downstream references the pin, not the pretty name."

If a candidate is a **managed candidate** (mod-110), still score it on the same nine axes; note in `candidates.md` where the axes are read differently (Axis 3 becomes vendor cost / hosting rent; Axis 6 becomes the narrower managed method surface).

### Part C — Author the rubric

Author `rubric.md`. Cover:

1. **The nine axes**, verbatim from chapter 2 (license, capability floor, size / HW ceiling, tokenizer, chat-template maturity, method compatibility, community / ecosystem, deprecation risk, cost).
2. **Weights, set *before* scoring, with one sentence of rationale per weight.** Chapter 2's defaults are a starting point; deviate where your scenario demands it (e.g., raise license weight if redistribution intent is commercial redistribution; raise cost weight if you are comparing across size tiers). Weights are integers 1–3. Setting weights after scoring is base-preference laundering; do not do that.
3. **Per-axis scoring criteria.** For each axis, what a 0, a 1, and a 2 look like *for this scenario*. E.g., for tokenizer on a Japanese-first product: `0 = > 3× English token count`, `1 = 1.5–3× English token count`, `2 = comparable to English`. This is what makes the scoring reproducible.
4. **Hard-constraint list.** The overrides that produce a boolean, not a score. Chapter 2's license blocker / regulatory floor / vendor exclusion; anything specific to your scenario.

The rubric is authored *once* per project family and reused across candidates. Version it (`rubric.v1.md`).

### Part D — Score the candidates

Author `scorecard.md` following the chapter 2 template (see the "The scorecard, filled in" section). For each candidate × axis cell, record:

- The score (0 / 1 / 2).
- **The evidence.** A citation for every score. License scores cite the license file; tokenizer scores cite a token-count experiment or the tokenizer config; capability-floor scores cite either a zero-shot probe you ran (see below) or a public benchmark result on the base's model card; method-compatibility scores cite the relevant library's model-support matrix (`peft`, `trl`, `vllm`, `text-generation-inference`, `llama.cpp`); community scores cite Hugging Face download counts and canonical recipe repos (Unsloth, Axolotl, LLaMA-Factory, first-party guides); deprecation cites the vendor's published lifecycle policy or its absence.
- **A one-sentence note** where the score needs qualification.

Compute the weighted total (0–max). Chapter 2's worked scorecard is the model.

**Optional: run a zero-shot capability probe.** Pick 30–100 prompts representative of your task-specific held-out set (or reuse mod-108 ex 02's if you did the mod-108 track). Run each candidate at zero-shot with the base's official chat template. Record the per-candidate accuracy or rubric-mean with bootstrap 95% CIs. This turns axis 2 from "1 vs 2 on intuition" into "1 vs 2 with CIs." If you skip this, mark axis 2 `needs-research: capability-probe deferred to full proposal` — do not silently score.

### Part E — Author the ADR

Author `adr.md` following chapter 2's ADR template. Sections:

1. **Context.** Two paragraphs: what the scenario is, what the decision is (base for `<project>`), what the constraints are. Cite `scenario.md`.
2. **Options considered.** Table of the three candidates with their weighted-total scores.
3. **Decision.** Which base wins. If a hard constraint moved the decision away from the rubric-topping candidate, say so plainly — "rubric top scorer was X; hard constraint Y disqualifies X; we chose Z."
4. **Rationale.** Two or three paragraphs. Cite the axes that decided. Cite the tie-breaker (incumbency, chapter 2) if it applied. Do not restate the scorecard; explain why *this* axis mattered more than others.
5. **Escape hatches.** Concrete signals that would revise the decision — chapter 2's "if X, revert to Y" pattern. At least three.
6. **Consequences.** What downstream work this decision creates (chat template rebase, eval baselines, tokenizer-specific data prep). What downstream work it avoids by not picking the other candidates.
7. **Reviewers.** Named sign-offs (yourself, a peer if available, or `<pending>` placeholders — the point is that the section exists).

Under 1200 words. If it is longer, it is a design doc, not a decision record.

### Part F — Risk register

Author `risks.md`. A table of 5–10 rows, chapter 1 section-6 format:

| Risk | Severity | Owner | Trigger | Mitigation |
|---|---|---|---|---|
| License reconsideration for candidate X | ... | ... | ... | ... |

For each risk: what would go wrong, how bad (low / medium / high), who would own the response, what event or signal would fire the risk, what the mitigation is (planned or in-place). At minimum include: a license risk, a deprecation risk, a capability-floor risk (rubric was wrong), a data risk (SFT budget shortfall changes the base's fit), and a tokenizer risk (poor coverage discovered mid-project).

### Part G — Directory layout

Final tree:

```
outputs/exercise-01/
├── scenario.md
├── candidates.md
├── rubric.md
├── scorecard.md
├── adr.md
├── risks.md
└── sources.md          # every URL cited across the above, with retrieval date
```

`sources.md` is the audit trail. Every URL referenced in `rubric.md`, `scorecard.md`, or `adr.md` appears here with `<url> — <what it was cited for> — retrieved YYYY-MM-DD`. Chapter 2's "everything downstream cites the pin, not the pretty name" applies to citations too.

## Starter guidance

- **Do Part A first, and pick a real-shaped scenario.** A vague "some kind of chat product" scenario produces a vague rubric. Pin the redistribution intent, the serving GPU, the objective stack, the data budget — most of the axes fall out of those four.
- **Set the weights before you look at the candidates.** Chapter 2 is emphatic on this. Justify each weight with a scenario-specific sentence; then close `rubric.md` before opening `candidates.md`.
- **Cite the license file, not the marketing page.** For every license axis score, the citation is the SPDX or the vendor's terms — not a homepage summary. Common ones: Apache-2.0, MIT, Llama 3.1 / 3.3 Community License, Gemma Terms of Use.
- **Do the tokenizer axis with actual numbers.** Take 5–10 prompts representative of your scenario, run `apply_chat_template` + tokenize with each candidate's tokenizer, count. A base that tokenizes your scenario 2× longer than another eats KV-cache and serving cost every request for the model's life; a score based on intuition here is the axis's biggest failure mode.
- **The capability probe is optional, but the scorecard should say so.** Do not score axis 2 as "2" from vibes; either run a probe or write `needs-research`. Chapter 2 is explicit: "clearly ahead" is *outside the overlap of 95% bootstrap CIs on the task-specific held-out set*.
- **Do not conflate serving size with training size.** Chapter 2's common mistake. A 70B QLoRA + AWQ 4-bit for serving fits a scenario that a full-parameter 70B does not; disqualifying a base because full-parameter would not fit is the wrong call.
- **The ADR names the tie-break.** If two candidates are within one point of each other, write down what broke the tie — incumbency, ecosystem, license comfort, migration cost. Chapter 2's tie-break section is the reference.
- **Do NOT commit real customer data.** If your scenario borrows from a real product, generalize enough that the scenario file is safe to publish. The exercise's usefulness is the reasoning, not the specifics.

## Acceptance criteria

You have completed the exercise when:

- `scenario.md` pins the seven scenario fields (product, redistribution intent, serving, objective stack, data budget, latency, regulatory).
- `candidates.md` names three real candidates with pinned revisions and a one-paragraph plausibility rationale each; at least two are open-weights, and the three are not near-duplicates of each other.
- `rubric.md` covers all nine axes with per-axis 0 / 1 / 2 criteria specific to your scenario, weights set *before* Part D was written, and a hard-constraint list.
- `scorecard.md` has one row per candidate and one column per axis, every cell has a score + evidence citation, and a weighted total is computed. No cell says "unknown"; gaps are `<!-- needs-research: ... -->` with a specific question.
- `adr.md` follows the chapter 2 template, comes in under 1200 words, names a specific winner, and lists at least three escape hatches.
- `risks.md` has at least five risks, each with severity / owner / trigger / mitigation.
- `sources.md` lists every URL cited with retrieval date.

## Stretch goals

- **Run the zero-shot capability probe on all three candidates.** ~30 minutes of managed-API budget or a couple of hours of local inference. Report per-candidate accuracy with 95% bootstrap CIs; chapter 2's "clearly ahead" definition becomes empirically defensible.
- **Add a fourth managed candidate.** A vendor from mod-110 chapter 1's seven (OpenAI, Anthropic, Vertex, Mistral, Cohere, Bedrock, Azure). Score it against the same nine axes with the two adjustments chapter 2 flags (axis 3 becomes vendor cost; axis 6 becomes the managed method surface). Does the managed candidate change the winner?
- **Sensitivity-test the weights.** Take your chosen weights and vary each one ±1 in turn. Does the winner change? A rubric whose winner is sensitive to a single ±1 weight is a rubric that has not decided a lot — flag it in the ADR.
- **Model the migration.** Write a one-page `migration.md` that walks through what it would take to switch off the winner if you had to — which artifacts port, which need to be rebuilt, how many engineering-days. This is the "escape hatches, quantified" companion to the ADR.
- **Draft the base-selection ADR as if you were the incumbent's team.** Run the exact same rubric against the same three candidates, but score under the constraint that the *incumbent* is one of the three and incumbency is worth an explicit tiebreaker point. Chapter 2's incumbency discussion. Does the verdict flip?
- **Add axis 10: reproducibility of the base's own release.** Not every base ships with a machine-consumable chat template, a pinned tokenizer revision, and a stable release channel. If your scenario is heavy on reproducibility (regulated tenant, external partner), a tenth axis on "reproducibility of the upstream release" may be worth including. Score it and defend the addition in `rubric.md`.
