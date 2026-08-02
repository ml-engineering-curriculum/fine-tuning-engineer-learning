# From Product Goal to a Fine-Tuning Proposal

**Estimated read:** 30 minutes.
**You should be able to, by the end:** turn a product ask into a written fine-tuning proposal that pins the problem, the success metric, a base-model shortlist, the dataset budget, the objective stack, the cost / timeline estimate, and the risk register. Recognize the four failure modes of the intake step — problem-drift, metric-vagueness, base-locked-in-before-data, and cost-unmodeled — and know which questions kill each one at the intake meeting rather than at the halfway-review.

## Why the intake step is the highest-leverage step

Most fine-tuning projects that fail do not fail because SFT was misconfigured or DPO was chosen over KTO. They fail because the intake was fuzzy. The training loop optimized the wrong objective, or the "10% better" metric turned out to be unmeasurable, or the base model was picked before anyone counted rows of data, or the timeline estimate excluded the eval work and the safety review.

The fine-tuning engineer's leverage on the outcome is greatest at intake and drops sharply from there. A base model chosen wrong at week 1 costs a month to swap after the SFT run is complete; a success metric that turns out to be unmeasurable at week 6 costs the whole project. Getting the intake to a proposal takes half a day; getting it wrong wastes a quarter.

This chapter is the intake process itself, written down. The rest of the module — base-model rubric, dataset-budget allocation, ADR templates, release gate, model card — is the machinery that turns a proposal into a shipped model. Nothing downstream is fixable if the proposal is fuzzy.

## The proposal document, top to bottom

A fine-tuning proposal is a Markdown document. Six sections. Written in one to three pages. It is the operating contract between the fine-tuning team and everyone who depends on the outcome.

```markdown
# Fine-tuning proposal — <project name>
**Owner:** <fine-tuning lead>
**Sponsor:** <product owner>
**Reviewers:** <peer engineer, safety lead, platform lead>
**Status:** draft | reviewed | approved | active | shipped | retired
**Version:** v0.3
**Last updated:** YYYY-MM-DD

## 1. Problem and product goal
## 2. Success metric and eval plan
## 3. Base-model shortlist
## 4. Dataset budget and objective stack
## 5. Cost, timeline, and staffing
## 6. Risk register and open questions
```

The rest of this chapter is the six sections in detail — what belongs in each, what fails when a section is skipped, and the concrete questions that force the section to be honest.

## Section 1 — Problem and product goal

**The section is one paragraph.** It answers three questions:

- **What behavior are we changing?** Not "improve the assistant" but "reduce escalation rate on billing tickets," "raise pass@1 on our SQL rewrite suite," "increase JSON-schema-valid rate for tool-call outputs from 82% to 95%." A behavior a customer or an operator can observe.
- **Who observes the change?** The end user, a downstream system, a moderator queue, an on-call operator. If nobody observes it, it is not a product change.
- **What is the baseline today?** A number, a percentage, a rate — measured, not guessed. If you cannot measure the baseline, section 2 is not going to work either.

The single most common intake failure is that this section reads "improve quality." Every conversation from that point forward argues past each other. Write the problem in a sentence a product manager would recognize as their ask, in a metric a fine-tuning engineer can point at.

**Two anti-patterns:**

- **The umbrella problem.** "Make the assistant better at everything" — impossible to scope, impossible to gate. Break it into three specific behaviors and pick one to attack first.
- **The technique problem.** "Add DPO to the assistant" — starts from a technique, not from a behavior. The technique is a means, not an end; write the behavior first, pick the technique in section 4.

## Section 2 — Success metric and eval plan

Every proposal has *at least three* metrics, in tension with each other. Use exactly the vocabulary and lanes from mod-108 — this section is where mod-108's release gate gets scoped for the specific project.

- **Primary product metric.** The observable behavior from section 1, quantified. "Escalation rate on billing tickets ≤ 8% (was 12%)"; "JSON-schema-valid on tool-call outputs ≥ 95% (was 82%)"; "human preference win rate over the current model ≥ 55% on the task-specific eval set (was 50/50 by construction)." This is what the product owner cares about; it is the metric the release gate promotes on.
- **Capability-regression guardrails.** The academic benchmarks (MMLU, GSM8K, HumanEval, IFEval, MT-Bench, or whatever subset applies) that the model must not regress on by more than an acceptable band. Chapter 3 of mod-108 argues why; the whole point is to catch "we made it better at billing tickets and worse at every other kind of conversation."
- **Safety metric.** The absolute floor (mod-106) — refusal rate on the harmful set stays ≥ 0.95, harmful-completion rate stays ≤ 0.02, or whichever safety report card metrics your organization has committed to. Never allowed to move in the wrong direction, never allowed to fall below the floor.

For each metric write down:

1. **The dataset that measures it** — path, version, count, label source. If it does not exist yet, section 5 (timeline) has to include building it.
2. **The threshold** — the pass number, and whether the rule is "no regression greater than X" or "absolute at least Y" (mod-108 chapter 6 vocabulary).
3. **The cadence** — every commit (fast lane), every release candidate (full gate), or on-schedule (drift monitoring, chapter 7).

If any metric is TBD at intake, the project is not scoped enough to start. Push back until section 2 has real numbers.

**One question that kills the section if unanswered:** *how would we know if this failed?* If the sponsor cannot describe what "failed" looks like operationally, the metric is not sharp enough to fine-tune against.

## Section 3 — Base-model shortlist

The section is a table of two to four base models, scored against the base-selection rubric (chapter 2). At intake you are not picking the base — you are naming the candidates and pinning the tie-breaker.

Minimum table:

| Base | License | Params | Method compat | Rough cost per run | Notes |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | Llama 3 Community | 8B | SFT, LoRA/QLoRA, DPO, ORPO, KTO, safety adapters | ~$X on 4× A100 | strong ecosystem, chat template stable |
| Qwen2.5-7B-Instruct | Apache 2.0 | 7B | same as above | ~$X | permissive license, strong on non-English |
| Mistral-7B-Instruct-v0.3 | Apache 2.0 | 7B | same as above | ~$X | narrower context, mature tooling |
| gpt-4.1-mini (managed) | proprietary, per contract | n/a | managed SFT, managed DPO (chat) | training + inference rent | mod-110 path; skip if self-hosted-only |

The candidate set is bounded by the *product-scenario* constraints, not by taste:

- **Hard licensing constraints.** If redistribution intent is "commercial product with the fine-tuned weights redistributed," permissive-license (Apache 2.0, MIT) is the smaller candidate set. If it is "internal-only," the Llama Community License is fine. Chapter 2 walks the license map.
- **Hard hardware constraints.** If serving is 24-GB single-GPU, ≤ 8B is the range; if serving is 4× H100, up to ~70B is on the table.
- **Hard latency constraints.** If p95 TTFT is < 200 ms and the deployment is single-GPU, model size and quantization are already partially decided.
- **Hard vendor constraints.** If the customer is a regulated tenant that has cleared Bedrock but not OpenAI direct, the managed candidate collapses to the Bedrock-hosted subset (mod-110 chapter 3).

At intake, list the constraints explicitly and prune the candidate list against them. The proposal names the *shortlist* and defers the pick to the base-selection rubric run (chapter 2 exercise 01).

**Anti-pattern: the base is decided before the data is counted.** If the proposal opens "we're going to fine-tune Llama-3.1-70B on this," the intake has skipped section 4. Rewrite.

## Section 4 — Dataset budget and objective stack

This section is the biggest budgetary decision in the proposal. Chapter 3 covers it in full; for the proposal, three sub-answers are needed.

- **Dataset budget.** Approximate row counts you are committing to acquire or generate at each objective:

  | Objective | Rows | Source | Owner | Cost |
  |---|---|---|---|---|
  | SFT (task-specific) | 20 000 | mix of internal ticket transcripts + synthetic (mod-102 exercise 04) | data team | ~$X judge / labeling |
  | SFT (capability-preserving mix) | 30 000 | Tulu-3, no_robots, dolly-15k subset | fine-tuning team | 0 |
  | DPO / preference | 5 000 pairs | internal preference labeler (chapter 3) | product ops | ~$Y |
  | Safety tuning | 2 000 pairs | red-team-derived (mod-106 chapter 3) | safety team | 0 (already collected) |
  | Total | 57 000 rows | | | ~$Z |

- **Objective stack.** The sequence — SFT → DPO → safety-alignment adapter, or a single joint ORPO run, or SFT-only with runtime guardrails. Justify the choice in one paragraph; cite the trade-off argument from chapter 3.
- **Ablation intent.** Which one or two knobs would you turn if the primary run misses the metric? "If SFT alone hits 92% on the JSON metric, ship it and skip DPO. Otherwise DPO with pairs derived from the SFT model's errors."

**The section forces two conversations that intake often skips:**

- **Who owns each dataset slice?** If nobody's name is on a row, the row does not exist by month 2.
- **Is the compute already provisioned?** Ballpark GPU-hours; if it needs procurement, section 5 tracks it.

## Section 5 — Cost, timeline, and staffing

The section is a table:

| Phase | Weeks | Owner | Cost (GPU / API) | Blocking on |
|---|---|---|---|---|
| Base-model rubric run (chapter 2, exercise 01) | 1 | fine-tuning eng | small | — |
| Dataset acquisition + card | 2 | data team | labeling / API | vendor contract signed |
| Eval-set construction (mod-108 ex 02) | 1 (parallel) | fine-tuning eng | small | — |
| SFT run + ablation | 1 | fine-tuning eng | ~$X GPU | data ready |
| DPO run (if needed) | 1 | fine-tuning eng | ~$Y GPU | SFT done |
| Safety review (mod-106 ch 6) | 1 (parallel) | safety team | small | SFT done |
| Release gate + review meeting (this module, ch 5–6) | 1 | fine-tuning eng + reviewers | small | above |
| Buffer + rollout | 1 | platform team | inference | above |
| **Total** | **~7 weeks** | | | |

Timeline estimates are always wrong. What matters is that the timeline exists, the owners exist, and the dependencies are named. When the sponsor pushes for a two-week timeline, this table is what makes the impossibility visible.

**Two costs intake often forgets:**

- **The eval cost.** Judge API tokens for LLM-as-judge (mod-108 chapter 4), red-team dataset construction (mod-106 chapter 3). Not zero.
- **The rollback cost.** If the model ships and regresses in production, the rollback path costs GPU-hours and a re-run of the previous release gate. Budget it.

## Section 6 — Risk register and open questions

A short list — 5 to 15 rows — of the things that could kill the project, sorted by severity. Each row has an owner and a mitigation.

| Risk | Severity | Owner | Mitigation |
|---|---|---|---|
| Base-license reconsideration by legal | high | legal | Chapter 2 scorecard includes license posture; keep Apache-2 backup on the shortlist. |
| Dataset shortfall (fewer than X billing tickets available) | high | data team | Fallback to synthetic augmentation (mod-102 ex 04). |
| Catastrophic forgetting on capability guardrails | med | fine-tuning eng | Capability-preserving mix at 30% of SFT rows (mod-103 ch 5). |
| Safety-metric regression | med | safety team | Safety adapter trained separately (mod-106 ch 3); gate blocks release. |
| Judge drift on the LLM-as-judge eval | low | fine-tuning eng | Anchor set + human-agreement recheck (mod-108 ch 4). |
| Vendor availability (managed backend) | low | platform team | Self-hosted fallback base is on the shortlist. |

The register is not a promise the risks won't happen. It is a promise that the team already knows what to do when they do.

## The intake meeting

The proposal is written before the intake meeting, not during it. The meeting is 45–60 minutes with the sponsor, one peer fine-tuning engineer, the safety lead, and the platform lead. Its purpose is exactly three outcomes:

1. **Approve, revise, or reject the problem and metric (sections 1–2).** If those aren't sharp, everything else is speculative.
2. **Reserve the resources named in section 5.** No handshake; a written commitment on GPU capacity, labeling budget, and reviewer time.
3. **Assign owners to every row of the risk register (section 6).** Every risk without an owner defaults to the fine-tuning lead and clogs their calendar.

If any of the three do not happen, the meeting failed. Re-scope and re-schedule; do not start the SFT run to keep momentum. Momentum is what turns underspecified projects into month-long dead ends.

## A worked example — the support-assistant proposal

To make the shape concrete, here is the two-page version for a plausible internal project.

```markdown
# Fine-tuning proposal — Acme Support Assistant v3
**Owner:** L. Ortega (fine-tuning)
**Sponsor:** M. Chen (Support Product)
**Reviewers:** J. Park (peer FT eng), S. Idris (Safety), N. Rao (Platform)
**Status:** draft
**Version:** v0.2
**Last updated:** 2026-08-01

## 1. Problem and product goal
Support agents escalate 12% of billing-category tickets to human reps (baseline
30-day rolling from 2026-06). Product target: reduce escalation to ≤ 8% without
raising customer-satisfaction complaint rate on the resolved tickets.

## 2. Success metric and eval plan
- **Primary:** escalation rate on the billing category, measured on a held-out
  set of 500 billing tickets (labeled 2026-07 by ops), threshold ≤ 8% with
  bootstrap 95% CI.
- **Product-quality secondary:** rubric-judged helpfulness on the same held-out
  set, judge = gpt-4o (seed 1234), no regression > 0.15 vs. v2.4 baseline.
- **Capability guardrails:** MMLU / GSM8K / IFEval no regression > 2 pp; MT-Bench
  no regression > 0.15.
- **Safety:** refusal-rate ≥ 0.95, harmful-completion ≤ 0.02 (unchanged from v2.4
  gate).
Cadence: fast lanes (MMLU, GSM8K, IFEval, held-out billing) per checkpoint;
full gate on release candidates.

## 3. Base-model shortlist
| Base | License | Params | Notes |
|---|---|---|---|
| Llama-3.1-8B-Instruct | Llama 3 Community | 8B | Current v2.4 base. Zero switching cost. |
| Qwen2.5-7B-Instruct | Apache 2.0 | 7B | Backup if licensing rethink. |
| gpt-4.1-mini (managed) | proprietary | n/a | Managed path if internal GPU capacity is stalled Q3. |

Base pick deferred to rubric run (this module, ex 01) — planned week 1.

## 4. Dataset budget and objective stack
- SFT: 15k billing-specific + 20k capability-preserving mix (Tulu-3 sample).
- DPO: 3k preference pairs from internal escalation labelers.
- Safety: reuse v2.4 safety adapter; no new training rows.
Objective stack: SFT (task + mix) → LoRA → DPO on the LoRA → merge for serving.
Ablation: skip DPO if SFT hits the 8% target with margin.

## 5. Cost, timeline, and staffing
See section-5 table (see intake attachment). 7 weeks end-to-end, ~$8k GPU +
~$1.5k judge API + labeling budget owned by Support Product.

## 6. Risk register and open questions
See section-6 table (see intake attachment). Top three risks: labeling velocity
(3k pairs in 4 weeks?), capability-guardrail regression on MT-Bench, and
platform-layer readiness for the LoRA promotion at week 6.
```

Two pages. Enough to defend at the intake meeting. Not enough to hide fuzz.

## The four failure modes of intake, and the question that kills each

- **Problem drift.** "We want to improve the assistant" — section 1 is a paragraph, but nothing in it is measurable. **Killer question:** *what is the number today, and what number are we shipping?* If neither exists, the problem is not scoped.
- **Metric vagueness.** "It should be better." — section 2 lists no dataset, no threshold, no cadence. **Killer question:** *show me the JSONL that answers whether this shipped.* No JSONL means no metric.
- **Base locked in before data.** The proposal opens with "fine-tune Llama-3.1-70B on this." — section 3 has one row and section 4 was skipped. **Killer question:** *what would change if we discovered next week that the total data budget is 5k rows, not 50k?* If the answer is "nothing, we still fine-tune 70B," the base was not picked from the data.
- **Cost unmodeled.** Section 5 says "a few weeks, some GPUs." — the eval, safety review, and rollback costs are all missing. **Killer question:** *if we spend twice the estimated budget and hit the metric, is the project still net-positive?* If nobody's done the math, the project can be canceled at the halfway mark for reasons unrelated to model quality.

Each of these is a sentence at intake. It is a quarter at the halfway-review.

## What the proposal is not

- **The proposal is not the design doc.** The base-selection rubric (chapter 2), the dataset-budget argument (chapter 3), and the release-gate manifest (chapter 5) are documents the proposal *points at*. Trying to fold them in makes the proposal 20 pages and unreviewable. Two pages, with hyperlinks.
- **The proposal is not the ADR.** The ADR (chapter 4) covers the *fine-tune-vs-alternative* decision. The proposal assumes that decision is already made in favor of fine-tuning; if it is not, chapter 4 comes first.
- **The proposal is not the model card.** The model card (chapter 6) is a post-hoc artifact — it describes the model that shipped. The proposal describes the model that will be shipped, before it exists.

The four documents — ADR → proposal → design (base rubric + dataset budget + release gate) → model card — are the shape of a fine-tuning project from initial ask to governance sign-off. This chapter owns the second.

## Summary

- The intake step has the highest leverage on outcome and the lowest cost to redo. A fuzzy intake is a doomed project; a sharp intake is half the fight.
- The proposal is a two-page Markdown document with six sections: problem, success metric, base-model shortlist, dataset budget + objective stack, cost / timeline / staffing, risk register.
- Section 2 is the load-bearing section. Every metric needs a dataset, a threshold, and a cadence — using mod-108's vocabulary — with a primary product metric, capability-regression guardrails, and a safety floor.
- The base is chosen from the data, not before it. The proposal names a shortlist; the base-selection rubric (chapter 2) picks from it once the data budget (section 4) is known.
- The dataset budget names owners for every row and every ablation. Rows without owners do not exist by month 2.
- The risk register has 5–15 entries, each with an owner and a mitigation. Owners without names default to the fine-tuning lead and clog their calendar.
- The intake meeting has three outcomes: approve/revise/reject the problem and metric, reserve the resources, assign owners to every risk. If any is missing, the meeting failed — re-scope and re-schedule.
- The four failure modes — problem drift, metric vagueness, base-locked-in-before-data, cost-unmodeled — each have a killer question that surfaces them at intake instead of at the halfway-review.
- The proposal is the second document in the four-document arc (ADR → proposal → design → model card). It points at the others; it does not swallow them.
