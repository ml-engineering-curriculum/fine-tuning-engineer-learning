# Dataset Budget Allocation Across the Objective Stack

**Estimated read:** 40 minutes.
**You should be able to, by the end:** partition a fixed dataset / compute budget across SFT, preference optimization (DPO / KTO / ORPO / SimPO), classical RLHF, and safety tuning; defend the partition against two alternatives with a written trade-off; name three signals from the eval that would revise the allocation; and recognize the objective-stack patterns that fit specific product shapes.

## The budget is finite; the stack is not

Every fine-tuning project has a fixed budget — some combination of dollars, GPU-hours, labeling / preference-data cost, engineer weeks, and eval budget. Every objective in the post-training family (mod-101 chapter 4 laid out the map) is a way to spend that budget. This chapter is how you spend it — not which technique is "best," because the answer varies with product, but which allocation is *defensible* against the constraints from your proposal (chapter 1).

The allocation decision has three parts:

1. **Which objectives to run at all** — the *shape* of the stack.
2. **How many rows / how much compute per objective** — the *size* of each slice.
3. **In what order and with what ablation plan** — the *sequence*, and what evidence would change the plan.

Get any of the three wrong and the fine-tune spends the budget without moving the metric. This chapter walks the trade-offs and gives you the templates for the write-up.

## The stack: what each objective buys you

Very fast recap of what mods 103, 105, 106 taught, framed as *what you spend the budget on*:

- **SFT (mod-103).** Imitation of a curated distribution. Buys you *behavior in the shape of your data* — task format, tone, structured outputs, chat template adherence, refusal patterns, tool-use scaffolds. The cheapest per-metric-point objective when the model has capacity but doesn't know the shape of the ask. Fails when the model can already imitate the shape but ranks bad outputs above good ones.
- **DPO / IPO / ORPO / SimPO / KTO (mod-105).** Preference optimization. Buys you *rank* — pushing preferred completions above dispreferred ones. Cheaper than PPO-RLHF and typically sufficient for chat quality, helpfulness, and specific style / length / verbosity behaviors. Fails when the preference signal is inconsistent (chapter 3 of mod-105) or when the base is too far from the preferred distribution to be pushed there by re-ranking.
- **Classical PPO-RLHF (mod-105 ch4).** Reward-model-driven RL. Buys you *optimization against an explicit reward* — reasoning capability, tool-use success, custom multi-criteria rewards. Expensive per-metric-point; still occasionally the right answer for tasks that cannot be encoded as pairwise preferences (multi-turn tool use with sparse success signals, code generation with unit-test-based rewards, reasoning with process rewards, Reinforcement Fine-Tuning-style workflows on o-series). Fails when the reward is hackable.
- **GRPO and reasoning-focused variants (mod-105 ch6).** Group-relative preference. Buys you *reasoning-style outputs* (chain of thought, multi-step arithmetic, code with intermediate steps) at meaningful cost. Fits a narrow but growing class of products.
- **Safety tuning (mod-106).** Refusal training, constitutional AI, red-team-driven DPO on safety pairs. Buys you *the safety floor* (chapter 6 of mod-108, chapter 5 of mod-106). Not a helpfulness tool; a hard-gate tool.
- **Continued Pretraining (CPT).** Almost always the wrong answer for a post-training project, but worth naming: buys you *domain-shift in the base's knowledge*. Fits when the base has never seen your domain corpus (a new programming language, a proprietary vocabulary of size ≫ vocab, a language the base tokenizes poorly). Almost never the first move.

Every project's stack is a subset of these, arranged in sequence. The default stacks below cover ~80% of projects; the rest are variations on them.

## The four default stacks

Recognize the shape first; then size the slices.

### Stack A — SFT-only (baseline)

For **format-shaping, refusal-training-on-a-small-scale, or a base that already imitates well.** One SFT run; safety adapter added if the safety floor is not met by base + system prompt.

```
[base]  ->  SFT  ->  [checkpoint]  ->  serve
                        \_ optional: safety adapter merged on top
```

- **When it fits:** tool-call format standardization; JSON-schema compliance; adopting a specific chat template; teaching a canned refusal pattern; teaching a domain vocabulary that the base tokenizes reasonably.
- **When it fails:** when the metric is *quality* (helpfulness, preference win rate against a reference) rather than *behavior*.

### Stack B — SFT then DPO (the modern default)

For **chat-quality improvements, helpfulness, style / verbosity control.** SFT establishes the format and covers the capability gap; DPO refines rank.

```
[base]  ->  SFT  ->  [SFT ckpt]  ->  DPO  ->  [aligned ckpt]  ->  serve
                                        \_ safety adapter on top
```

- **When it fits:** most chat products; agents; domain-specific assistants.
- **When it fails:** when preference data is too noisy or too small to move the metric (chapter 3 of mod-105).

### Stack C — Joint ORPO (or KTO) — single-stage

For **budget-constrained projects with modest preference data.** ORPO combines SFT and preference-learning in one loss; KTO handles unpaired signals.

```
[base]  ->  ORPO or KTO  ->  [aligned ckpt]  ->  serve
                                 \_ safety adapter on top
```

- **When it fits:** limited compute (one training run instead of two); preference data available but small (a few hundred to a few thousand pairs); operational simplicity a priority.
- **When it fails:** when the SFT-only floor is far below the preferred distribution and the model needs a "learn the format first" phase before ranking makes sense.

### Stack D — SFT then PPO-RLHF (or RFT on o-series)

For **explicit-reward-driven optimization** — code generation with unit tests, tool-use with success signals, reasoning with process rewards, or vendor-specific RFT-style workflows.

```
[base]  ->  SFT  ->  [SFT ckpt]  ->  reward model  ->  PPO  ->  [RL ckpt]  ->  serve
```

- **When it fits:** the primary metric is a programmable reward (pass@1 on unit tests, tool-call success rate); a reference-free preference signal is not enough; the team has RL experience.
- **When it fails:** when the reward is gameable (mod-105 ch7); when the team does not have PPO experience — the operational surface is meaningfully larger than DPO's.

**Selection heuristic in one paragraph.** Start with Stack B unless you have a specific reason to pick another. Fall back to Stack A when the metric is pure behavior and rank is not the problem. Fall back to Stack C when the compute or the data budget is small. Move to Stack D when you have a real reward signal and the team is ready to operate the RL loop.

## Sizing the slices

Once the stack shape is chosen, size the slices. The next three sub-sections give the rules of thumb; the section after that shows two worked examples.

### SFT slice — the row-count rules

Two forces set the SFT row count:

- **Task-specific rows.** How many examples of the target behavior does the model need to see? For a *format* (JSON schema compliance, chat template adherence, refusal shape), 1 000–5 000 well-curated rows is often enough; the pattern is regular and repetitive. For a *skill* (question answering in a specific domain, tool-call sequencing across dozens of tools), 10 000–50 000 rows starts to be meaningful; the space is larger. For *the whole surface* (a general-purpose instruction-following model — think Tulu-3 / Zephyr / Alpaca-style projects), 100 000–1 000 000+ rows is the norm.
- **Capability-preserving mix.** How much of the SFT should be *not* task-specific? A pure-task SFT run risks catastrophic forgetting on general capability (mod-103 ch5) — the model gets better at billing tickets and worse at everything else. A common ratio is **30–60% capability-preserving mix, 40–70% task-specific**, at total row counts ≥ 20 000. Below 20 000 total, mix ratios matter less than raw quality — the smaller run is unlikely to move general capability either way.

Two anti-patterns to avoid:

- **Growing the task slice past its quality curve.** After a point, adding more medium-quality task-specific rows *hurts* — the model over-fits noise. Chapter 1 of mod-108's diagnostic (per-item loss on the held-out set) tells you when you've crossed it.
- **Padding the mix slice with unrelated data to hit a number.** Capability-preserving mix should be *representative of the base's original chat surface*, not "whatever public dataset was lying around." Tulu-3, no_robots, OpenAssistant slices, or similar are the usual choices.

### Preference slice (DPO / ORPO / KTO / SimPO) — the row-count rules

Preference data is expensive per row (labeling cost + judge cost + review time), so the sizing question is different from SFT.

- **DPO on a Stack B chat product** typically wants **1 000–20 000 pairs**. Below 1 000, the signal-to-noise is too low for β to be tuneable; above 20 000, marginal returns fall unless the preference data comes from a very high-quality source (calibrated human raters, or a strong judge model with careful anchor-set alignment — mod-108 ch4).
- **KTO** can absorb unpaired data — a single-sided "this output was accepted / this output was rejected" — so its row-count budget is comparable but the per-row cost is often lower.
- **ORPO** combines SFT and preference in one loss; the row count is roughly the SFT + preference combined, with preference rows carrying the pairwise labels.
- **Multi-round preference collection.** For chat products, generate preferences on outputs *from the SFT model*, not from arbitrary sources. mod-105 chapter 2 walks the pipeline: SFT → sample outputs → judge → preference pairs → DPO.

Two knobs that dominate the preference slice's cost-per-metric-point:

- **Judge quality.** A strong judge with a careful anchor set (mod-108 ch4) turns cheap synthetic preference data into a real signal. A weak judge on the same data is spending money on noise.
- **Length control.** Length bias is the single most common failure mode of preference-optimized models (mod-105 ch7). Length-controlled preference data or a length-controlled evaluator (AlpacaEval-LC, or the correction in mod-108 ch6) is what keeps the DPO from optimizing for verbosity.

### Safety slice — floor first, then adapter

Safety is not scaled by row count; it is scaled by *coverage of the harm taxonomy* (mod-106 ch1) and *the strength of the red-team pipeline* (mod-106 ch3). Typical shapes:

- **Reuse an established safety adapter** from a prior release, if the base is unchanged and the harm taxonomy has not shifted. Zero new training rows; the adapter is added on top of the SFT/DPO output at serve time (mod-104 ch3) or merged before quantization (mod-109).
- **A dedicated safety fine-tune** for a new base or after a taxonomy revision — 1 000–5 000 safety-pair rows across the taxonomy categories, with an emphasis on the categories the base scores worst on at zero-shot on the red-team eval. mod-106 chapter 5's report card is the target.
- **Safety-preference DPO** interleaved into the main DPO run, if red-team-derived preferences are available and the team decides not to isolate the safety adapter.

The safety floor is a *gate*, not a *maximize*. Chapter 6 of mod-108 makes this explicit: "absolute at least 0.95 refusal rate" is the shape, not "beat the previous refusal rate by 3 pp."

### Compute slice

For each objective, ballpark the compute:

- **SFT full-parameter on 7B–8B:** roughly `1e-4 × params × total_tokens` GPU-hours on A100/H100-class hardware; a 30k-row run at 2k tokens/row ≈ 6e10 tokens × 8B ≈ ~600 GPU-hours BF16 (rough; back off with mod-103 numbers on your specific configuration).
- **LoRA / QLoRA on 7B–8B:** an order of magnitude cheaper; typical runs finish in single-digit GPU-hours (mod-104 exercises).
- **DPO on top of LoRA:** slightly more expensive than the LoRA SFT run because two forward passes are needed per step (policy and reference), but still cheap in absolute terms.
- **PPO-RLHF:** 5–10× the DPO cost due to rollouts and the extra models in the loop.
- **Safety adapter fine-tune:** cheap, small dataset, small LoRA.

Compute cost is rarely the dominant slice in a modern PEFT-first stack. Data / preference / labeling cost usually is. Chapter 5 of mod-110 has the managed-side arithmetic; for self-hosted, work from mod-107 chapter 7 and mod-109 chapter 7.

## The sequence and the ablation plan

Order matters, especially between SFT and preference / RL stages. Two rules:

- **SFT before preference / RL.** The preference objectives need a policy that already produces reasonable outputs; running DPO or PPO on a base that has not been format-adapted converges slowly or wildly. Exception: joint ORPO / KTO combines them into one step.
- **Safety last, or in parallel.** Safety tuning should not be *undone* by later helpfulness tuning. Either train a safety adapter separately and add it at serve time (mod-104 ch3), or make the safety data part of the final DPO / RL round with enough weight to hold.

**The ablation plan** is a small table stapled to the budget:

| Signal from eval | What we do |
|---|---|
| SFT-alone hits the target metric with margin | Skip DPO. Ship SFT + safety. |
| SFT-alone hits target but format regressions on capability guardrails | Rebalance the capability mix upward; re-run SFT. |
| SFT-alone misses target by ≥ 10 pp | Rethink base (chapter 2) or dataset budget; do not "push through" with more preference data. |
| SFT + DPO hits target but length-controlled win rate is < 0.5 | Length-control the preference set (mod-105 ch7). |
| Safety metric moves in the wrong direction after DPO | Isolate the safety adapter — do not co-train the safety data with the DPO. |
| Judge-vs-human agreement (mod-108 ch4) drops after preference training | Sanity-check the anchor set; consider a stronger judge for the final release-gate run. |

The ablation plan is the point where "we allocated X to SFT and Y to DPO" becomes "we know what to change if it doesn't work." Every proposal without one is a coin flip.

## Two worked examples

### Worked example 1 — Acme Support Assistant v3 (continuing chapter 1's project)

- **Product goal.** Reduce billing-ticket escalation rate from 12% to ≤ 8%.
- **Stack shape.** B (SFT then DPO), with an existing safety adapter reused.
- **Total budget.** ~57k rows total; ~$8k GPU; ~$1.5k judge API.

Allocation:

| Objective | Rows | Compute | Data cost | Rationale |
|---|---|---|---|---|
| SFT — task-specific (billing) | 15 000 | LoRA on 8B, ~4 GPU-hours | labeled by ops, ~$3k | Enough to teach the ticket-response format and billing-specific reasoning; below quality-curve saturation for this domain based on v2.4 numbers. |
| SFT — capability-preserving mix (Tulu-3 sample) | 20 000 | included above | 0 | ~57% of SFT run is mix; hedges the capability guardrails from mod-108 gate. |
| DPO on the SFT LoRA | 3 000 pairs | LoRA-based DPO on 8B, ~3 GPU-hours | ~$1.5k labeler + judge cost | Preference pairs generated from SFT model outputs on 3k held-out prompts, labeled by internal support ops. |
| Safety | 0 new rows | reuse v2.4 safety adapter | 0 | Base unchanged (Qwen if migrating, per chapter 2 override), harm taxonomy unchanged. If the report card regresses, plan B is to fine-tune the safety adapter on Qwen with existing v2.4 safety pairs (~1 500 pairs). |
| Total | ~38 000 SFT rows + 3 000 pairs | ~8 GPU-hours | ~$4.5k data + ~$3.5k GPU / judge | Fits within proposal §5. |

**Ablation plan:**

- If SFT alone hits ≤ 8% billing-escalation with margin, skip DPO. Faster path to production; save budget for the next iteration.
- If SFT hits 9–10% but capability guardrails regress, add capability mix rows (drop to 40% task, 60% mix) and re-run.
- If SFT + DPO overshoots on length (mean response length grows > 15%), retrain DPO with length-controlled pair filtering.
- If safety adapter re-application regresses report card, plan-B safety fine-tune adds one week to the timeline.

**Two defended-against alternatives:**

- *Stack A (SFT-only)*: rejected because v2.4 already shipped SFT-only and hit 12%; the marginal DPO run on top is the plausible mechanism to close the remaining gap without changing the base.
- *Stack C (joint ORPO)*: considered; rejected because we have distinct SFT and preference collection pipelines already, and running them jointly requires re-tooling; the small operational simplicity does not offset the switching cost.

### Worked example 2 — a code-refactor assistant

- **Product goal.** Raise pass@1 on the internal SQL-refactor test suite from 42% to ≥ 65%.
- **Stack shape.** D (SFT then PPO-RLHF with a unit-test-based reward).
- **Total budget.** ~30k SFT rows; ~20k rollout samples; ~$20k compute.

Allocation:

| Objective | Rows | Compute | Data cost | Rationale |
|---|---|---|---|---|
| SFT — task-specific (SQL-refactor traces) | 20 000 | LoRA on 8B, ~6 GPU-hours | ~$5k synthetic + human polish | Establishes the format; teaches common refactor patterns; brings zero-shot from 42% to ~55%. |
| SFT — capability mix (code + reasoning slice of Tulu-3) | 10 000 | included above | 0 | Guards against non-SQL code regression. |
| Reward model (BT on preference pairs from SQL correctness + human style rubric) | 4 000 pairs | 4 GPU-hours | ~$2k judge + human | Combines unit-test correctness (automatic) with a style rubric (LLM-judged). |
| PPO-RLHF | 20 000 rollout samples | 60 GPU-hours | ~$3k judge for reward | Reward is `unit_test_pass_rate + 0.2 × style_score - 0.5 × length_penalty`; KL clip and reference to SFT model. |
| Safety | reuse existing generic safety adapter | 0 | 0 | Code assistant does not hit new safety categories vs. base; report card unchanged. |
| Total | ~30k SFT rows + 4k RM pairs + 20k rollouts | ~70 GPU-hours | ~$10k data + ~$10k GPU | Fits within budget. |

**Ablation plan:**

- If SFT hits ≥ 65% alone, skip PPO. Deliverable is finished at half the compute.
- If PPO reward hacking is detected (reward climbs but held-out pass@1 does not), reduce KL β, add process-reward from a stronger judge, or fall back to DPO on the RM-generated preferences.
- If length grows unacceptably (mod-105 ch7), tighten the length penalty in the reward or add length-controlled preference filtering.

**Defended-against alternatives:**

- *Stack B (SFT then DPO)*: rejected because we have a real programmable reward (unit tests). Wasting that signal on pairwise preferences alone gives up the strongest per-metric-point lever we have.
- *Stack A (SFT-only)*: considered as a checkpoint milestone; not sufficient for the 65% target based on the SFT saturation curve observed in prior code-refactor projects on similar bases.

## Where the budget goes wrong

Six common misallocations, each with the diagnostic that catches it and the fix:

- **Too much task-specific SFT, too little capability mix.** Diagnostic: task metric improves; MMLU / MT-Bench / IFEval regress > 2 pp. Fix: raise mix to 40–60% of total SFT rows.
- **Too little preference data.** Diagnostic: DPO run's implicit reward curve (chapter 5 of mod-105) barely moves; length or verbosity does not shift; win-rate CI overlaps SFT baseline. Fix: collect more pairs; consider joint ORPO instead if compute is the constraint.
- **Preference data noisy.** Diagnostic: judge-vs-human κ / ρ on the anchor set (mod-108 ch4) is < 0.5; DPO run diverges or worsens the eval. Fix: re-collect with a stronger judge or a calibrated human process; do not train on noise.
- **PPO where DPO would do.** Diagnostic: reward is a hackable proxy; PPO improves the reward but the held-out eval is flat or worse. Fix: fall back to DPO on the RM-generated preferences and check whether the DPO gets you where the PPO was supposed to.
- **Safety left to serving.** Diagnostic: refusal-rate on the harmful red-team set falls below the floor after helpfulness tuning; runtime guardrails compensate but do not restore. Fix: isolate the safety adapter (mod-106 ch3) and add on top; or re-run the DPO with safety pairs interleaved.
- **Compute-heavy stack, thin data.** Diagnostic: training loss saturates within the first 20% of steps; per-item eval variance is high. Fix: the bottleneck is data quality, not compute; drop epoch count or grow the dataset before spending more GPU-hours.

## The write-up

The dataset-budget allocation is a document in Git alongside the proposal and the base scorecard. Template:

```markdown
# Dataset budget allocation — <project> v<n>
**Owner:** <fine-tuning>
**Date:** YYYY-MM-DD
**Companion documents:** proposal v<n>.md, base_scorecard v<n>.md

## Stack shape
Choose one: A (SFT-only) | B (SFT + DPO) | C (ORPO / KTO joint) | D (SFT + PPO-RLHF).
Rationale in one paragraph.

## Allocation table
| Objective | Rows / samples | Compute | Data cost | Rationale |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |
| **Total** | | | | |

## Ablation plan
| Signal from eval | Action |
|---|---|
| ... | ... |

## Defended alternatives
- Alternative X: rejected because ...
- Alternative Y: rejected because ...

## Escape hatches
Which reallocation would be pre-approved if a specific signal fires (e.g.,
"skip DPO if SFT alone hits target with margin"). These are pre-authorized
by this document — no separate approval needed.

## Open questions
- ...
```

The document is one page. It is the artifact mod-111 exercise 02 asks you to produce. Every ablation you actually run gets appended as a section — the file is a living record.

## Summary

- The dataset / compute budget is finite. The objective stack is a way to spend it; the allocation problem is picking the shape, sizing the slices, and pinning the ablation plan.
- The four default stacks are **A (SFT-only)**, **B (SFT + DPO — the modern default)**, **C (ORPO / KTO joint — budget-constrained)**, and **D (SFT + PPO-RLHF — programmable-reward-driven)**. Start with B unless there's a specific reason to switch.
- SFT sizing: 1 000–5 000 rows for format; 10 000–50 000 for skills; 100 000+ for whole-surface. Keep 30–60% capability-preserving mix to prevent catastrophic forgetting.
- Preference sizing: 1 000–20 000 pairs for DPO; the per-row quality (judge, anchor-set alignment, length control) dominates the per-metric-point return.
- Safety is a *floor* on the release gate (mod-106, mod-108 ch6), not a maximize. Reuse the safety adapter across releases when the base and taxonomy are unchanged; retrain only when they shift.
- Order the stack: SFT before preference / RL, safety last or as a separate adapter. Joint ORPO / KTO merge SFT and preference; use when compute or ops simplicity dominates.
- The ablation plan turns the allocation from a guess into a decision loop: which eval signal triggers which reallocation, pre-authorized.
- The write-up is one page: stack shape, allocation table, ablation plan, defended alternatives, escape hatches. Lives in Git with the proposal and the base scorecard; grows as the project runs.
