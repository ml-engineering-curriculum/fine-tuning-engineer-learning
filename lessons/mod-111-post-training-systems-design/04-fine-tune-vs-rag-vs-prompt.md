# The Build-vs-Buy and Fine-Tune-vs-Alternative Decision

**Estimated read:** 35 minutes.
**You should be able to, by the end:** walk a product ask through a decision tree that separates prompt engineering, RAG, tool wiring, buying an endpoint, and fine-tuning; write an ADR that defends the pick against the other four; name the concrete signals that would move the decision; and recognize the three product patterns where fine-tuning is (almost always) the wrong first move.

## Fine-tuning is not the default

Every previous module in this track assumed a fine-tune was the right answer. This chapter is the sanity check: *is it?* A fine-tune costs money, GPU-hours, engineer weeks, a release gate, a governance sign-off, ongoing maintenance, and a rollback plan. Every one of those costs is avoided when a shorter path fits the ask.

The failure mode this chapter attacks is the one that fine-tuning engineers are structurally most likely to commit: we picked fine-tuning because we are fine-tuning engineers. The rubric here forces the pick to be defended against alternatives even when the fine-tune is the eventual answer.

**Rule of thumb, before we start:** if you can get to the target metric without touching a weight, do that. Fine-tune when you can't.

## The five options

Line them up before diving in. Each is a *class* of solution; multiple can be combined.

1. **Prompt engineering.** Change what the model reads at inference time — system prompt, few-shot examples, output-format instructions, chain-of-thought scaffolds. No weight change, no infrastructure change beyond the prompt store. Cheapest, fastest, most-often-underestimated.
2. **RAG (retrieval-augmented generation).** Fetch relevant context at inference time from a corpus (documentation, tickets, database, wiki) and stuff it into the prompt. Buys the model access to *knowledge* it does not have. No weight change; infrastructure change is a retrieval index.
3. **Tool wiring / function calling / agent scaffolding.** Give the model programmatic access to specific capabilities (search, code execution, structured queries, external APIs). Buys the model *actions* it does not have. No weight change; infrastructure change is the tool set and the orchestration layer.
4. **Buy an endpoint (managed model, larger tier).** Move to a larger or better-suited off-the-shelf model. Buys *raw capability*. Usually a per-token cost increase; no engineering work beyond the swap.
5. **Fine-tune.** Everything mods 101–110 taught. Buys *behavior in the shape of your data* or *ranked preferences*.

The order 1 → 5 is deliberate: it is (approximately) the order of increasing cost and coupling. Do them in that order unless you have a reason not to.

## The decision tree

```
                       PRODUCT ASK
                             |
                             v
   +----- Have you defined the metric? -----+
   | NO -> back to proposal (chapter 1)     |
   +-----------------------------------------+
                             |
                             v
        Is the metric currently ≥ 95% of target
        on the largest / best off-the-shelf model
        with a plain prompt?
                 |
   YES -> ship prompt engineering; iterate.
                 |
                 v (NO)
        Is the failure a KNOWLEDGE gap?
        (model does not know the facts, docs, entities)
                 |
   YES -> RAG.  (proceed even if fine-tune also considered later)
                 |
                 v (NO)
        Is the failure an ACTION gap?
        (model could reason correctly if it could call X)
                 |
   YES -> tool wiring / function calling.
                 |
                 v (NO)
        Is the failure a CAPABILITY gap?
        (base model is fundamentally too small or too weak
         to reason about the task, even given context and tools)
                 |
   YES -> buy a bigger model tier OR change base (chapter 2)
                 |
                 v (NO)
        Is the failure a BEHAVIOR gap?
        (right knowledge, right tools, right size — wrong style,
         wrong format, wrong ranking of alternatives, wrong refusal
         pattern, wrong verbosity, wrong tone)
                 |
   YES -> fine-tune.  Return to chapter 1 and write the proposal.
```

Each branch has a diagnostic (below). Each branch is defended in the ADR (below). Even when the eventual answer is "fine-tune," the tree forces you to have argued against the alternatives.

## The four gaps

The tree turns on a diagnosis: which *gap* is causing the model to miss the metric? Four categories cover almost every case. The vocabulary matters — writing "knowledge gap" or "behavior gap" in the ADR anchors reviewers on the right alternative.

### Knowledge gap → RAG

- **Signal.** The model produces confidently wrong facts about entities, documents, or events it could not have seen in pretraining. Errors are *lookups*, not *reasoning*.
- **Test.** Give the model the correct fact in the prompt. Does the answer become correct? If yes, the failure is knowledge, and RAG closes it.
- **Anti-pattern:** fine-tuning to "teach the model the docs." Fine-tuning on your knowledge base is a slow, lossy, expensive way to do what RAG does natively — and worse, the fine-tune has to be redone every time the knowledge base changes. RAG is a live query; fine-tuning is a stale snapshot.

### Action gap → tool wiring

- **Signal.** The model correctly reasons about *what to do* but cannot actually do it (get the current temperature, run the SQL, look up the ticket, call the payment API).
- **Test.** In a few-shot prompt, tell the model to *simulate* calling a tool by outputting the arguments and the pretend result. Does the reasoning become correct? If yes, the failure is action, and function calling / tool use closes it.
- **Anti-pattern:** fine-tuning "correct SQL for our schema" when the base can already write correct SQL given the schema — the actual failure is that nobody wired up schema retrieval. Fine-tuning to memorize schemas is a losing arms race.

### Capability gap → bigger model / different base

- **Signal.** Even given the knowledge and the tools, the model produces incoherent reasoning, misapplies logic, or fails on tasks that a stronger model does correctly. Errors are *fundamental*, not stylistic.
- **Test.** Run the same prompt (with the same RAG context, the same tools) on the largest available model in the family. Does the accuracy jump? If yes, the failure is capability, and the answer is a larger tier or a different base (chapter 2).
- **When to fine-tune anyway:** if you cannot afford to serve the larger model but can fine-tune a smaller one to close some of the gap. This is the *distillation* pattern — fine-tune the smaller model on outputs of the larger one (mod-102 ch4). Even so, RAG and tools usually come first; distillation is fine-tuning as a cost-reduction lever, not as a first-fix.

### Behavior gap → fine-tune

- **Signal.** The knowledge is present or fetchable, the tools work, the base is capable enough — the model produces answers that are almost right but consistently wrong in *shape*: chats that skip the required refusal, JSON that is malformed, responses that pick the wrong option among two that are both correct, tone that is off, style that is verbose, chain-of-thought that is too long or too short, refusals that are too eager or not eager enough, tool calls that are formatted for the wrong system.
- **Test.** Add a strong system prompt describing the desired behavior. Does the metric close? If yes, prompt engineering is enough. If no — the model *cannot follow the instruction reliably enough* even when told — the failure is behavior, and fine-tuning closes it.
- **Special case:** *format compliance* (JSON schema, structured outputs, chat template adherence). Even the best models routinely fail this at scale, and fine-tuning is the appropriate remedy at that scale. If the metric is "≥ 98% JSON-schema-valid on 100k daily calls," system-prompted format is not enough; fine-tune.

## The tests you actually run

Turn the tree into a small experiment protocol. Each test is a few hours and a small budget; the whole protocol runs in a couple of days.

- **Test 1: Baseline on the largest available off-the-shelf model with a plain prompt.** Measures the ceiling of "buy an endpoint." If this hits the metric, the answer is *buy*.
- **Test 2: Baseline on the target-tier off-the-shelf model with a plain prompt.** Measures the starting point for the tier your product is actually willing to serve.
- **Test 3: Same as Test 2 with a strong system prompt + few-shot exemplars.** Measures the ceiling of "prompt engineering." If this hits the metric, the answer is *prompt*.
- **Test 4: Same as Test 3 with retrieval context stuffed in.** Measures the ceiling of "RAG." Use a good retriever (mod outside this track, but a mainstream embedding-based retriever suffices for the diagnostic). If Test 4 closes a gap Test 3 did not, the failure is knowledge.
- **Test 5: Same as Test 4 with function-calling to your real tools.** Measures the ceiling of "tool wiring." If Test 5 closes a gap Test 4 did not, the failure is action.
- **Test 6 (only if 1–5 do not close): baseline fine-tune on a small SFT slice** (a few thousand rows) at the target tier. Measures whether fine-tuning is even the right direction. If Test 6 moves the needle, proceed to the full proposal (chapter 1). If Test 6 does not, fine-tuning is unlikely to be the answer — reconsider.

The protocol above is what you present in the ADR. It is why the ADR is defensible.

## The ADR — one page, five sections

The ADR (Architecture Decision Record) is the durable document. Reviewers should be able to reconstruct the reasoning from it a year from now, when the decision needs to be revisited because the frontier models moved.

```markdown
# ADR — Fine-tune vs. alternatives for <project>
**Owner:** <fine-tuning>
**Status:** proposed | accepted | superseded by <link>
**Date:** YYYY-MM-DD

## Context
One paragraph on the product ask, the metric, and the constraints (latency,
cost, regulatory, tenant, redistribution intent).

## Alternatives evaluated
| Option              | Test result                    | Cost / month | Notes |
|---------------------|-------------------------------|--------------|-------|
| Prompt engineering  | Test 3: 76% (target 90%)      | $X           | ...   |
| RAG                 | Test 4: 82%                    | $Y           | ...   |
| Tool wiring         | Test 5: 84%                    | $Z           | ...   |
| Buy a bigger tier   | Test 1: 91% at 8× the cost    | $W           | ...   |
| Fine-tune           | Test 6: 88% at target tier    | $V           | ...   |

## Decision
Chosen: <one of the above>. If fine-tuning: point to the proposal (chapter 1).

## Rationale
Why the chosen option beats the others on the criteria that matter for this
project. Cite Test numbers.

## Escape hatches
Concrete signals that would move the decision:
- If the frontier tier's price drops by 50%, revisit "buy" (Test 1's cost was
  the blocker).
- If the fine-tune's release-gate delta falls below +3 pp vs. RAG at the same
  tier, revisit whether the fine-tune is earning its keep.
- If the knowledge base grows to include entity types not covered by the
  training data, RAG becomes more attractive; revisit.

## Consequences
- Ongoing costs (registry, gate, model card, rollback runbook if fine-tuning).
- Ownership: who watches the escape-hatch signals and triggers the revisit.
- Sunset conditions: when the decision no longer applies.
```

The ADR is short. It refers out to the proposal, the base scorecard, the dataset-budget allocation, and the eval results. It is the *decision*, not the *design*.

## Three product patterns where fine-tuning is (almost) always wrong first

- **"Teach the model our documentation."** The docs change monthly; the fine-tune is stale the day it ships. **RAG.**
- **"Have the model reliably know our current inventory / customer list / active tickets."** Live data. **RAG + tools.**
- **"Give the model access to our internal APIs."** Wire the APIs; do not fine-tune on their behavior. **Tools + function calling.**

Every one of these has a fine-tune-shaped hole in it that a fine-tuning engineer will feel gravity toward. Fill the hole with the right pattern first; consider fine-tuning only if the pattern falls short *after* it is in place.

## Three product patterns where fine-tuning is (almost) always right eventually

- **Format compliance at scale.** JSON schema validity, chat template adherence, structured outputs at ≥ 98% on high-volume traffic. Prompt engineering caps out below that; fine-tuning gets there.
- **Style / tone / voice adaptation.** The brand voice, the character of a specific persona, the register of a specialist domain. System prompts get you 80% of the way; fine-tuning gets you to consistent.
- **Ranked-preference behavior.** Choosing among multiple correct answers ("which of these three reasonable responses is best?"). Fine-tuning via DPO / preference optimization is the right tool; prompt engineering cannot rank consistently at scale.

These three patterns are what mod-103 through mod-105 exist to serve. When the ADR concludes "fine-tune," it is almost always because the ask is one of these.

## The hybrid answer

The ADR often does not pick one option; it picks a *stack*. Common hybrids:

- **Fine-tune + RAG.** Fine-tune the base for tool-use, format, and citation style; feed retrieved docs at inference. Most production RAG-heavy chat products are this shape.
- **Fine-tune + tools.** Fine-tune the base to call your specific tool schemas reliably; keep the tool set live.
- **Prompt + RAG (no fine-tune).** The simplest chat products on top of a strong base model. The right first move whenever it hits the metric.
- **Fine-tune (small) + buy (large) with routing.** A small in-house fine-tuned model handles routine traffic; a bought frontier tier handles complex traffic (routed by a classifier). Common cost-management pattern for high-QPS products.

The hybrid is not a cop-out — it is often the right answer. What the ADR does is *name* the hybrid and *defend the pieces* against alternatives. "Fine-tune + RAG" is fine; "we do a bunch of stuff" is not.

## Managed fine-tuning enters the tree at the fine-tune branch

Once the tree lands on "fine-tune," the next decision is *managed or self-hosted*, and mod-110's chapter 6 owns it in full. In the ADR, cite mod-110's rubric — do not re-derive it here. This chapter's job ends at the "fine-tune" verdict; mod-110's ADR template starts where this one leaves off.

## When the decision is revisited

Revisit the ADR when any of:

- The frontier tier's price drops by ≥ 30%. "Buy" gets cheaper; the fine-tune's cost-per-metric-point argument weakens.
- A new base model with meaningfully higher capability at your tier ships. Chapter 2's scorecard re-runs; the "fine-tune" branch's baseline shifts.
- RAG or tool-wiring infrastructure that was not previously available becomes available. "Test 4 was 82% without our new retriever, which is now available" is a real reason to revisit.
- The metric moves — the product target changes, or the eval finds a systematic gap the ADR did not consider.
- Regulatory / vendor / license change flips one of the constraints in the Context section.

Version the ADR. `ADR-001-v1.md`, `ADR-001-v2.md`. Each version's rationale includes what changed. Chapter 5's registry stores the current-version pointer.

## Common failure modes of this decision

- **"We fine-tuned first because it was fastest to try."** No — writing a system prompt and running a benchmark is faster than any fine-tune. If a fine-tune is your fastest experiment, your prompt-engineering setup is broken.
- **"RAG was too complex to set up, so we fine-tuned."** RAG can be a single-file experiment with an in-memory FAISS index over your docs. If the RAG POC took longer than a day, the POC scope is wrong.
- **"The frontier model was more expensive per token, so we fine-tuned a smaller one."** Do the math. A fine-tuned smaller model has *ongoing engineering cost* (release gate, model card, rollback plan, retraining every N months) that a bought endpoint does not. At low volume, buying the frontier tier is often *cheaper total-cost-of-ownership*.
- **"Fine-tuning is what our team knows."** True but not a reason. If prompt / RAG / tools would fit the ask, invest in the team's skills for those patterns rather than defaulting to fine-tuning by inertia.
- **"We didn't measure the baseline."** The ADR cannot exist without the Test 1–5 numbers. If those numbers do not exist, run them; do not skip the ADR.

## Worked example — chat assistant for a legal team

- **Ask.** In-house legal assistant that answers questions about the company's contracts and playbook documents.
- **Metric.** Cited-answer accuracy on 200 held-out questions ≥ 90%; refusal on out-of-scope questions ≥ 95%.

Tests:

- Test 1 (frontier tier + plain prompt): 62% cited accuracy. Cannot cite docs it never saw.
- Test 3 (target tier + strong system prompt + few-shot exemplars): 58%. Similar failure — no access to the docs.
- Test 4 (Test 3 + RAG over the contract corpus): **92%**. Refusal rate 91%.
- Test 5 (Test 4 + tools for `lookup_contract_by_id`, `search_playbook`): 93%. Refusal rate 96%.
- Test 6 (fine-tune the target-tier model on 5k Q&A pairs from the corpus): 84%. Refusal rate 88%. And the fine-tune is stale the moment a new contract is signed.

**Verdict.** RAG + tools (hybrid of options 2 and 3). No fine-tune. The ADR notes that fine-tuning was tested and rejected — a signature of a serious evaluation is that the rejected option was tried, not just dismissed.

Now imagine a variation: same product, but the ask is "answer in the style of the general counsel." Suddenly the *behavior* gap is real (voice, register, defensiveness). Now the ADR probably reads "RAG + tools + fine-tune on 5k voice-labeled Q&A pairs to shape the response voice." The fine-tune is not fetching knowledge — RAG is; it is not calling APIs — tools are. It is doing the one thing only fine-tuning does well: shaping behavior.

## Summary

- Fine-tuning is not the default. Try prompt engineering, RAG, tool wiring, and buying a bigger tier — in that order — before writing a fine-tuning proposal.
- The decision tree turns on the *gap*: knowledge → RAG, action → tools, capability → bigger tier / different base, behavior → fine-tune.
- The tests are cheap: baseline the target and larger tiers plain-prompted; add prompt engineering, then RAG, then tools; run a small fine-tune only if the earlier tests did not close the gap.
- The ADR is one page, five sections: context, alternatives evaluated (with test numbers and cost), decision, rationale, escape hatches. Every alternative is defended; every escape hatch is a concrete signal that would revise the decision.
- Three product patterns where fine-tuning is almost always wrong first: teach-the-docs, live-data, and internal-APIs. Reach for RAG and tools instead.
- Three patterns where fine-tuning is almost always right eventually: format compliance at scale, style / voice / tone adaptation, ranked-preference behavior.
- Hybrids are the norm, not the exception. Fine-tune + RAG, fine-tune + tools, small-fine-tune + bought-large with routing — each is a valid answer as long as the pieces are named and defended.
- Once the tree lands on "fine-tune," mod-110 chapter 6 owns the *managed vs. self-hosted* sub-decision. This ADR ends at the fine-tune verdict.
- Revisit on price shifts, new bases, new infrastructure, metric drift, regulatory / vendor changes. Version the ADR; the current pointer lives in the registry (chapter 5).
