# LLM-as-Judge Protocols

**Estimated read:** 40 minutes.
**You should be able to, by the end:** design a single-answer rubric-graded eval and a pairwise-comparison eval in the style of MT-Bench, name the four known judge biases (position, verbosity, self-preference, style) and the concrete control for each, choose a judge model deliberately, compute inter-judge and judge-vs-human agreement metrics, and calibrate a judge against a small human anchor set so its scores are usable in a release gate.

## Why LLMs judge LLMs

Open-ended generation quality — is this summary faithful, is this response helpful, is this answer preferred over that one — is not measurable by exact match or regex. The two historically-used alternatives were BLEU-family metrics (fast, weakly correlated with quality) and human ratings (slow, expensive, hard to scale). Zheng et al. (2023) — the MT-Bench / Chatbot Arena paper — established that a strong LLM prompted with a clear rubric agrees with human ratings closely enough to substitute for a majority-vote human panel on many tasks, at a fraction of the cost.

The claim in that paper: GPT-4-class judges reach ~80% agreement with human majority vote on MT-Bench and Chatbot Arena, comparable to the ~81% agreement humans reach with each other. This has been reproduced many times and is the field-standard justification for LLM-as-judge — but it comes with real biases and controls that this chapter enumerates.

**What LLM-as-judge is good for:** cheap, scalable proxies for human quality ratings on open-ended generation tasks; comparing two model responses pairwise where a rubric would be hard to write; running a rubric-graded eval on hundreds of items without a labeling team.

**What LLM-as-judge is bad for:** absolute ground truth (it is a proxy, not the truth); scoring content the judge itself would refuse (the judge tends to score anything it flags as harmful as zero, even if the rubric asks about a different dimension); tasks that require domain expertise the judge model lacks (a general-purpose judge cannot reliably grade obscure medical or legal reasoning).

The rest of this chapter is about running it correctly enough that a release gate can consume its numbers.

## The two protocols

### Single-answer grading (MT-Bench, Sec. 3.1)

The judge sees one model's answer to a prompt and returns a score against a rubric. Aggregate: mean score per model per stratum.

```
System: You are an impartial judge evaluating an assistant response.

User prompt shown to the assistant:
{prompt}

Assistant response:
{response}

Grade the response on the following rubric:
1 — Actively harmful, refuses without cause, or fully incorrect.
2 — Attempts the task but is mostly wrong.
3 — Partially correct, with meaningful gaps.
4 — Correct and reasonably well-explained.
5 — Fully correct, well-organized, and appropriately detailed.

Reply first with two or three sentences of reasoning, then on a new line output:
Score: <integer between 1 and 5>
```

Properties:

- **Independent per model.** Adapter A's score does not depend on adapter B's score. You can add adapter C later without re-scoring A and B.
- **Comparable across items.** A 4 on item 12 is directly comparable to a 4 on item 47, subject to rubric consistency.
- **Susceptible to score compression** — judges cluster ratings near the middle or the top of the scale unless the rubric is precise.
- **Susceptible to style bias** — a well-formatted, well-hedged answer scores higher than a terse but correct one.

### Pairwise comparison (MT-Bench, Sec. 3.2)

The judge sees two responses to the same prompt and picks a winner (or "tie"). Aggregate: win rate of model A over model B.

```
System: You are an impartial judge comparing two assistant responses.

User prompt shown to both assistants:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Consider correctness, helpfulness, clarity, and following of any explicit instructions in the prompt. Ignore superficial length differences.

Reply first with two or three sentences of reasoning, then on a new line output one of:
Verdict: A
Verdict: B
Verdict: Tie
```

Properties:

- **Direct comparison.** No rubric compression; the judge only has to say which is better.
- **Higher agreement with humans** than single-answer grading on many benchmarks (MT-Bench reports ~85% pairwise agreement vs. ~80% single-answer).
- **Requires 2× the responses** — you need both adapters' outputs before you can judge.
- **Position-biased.** The judge over-picks whichever answer appears first. Chapter section below.
- **Not transitive.** A > B and B > C does not imply A > C at the judged-item level. Only in aggregate does the ordering behave sensibly.

Choose based on what you are measuring. **Release gates typically use both** — single-answer for absolute quality drift, pairwise against the previous adapter for regression detection.

## The four biases and how to control them

MT-Bench and follow-up work catalog four biases that LLM judges exhibit consistently. Each has a concrete mitigation. All four apply to every judged eval; none of the mitigations is optional in a release-gate setting.

### 1. Position bias (pairwise only)

The judge over-prefers the response in a particular position — usually the first, sometimes the second, depending on judge and prompt. In MT-Bench experiments GPT-4 shows a ~10–15 percentage-point position preference on some tasks.

**Control: randomize and average.** For each item, judge it *twice* — once with (A first, B second) and once with (B first, A second). The final verdict for that item:

- Both orderings pick A → A wins.
- Both orderings pick B → B wins.
- Orderings disagree → tie (not a win for either).
- Both orderings say tie → tie.

Cost: 2× judge calls for pairwise. Non-negotiable for a real release gate. Report the position-bias fraction (the % of items where the two orderings disagree) as a health metric — high disagreement means either a weak judge or genuinely close-quality responses.

### 2. Verbosity bias (single-answer and pairwise)

Judges reward longer answers. A response that is longer, more hedged, and has more structure scores higher than a terse-but-correct response, all else equal. This effect is well-documented (Zheng et al. 2023, Chiang & Lee 2023, and many replications).

**Controls:**

- **Length-normalize the rubric.** In single-answer grading, include an explicit "Do not reward length — a concise correct answer is preferred to a verbose one" instruction. Modest effect; helps at the margin.
- **Report length alongside score.** Chapter 6's regression report includes mean response length per model. A win-rate gain that coincides with a length increase should be discounted.
- **Bin by length.** For pairwise, split items into "short vs long," "short vs short," "long vs long" buckets and report separately. If your model wins only in the "long response" bucket, it's a verbosity artifact.
- **Length-controlled AlpacaEval (Dubois et al. 2024).** A published statistical correction that models the length effect and produces a length-corrected win rate. Worth adopting for release-gate pairwise numbers; the reference implementation lives in the `alpaca_eval` repo.

Do not accept a pairwise win-rate improvement of ≤5 pp when your model's responses got ≥30% longer. Chapter 6 codifies this as a gate rule.

### 3. Self-preference / self-enhancement bias

A judge model prefers responses that resemble its own output distribution. GPT-4 slightly prefers GPT-4-generated answers over Claude-generated answers of equal human-rated quality; Claude does the same for its own outputs. Empirical effect size is smaller than position or verbosity but is real (Panickssery et al., 2024, and others).

**Controls:**

- **Use a different judge family than the model being evaluated.** If you are fine-tuning a Llama derivative, judge with Claude or GPT, not with another Llama. If you are fine-tuning a Claude-tuned base, judge with GPT, and vice versa. There is no way to fully eliminate this — but not using an in-family judge removes the strongest effect.
- **Average across multiple judges** where budget allows. Two judges (GPT + Claude) with the pairwise verdict decided by majority, or averaged single-answer scores, materially reduces self-preference.
- **Report inter-judge agreement.** If your two judges disagree on 30% of pairwise items, one of them is probably wrong on this class of task; investigate before trusting either.

### 4. Style / persona bias

Judges reward polite, well-formatted, well-hedged, "assistant-shaped" responses regardless of correctness. A curt but correct answer to "what is 2+3" scores lower than "Great question! The sum of 2 and 3 is 5. Let me know if you need help with more math!"

**Controls:**

- **Rubric specificity.** A rubric that says "score correctness first, style second, and reject responses that hedge unnecessarily" reduces the effect. Vaguer rubrics amplify it.
- **Reference answers.** Provide a gold or reference answer to the judge along with the model response; this anchors the judge to the substance rather than the style. Adds cost per call but is worth it for reasoning tasks.
- **Cross-check with automated metrics.** Include a task-appropriate automated metric (exact match on a claim, JSON validity, tool-call correctness) alongside the judge score. If the judge scores a response 4/5 that fails the automated check, the judge is being fooled by style.

## Prompt-engineering the judge

The judge's prompt template is *itself* a piece of your eval infrastructure. Version it, pin it, and change it deliberately. A few concrete design choices that pay off:

### 1. Require reasoning before the verdict

Every judge prompt should ask the judge to write a short reasoning trace *before* the final score / verdict. This does two things:

- **Improves accuracy.** Chain-of-thought-style reasoning improves judge decisions the same way it improves any other LLM task.
- **Makes the score auditable.** When a per-item score seems wrong, the reasoning trace tells you whether the judge misunderstood the item, missed the rubric, or picked up on something you didn't. Store the reasoning.

### 2. Force a structured verdict format

The last non-whitespace token of the judge's response should be parseable with one regex. `Score: 4` or `Verdict: A` is enough. Do not let the judge free-form the verdict — you will find yourself post-processing "The correct verdict is A, though B has some merit" on 10% of items and mis-scoring silently.

### 3. Include the item's context in the judge prompt

The judge should see everything the model saw. If the item had a system prompt, show it. If the item had a tool-call context, show it. A judge scoring a support-agent response without the system prompt has to guess what tone was expected and will guess wrong.

### 4. Constrain the rubric to what the judge can reliably score

If the rubric asks "score factual correctness on medical claims," a general-purpose judge without medical grounding will make things up. Either constrain the rubric to what a strong LLM can plausibly grade (helpfulness, coherence, format compliance, obviously-wrong claims) or use a domain-tuned judge (rare and expensive) or fall back to human labeling for that dimension.

### 5. Test the rubric on the anchor set before the full run

Every rubric change requires a re-run on the anchor set. If the judge with the new rubric agrees with human labels less well than with the old rubric, the change made the eval worse even if it read better to you. Chapter 6 discusses versioning the rubric alongside the eval.

## Choosing a judge model

The right judge depends on task, budget, and the model you are evaluating. A practical hierarchy:

| Judge choice | Cost per 300-item eval | Quality | When to use |
|---|---|---|---|
| GPT-4o / Claude Sonnet 4-class | $10–40 | High, well-studied biases | Release-gate default |
| GPT-4-Turbo / Claude Opus | $50–200 | Highest published agreement with humans | High-stakes releases, anchor calibration |
| Small hosted models (Haiku, Mistral Large) | $2–10 | Weaker, more noise; still usable for pairwise | Iteration during development |
| Open judge models (`prometheus`, `judge-lm`, `pandalm`) | Free (self-hosted) | Trained for judging; agreement varies by task | Cost-sensitive gates; when you must not send data to APIs |
| Ensemble (two judges, majority) | 2× | Best bias reduction | High-stakes; anchor set calibration |

Two important cautions:

- **Do not judge with the model you are fine-tuning.** Even the base model of your fine-tune is self-preference-biased against alternatives.
- **Pin the judge's model version.** `gpt-4o-2024-08-06` is a specific model; `gpt-4o` is a moving target. If the judge model silently upgrades between your baseline run and your regression run, your delta is a mixture of adapter delta and judge delta.

## Agreement metrics

You cannot trust a judge's scores in isolation. You need at least one of these agreement measures, and preferably both:

### Inter-judge agreement

Two (or more) judges score the same items. Compute:

- **Pairwise (win/lose/tie):** Cohen's kappa, or plain percent agreement for a first pass. MT-Bench reports GPT-4/human pairwise agreement of ~85% and GPT-4/GPT-4 self-agreement of ~89% on their setup — anything meaningfully below the base model card's number is a red flag.
- **Single-answer scores (integer):** Spearman correlation, or Krippendorff's alpha with an interval scale. Above 0.7 is good; below 0.5 means the judges are largely disagreeing.

Two judges that agree well (κ ≥ 0.7 on pairwise, ρ ≥ 0.7 on single-answer) can be averaged with confidence. Two judges that disagree tell you the task is too hard for automated judging on some subset of items — inspect the disagreements.

### Judge-vs-human agreement (the anchor set)

The judge's score is a proxy for human quality; the anchor set measures how good a proxy it is. Recipe:

1. Draw 50–100 items from the eval set that cover the score / stratum distribution.
2. Have a human (or a panel) label each item — single-answer rubric and pairwise against a fixed reference.
3. Run the judge over the same items with the same rubric and pairwise setup.
4. Compute agreement (Cohen's kappa on pairwise, Spearman on scores).
5. Report the agreement number *alongside* every judged eval you report. `Model X scored 4.12/5 on the held-out rubric eval (judge-vs-human κ = 0.68 on the 80-item anchor set).`

The anchor set is a lightweight investment (a day of expert labeling) that gives every subsequent judged number credibility. Without it, "judge said 4.12" is a number without a scale.

## Cost management

Judged evals are the most expensive lane. A few practical patterns:

- **Two-pass judging.** First pass with a cheap judge (Haiku, small open model). Items where the cheap judge is confident, keep. Items where confidence is low, re-judge with an expensive model. Cuts total cost by 40–70% on typical evals.
- **Cache aggressively.** The judge's response to (prompt, response_a, response_b, rubric_version, judge_version, seed) is deterministic-ish. Cache by that key; if the model responses are unchanged between runs, skip the judge call. This is table stakes for the release gate that runs the same suite every release.
- **Batch API endpoints.** OpenAI, Anthropic, and Vertex all offer batch pricing at ~50% of interactive pricing with a 24-hour SLO. If the release gate can wait, batching halves the cost.
- **Reserve pairwise for what matters.** A single-answer eval on a 500-item held-out set is 500 judge calls. Adding pairwise-vs-previous doubles it (and doubles it again for position-bias correction). Reserve pairwise for the top-priority strata; use single-answer for the long tail.

## When LLM-as-judge is the wrong tool

Three situations where you should refuse to use LLM-as-judge and fall back to human labels or a different metric:

1. **The judge would refuse the content.** Safety-adjacent evals (red-team responses, borderline harmful requests) — the judge often refuses to evaluate, or automatically scores anything ambiguous as 1. Use human labels or a specialized safety classifier.
2. **The judge lacks the domain knowledge.** Highly technical domains (advanced math, subspecialty medicine, novel legal frameworks) — the judge will hallucinate authoritative-sounding but wrong scores. Human domain experts, or a domain-tuned judge, or an automated verifier (unit tests, symbolic math checks).
3. **You are measuring adherence to a nuanced style guide.** The judge's own preferences override the style guide about ~30% of the time in practice. Human raters trained on the guide, or a rule-based automated check where the guide is codifiable.

Also — if the judge model *is* the model being evaluated, or comes from the same family, everything above is compounded by self-preference. Cross-family judging is the minimum bar.

## What to report

Every judged eval, in the release-gate report:

1. **The judge identity (model, version, provider, temperature, seed).**
2. **The rubric version.**
3. **The prompt template (or a hash of it).**
4. **Aggregate score(s) — mean, median, stratified.**
5. **Confidence intervals — bootstrap over items is the standard.**
6. **Position-bias rate (pairwise) — % of items where the two orderings disagreed.**
7. **Judge-vs-human agreement on the anchor set — κ or ρ.**
8. **Inter-judge agreement if you used more than one judge — κ or ρ.**
9. **Mean response length per model — for verbosity-bias detection.**
10. **Per-item scores dumped to JSON — for the per-item diffs chapter 6 computes.**

If any of these are missing, the number is not release-gate-quality. Chapter 6 wraps this into a template.

## Exercise pointer

`exercises/exercise-03-llm-as-judge-mt-bench-style.md` implements single-answer rubric grading and pairwise comparison against your task-specific eval from exercise 02, controls for position bias, and reports judge-vs-human agreement against a 50-item human anchor.

## Summary

- LLM-as-judge is the field-standard proxy for human quality ratings on open-ended generation tasks. It works well enough to substitute for a human panel on many tasks — but only with explicit controls.
- Two protocols: **single-answer grading** (score against a rubric, comparable across items and models, susceptible to style/verbosity bias) and **pairwise comparison** (direct A/B, higher agreement with humans, requires position-bias correction).
- Four documented biases: **position** (control: randomize order and require agreement), **verbosity** (control: length-normalize the rubric, report length, length-controlled AlpacaEval), **self-preference** (control: cross-family judge, multiple judges), **style** (control: rubric specificity, reference answers, cross-check with automated metrics).
- Judge prompt design: require reasoning before verdict; enforce a structured verdict format; include the model's full context; test rubric changes on the anchor set before the full run.
- Pin the judge model version. Never judge with the model family you are fine-tuning.
- Report agreement metrics: inter-judge (κ / ρ) between multiple judges when you run more than one, and judge-vs-human (κ / ρ) against a 50–100-item human anchor set. Without an anchor, the score has no scale.
- Cost patterns: two-pass judging (cheap + expensive), aggressive caching by (prompt, response, rubric, judge) key, batch APIs where the gate can wait.
- LLM-as-judge is the wrong tool for safety-adjacent content, deep-domain expertise, and nuanced style adherence. Fall back to humans or specialized metrics for those.
- Every judged number in a release-gate report includes: judge id, rubric version, prompt template hash, aggregate + CIs, position-bias rate, judge-vs-human agreement, inter-judge agreement, mean response length, per-item scores.
