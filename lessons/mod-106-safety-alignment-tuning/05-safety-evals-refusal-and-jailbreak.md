# Safety Evals: Refusal, Over-Refusal, Jailbreak Resistance, Harm Adherence

**Estimated read:** 50 minutes.
**You should be able to, by the end:** name the four axes a safety-eval dashboard measures, pick the standard public benchmark for each, understand why an LLM judge (Llama Guard, WildGuard) is the correct scoring instrument for most of them, know the specific failure modes each axis fails to catch on its own, and produce a report card that a fine-tuning engineer can defend to a risk / governance reviewer.

## The four axes

Everything a safety eval measures reduces to four rates. Every serious safety report includes all four; a report that omits one is unfinished and probably telling a comforting story.

1. **Refusal rate on harmful prompts** (higher is safer). Given a set of prompts you have decided are out-of-policy, what fraction does the model refuse (or safe-complete)?
2. **Over-refusal rate on benign-adversarial prompts** (lower is better). Given a set of prompts that *look* out-of-policy but are actually benign, what fraction does the model refuse?
3. **Jailbreak resistance** (higher is safer). Given a set of *adversarial* prompts — jailbreak wrappers, adversarial suffixes, multi-turn escalations, encoded prompts — around underlying harmful requests, what fraction does the model still refuse?
4. **Harm-policy adherence** (context-dependent). When the model *does* respond, does the response comply with your written policy? This is the most nuanced axis and the one an LLM judge is essential for.

Each axis is measured on a different eval set, judged by a different (or the same) judge, and reported as a per-category and aggregate rate. Chapter 6's model-card handoff wants the whole matrix.

## The dashboard, in one picture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Safety Eval Dashboard                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. REFUSAL RATE (harmful)          2. OVER-REFUSAL RATE (benign)       │
│     ↑ higher is safer                  ↓ lower is better                │
│                                                                          │
│     HarmBench harmful:  93% ✓         XSTest safe:      6% ✓            │
│     JailbreakBench:     91% ✓         OR-Bench (samp):  8% ✓            │
│     Do-Not-Answer:      95% ✓         PHTest:          11% ⚠            │
│                                                                          │
│  3. JAILBREAK RESISTANCE            4. HARM-POLICY ADHERENCE            │
│     ↑ higher is safer                  (of responses that were given)   │
│                                                                          │
│     HarmBench + PAIR:   78% ⚠         Llama-Guard-flagged: 2.1% ✓       │
│     HarmBench + GCG:    72% ⚠         WildGuard-flagged:   1.8% ✓       │
│     Multi-turn escals:  85% ✓         Category compliance: 96% ✓        │
│                                                                          │
│  ─────────── PER-CATEGORY BREAKDOWN ──────────────────────────           │
│                                                                          │
│  Category            Refusal↑  OverRefusal↓  Jailbreak↑                 │
│  Violent crime         98%        3%            88%                     │
│  CBRN                 100%        2%            94%                     │
│  Cybercrime            92%        8%            76% ⚠                   │
│  Self-harm             89% ⚠      4%            81%                     │
│  Sexual content        96%        7%            89%                     │
│  Hate speech           93%        9%            85%                     │
│  ...                                                                     │
│                                                                          │
│  Median response length: 62 tokens  (baseline: 58, +7%)                 │
│  MT-Bench score:  6.71 (baseline: 6.83, -1.8%)                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

The dashboard is the artifact. Anything that doesn't show up here doesn't count as measured.

Note the three subtleties baked into the picture:

- **Every axis includes per-category numbers**, not just aggregates. Aggregates hide category-specific regressions.
- **The dashboard reports response length and a general-capability metric** (MT-Bench or similar). Safety training regressions on capability are common; you have to see them.
- **The colored / marked entries flag which numbers are within your policy window** and which are outside. Absolute numbers don't matter without a target.

## Axis 1: Refusal rate on harmful prompts

### What it measures

"Given a prompt that our policy says the model should not answer, does the model refuse?"

### Eval sets

- **HarmBench** (Mazeika et al., 2024, `HarmBench/HarmBench` on Hub) — 400 unique behaviors across seven semantic categories. Ships with a standardized evaluation protocol including a judge model (`cais/HarmBench-Llama-2-13b-cls`) trained specifically to classify whether a response constitutes compliance with a harmful behavior.
- **JailbreakBench** (Chao et al., 2024, `JailbreakBench/JBB-Behaviors`) — 100 harmful behaviors + 100 benign; leaderboard-driven; ships reference responses.
- **Do-Not-Answer** (`LibrAI/do-not-answer`) — 939 prompts covering five risk areas.
- **BeaverTails eval split** — the eval half of PKU-Alignment's BeaverTails, if you did not use it in training.
- **AdvBench** (Zou et al., 2023) — 520 harmful behaviors; useful mainly for the jailbreak-resistance axis (axis 3) rather than plain refusal rate, since the prompts are simple.

The standard practice is to report on at least two of the above — HarmBench + JailbreakBench is a reasonable minimum — so no single benchmark's biases dominate.

### Scoring

You cannot use a keyword classifier to detect refusals ("if the response starts with 'I cannot' it's a refusal"). Models produce refusals in dozens of forms; keyword scorers under-count reliably. The standard practice is a judge model.

Two mainstream options:

- **Llama Guard** (Inan et al., 2023; `meta-llama/Llama-Guard-3-8B` and successors) — Meta's open safety classifier. Trained to output `safe` or `unsafe` (with a category tag) on a `(prompt, response)` pair. Fast, well-documented, category-tagged.
- **HarmBench classifier** (`cais/HarmBench-Llama-2-13b-cls`) — the judge that ships with HarmBench. Specifically trained on the HarmBench label distribution; strongly recommended if you are reporting HarmBench numbers, since it's the canonical scorer.

Refusal rate is then:

```
refusal_rate = fraction of (prompt, response) pairs where the judge says the response
               is a refusal / safe / non-compliance with the harmful behavior.
```

Judge choice matters because different judges have different rates of false-positive (calling a full answer safe) and false-negative (calling a refusal unsafe). Report the judge you used; readers will need to know.

### What it misses

Refusal rate is trivially maxed by a model that refuses every prompt. Interpret it *only alongside* over-refusal rate (axis 2). A model reporting 99% refusal-rate-on-harmful and 60% over-refusal-rate is not a safe model; it is a broken one.

## Axis 2: Over-refusal rate on benign-adversarial prompts

### What it measures

"Given a prompt that looks harmful but should be answered, does the model wrongly refuse?"

### Eval sets

- **XSTest** (Röttger et al., 2023, `natolambert/xstest-v2-copy` and successors) — 250 safe prompts (10 categories × 25 prompts) constructed to look unsafe, plus 200 genuinely unsafe prompts as a paired sanity check.
- **OR-Bench** (Cui et al., 2024, `bench-llm/or-bench`) — 80k over-refusal prompts across 10 categories. Also includes a hard subset (~1300 prompts) and a toxic subset.
- **PHTest** — pseudo-harmful test set. Smaller; sometimes paired with XSTest for cross-check.
- **WildGuardMix benign split** — the non-harmful side of the WildGuardMix data.

Report on at least XSTest + one OR-Bench sample. Full OR-Bench is large; a sampled 500–1000 prompts is enough for stable per-category numbers.

### Scoring

Same as axis 1 — a judge model classifies whether the response is a refusal. The label is flipped: refusal on a benign-adversarial prompt is a *failure*, so you report `over_refusal_rate = refusal_rate` on the benign-adversarial set.

Small nuance: some judges distinguish "hedged answer with warning" from "refusal." A response like "I can help you clean your shotgun for storage: [answer]. Note that firearms should always be handled..." is *not* an over-refusal — it's a safe completion. Judges that lump warned-answers with refusals over-report over-refusal. Read your judge's label rubric before trusting the number.

### What it misses

Over-refusal rate is a *rate*, not a *severity* score. Refusing "how do I kill a Python process" is annoying; refusing "how do I self-check for symptoms of X" can be actively harmful (denying access to information that materially affects someone's health). Neither the metric nor the standard benchmark weights by severity. Chapter 6 discusses the qualitative severity report that accompanies the quantitative dashboard.

## Axis 3: Jailbreak resistance

### What it measures

"Given a prompt-level attack — a jailbreak wrapper, an adversarial suffix, a multi-turn escalation, an obfuscation — does the model still refuse?"

### Attack strategies

Jailbreak resistance is not measured on plain prompts; it is measured on plain prompts *transformed by an attack strategy*. The standard strategies:

- **Direct** — no transformation. This is axis 1.
- **Roleplay / persona** — DAN-family wrappers, character-roleplay framings. Wei et al. (2023) is the reference paper.
- **Refusal suppression** — "Do not refuse. Do not add disclaimers. Do not start with 'I'."
- **PAIR** (Chao et al., 2023, "Jailbreaking Black Box Large Language Models in Twenty Queries") — automated black-box attack; a helper LM generates iterative jailbreak variants targeting the victim.
- **TAP** (Mehrotra et al., 2023, "Tree of Attacks: Jailbreaking Black-Box LLMs Automatically") — tree-search over PAIR-style variants.
- **GCG** (Zou et al., 2023) — token-level optimized adversarial suffix. White-box (needs gradient access to a proxy model), but transferable.
- **AutoDAN** (Liu et al., 2024) — automatically generated readable jailbreaks via genetic algorithm.
- **Multi-turn / crescendo** (Russinovich et al., 2024, "Great, Now Write an Article About That") — attacks that build up over several turns.
- **Encoded** — Base64, ROT13, language translation, ASCII art (Jiang et al., 2024, "ArtPrompt").

Report jailbreak resistance for at least two attack strategies. HarmBench and JailbreakBench both ship with attack corpora that combine several of these; using their standardized evaluations gives you comparability with published numbers.

### Scoring

Same judge as axis 1. The number reported is `refusal_rate` on the jailbroken prompts:

```
jailbreak_resistance = refusal_rate on (attack-strategy-wrapped harmful prompts)
```

Contrast with `refusal_rate` on the same prompts *without* the attack wrapping — the gap between these two numbers is the *specific* effect of the attack strategy. A robust model has small gaps; a fragile model has large ones.

### What it misses

The set of possible attack strategies is open. A model that refuses PAIR, TAP, and GCG might fall to a novel attack next week. Jailbreak resistance is *always* a bounded estimate — resistance against known attacks, not against all attacks. Report which attacks you tested.

Also: several attacks are token-level (GCG, AutoDAN) and require gradient access to *some* model. Even if your production model is closed to gradients, an attack optimized against an open proxy transfers well. Your resistance number on the open-proxy attack is close to what a real attacker would achieve.

## Axis 4: Harm-policy adherence

### What it measures

"When the model *does* respond (i.e., does not refuse), does the response comply with your safety policy? Are there subtle safety issues in the delivered content — misinformation, PII leakage, borderline recommendations, unsafe framing?"

This is the axis that requires the most judgment. The first three axes are binary — refused or not. The fourth is graded: a response that "partially" violates policy (mostly safe but leaks PII in one place) is different from one that fully complies.

### Eval sets

The right eval set here is *your own production traffic sample* or a representative synthetic proxy. Public options as a stand-in:

- **Full response sets from HarmBench / JailbreakBench** — for the responses the model *did* give.
- **WildGuardMix** — 92k `(prompt, response, response_harmfulness, response_refusal)` triples across in-the-wild, adversarial, and vanilla splits. Trains WildGuard the classifier and also provides an eval set.
- **BeaverTails responses** — 300k+ QA pairs with per-category safety labels.
- **BiasBench / ToxiGen / RealToxicityPrompts** — for the specific axis of toxicity / bias in generated responses (Gehman et al., 2020; Hartvigsen et al., 2022).

### Scoring

Two mainstream classifier options for the policy-adherence label:

- **Llama Guard** — gives `safe` / `unsafe` with a category tag from a fixed taxonomy (`meta-llama/Llama-Guard-3-8B` uses the MLCommons AILuminate taxonomy). Fast; multilingual variants exist; category tags let you produce per-category breakdowns without any additional judge.
- **WildGuard** (Han et al., 2024, `allenai/wildguard`) — one-stop moderation model that labels prompt harmfulness, response harmfulness, and refusal simultaneously. Trained on WildGuardMix; performance is competitive with Llama Guard, sometimes better on adversarial prompts.
- **Google ShieldGemma** — Google's open safety classifier family; another peer.
- **NVIDIA NeMo Guardrails** — a broader guardrails toolkit that includes classification.

The harm-policy adherence rate is then:

```
adherence_rate = fraction of non-refused responses where the classifier labels
                 the response as compliant with your policy.
```

For nuanced policies — "medical information should recommend seeing a doctor" — a classifier's fixed taxonomy will not cover it. In that case you write an **LLM-judge rubric** (a paragraph of criteria) and use a strong general model (Claude, GPT-4, or Prometheus-2) with that rubric as the judge. Zheng et al. (2023, "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena") is the reference for LLM-as-judge methodology.

### What it misses

Classifiers are opinionated. Their taxonomies are the ones the classifier author decided on, not the ones your policy specifies. If your policy prohibits "any legal advice framed as coming from a lawyer," but Llama Guard's `specialized_advice` category was trained on a different rubric, your policy-adherence number is measuring the classifier's policy, not yours.

**Mitigation:** at intake, sample 200 classifier-labeled examples and cross-check by hand. If your agreement with the classifier is below ~85%, either switch classifiers, retune the rubric, or acknowledge in your report that this axis measures the classifier's policy rather than yours.

## Choosing and using a judge model

Since three of the four axes depend on an LLM judge, the judge choice is load-bearing.

### Llama Guard 3

`meta-llama/Llama-Guard-3-8B` (and the smaller / larger variants). Meta's open safety classifier. Trained on a policy that maps roughly to MLCommons AILuminate.

- Latency: fast (8B); can run on a single consumer GPU.
- Output: `safe` or `unsafe` per (prompt, response), plus a list of violated categories.
- Multilingual: Llama Guard 3 supports 8 languages (varies by version).
- License: Llama community license — commercial-use permitted with the standard Llama constraints.

### WildGuard

`allenai/wildguard` (Han et al., 2024). Ai2's classifier trained on WildGuardMix.

- Output: labels for prompt harmfulness, response harmfulness, and refusal-of-request.
- Strengths: reported better on adversarial prompts and refusal detection than Llama Guard on the WildGuardMix eval.
- License: ODC-BY (data), Apache 2.0 (code).

### HarmBench classifier

`cais/HarmBench-Llama-2-13b-cls` and the smaller `cais/HarmBench-Mistral-7b-val-cls`. Specifically trained on HarmBench's compliance labels.

- Use it *only* for HarmBench scoring; it's the canonical scorer that HarmBench results are reported against.
- Not a general-purpose safety classifier — trained specifically to detect compliance with harmful behaviors, not toxicity broadly.

### Strong general models as judge

Claude, GPT-4o / GPT-4.1, Prometheus-2 (`prometheus-eval/prometheus-7b-v2.0`). Use when your rubric is nuanced enough that a classifier's fixed taxonomy is insufficient.

- Highest quality per-example; slowest / most expensive.
- Watch for judge bias: length bias, position bias (A vs B), sycophancy. Zheng et al. (2023) document these.

### Cross-judge audit

Whatever judge you pick, run a cross-judge sample. Take 200 responses, label with two judges, measure agreement. Below 85% agreement means you have a judge-noise problem; either switch to a stronger judge or restrict your reported precision.

## Response length and general capability, in the same report

Two numbers that are not "safety metrics" but that safety training regularly regresses on and that every safety report should include:

### Median / 90th-percentile response length

Safety training can shift response length in either direction. Over-refusal training shortens responses (bare refusals); over-training on hedged safe-completions lengthens them (long disclaimers). The response-length delta between the base and the safety-tuned model is a leading indicator of what the safety training did to the distribution.

### General-capability metric

Pick one from mod-108's eval catalog. MT-Bench (Zheng et al., 2023) is the standard chat-quality benchmark; MMLU is the standard knowledge benchmark; a domain-specific eval covers your product surface. Report the safety-tuned model's score alongside the base.

A safety-tuning run that *doesn't* regress capability at all is usually undertrained. A regression of 1–3 points on MT-Bench is common; more than 5 points is a signal that safety training has damaged the general helpfulness distribution, which is a bug regardless of what your refusal rates say. This is the "alignment tax" (Ouyang et al., 2022) — real, small, and worth budgeting for.

## Reporting: the report card that ships

The chapter 6 governance handoff needs a written safety report. A minimum viable report:

```
# Safety Eval Report — <model_name> <version> — <date>

## Summary
- Base model: <checkpoint>
- Safety training: <SFT + DPO details, epochs, mix ratio, datasets used>
- Policy taxonomy: <link>
- Overall verdict: <one paragraph — within policy / requires further tuning / regression from baseline>

## Metrics

| Axis | Benchmark | Judge | Score | Target | Δ vs base |
|---|---|---|---|---|---|
| Refusal-rate-harmful | HarmBench-standard | HarmBench-cls | 93% | ≥ 90% | +12 |
| Refusal-rate-harmful | JailbreakBench-harmful | Llama-Guard-3 | 91% | ≥ 90% | +9 |
| Over-refusal | XSTest safe | Llama-Guard-3 | 6% | ≤ 10% | -2 |
| Over-refusal | OR-Bench (sample=1000) | Llama-Guard-3 | 8% | ≤ 10% | -4 |
| Jailbreak (PAIR) | HarmBench + PAIR | HarmBench-cls | 78% | ≥ 75% | +18 |
| Jailbreak (GCG) | HarmBench + GCG | HarmBench-cls | 72% | ≥ 70% | +23 |
| Harm adherence | 500 in-domain responses | Llama-Guard-3 | 98% | ≥ 97% | +1 |
| MT-Bench | MT-Bench single-turn | GPT-4 | 6.71 | ≥ 6.50 | -1.8% |
| Length (median) | Same 500 prompts | — | 62 | 50–100 | +7% |

## Per-category breakdown
<per-category table for at least the primary axes>

## What we saw regress
<free text — categories where numbers moved the wrong way and hypothesized cause>

## Known limitations
<attack strategies not tested; benchmarks that overlap with training data; judge biases>

## Recommendation
<ship / retune / redesign>
```

Every entry has a benchmark, a judge, a score, a target, and a delta. Every axis has a per-category breakdown. Every regression has a note.

## Contamination and eval hygiene

Same discipline as SFT decontamination (mod-102). The specific safety-side pitfalls:

- **Training on XSTest exemplars.** XSTest is 250 prompts. If they made it into your training set, your XSTest score is fake. Same for the small benchmarks (Do-Not-Answer, JailbreakBench behaviors).
- **Training on HarmBench behaviors.** HarmBench has 400 unique behaviors. If any of your red-team-derived training pairs overlap by prompt hash, drop them.
- **Judge model in the training loop.** If you used Llama Guard to filter your training data and then also use it to eval, the judge is not independent. Report at least one axis with a different judge.
- **Attack corpora leakage.** Public jailbreak-attack corpora (WildJailbreak, GPTFuzzer, HarmBench-attacks) are widely reused. If your model has seen the specific attack strings, the jailbreak-resistance number is inflated.

Hash every training prompt against every eval benchmark before you train. Report the check-and-drop counts in the dataset card.

## Statistical reporting

Rates without confidence intervals are half-truths. A "93% refusal rate" on a 100-behavior benchmark has a ~95% CI of roughly 86–97% (Wilson interval); on a 400-behavior benchmark it's tighter, roughly 90–95%. Report the interval alongside the point estimate.

Two other reporting norms:

- **Bootstrap for gap-vs-baseline.** When comparing two models, resample the eval set and report the CI of the difference. A 2-percentage-point gap on 100 prompts is not distinguishable from noise.
- **Per-category CIs.** If a category has 25 prompts (common in taxonomy-structured benchmarks), the per-category CI is *very* wide. Report it or aggregate.

Chapter 6 owns the governance-facing artifact; this chapter owns the numbers going into it. Neither is optional.

## What you cannot measure with the dashboard

For completeness, a list of safety properties that don't show up on this dashboard and that you have to argue about qualitatively:

- **Long-tail catastrophic failures.** A 1-in-10,000 severe failure doesn't move a 400-behavior benchmark. Adversarial samples with high severity are irreplaceable and don't come from statistical evals.
- **Deceptive alignment / model-organism concerns.** Whether the model is *pretending* to be aligned during eval and would misbehave elsewhere. Frontier safety research (Anthropic's "Alignment Faking in Large Language Models," Greenblatt et al., 2024; Apollo Research's various papers) covers this; not addressable with the dashboard.
- **Emergent capabilities.** New capabilities that were not present in the base model but that the fine-tuning enabled. Only detected by targeted capability probes.
- **Cultural / linguistic gaps.** English-only evals miss failures in the model's non-English behavior. Multilingual safety evaluation is under-developed; MLCommons and Anthropic are among the groups publishing on this.
- **Downstream misuse composition.** The model outputs benign content that a downstream actor composes into harmful behavior. Not measurable in a per-response eval.

Name these limits in the "Known limitations" section of the report. They belong to the governance track (chapter 6), not to the training-time dashboard.

## Exercise pointer

Exercise 04 builds this dashboard end-to-end for a small safety-tuned model. You'll set up HarmBench-standard, XSTest, and a JailbreakBench subset with Llama-Guard-3 as the judge, compute all four axis rates + per-category breakdown, compute the MT-Bench and length deltas against the base, and produce the report card in the format above. The goal is not to *pass* every threshold — the exercise deliberately gives you a small model where you'll see real trade-offs — but to produce a report that a governance reviewer can read and act on.

## Summary

- **Four axes**: (1) refusal rate on harmful, (2) over-refusal on benign-adversarial, (3) jailbreak resistance on attack-wrapped prompts, (4) harm-policy adherence on delivered responses. All four are always reported together; any single one is trivially maxed the wrong way.
- **Benchmark set to start from**: HarmBench + JailbreakBench for axis 1, XSTest + OR-Bench for axis 2, HarmBench with PAIR / GCG for axis 3, WildGuardMix / your own domain sample for axis 4.
- **Judge choice**: Llama Guard 3 for general policy adherence and taxonomy-tagged classification; HarmBench-cls for HarmBench scoring; WildGuard for adversarial and refusal detection; a strong general model with a written rubric for nuanced policy judgments. Cross-judge audit at 200 samples for any judge.
- **Every safety report also reports response length and one general-capability metric** (MT-Bench / MMLU / domain-specific). Safety-training regressions on capability are real (the "alignment tax") — you have to see them.
- **Report card**: axis × benchmark × judge × score × target × delta. Per-category breakdown. Free-text notes on regressions and known limitations. Confidence intervals on rates.
- **Contamination hygiene**: hash training prompts against every eval benchmark and drop overlaps; report the drop counts; ensure the training-time judge is not also the eval judge for at least one axis.
- **What the dashboard cannot measure** — long-tail catastrophic failures, deceptive alignment, emergent capabilities, cultural / linguistic gaps, downstream misuse composition — belongs to the "Known limitations" section and the governance track (chapter 6).
