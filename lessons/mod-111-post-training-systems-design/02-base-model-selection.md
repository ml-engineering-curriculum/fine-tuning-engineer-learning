# Selecting the Base Under Real Constraints

**Estimated read:** 35 minutes.
**You should be able to, by the end:** score any base-model candidate against a nine-axis rubric (license, capability floor, size / hardware ceiling, tokenizer, chat-template maturity, method compatibility, community / ecosystem, deprecation risk, cost). Write a defensible base-selection ADR that a peer engineer and a lawyer can both read. Recognize when a hard constraint overrides the rubric and record the override so the next model release can revisit it.

## Why base selection is a rubric, not a preference

Ask five fine-tuning engineers which base they would pick and you will get five answers, each defensible in isolation and each smuggling in a different set of assumptions. Ask them to score against the same rubric and the answers converge — and the disagreements become explicit and negotiable ("I weighted license twice because we redistribute externally"). The rubric is the mechanism.

Base selection also compounds. Every downstream module in this track — dataset conversion (mod-102 ch3, mod-110 ch4), SFT (mod-103), PEFT (mod-104), preference optimization (mod-105), safety tuning (mod-106), distributed (mod-107), eval (mod-108), quantization / serving (mod-109) — inherits the base's chat template, tokenizer, license, and method-compatibility limits. A base picked casually at week 1 is a limiter on every downstream decision for the model's life.

## The nine-axis rubric

The rubric is a spreadsheet with one row per candidate and one column per axis, plus a `notes` column per axis and a `verdict` column overall. Below, each axis, what it means, and the signals that score it.

### Axis 1 — License posture

The single most likely axis to kill a candidate outright. Four questions:

1. **What license does the base carry?** SPDX identifier if available. Common cases:
   - Apache-2.0 / MIT (Qwen 2.5, Mistral base weights, Gemma sanctioned use) — permissive, redistributable, generally the friendliest posture.
   - Llama 3 / Llama 3.1 / Llama 3.3 Community License — permissive with restrictions (700M-MAU threshold; naming and attribution rules; acceptable-use policy; distribution attribution).
   - Custom-non-commercial (some research releases) — internal use only, no redistribution.
   - Vendor proprietary (OpenAI GPT-4.1, Anthropic Claude 4.x, Google Gemini 2.5) — accessible only through the managed API (mod-110), no weights to download, redistribution is not a category.
2. **What is your redistribution intent?** Internal-only, open-weights research release, commercial product with the weights shipped, or SaaS behind an API? Each intent has a different license bar. `mix_card.md` from mod-102 chapter 6 captures this exact question for the dataset side; the base has to match.
3. **What does the license say about *fine-tuned* weights?** Some licenses (Llama Community) impose downstream naming ("Llama" must appear in the derivative model's name); some data licenses restrict what you may train on; some model licenses restrict commercial use of *outputs* from the fine-tune. Read the license, or have legal read the license and give you a one-page summary.
4. **What does the license say about *distillation* from the base?** OpenAI, Anthropic, and other frontier vendors' terms currently restrict using their outputs to train competing models. If your dataset (mod-102 ch4) includes synthetic completions from those vendors, the license posture of the base matters less than the license posture of the *dataset*.

**Scoring:** 0 = outright blocker for the intended distribution, 1 = usable with caveats, 2 = clean.

### Axis 2 — Capability floor

The base's zero-shot performance on the tasks that matter for the target product. Before fine-tuning, before any of your data. This axis answers: *is the base already close enough that fine-tuning is a finish, and not a full renovation?*

- Run the mod-108 fast-lane eval (MMLU, GSM8K, IFEval, MT-Bench, plus the task-specific held-out set from mod-108 ex 02) against each candidate at zero-shot with the base's official chat template.
- Record the numbers with 95% bootstrap CIs.

A candidate with a low capability floor is not disqualified. It just changes what the fine-tune has to do: the further below the target metric the zero-shot score is, the more capability you are asking the fine-tune to add, which almost always means more data (chapter 3), more objectives (SFT alone may not be enough), or a longer training run.

**Scoring signals:**

- Zero-shot within 10 pp of the target on the task-specific held-out set → floor 2.
- Zero-shot 10–20 pp below target → floor 1.
- Zero-shot more than 20 pp below → floor 0; fine-tuning it there is doable but expensive, and a larger base often gets there faster than a stack of tricks on a small one.

### Axis 3 — Size and hardware ceiling

Two hard limits, decided by serving (mod-109) and not by training:

- **Serving hardware.** What single-GPU / multi-GPU class is production allowed to use? A 70B in BF16 needs ~140 GB HBM; an 8B fits in 16 GB. The serving constraint bounds the training decision from above.
- **Training hardware.** What GPU capacity are you allowed to burn for the run? A 70B full-parameter SFT is ~800 GB HBM with ZeRO-3 / FSDP; QLoRA on a 70B fits in a couple of H100s. The training constraint bounds *the method choice* rather than the base itself.

A larger base with QLoRA (mod-104 ch2) is often the right answer even under a tight serving budget: quantize to AWQ / GPTQ 4-bit for serving (mod-109 ch3), fit an 8B ~= 4 GB, and preserve the capability that a smaller base cannot match. Do not conflate "we can only afford to serve 8B" with "we can only fine-tune 8B" — the two are the same only for full-parameter runs.

**Scoring:** 0 = does not fit the serving budget, 1 = fits with a quantization step, 2 = fits comfortably.

### Axis 4 — Tokenizer

The tokenizer is easy to overlook and expensive to be wrong about. Three questions:

- **Vocab size and coverage for your language(s) and domain.** Poor tokenization of your primary language balloons the sequence length, which raises training cost linearly and eats the KV cache at inference (mod-109 ch7).
- **Special-token compatibility for your objective stack.** DPO and safety adapters commonly require the base's chat template's role tokens to already exist; a base whose tokenizer needs custom special tokens added means resizing embeddings and retraining the added rows, which is a small step but a step that has to be planned.
- **Reproducibility of the tokenizer version.** Some vendors have re-released tokenizers with fixed pre-tokenization rules. Pin the tokenizer version alongside the base version.

The mod-101 chapter 2 walkthrough on tokenizers and mod-102 chapter 3 walkthrough on chat templates are the reference; this axis is where their outputs land in the rubric.

**Scoring:** 0 = ballooned sequence lengths or missing critical tokens, 1 = usable but suboptimal, 2 = clean coverage and stable special tokens.

### Axis 5 — Chat-template maturity

Every current base ships with a chat template — a Jinja2 (or equivalent) function that maps `messages` to a string. The maturity of that template determines how much drift you will fight for the model's life.

- **Is the template published in a machine-consumable form?** Hugging Face `tokenizer.apply_chat_template` is the canonical carrier; if the template is only documented in prose, expect drift.
- **Does the template support the roles you need?** System, user, assistant is universal; `tool` / function-calling / multimodal roles are not. If your objective stack requires tool-call fine-tuning (mod-102 ch3), a base whose template omits `tool` role is a non-starter until upstream patches it.
- **Has the template changed across releases of the same base?** Some model families have re-released the tokenizer/template with silent changes (whitespace, BOS/EOS handling). Pin the exact revision.

**Scoring:** 0 = ad-hoc / drift-prone / missing roles, 1 = mature but with quirks, 2 = mature, roles complete, revision-pinned.

### Axis 6 — Method compatibility

Which of the objective-stack elements you plan (chapter 3) work on this base *without* invention?

- **SFT** — universally supported; not a discriminator.
- **LoRA / QLoRA** — near-universal on decoder-only bases; some multimodal bases have gaps in `peft` support.
- **DPO / ORPO / KTO** — TRL support depends on the tokenizer's chat-template and the base's ability to accept a reference model of the same architecture. Most modern bases are fine; verify with a smoke run.
- **Classical PPO-RLHF** — depends on TRL's `PPOTrainer` support for the base; some architectures (Mistral variants, some Qwen releases) have historically had rough edges.
- **Safety adapters / CAI-style tuning** — depends on tokenizer compatibility with the safety-preference format and on the availability of a compatible constitutional prompt template.
- **Quantized serving (AWQ / GPTQ / GGUF) and multi-adapter serving (vLLM, LoRAX)** — critical for downstream (mod-109); check that at least one of the mainstream serving stacks lists the base as first-class.

For each planned objective, mark ✔ (works out of the box), ⚠ (works with a patch or a workaround), ✘ (does not work). A ✘ on any objective the stack requires is a blocker.

**Scoring:** 0 = one or more required objectives are ✘, 1 = requires meaningful workarounds, 2 = clean across the stack.

### Axis 7 — Community and ecosystem

Rubric axes 1–6 are the base's *capabilities*. Axis 7 is the *support surface* — the density of people who have solved the same problem before you.

- **Model downloads and forks on Hugging Face.** Popular bases have hundreds of derivative models, most of which are one Git-blame away from the recipe you need.
- **Documentation and canonical recipes.** Meta's Llama recipes, Alibaba's Qwen guide, Mistral's docs, and equivalent first-party guides. Third-party recipes (Unsloth, Axolotl, LLaMA-Factory) matter too — a base with 50 Axolotl configs on GitHub is a base whose failure modes are catalogued.
- **Serving-stack support.** vLLM's model-support matrix, TGI's, llama.cpp's, LoRAX's — each is a first-class or second-class citizen matrix.
- **Bug traffic on GitHub.** A base whose issues are actively responded to is a base whose problems get fixed; a base whose issues sit for a year is a base you are alone with.

This axis is the difference between "we ran the standard recipe" and "we spent three weeks reverse-engineering the tokenizer's pre-processing rules." Do not undervalue it.

**Scoring:** 0 = niche / lonely, 1 = adequately supported, 2 = strong ecosystem with multiple recipes.

### Axis 8 — Deprecation risk

Every model has a lifecycle. A base can be:

- **Actively maintained.** Upstream releases point-releases; the tokenizer / chat template are stable across them; you can expect security and bug patches for the model card's stated horizon.
- **Frozen but stable.** Not actively maintained but not going away. Fine for a project that ships and stops iterating.
- **Deprecated / on notice.** Vendor has announced or telegraphed end-of-life. Any fine-tune built on a deprecating base has a countdown on it.

For managed bases (mod-110), deprecation risk is stronger: OpenAI has retired fine-tunable model IDs on 6–12 month notice, and every retired ID means the fine-tune's inference endpoint disappears with it. Chapter 4 of this module and chapter 6 of mod-110 both flag this trade-off; on the rubric, it lives here.

**Scoring signals:**

- First-party deprecation policy published, ≥ 12 months' notice on retirements → 2.
- No stated policy but recent activity in the model family → 1.
- Recently deprecated sibling models or vendor signals of narrowing focus → 0.

### Axis 9 — Cost — training and inference

Two costs, both worked in dollars:

- **Training cost.** Ballpark from mod-103 (full SFT) or mod-104 (LoRA / QLoRA) math applied to the data budget from chapter 3.
- **Serving cost.** Dollars-per-million-tokens on the target serving stack (mod-109 ch7) or the vendor's per-token API rate (mod-110 ch5). Include the hosting rent for managed fine-tunes.

The cost axis is rarely the tiebreaker between two candidates in the same size class — it becomes decisive between size classes (7B vs. 70B) or between paths (self-hosted vs. managed). Chapter 4 owns that decision. On the rubric, the cost is one axis; it does not eat the others.

**Scoring:** 0 = exceeds the section-5 budget from the proposal (chapter 1), 1 = tight, 2 = comfortable.

## The scorecard, filled in

```markdown
# Base-model scorecard — Acme Support Assistant v3
**Date:** 2026-08-01
**Owner:** L. Ortega (fine-tuning)
**Rubric version:** v2

## Constraints inherited from proposal
- **Redistribution:** internal-only, served through Acme's own inference stack.
- **Serving:** single-node 4× A10G (24 GB each) via vLLM.
- **Objectives planned:** SFT + LoRA + DPO on LoRA (chapter 3 default stack).
- **Data budget:** ~57k rows total (chapter 1, section 4).
- **Latency SLO:** p95 TTFT ≤ 250 ms.

## Weights (documented, not floating)
- License: 3× (redistribution intent is internal, so any of the listed
  candidates is legal; still we prefer permissive to reduce future risk if
  redistribution changes).
- Capability floor: 3×.
- Size / HW ceiling: 3×.
- Tokenizer: 2×.
- Chat-template maturity: 2×.
- Method compat: 3×.
- Community: 2×.
- Deprecation: 2×.
- Cost: 1× (well within budget for all candidates).

## Candidates and scores (0–2 per axis, weighted)
| Axis (weight)              | Llama-3.1-8B-Inst | Qwen2.5-7B-Inst | Mistral-7B-Inst-v0.3 |
|----------------------------|-------------------|-----------------|-----------------------|
| License (3×)               | 1 (Community)     | 2 (Apache-2)    | 2 (Apache-2)          |
| Capability floor (3×)      | 2 (60% zero-shot) | 2 (58%)         | 1 (48%)               |
| Size / HW ceiling (3×)     | 2                 | 2               | 2                     |
| Tokenizer (2×)             | 2                 | 2               | 2                     |
| Chat template (2×)         | 2                 | 2               | 1 (system-role quirk) |
| Method compat (3×)         | 2                 | 2               | 2                     |
| Community (2×)             | 2                 | 2               | 2                     |
| Deprecation (2×)           | 2                 | 2               | 1 (v0.4 announced)    |
| Cost (1×)                  | 2                 | 2               | 2                     |
| **Weighted total (max 44)**| **41**            | **44**          | **35**                |

## Verdict
**Qwen2.5-7B-Instruct** by rubric. **Llama-3.1-8B-Instruct** is a close second
and is the incumbent (v2.4 base).

## Override discussion
The rubric says Qwen. The incumbent-base migration cost is real: chat template
diff, tokenizer diff, ~1 week of eval-set re-baselining. We plan to run
exercise-01 rubric with both Qwen and the incumbent Llama, in parallel, on a
20% subsample of the SFT set; whichever is within 1 pp of the other at
half-budget wins. If they tie, incumbency wins.

## Reproducibility pin
- Base A: `meta-llama/Llama-3.1-8B-Instruct`, revision `<sha>`,
  tokenizer.model sha `<sha>`, chat_template hash `<hash>`.
- Base B: `Qwen/Qwen2.5-7B-Instruct`, revision `<sha>`, tokenizer.model sha
  `<sha>`, chat_template hash `<hash>`.

## Open questions
1. Legal has not weighed in on Llama-3 Community for the imminent revamp of
   the redistribution-intent clause in the Acme product agreement. Ping
   J. Yates by 2026-08-05.
2. Qwen tokenizer coverage on Acme's product-specific vocabulary (SKU codes,
   ticket IDs). Small sample tokenization check by 2026-08-04.
```

The scorecard is a page. It is the artifact that mod-111 exercise 01 asks you to build. Every downstream decision cites it.

## Weight-setting is a judgment, not a formula

The rubric's per-axis weights are a *statement of what your organization values*. Two organizations can score the same candidate list against the same axes and reach different verdicts because their weights differ.

Some sane defaults:

- **License** is 3× if you distribute derivative weights externally, 2× if you serve behind an API, 1× if the work is internal-only.
- **Capability floor** is 3× when the target metric is far above zero-shot; 1× if the base is already 90% of the way there.
- **Cost** is 1× when candidates are in the same class; step up to 3× when comparing across size tiers (7B vs. 70B) or across managed vs. self-hosted (mod-110 chapter 6 territory).
- **Deprecation risk** is 3× for a model whose life must exceed 18 months; 1× for an experimental short-lived project.

Set weights *before* scoring. Doing it in the other order — score, then tune weights until the preferred base wins — is base-preference laundering.

## When a hard constraint overrides the rubric

The rubric produces a weighted score. A hard constraint produces a boolean. Two examples:

- **License:** if the redistribution intent is "commercial redistribution of the fine-tuned weights" and a candidate's license does not permit it, the candidate is out — no weighting will save it. Score 0 on axis 1 is a blocker for that intent.
- **Regulatory:** if the deployment is in a jurisdiction that requires model-provenance disclosure and one candidate has undisclosed pre-training data provenance, the candidate is out for the regulated tenant. Not a rubric issue; a compliance floor.
- **Vendor exclusion:** if the platform team has cleared Bedrock but not OpenAI direct for the tenant, the OpenAI-direct managed candidate collapses to Bedrock-hosted only.

Record every override *with its trigger* in the scorecard's `Override discussion` section. When the trigger changes — the customer signs the new DPA, the region opens up, the license clarifies — the override is revisited automatically, not forgotten.

## The tie-break: incumbency

Two close candidates almost always tie in practice. The tie-break is usually incumbency: **the current base is the base you know, and the switching cost is real.** A migration from Llama-3.1-8B to Qwen2.5-7B is:

- Chat template diff (mod-102 ch3): all your SFT JSONL needs re-rendered.
- Tokenizer diff: your token-length estimates and the KV-cache math (mod-109 ch7) change.
- Eval-set re-baselining: every zero-shot number in your gate report changes.
- Recipe drift: the community recipes for the old base do not directly transfer.

Give incumbency 1–2 points on the tiebreaker unless the rubric says the new base is clearly ahead. "Clearly ahead" is *outside the overlap of 95% bootstrap CIs on the task-specific held-out set*; not "1 point higher on the spreadsheet."

## Managed bases on the same rubric

Every axis above applies to managed bases too, with two adjustments:

- **Axis 3 (size / hardware).** For managed, the axis becomes "cost per training token + cost per inference token + hosting rent." You are not sizing GPUs; you are sizing the vendor bill.
- **Axis 6 (method compatibility).** The managed method surface is narrower (mod-110 ch3): SFT is universal, DPO is available on some (OpenAI chat, some Anthropic paths), classical RLHF is rare, Reinforcement Fine-Tuning is O(o-series). Score honestly.

The rest — license, capability floor, tokenizer, chat template, community, deprecation, cost — is the same rubric. This is why chapter 4 can defer to *this* rubric when arguing fine-tune-vs-alternative: whether the fine-tune is on your GPUs or on OpenAI's, the base-selection artifact has the same shape.

## The base ADR

The scorecard is one artifact. The ADR — Architecture Decision Record — is another. The two are complementary: the scorecard is quantitative; the ADR is the durable *decision* with the *rationale* attached, so the decision survives its author.

Minimum ADR template:

```markdown
# ADR — Base model for Acme Support Assistant v3

## Context
Acme Support Assistant v2.4 is on Llama-3.1-8B-Instruct. We are proposing a v3
release (see proposal v0.2, 2026-08-01) targeting an escalation-rate reduction
on billing tickets. Base selection is due before SFT kicks off.

## Decision
Fine-tune from **Qwen2.5-7B-Instruct** (revision `<sha>`), migrating off
Llama-3.1-8B-Instruct.

## Options considered
1. Qwen2.5-7B-Instruct (chosen). Rubric score 44/44.
2. Llama-3.1-8B-Instruct (incumbent). Rubric score 41/44.
3. Mistral-7B-Instruct-v0.3. Rubric score 35/44.

## Rationale
- License: Qwen's Apache-2 posture reduces future risk if Acme changes
  redistribution intent (proposal v0.2 does not require redistribution today,
  but legal flagged that likelihood is non-trivial in 2027).
- Capability floor: within 2 pp of Llama on zero-shot task-specific held-out;
  not a distinguishing factor.
- Ecosystem: comparable to Llama for our objective stack.
- Deprecation: comparable.
- Migration cost: ~1 week of chat-template / eval-baseline rebase, absorbed
  into proposal's 7-week timeline.

## Consequences
- Chat template diff means all SFT JSONL must be re-rendered against Qwen
  template. Existing conversion scripts (from mod-102) generalize; small edit.
- Eval baselines re-run against Qwen zero-shot before the gate can be
  interpreted comparably to v2.4.
- v2.4 rollback path (mod-108 ch6 gate) still points at Llama-8B; the registry
  entries stay live even after we ship Qwen-based v3.

## Escape hatches
If the migration cost balloons past 1.5 weeks, or if Qwen zero-shot on the
task-specific held-out set is > 2 pp below Llama at the half-budget mark,
revert to Llama-3.1-8B for v3. This override does not require another ADR —
it is pre-approved here.

## Reviewers and sign-off
- L. Ortega (fine-tuning, owner) — 2026-08-01
- J. Park (peer FT eng) — 2026-08-01
- S. Idris (safety) — 2026-08-01
- N. Rao (platform) — 2026-08-01
- J. Yates (legal) — pending, blocker for merge
```

The ADR ships in Git alongside the model card and the scorecard. Six months later, when someone asks "why Qwen?", the answer is one file, one commit, three reviewers.

## Common mistakes on this axis

- **Picking the base first, then justifying.** Rubric done last is not a rubric. Write the axes and weights *before* naming candidates.
- **Confusing serving size with training size.** Fine-tune a 70B with QLoRA (mod-104 ch2), quantize to 4-bit for serving (mod-109 ch3), fit in the serving budget. Do not disqualify a base because full-parameter would not fit.
- **Overweighting benchmark leadership.** A one-week-old release with a headline MMLU number and no ecosystem is worse than a 6-month-old release with 100 Axolotl configs and a mature vLLM adapter. Axis 7 exists for a reason.
- **Ignoring tokenizer for non-English.** English is well-tokenized by every major base; other languages are not. If the product is Japanese-first, the tokenizer axis is worth its weight several times over.
- **Not pinning revisions.** "Llama-3.1-8B-Instruct" is not a pin. `meta-llama/Llama-3.1-8B-Instruct` at revision `<sha>` with `tokenizer.model` sha `<sha>` is a pin. Everything downstream should reference the pin, not the pretty name.

## The base-selection loop across releases

Base selection is not a one-time decision even for a single product. Every 6–12 months a new frontier of open models lands (Llama-4, Qwen3, Mistral-Nemo successors, DeepSeek releases) or a managed vendor deprecates an existing base. Re-run the rubric on a cadence:

- **Per release.** Cheap: re-score the incumbent against the rubric and note any degradation on axes 8 (deprecation) or 6 (method compat, if the objective stack changed).
- **Per model-family generation.** More expensive: re-score against a full candidate list. Trigger this when a new base with a ≥ 1-generation capability improvement is published, or when a hard constraint (license posture, regulator, region) shifts.
- **On explicit trigger.** A deprecation notice on the incumbent base; a compliance change; a cost renegotiation with the managed vendor; a security incident in the base.

The re-scoring is the same document, versioned. `base_scorecard.v3.md`, `base_scorecard.v4.md`. Each version's ADR chain records what changed. This is how "which base?" stops being an argument and becomes a maintained artifact.

## Summary

- Base selection is a rubric, not a preference. Nine axes: license, capability floor, size / hardware ceiling, tokenizer, chat-template maturity, method compatibility, community / ecosystem, deprecation risk, and cost. Score each candidate 0–2 with a weight per axis, and produce a total that survives review.
- Every axis has failure signals — bloated tokenization, missing chat-template roles, ✘ on a required method — that force the score without requiring taste.
- Weights are set *before* scoring. Setting them after is base-preference laundering.
- A hard constraint (license blocker, regulatory floor, vendor exclusion) overrides the score. Every override is recorded with its trigger so it is revisited when the trigger moves.
- Incumbency is the tie-breaker. Migration cost is real (chat template, tokenizer, eval-baseline, recipe drift); give the incumbent 1–2 points unless the challenger is clearly outside the CI overlap.
- Managed bases (mod-110) score on the same rubric with two adjustments — axis 3 becomes vendor cost, axis 6 becomes the narrower managed method surface. This shared rubric is why chapter 4's fine-tune-vs-managed ADR can be defended in the same vocabulary.
- The scorecard and the ADR are separate artifacts: scorecard is quantitative, ADR is durable rationale with escape hatches. Both ship in Git alongside the model card.
- Re-run the rubric on a cadence — per release, per generation, on explicit trigger. Version the scorecard; the ADR chain records what changed.
