# ORPO, KTO, IPO, SimPO, GRPO: The Modern Preference Family

**Estimated read:** 55 minutes.
**You should be able to, by the end:** describe what each of ORPO, KTO, IPO, SimPO, and GRPO changes relative to DPO or PPO, know which data shapes and memory constraints each is designed for, and pick a method for a specific scenario using a decision matrix rather than a paper-of-the-week reflex.

## What each variant fixes

DPO (chapter 5) has four features that different downstream teams found either awkward or wrong:

1. **It requires paired data** — chosen *and* rejected for the same prompt.
2. **It keeps a reference model in memory** — even the "disabled adapter" trick still holds base weights that participate in reference computation.
3. **It can overfit** — the Bradley-Terry-shaped loss saturates and pushes the implicit reward margins beyond what the underlying preference data supports.
4. **It never sees on-policy samples** — the policy is trained purely on fixed pairs, not on what it currently generates.

Each of the modern variants targets one or more of these:

| Method | Reference-free? | Paired data? | On-policy? | Distinctive fix |
|---|---|---|---|---|
| **IPO** | No (uses reference) | Yes | No | Squared-loss regularization; fights DPO overfitting |
| **KTO** | No (uses reference) | **No — unpaired** | No | Kahneman-Tversky utility; single-label per example |
| **ORPO** | **Yes — no reference** | Yes | No | SFT loss + log-odds penalty in one stage |
| **SimPO** | **Yes — no reference** | Yes | No | Length-normalized margin; explicit reward margin γ |
| **GRPO** | Depends on setup | No — verifiable/RM reward | **Yes** | Group-relative advantages; no value model |
| **CPO** | Depends on setup | Yes | No | Contrastive preference optimization; hybrid with SFT |

CPO shows up briefly at the end; the primary five are ORPO, KTO, IPO, SimPO, GRPO.

## IPO — Identity Preference Optimization

Azar et al., 2023 ("A General Theoretical Paradigm to Understand Learning from Human Preferences"). The premise: DPO's Bradley-Terry-shaped loss lets the implicit reward margin grow without bound, which pushes overfitting on the training pairs.

The IPO fix: replace the log-sigmoid loss with a **squared loss** on the implicit reward margin, targeting a fixed value (typically `1 / (2β)`).

```
L_IPO = E[ ( β · [Δ_w − Δ_l] − 1/(2β) )^2 ]
```

Where `Δ = log(π_θ / π_ref)` as in DPO. The intuition: the loss reaches zero when the margin hits the target, and *stops* pulling the policy further. DPO's log-sigmoid keeps pulling forever (asymptotically).

**When to use IPO:** when your DPO run visibly overfits (eval loss reverses within one epoch, implicit reward margins keep growing but eval performance drops). IPO is a drop-in replacement — same data format, same trainer scaffolding, different loss.

**TRL implementation:** set `loss_type="ipo"` in `DPOConfig`. `β` interpretation shifts slightly — the target margin is `1/(2β)` so smaller β means larger target margin (opposite direction from DPO's β).

**Trade-off:** IPO's regularization can *under*-optimize on datasets where DPO would have succeeded. It is the "safer" loss but not strictly better; try DPO first, drop to IPO when the overfit signal shows up.

## KTO — Kahneman-Tversky Optimization

Ethayarajh et al., 2024 ("KTO: Model Alignment as Prospect Theoretic Optimization"). The premise: paired preference data is expensive and awkward. Real-world signals are often *unpaired*: "user thumbs-up on this response" or "this response was flagged" — one label per response, not two.

The KTO fix: **train on unpaired data** using an asymmetric loss inspired by Kahneman and Tversky's prospect theory (humans weight losses more than gains). Each example is `(x, y, label ∈ {desirable, undesirable})`, and the loss for a desirable example is roughly

```
L_KTO_desirable = 1 − σ( β · [Δ − reference_KL] )      (schematically)
```

and for an undesirable example

```
L_KTO_undesirable = 1 − σ( β · [reference_KL − Δ] )    (schematically)
```

where `reference_KL` is an estimate of the average KL between policy and reference on *other* examples in the batch — the paper's mechanism for anchoring the loss without a paired counterpart. The paper describes the exact terms; the working intuition is: desirable examples pull the policy up; undesirable examples push it down; the anchor keeps the whole thing from drifting.

**When to use KTO:**

- You have unpaired signals (thumbs-up/down data from a deployed product, safety-flag data).
- You have paired data but the pairs are unbalanced (many more chosen than rejected, or vice versa).
- You want to combine multiple data sources with different structures — KTO handles a mix of pure-chosen, pure-rejected, and paired.

**When to be careful:**

- KTO's asymmetric loss has a beta-and-desirable-weight interaction that is more sensitive than DPO's β alone. The paper recommends `desirable_weight ≈ 1` and `undesirable_weight ≈ (fraction of desirable examples) / (fraction of undesirable examples)` to balance the gradient signal.
- Very small unpaired datasets (< 1k) are unstable; KTO benefits from scale.

**TRL implementation:** `KTOTrainer` and `KTOConfig`. Dataset shape is `{prompt, completion, label: bool}` per row.

## ORPO — Odds Ratio Preference Optimization

Hong et al., 2024 ("ORPO: Monolithic Preference Optimization without Reference Model"). The premise: DPO's two-stage pipeline (SFT then DPO) is operationally awkward — you hold two model copies, iterate through two training runs, and manage two dataset formats. Can preference tuning happen *inside* SFT?

The ORPO fix: **combine an SFT loss with an odds-ratio preference penalty in a single training run, with no reference model.**

```
L_ORPO = L_SFT(y_w | x) + λ · L_OR(y_w, y_l | x)
```

Where:

- `L_SFT(y_w | x)` is standard cross-entropy on the chosen response — the same loss you use in SFT (mod-103).
- `L_OR(y_w, y_l | x) = −log σ( log[odds(y_w|x) / odds(y_l|x)] )` where `odds(y|x) = P(y|x) / (1 − P(y|x))`.
- `λ` weights the preference term. Paper defaults around 0.1–1.0.

Read this: the SFT term teaches the model to imitate `y_w` (as in mod-103). The odds-ratio term additionally teaches "the odds of generating `y_w` should exceed the odds of generating `y_l`." No reference model — the odds are computed from `π_θ` alone. The chosen response gets an SFT-style gradient and a preference-style gradient in the same step.

**When to use ORPO:**

- You want a single-stage training pipeline. ORPO is DPO+SFT in one loop; you skip the SFT stage and train directly to a preference-tuned model from the base.
- You are memory-constrained beyond what QLoRA-DPO buys you (no reference model → one fewer copy of base weights on GPU).
- You have paired data that already includes what SFT would want (chosen is high-quality enough to also be an SFT target).

**When to be careful:**

- The absence of a reference model means the KL anchor is gone. Empirically the SFT term acts as a soft anchor (the model is pulled toward chosen at the same time), but there is no explicit KL constraint. In practice ORPO models can drift in ways DPO models do not.
- Combining SFT and preference gradients means the loss is composite; tuning `λ` matters. The paper's β/λ recommendation is worth following closely.

**TRL implementation:** `ORPOTrainer` and `ORPOConfig`. Same paired-data format as DPO.

## SimPO — Simple Preference Optimization

Meng et al., 2024 ("SimPO: Simple Preference Optimization with a Reference-Free Reward"). The premise: DPO's implicit reward `β · log(π_θ / π_ref)` includes the reference model — an extra forward pass per pair — and has no length normalization, so it inherits DPO's mild length bias.

The SimPO fix: use the **average log-probability of the response** as the implicit reward, with an added margin γ:

```
r_SimPO(x, y) = (β / |y|) · Σ_t log π_θ(y_t | x, y_{<t})

L_SimPO = −log σ( r_SimPO(x, y_w) − r_SimPO(x, y_l) − γ )
```

Two changes from DPO:

- **Length normalization:** divide the log-prob by response length `|y|`. Longer responses no longer automatically have larger magnitude log-probs.
- **Explicit margin γ:** subtract a target margin from the implicit reward difference. Trains the model to keep chosen a *specific* amount above rejected, not just any amount.
- **No reference model:** the implicit reward is defined on `π_θ` alone.

**When to use SimPO:**

- You want the memory savings of reference-free (like ORPO, no second model copy).
- You are worried about length bias in your preference set (the length normalization directly attacks it).
- Your task is short-response and you want to avoid the "chosen just has more tokens" gaming.

**When to be careful:**

- Length normalization *can* over-correct on tasks where longer responses are genuinely better. Watch response lengths on the eval side; if the trained model produces suspiciously short responses, γ is too aggressive.
- No reference model means no KL anchor; the same drift risk as ORPO applies.
- SimPO is newer than DPO — less experience across the community; expect more surprises in edge cases.

**TRL implementation:** `loss_type="simpo"` in `DPOConfig` (recent TRL versions). Also standalone in Meng et al.'s reference implementation. Two SimPO-specific args: `simpo_gamma` (the margin) and no `ref_model` needed.

## GRPO — Group Relative Policy Optimization

Shao et al., 2024 ("DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"). The premise: PPO's value model is a big engineering cost — half of your memory, half of your bugs. On problems where a rule-based reward is available (math, code, structured output), can you skip the value model *and* the reward model?

The GRPO fix: **sample K responses per prompt**, compute a reward per response (from a verifier, a rule, or an RM), and compute **group-relative advantages** by normalizing the K rewards within the group:

```
A_i = (R_i − mean(R_1..K)) / std(R_1..K)
```

Then run a PPO-style clipped update using `A_i` as the advantage. No value model — the group of K responses is your baseline. The KL penalty against `π_ref` remains (KL-constrained objective from chapter 1).

Compared to PPO:

- **No value model.** Saves ~14 GB for a 7B run, plus eliminates value-related bugs.
- **On-policy with a group of samples per prompt.** Explores response variety within the same prompt.
- **Reward source is arbitrary.** A rule ("was the final answer correct?"), an executable check, an RM, or a hybrid. RLVR (Reinforcement Learning with Verifiable Rewards) most commonly pairs with GRPO.
- **Trained without a critic** — the group provides the baseline directly. Also no need for GAE or advantage bootstrapping.

**When to use GRPO:**

- You have a verifiable reward signal — math correctness, code execution, JSON validity, tool-use success. GRPO is the modern default for reasoning-style RL. DeepSeek-R1 and Qwen-QwQ were trained with GRPO variants.
- You want on-policy training but cannot afford PPO's full compute.
- You want to explore per-prompt variety.

**When to be careful:**

- GRPO requires generating K samples per prompt every step. K = 4 to 16 is typical; K = 4 is a floor. Rollout cost is K× a normal PPO rollout.
- With no value model, if the group is uniformly bad (all K responses fail the verifier), you have zero variance and no learning signal on that prompt. Filter or resample.
- KL penalty tuning still matters — the "no reward model" version does not exempt you from KL collapse (chapter 7).

**TRL implementation:** `GRPOTrainer` and `GRPOConfig`, added in recent TRL versions (~0.14+). Pass a `reward_funcs` callable that takes `(prompts, completions) → rewards`. The trainer does the group sampling and normalization.

<!-- needs-research: pin the exact TRL version that stabilizes GRPOTrainer's API. It was introduced in the 0.14 series but its argument surface has evolved. Confirm before quoting API names in exercise-05. -->

## CPO — Contrastive Preference Optimization

Xu et al., 2024 ("Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation"). A DPO variant tuned initially for machine translation but useful more broadly. The idea: combine the log-sigmoid DPO loss with a *cross-entropy* term on the chosen side (like ORPO), but keep the reference model.

```
L_CPO = L_DPO(π_θ; π_ref) + λ · L_SFT(y_w | x)
```

Effectively DPO + SFT auxiliary loss, with a reference model still present. Sits between DPO and ORPO in the design space.

**When to use CPO:**

- Your DPO run shows log-prob deflation (chapter 5's "everything drops" failure). CPO's SFT auxiliary term is a direct fix.
- You want the SFT-side anchor without giving up the reference model.

TRL supports it via `CPOTrainer` / `CPOConfig`.

## The decision matrix

The right way to pick is to be honest about your situation on five axes:

1. **Data shape.** Paired (chosen+rejected per prompt) vs. unpaired (label per response) vs. verifiable (reward function on the response).
2. **Reference model available.** Yes (any SFT you have on hand) vs. no (memory constraint or single-stage requirement).
3. **On-policy required.** Yes (agent tasks, verifiable rewards, iterative RM refresh) vs. no.
4. **Memory budget.** Very tight (single 24 GB GPU) vs. medium (single 80 GB) vs. abundant (multi-GPU).
5. **Overfit risk observed.** Have you already seen DPO overfit on this data? Have you seen log-prob deflation?

The matrix:

| Situation | Recommended | Second choice | Notes |
|---|---|---|---|
| SFT+paired data, medium memory, first run | **DPO** | ORPO | Most stable default; enormous community experience |
| DPO overfits within 1 epoch | **IPO** | DPO with higher β | IPO's regularization is the direct fix |
| DPO shows log-prob deflation | **CPO** | DPO + SFT-loss aux term | Auxiliary SFT keeps chosen anchored |
| Unpaired thumbs-up/down data | **KTO** | Convert to pairs (loses signal) | KTO exists exactly for this |
| Very tight memory, want single-stage pipeline | **ORPO** | SimPO | Skip separate SFT stage entirely |
| Length bias observed in eval | **SimPO** | DPO with length-filtered data | Length normalization is a built-in fix |
| Verifiable reward (math / code / structured) | **GRPO** | PPO with the same RM | Modern RLVR default; DeepSeek / Qwen-style |
| Multi-turn agent task | **PPO** or **GRPO** | Iterative DPO with refreshed pairs | On-policy essential |
| Frontier-scale compute + safety focus | **PPO-RLHF** with iterative RM | DPO with expert refresh | The Llama-2 / Anthropic-style pipeline |
| Distillation from a stronger teacher | **DPO** or **KTO** | Simple SFT if pairs are noisy | Zephyr's recipe pattern |

Two meta-observations:

- **DPO remains the honest default for chat-style SFT preference tuning.** ORPO, IPO, SimPO all exist because they beat DPO in *specific* scenarios; none has replaced it as the general default.
- **GRPO is displacing PPO for verifiable-reward RL** at every scale where teams have measured the two. If your reward is a rule / verifier, start with GRPO.

## A parallel implementation sketch

To make the "these are all trainer subclasses of the same loop" claim concrete — a psuedo-code of what each trainer does per batch, side by side:

**DPO**

```
for (x, y_w, y_l) in batch:
    logp_w = log_pi_theta(y_w | x); logp_w_ref = log_pi_ref(y_w | x)
    logp_l = log_pi_theta(y_l | x); logp_l_ref = log_pi_ref(y_l | x)
    loss  += -log_sigmoid(beta * ((logp_w - logp_w_ref) - (logp_l - logp_l_ref)))
```

**IPO** — same as DPO, replace last line:

```
    loss  += (beta * ((logp_w - logp_w_ref) - (logp_l - logp_l_ref)) - 1/(2*beta))**2
```

**ORPO**

```
for (x, y_w, y_l) in batch:
    logp_w = log_pi_theta(y_w | x); logp_l = log_pi_theta(y_l | x)
    loss += -logp_w                                                  # SFT term on chosen
    log_odds_w = logp_w - log(1 - exp(logp_w))
    log_odds_l = logp_l - log(1 - exp(logp_l))
    loss += -lambda * log_sigmoid(log_odds_w - log_odds_l)           # OR term
```

**SimPO**

```
for (x, y_w, y_l) in batch:
    r_w = (beta / len(y_w)) * sum(log_pi_theta(y_w_t | x, y_w_{<t}))
    r_l = (beta / len(y_l)) * sum(log_pi_theta(y_l_t | x, y_l_{<t}))
    loss += -log_sigmoid(r_w - r_l - gamma)
```

**KTO** (schematic; details in the paper)

```
for (x, y, label) in batch:
    logp = log_pi_theta(y | x); logp_ref = log_pi_ref(y | x)
    kl_ref = estimate_batch_kl()
    delta = beta * (logp - logp_ref - kl_ref)
    if label == "desirable":
        loss += 1 - sigmoid(delta)
    else:
        loss += 1 - sigmoid(-delta)
```

**GRPO** — sample-then-update:

```
for x in batch:
    ys = [sample_from_pi_theta(x) for _ in range(K)]
    rewards = reward_fn(x, ys)          # verifier, RM, rule
    advantages = (rewards - mean(rewards)) / std(rewards)
    for y_i, A_i in zip(ys, advantages):
        ratio = pi_theta(y_i|x) / pi_theta_old(y_i|x)
        loss += -min(ratio * A_i, clip(ratio, 1-eps, 1+eps) * A_i)
        loss += beta * kl(pi_theta || pi_ref)                        # KL anchor
```

The core loop shape — forward the policy on candidates, compute a loss, backprop — is identical across DPO / IPO / ORPO / SimPO / KTO. GRPO adds a sampling step; PPO (chapter 4) adds a full value/advantage estimation pipeline. That is why swapping methods in TRL is a `loss_type=` argument for most of the DPO-family and a different trainer class for the sampling-based methods.

## Empirical picture across 2024–2025 papers

An honest summary of what the current literature says (subject to what the next arxiv drop will say next month):

- **DPO** wins on most SFT-shaped chat preference-tuning benchmarks at 7B–70B scale when compared to PPO on the same preference data, at equal compute budgets. This is the finding that made DPO the default.
- **IPO** narrows the gap between DPO and full-parameter SFT on tasks where DPO overfits; not universally stronger than DPO.
- **KTO** matches DPO on paired data and dramatically outperforms it on unpaired data (which is where DPO would need to fabricate pairs). Extends the reachable data.
- **ORPO** claims to match DPO+SFT quality with half the compute; independent reproductions are mixed. Best characterized as "DPO+SFT combined at a small quality cost."
- **SimPO** is roughly on par with DPO on chat benchmarks and specifically better on length-controlled evals. Reference-free is real memory savings.
- **GRPO** is the current SOTA for RLVR (reasoning, math, code, tool-use) — DeepSeek-R1, Qwen-QwQ, and downstream re-implementations all use it. Not a chat-preference method in the DPO sense; different problem shape.
- **RLHF via PPO** is still used at the frontier, especially for safety-critical alignment where iterative RM refresh matters (Anthropic, Llama-2). Rarely the right default for a mid-size team.

No universal winner. Expect the frontier to keep moving — the honest position for a curriculum is "know the family, run a small comparison on your data, decide."

## Exercise pointer

`exercises/exercise-05-orpo-kto-simpo-grpo-survey-experiment.md` has you pick three methods from this family, run each on the same base, and produce a comparison matrix plus a defended recommendation for a specified scenario.

## Summary

- Each modern variant fixes a specific weakness of DPO or PPO: IPO regularizes DPO's overfit, KTO handles unpaired data, ORPO merges SFT and preference into one stage, SimPO drops the reference and adds length normalization, GRPO drops the value model for verifiable-reward RL.
- The core loops are all variants of the same shape — DPO / IPO / ORPO / SimPO / KTO differ only in the loss formula; GRPO adds a per-prompt K-sample rollout and a group-relative advantage.
- Decision axes: paired vs. unpaired data, reference model available or not, on-policy required or not, memory budget, and observed failure modes on the current data.
- Sensible starting points: **DPO** for SFT+paired data; **IPO** if DPO overfits; **KTO** for unpaired signals; **ORPO** / **SimPO** for memory-constrained reference-free runs; **GRPO** for verifiable-reward RL.
- The current empirical picture: DPO is the honest default for chat; GRPO owns verifiable-reward tasks; the others earn their keep in specific niches. Expect the frontier to move; know the family map so you can adapt.
