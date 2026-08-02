# From SFT to Preferences

**Estimated read:** 45 minutes.
**You should be able to, by the end:** state precisely what SFT does not give you that preference optimization does, write the Bradley-Terry model for pairwise preferences and derive the InstructGPT-style KL-constrained RL objective from it, and place PPO-RLHF, DPO, ORPO, KTO, IPO, SimPO, and GRPO on a single family map — knowing which one drops which piece of the RLHF pipeline and why.

## Why SFT is not enough

Mod-103 turned a base model into an instruction-follower with supervised fine-tuning: you gave it `(prompt, chosen_response)` pairs, minimized cross-entropy against `chosen_response`, and the model started imitating that distribution. That gets you a model that can *complete* an instruction template and *produce* responses that look like the training set.

What SFT does not give you:

- **A signal for what is *better* between two plausible responses.** Cross-entropy against one response says "this is the right answer"; it does not say "this response is better than that response." If both responses in your dataset were correct but one was clearly better, SFT trained on either one loses that signal.
- **A way to teach behavior that is defined *relative* to alternatives.** "Be more concise than the last time you answered this kind of question" is a preference over trajectories, not a target string. SFT cannot express it.
- **A gradient signal on responses your model actually generates.** SFT trains against a *fixed* dataset. If the model's rollouts drift into behaviors the dataset never demonstrated (over-hedged, over-formatted, subtly evasive), SFT has no way to push them back — the training examples do not cover the drift.

The InstructGPT paper (Ouyang et al., 2022) named this gap directly: their SFT stage produced a model that was "helpful in expectation" but had a long tail of over-cautious, verbose, or subtly wrong responses that a human annotator could easily *identify as worse* when shown alongside a better alternative. Their fix — reward modeling on pairwise preferences plus RL against that reward — is where the whole preference-optimization literature starts.

The abstract shape of the problem: you have a policy π (your SFT model) that produces responses `y` for prompts `x`, and you have some way to say "this response is better than that one." You want to shift π toward the better-labelled responses without collapsing into an arbitrarily-narrow distribution. Everything in this module is a different answer to *that* problem.

## The Bradley-Terry model

Almost every preference-optimization method in production use assumes a **Bradley-Terry model** of pairwise comparisons. Bradley and Terry (1952) — the paper is about tournament rankings — proposed that if item `i` has a latent "strength" `r_i`, then the probability of `i` beating `j` in a comparison is

```
P(i > j) = exp(r_i) / (exp(r_i) + exp(r_j)) = σ(r_i - r_j)
```

where σ is the logistic sigmoid. This is a one-parameter-per-item logistic model. The key property: **only the *differences* between rewards matter** — adding a constant to every `r_i` leaves every pairwise probability unchanged. Reward is defined up to an additive constant.

Applied to language: `y` is a full response, and there is some latent reward function `r*(x, y)` that assigns a scalar to each `(prompt, response)`. Given a pairwise preference `y_w > y_l` (annotator picked `y_w` as "chosen" over `y_l` as "rejected") for a prompt `x`, the model says

```
P(y_w > y_l | x) = σ(r*(x, y_w) - r*(x, y_l))
```

If you had `r*` — a magic oracle — you could just generate a bunch of candidates, score each with `r*`, and pick the best. You do not have `r*`. Preference optimization is about what to do instead.

Two families of moves fall out of the Bradley-Terry setup:

1. **Learn an approximation `r_φ ≈ r*` from the preference data** (this is what reward modeling does — chapter 3), then optimize the policy against `r_φ`. Classical RLHF.
2. **Skip `r_φ` entirely by writing the policy objective directly in terms of the preference probability.** DPO and the whole "reference-free" family — chapters 5 and 6.

The Bradley-Terry model is the load-bearing assumption behind both. Where it fails — cyclic preferences, ties, three-way comparisons, non-comparable dimensions — the whole edifice wobbles. Chapter 6 mentions IPO's motivation explicitly here: the paper argues that treating Bradley-Terry as literally true leads DPO to overfit in a way IPO's more forgiving objective avoids.

## The KL-constrained RL objective

If you had `r*`, the naive move would be: pick π that maximizes `E_{x, y ~ π}[r*(x, y)]`. This is exactly the reward-hacking wormhole. π will drift arbitrarily far from a sensible language model in pursuit of high reward, generating text that scores well but no longer reads as English, or gaming any imperfection in `r_φ`.

The RLHF fix, from Christiano et al. (2017) and stabilized in InstructGPT, is to add a **KL regularization** to a reference policy — typically the SFT model — and maximize

```
J(π) = E_{x ~ D, y ~ π(·|x)}[ r*(x, y) ] − β · KL(π(·|x) || π_ref(·|x))
```

Two things are happening here:

- The **reward term** pulls π toward high-scoring responses.
- The **KL term** pushes π back toward `π_ref`. `β > 0` is the trade-off knob: high β means "do not drift far from `π_ref`"; low β means "chase reward more aggressively."

InstructGPT and every downstream RLHF pipeline (Anthropic's Constitutional AI, Llama-2's RLHF, GPT-4's RLHF phase) run PPO — Proximal Policy Optimization, Schulman et al., 2017 — against this KL-penalized reward. Chapter 4 walks the PPO training loop in TRL. The high-level shape you need before then:

1. **SFT model** — the reference policy `π_ref`. Frozen.
2. **Reward model** — `r_φ`, trained on preference pairs (chapter 3). Frozen during RL.
3. **Policy model** — π, initialized from the SFT model. Trained.
4. **Value model** — an auxiliary head that estimates `V(x)` for the PPO advantage estimate. Also trained (often shares weights with the policy in TRL).

The training loop is: sample `y ~ π(·|x)`, score it with `r_φ`, add the KL penalty, compute PPO gradients, update π. Chapter 4 is where the details live.

The load-bearing fact for the rest of this chapter: **KL to a reference policy is not an implementation detail. It is what keeps preference-optimized language models legible.** Every method in this module either includes an explicit KL term (PPO, DPO with `β`, IPO with `β`) or has a design reason for why it can drop it (ORPO's log-odds regularizer, SimPO's length-normalized margin, KTO's paired-signal structure). The choice of KL term is the choice.

## The RLHF pipeline, in one diagram

```
Pretrained base ─ mod-101/102/103 ─▶ SFT model  ─┐
                                                 ├─▶ Reference policy π_ref (frozen)
                                                 │
                                                 └─▶ Policy π (init from SFT, trained)

Preference pairs ─ chapter 2 ─▶ Reward model r_φ ── chapter 3

                    ┌────────── PPO loop (chapter 4) ────────────┐
                    │                                            │
        for each prompt x:                                       │
            sample y ~ π(·|x)                                    │
            reward = r_φ(x, y) − β · log[π(y|x) / π_ref(y|x)]   │
            take PPO step against reward                         │
                    │                                            │
                    └────────────────────────────────────────────┘
                                        │
                                        ▼
                            Preference-tuned policy π*
```

The four boxes on the left of the loop — SFT model, reference policy, policy, reward model — are what makes RLHF operationally expensive. You are holding four models' worth of weights, one of which (the policy) is being trained. Two of them (the SFT/reference and the reward) generate scores at every step. Chapter 4 has the memory math.

## The DPO shortcut

Rafailov et al. (2023) — the DPO paper — proved a striking result: **the reward model and PPO training are not necessary.** Under the Bradley-Terry assumption and the KL-constrained RL objective, the optimal policy `π*` has a closed form in terms of `π_ref` and `r*`:

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r*(x, y) / β )
```

Rearranging, the *reward* is an affine function of the log-ratio between π* and π_ref:

```
r*(x, y) = β · log[π*(y|x) / π_ref(y|x)] + β · log Z(x)
```

Substitute *that* back into the Bradley-Terry preference likelihood, and the `log Z(x)` term cancels between the chosen and rejected response (both share the same prompt x). What is left is a loss that depends only on the *policy* π and the frozen `π_ref`, evaluated on the preference pairs directly:

```
L_DPO(π; π_ref) = −E_{(x, y_w, y_l)}[ log σ( β · log(π(y_w|x)/π_ref(y_w|x))
                                             − β · log(π(y_l|x)/π_ref(y_l|x)) ) ]
```

That is a supervised loss on preference pairs. **No reward model. No rollouts. No PPO.** The training loop is basically SFT with a paired objective and a frozen `π_ref` in memory. Chapter 5 derives this in detail and walks the `DPOTrainer` recipe.

DPO is not "PPO but easier." It is *provably* the same objective under Bradley-Terry, with the reward model marginalized out analytically. The two produce different empirical results only because (a) DPO does not require generating rollouts, so it never sees on-policy samples, and (b) the KL term in DPO is *implicit* (the log-ratio) instead of *explicit* (a penalty term), which changes the training dynamics in ways chapter 7 will spend time on.

Since DPO's introduction in mid-2023, most production preference tuning at 7B–70B scale has been DPO or a DPO relative. PPO-RLHF is still used at the frontier (Llama-2 and Llama-3 shipped with PPO stages; the OpenAI o-series shipped with RL against verifiable rewards), but for most teams DPO is the default and PPO is the "when I have a reason to reach for it" backup.

## The modern family

Once DPO opened the door, a family of variants followed. Each one drops or changes a piece of the DPO recipe. Chapter 6 covers them in detail; the map you should carry into that chapter:

| Method | Reference model? | Data shape | Distinctive move |
|---|---|---|---|
| **PPO-RLHF** | Yes | Preference pairs → RM → on-policy rollouts | Full RL loop with explicit KL penalty |
| **DPO** | Yes | Preference pairs | Analytical marginalization of the reward model |
| **IPO** | Yes | Preference pairs | Regularized DPO; MSE-style loss to fixed margin, less overfit-prone |
| **KTO** | Yes | *Unpaired* labels (chosen OR rejected per example) | Kahneman-Tversky utility; loosens the paired-data requirement |
| **ORPO** | **No** | Preference pairs | SFT loss + odds-ratio penalty in one stage, no separate reference model |
| **SimPO** | **No** | Preference pairs | Length-normalized log-probability margin, no reference model |
| **GRPO** | Yes | Prompts + verifiable / model-scored reward | On-policy sampling, group-relative advantages, no value model (drops PPO's critic) |

Two axes to keep track of when reading:

- **Do you keep a reference model?** PPO / DPO / IPO / KTO yes; ORPO / SimPO no. Reference-free methods save memory (one fewer copy of the base) and eliminate reference-drift as a failure mode (chapter 7) — but give up the theoretical connection to the KL-constrained RL objective.
- **Do you generate on-policy samples during training?** PPO / GRPO yes; DPO / IPO / KTO / ORPO / SimPO no. On-policy methods can chase reward more aggressively (they see what the policy actually generates) but pay for the rollouts and inherit RL's stability problems.

Nothing here supersedes DPO for general SFT-style preference tuning at time of writing. But the variants exist because DPO has real weaknesses: it overfits (IPO), it demands paired data (KTO), it keeps two model copies in memory (ORPO / SimPO), and it does not train on on-policy samples (GRPO). Pick the variant that fixes the weakness that is actually biting you.

## A worked cost comparison

For a 7B policy at BF16, held for a single-GPU run, rough memory numbers. Trained parameters are in **bold**:

| Method | Models loaded | Approx. weight memory |
|---|---|---|
| PPO-RLHF | **Policy** + reference + reward + **value head** | ~14 GB (policy) + 14 GB (ref) + 14 GB (RM) + ~1 GB (V head) ≈ 43 GB |
| DPO | **Policy** + reference | ~14 GB + 14 GB = 28 GB |
| DPO with QLoRA | **Policy (4-bit + LoRA)** + reference (shared frozen 4-bit base) | ~5 GB (single 4-bit base with LoRA on top; reference computed by disabling the adapter, per `peft`'s pattern) |
| ORPO | **Policy** only | ~14 GB |
| KTO | **Policy** + reference | ~28 GB (or ~5 GB with QLoRA) |
| SimPO | **Policy** only | ~14 GB |
| GRPO | **Policy** + reward + (no value model) | ~14 GB + 14 GB (RM or verifier) ≈ 28 GB, plus rollout memory |

Two things to notice:

- PPO-RLHF is by far the most memory-hungry. This is what motivated DPO in the first place and what motivates ORPO / SimPO on top of DPO.
- With QLoRA (chapter 2 of mod-104), DPO's memory footprint drops to a single 4-bit base plus a tiny LoRA adapter — the reference "model" is just the same 4-bit base with the adapter disabled. This is the recipe that made DPO tractable on 24 GB consumer GPUs.

The DPO + QLoRA + LoRA recipe (Zephyr-7B, later "alignment handbook" recipes) is the one preference-tuning setup that a mid-career engineer can run on their own hardware. Chapters 5 and 6 spend real time on it.

## What preference optimization does *not* do

Two clarifications that come up often enough to name upfront:

**Preference optimization is not a substitute for SFT.** Every method in this module assumes you start from a competent SFT model. If your model does not know how to follow instructions at all, running DPO on chat-style preference pairs will not teach it to — it will just push it slightly in the direction of the chosen responses versus the rejected ones, but both were assumed to be in-distribution for the SFT. Do SFT first (mod-103). Do preference optimization second.

**Preference optimization is not RLAIF-vs-RLHF.** The choice of *where the preferences come from* (human annotators = RLHF, LLM-as-judge = RLAIF, verifiable ground-truth answers = RLVR) is orthogonal to the choice of *how you optimize against them* (PPO / DPO / KTO / ...). RLAIF-vs-RLHF is a dataset question, covered in chapter 2. Preference optimization methods work identically on any of them.

## What comes next

- Chapter 2 covers where preference pairs *come from* and how to keep them clean.
- Chapter 3 turns preference pairs into a reward model — the piece PPO-RLHF depends on and DPO analytically avoids.
- Chapter 4 runs the full PPO-RLHF loop in TRL, with the four-model setup above.
- Chapter 5 walks the DPO derivation in detail and gets you to a working `DPOTrainer` run.
- Chapter 6 tours ORPO, KTO, IPO, SimPO, and GRPO and gives you a decision matrix.
- Chapter 7 catalogs the failure modes that show up across every method — reward hacking, length bias, KL collapse, reference drift.

## Summary

- SFT trains against a fixed distribution of "correct" responses. Preference optimization adds a *comparative* signal: "response A is better than response B."
- The Bradley-Terry model is the workhorse: `P(y_w > y_l | x) = σ(r*(x, y_w) − r*(x, y_l))`. Rewards are defined up to an additive constant. Every method in this module either learns a reward function or reasons directly about the preference probability.
- InstructGPT's KL-constrained RL objective — `E[r*] − β · KL(π || π_ref)` — is the theoretical anchor. PPO-RLHF optimizes it directly; DPO shows it has a closed-form solution that lets you skip the reward model and rollouts.
- The family map: **PPO-RLHF** (full RL loop) → **DPO** (analytical shortcut) → **IPO** (regularized DPO) → **KTO** (unpaired data) → **ORPO / SimPO** (drop the reference model) → **GRPO** (on-policy with group-relative advantages, no value model). Each variant fixes a specific weakness of the parent.
- DPO + QLoRA + LoRA is the recipe most 24-GB-GPU teams start with. PPO-RLHF is the frontier / heavy-lift option. The rest are for specific data or memory situations.
- KL to a reference model is load-bearing across the family. It is not a hyperparameter — it is *the* hyperparameter of preference optimization. Chapter 7 dedicates itself to what happens when it goes wrong.
