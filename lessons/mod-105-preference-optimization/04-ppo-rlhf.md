# Classical PPO-RLHF with TRL

**Estimated read:** 55 minutes.
**You should be able to, by the end:** wire an SFT model, a reference model, a reward model, and a value head into TRL's `PPOTrainer`, explain what each of them contributes at each step, read the run's KL / reward / value-loss diagnostics, and reason about the failure modes that make PPO-RLHF fussier in practice than SFT.

## The four models

Chapter 1 introduced the four models. Now that you have an RM (chapter 3) and are about to run PPO, pin down what each is doing:

| Model | Role | Weights | Called per step |
|---|---|---|---|
| **Policy** (`π`) | Generates responses; the thing you are training | Trained | Yes (rollout + gradient update) |
| **Reference** (`π_ref`) | The "before-RL" version; anchors KL penalty | Frozen | Yes (per-token log-prob computation) |
| **Reward** (`r_φ`) | Scores generated responses | Frozen | Yes (score rollouts) |
| **Value** (`V`) | Estimates expected return for PPO advantage | Trained | Yes (baseline for advantage estimation) |

Practical note: in most TRL setups the value head shares the backbone with the policy — TRL's `AutoModelForCausalLMWithValueHead` (or the newer PPOv2 variants) provides both — so you hold "policy + value" in one set of weights and "reference" and "reward" as two additional frozen copies. Three model-size chunks of GPU memory, four "conceptual models."

The DPO shortcut in chapter 5 exists precisely because this setup is expensive and finicky. PPO-RLHF is what you reach for when DPO does not work — or, at the frontier, when you have specific reasons to want on-policy training and can afford it.

## The PPO objective, adapted for language

PPO is not preference-optimization-specific; it is a general RL algorithm (Schulman et al., 2017). Adapted to language modeling:

- **Trajectory.** A trajectory is a prompt + a sampled response: `τ = (x, y_1, y_2, ..., y_T)`. Each generated token `y_t` is an "action" taken given the "state" `(x, y_{<t})`.
- **Reward.** In language RLHF, the environment is silent for every token except the last — the RM scores the *full* response once, so the reward `R(τ) = r_φ(x, y)` is a single scalar at the end.
- **KL penalty.** InstructGPT and TRL add a per-token KL penalty against `π_ref`: `r'_t = −β · [log π(y_t | x, y_{<t}) − log π_ref(y_t | x, y_{<t})]`. This penalty is subtracted from the terminal RM reward at the end of the trajectory (or credited per-token, depending on the implementation). It is the operational form of the KL-constrained objective from chapter 1.
- **Advantage.** PPO estimates the advantage `A_t` for each token using generalized advantage estimation (GAE, Schulman et al., 2015) with the trained value head as the baseline.
- **Clipped surrogate loss.** PPO's central move: the policy update is bounded so the ratio `r_t(θ) = π_θ(y_t | ·) / π_θ_old(y_t | ·)` cannot move too far in a single step. The loss is

    ```
    L_PPO(θ) = E_t[ min( r_t(θ) · A_t, clip(r_t(θ), 1 − ε, 1 + ε) · A_t ) ]
    ```

    with typical ε in [0.1, 0.3]. The clip prevents blow-ups when a large advantage would otherwise push the policy off a cliff. TRL sets it via `cliprange` and `cliprange_value`.
- **Value loss.** The value head is trained by MSE against the observed return: `L_V = (V(x, y_{<t}) − R_t)^2`. Also clipped, symmetrically.
- **Entropy bonus (optional).** Some PPO implementations add an entropy term to encourage exploration. TRL's PPO trainer supports it but it is typically small or off in RLHF.

Total loss per step:

```
L = L_PPO + c_V · L_V − c_ent · H(π)
```

The clip on the policy update, the KL penalty, and the value head are what make PPO "PPO" in the language-modeling context. Every one of them has a hyperparameter you will end up tuning.

## The training loop, mechanically

For each PPO iteration:

1. **Sample a batch of prompts** from your dataset.
2. **Rollout: generate responses** with the current policy. This is inference — no gradients, no backprop, just `model.generate(...)` on each prompt.
3. **Score responses with the reward model.** Get a scalar reward per response.
4. **Compute per-token log-probs** with both `π` and `π_ref`. Use them to compute the per-token KL penalty and to seed the ratio in the PPO surrogate loss.
5. **Compute advantages** using the value head and GAE.
6. **Take PPO gradient steps.** Multiple mini-epochs over the collected batch (typical: 4 mini-epochs; TRL's `ppo_epochs`).
7. **Log everything.** Loss, reward, KL, value loss, ratio bounds, response length.
8. **Repeat.**

The wall-clock breakdown is heavily skewed toward step 2 (generation) — a single PPO iteration can spend 60–80% of its time in rollout at typical sequence lengths. This is why PPO-RLHF is slow relative to DPO on identical hardware: DPO reuses fixed responses, PPO generates fresh ones every iteration.

## Setting up `PPOTrainer` (current TRL)

TRL's PPO trainer has been re-worked significantly. Two lineages you will encounter:

- **Legacy `PPOTrainer`** — the original TRL PPO interface (2022–2024). Simpler API; less feature-complete; uses `AutoModelForCausalLMWithValueHead`.
- **PPOv2 / rewritten `PPOTrainer`** — introduced in TRL ~0.9, aligned closely with the InstructGPT recipe (Huang et al., 2024 "The N+ Implementation Details of RLHF with PPO"). More configuration, more diagnostic surface. Current default in modern TRL.

The rest of this chapter uses the modern (post-0.9) `PPOTrainer` API. Older tutorials will read differently; pin your TRL version.

<!-- needs-research: pin the exact TRL version this API is stable in when authoring the exercise-03 lab notebook. `PPOTrainer` argument surface has shifted between 0.9, 0.11, and 0.15 series. -->

A working setup:

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig, PPOTrainer
from transformers import AutoModelForSequenceClassification

SFT_ID = "./sft-checkpoint"          # from mod-103
RM_ID = "./reward-model"              # from chapter 3 / exercise 02

# 1. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(SFT_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"       # left-pad for generation

# 2. Policy (also holds the trained value head in the modern API)
policy = AutoModelForCausalLM.from_pretrained(SFT_ID, torch_dtype="bfloat16")

# 3. Reference policy (frozen SFT model)
ref_policy = AutoModelForCausalLM.from_pretrained(SFT_ID, torch_dtype="bfloat16")

# 4. Reward model (frozen)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    RM_ID, num_labels=1, torch_dtype="bfloat16",
)

# 5. Value model (in modern TRL, a separate SeqCls-style model,
#    or the shared-with-policy variant depending on config)
value_model = AutoModelForSequenceClassification.from_pretrained(
    SFT_ID, num_labels=1, torch_dtype="bfloat16",
)

# 6. Prompts dataset — just prompts, no responses
train_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs[:5000]")
train_ds = train_ds.map(lambda ex: {"input_ids": tokenizer(ex["prompt"], truncation=True,
                                                            max_length=512).input_ids})

config = PPOConfig(
    output_dir="./ppo-run",
    learning_rate=1e-6,               # tiny — PPO is very sensitive to LR
    num_ppo_epochs=4,                 # mini-epochs per batch
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    total_episodes=10_000,            # total rollout episodes; PPO ≠ Trainer "epochs"
    kl_coef=0.05,                     # β for KL penalty in reward
    cliprange=0.2,                    # PPO ratio clip
    cliprange_value=0.2,              # value clip
    vf_coef=0.1,                      # value loss weight
    response_length=256,              # generation length per rollout
    temperature=0.7,
    bf16=True,
    logging_steps=1,
)

trainer = PPOTrainer(
    args=config,
    processing_class=tokenizer,
    model=policy,
    ref_model=ref_policy,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=train_ds,
)

trainer.train()
```

Every non-default argument above has been picked or defended for a reason:

- **`learning_rate=1e-6`.** PPO on a 7B policy is famously LR-sensitive. Values above 5e-6 tend to blow up KL. Values below 1e-7 do not learn. The safe range is 1e-6 to 3e-6 for full-parameter runs; LoRA-on-PPO tolerates 5e-6 to 1e-5. The InstructGPT paper used 9e-6 for their 175B run with careful warmup — that scales down with model size in practice.
- **`kl_coef=0.05`.** The β in `r'_t = R − β · KL`. Higher (0.1–0.3) keeps the policy tight to the reference; lower (0.01–0.02) lets it drift more. See chapter 7 on adaptive KL vs. fixed KL.
- **`cliprange=0.2`.** The PPO surrogate loss clip. 0.2 is the paper default. Rarely worth touching.
- **`vf_coef=0.1`.** Weight of the value loss in the total. Lower than default in some TRL settings because the value model shares a backbone with the policy — you do not want the value loss to swamp the policy gradient.
- **`response_length=256`.** Generation length per rollout. Longer means more expensive per step and more sensitive to length bias. Shorter means less reward signal per rollout. Start at 128–256; extend only if your prompt distribution demands it.
- **`temperature=0.7`.** Sampling temperature for rollouts. `0.0` (greedy) is a common bug — no exploration means the policy update has no signal. `>1.0` explodes KL. `0.6–0.9` is the working range.

## Reading the diagnostics

A PPO-RLHF run is *unreadable* without the right diagnostic plots. TRL logs many; the ones you actually watch:

### `objective/rlhf_reward`

Mean reward on the current batch of rollouts, *after* the KL penalty. Should generally trend up if the run is learning.

- **Flat throughout:** LR too low, or KL coefficient so high the policy cannot move.
- **Climbs then oscillates:** normal for later stages of a stable run.
- **Climbs then crashes:** KL blew up (see below) and the run collapsed.
- **Climbs monotonically for thousands of steps:** possibly reward hacking. Cross-check with `policy/kl` and a held-out human eval.

### `objective/kl`

Mean per-token KL divergence between `π` and `π_ref`, averaged over the batch. This is the single most important plot.

- **Grows steadily but slowly (0 → 5–20 nats over the run):** normal, healthy.
- **Grows fast and unboundedly:** KL blew up. The policy is drifting far from `π_ref`; reward hacking or catastrophic mode collapse is imminent. Raise `kl_coef`, drop LR, or restart with a shorter response length.
- **Stays near zero:** policy is not learning (or the KL penalty is so high it cannot).
- **Oscillates wildly:** possibly a batch-size or gradient-accumulation issue; LR may be too high.

### `objective/entropy`

Mean per-token entropy of the policy. Should decrease slightly over the run (the policy becomes more decisive) but not crash to zero.

- **Crashes to near-zero:** mode collapse. The policy is producing a very narrow distribution — often the same response over and over. Common when KL penalty is too weak.

### `objective/scores`

Mean reward *before* the KL penalty. Together with `objective/rlhf_reward` and `objective/kl`, this decomposes what is driving improvements.

### `loss/policy_avg`, `loss/value_avg`

The two loss components. Policy loss should be small in absolute value (PPO's clipped surrogate is normalized). Value loss should decrease steadily; a value loss that never converges signals the value head is misfit and advantages are noisy.

### `policy/clipfrac`

The fraction of ratio-clip events per batch. If this is >0.3, the policy is trying to step further than PPO allows every iteration — LR is too high. Should sit in [0.02, 0.15] for a healthy run.

### `policy/approxkl`

The approximate KL between the pre-update and post-update policy (different from `objective/kl` above, which is KL to `π_ref`). If this exceeds a target — TRL supports early-stopping mini-epochs based on it — the trainer stops the current step to prevent overstep. Configured via `target_kl` in older TRL.

### `val/mean`, `val/var`

Value predictions' mean and variance. If mean is drifting monotonically (either way) while `objective/rlhf_reward` is flat, the value head is miscalibrated and your advantages are systematically biased.

## The memory picture

For a 7B PPO-RLHF run at BF16 without any tricks, rough per-model memory:

| Model | Weights (BF16) | Notes |
|---|---|---|
| Policy | 14 GB | Trained; has ~14 GB of Adam state on top |
| Reference | 14 GB | Frozen; forward-only |
| Reward | 14 GB | Frozen; forward-only |
| Value | 14 GB | Trained; ~14 GB of Adam state |
| Activations + KV | 10–20 GB | Depends on batch × response_length |
| **Total** | **80–100+ GB** | |

On a single 80 GB GPU this is right at the ragged edge. In practice teams reach for:

- **LoRA / QLoRA on the policy.** Cuts trained-model optimizer state ~100×. The reference model can share the same 4-bit base as the policy (`peft` disables the adapter to compute reference log-probs) — same trick DPO uses.
- **Shared backbone for value and policy.** Cuts value model overhead essentially to zero.
- **Rollout generation via a separate inference engine** (vLLM), with policy weights synced periodically. This is what Anthropic-scale RLHF looks like; way beyond what a single GPU wants.
- **DeepSpeed ZeRO-3 or FSDP** across multiple GPUs. Standard for teams running PPO seriously; mod-107 covers the distributed side.

For a first PPO run on a single 24-40 GB GPU, use LoRA on the policy and either share the reference with the policy (via `peft` adapter disable) or use a 1B–3B base — enough to feel the mechanics without a battery of A100s.

## The `AutoModelForCausalLMWithValueHead` alternative

The older TRL API used a single class `AutoModelForCausalLMWithValueHead` that added a value head to a causal LM. Modern PPO in TRL prefers a separate value model, but the shared-backbone shortcut is still useful for small runs. If your `trl` version documents `AutoModelForCausalLMWithValueHead` as the primary path, the recipe is:

```python
from trl import AutoModelForCausalLMWithValueHead

policy_with_value = AutoModelForCausalLMWithValueHead.from_pretrained(SFT_ID)
```

This attaches a small linear head on top of the shared backbone that outputs `V(x, y_{<t})`. Half the memory of a separate value model; slightly less stable in some settings because the value gradient and policy gradient contend for the same weights. Which lineage your `trl` version uses controls which recipe compiles. Check first.

## Common bugs and their fingerprints

Beyond the diagnostic-reading guide above, the specific bugs that trip beginners:

**Wrong padding side.** Generation requires left-padding (the pad tokens go before the real prompt so the model starts generating right after the prompt's last real token). `tokenizer.padding_side = "left"` — an obscure but critical setting. If you leave it at `"right"`, `.generate()` will produce garbage that scores zero on the RM, KL will explode, and the diagnostics will look like the policy exploded when actually the input format was wrong.

**RM tokenizer mismatch.** The RM was trained with a specific chat template; if the PPO loop concatenates prompt + rollout in a different template, the RM sees an out-of-distribution input and scores it randomly. Assert the chat template matches by tokenizing a fixed prompt with both tokenizers and diffing.

**Reference model drift.** In PPO-with-LoRA setups, `π_ref` is often computed by disabling the adapter on the same base. If the adapter is not properly disabled at reference-computation time — e.g., a stale `set_adapter` call — the "reference" is actually the current policy, and the KL penalty becomes zero. Log `KL(π || π_ref)` on a canned prompt at the start of every run; if it is exactly zero, you have this bug.

**Reward scale drift.** RMs' reward scale drifts across training runs. The `kl_coef` that worked with one RM will be wrong for the next. Standardize the reward (mean-zero, unit-variance) inside the PPO loop if you swap RMs — TRL exposes `reward_normalize` or similar in various versions.

**Value head initialized without a warmup step.** A freshly-initialized value head's estimates are noise; the first few PPO steps use garbage advantages. Some recipes run a few "value-only warmup" steps (value loss only, no policy update) at the start. TRL does not do this by default; if your first ~50 steps look pathological, consider it.

## Stability tricks that actually help

The InstructGPT paper and Huang et al. 2024 ("The N+ Implementation Details of RLHF with PPO") together enumerate the recipe that makes PPO-RLHF work at scale. The ones you should know:

- **Adaptive KL controller.** Instead of a fixed `kl_coef`, adjust it every step so that `KL(π || π_ref)` tracks a target value (e.g., 6 nats per response). TRL supports this via `AdaptiveKLController`; without it you often over- or under-penalize.
- **Reward standardization / centering.** Standardize batch rewards (mean-zero, unit-variance) or center them per-batch before combining with the KL penalty. Keeps the effective scale of the reward stable across different RMs.
- **Whiten advantages.** Standardize advantages within each PPO batch. Reduces variance of the gradient significantly. TRL does this by default.
- **Bound the response length or apply a length penalty in the reward.** Directly attacks the length-bias-during-training failure mode.
- **Value loss clipping.** Clip the value predictions symmetrically around the previous iteration to prevent runaway value estimates.

None of these are optional at scale; every one shows up in the reference recipes.

## When to reach for PPO-RLHF (vs. DPO)

Given DPO's simplicity, the honest question is *when* to use PPO. Situations where PPO earns its complexity:

- **You have a reward model but not preference pairs.** For example, a verifiable-reward setup where correctness is a function call, not a human comparison. GRPO (chapter 6) is often the modern move here, but PPO works.
- **The task's reward signal is fundamentally on-policy.** Agent / tool-use / multi-turn planning tasks where the reward depends on the actual trajectory the policy takes, not on a fixed dataset of "good vs bad" responses.
- **You are at a scale where PPO's compute is not a limiter.** Frontier labs run PPO. Everyone else usually starts with DPO.
- **You want to iteratively refresh the RM against on-policy rollouts.** DPO's fixed preference set does not adapt; PPO with a periodically-refreshed RM (Anthropic's "iterative RLHF" or Meta's Llama-2-Chat recipe) does.

For a first preference-tuning run in a production team, DPO first, PPO second when the specific need arises.

## Exercise pointer

`exercises/exercise-03-ppo-rlhf-run.md` walks you through a small-scale PPO-RLHF run on the SFT base from mod-103 (or a 1B–3B model if hardware is tight), plugs in the RM from exercise 02, and asks you to read the diagnostic plots and produce a run report.

## Summary

- Classical PPO-RLHF holds four models in memory: policy (trained), reference (frozen SFT), reward (frozen), and value (trained, often shares the policy backbone). Rollouts are on-policy; reward is a scalar at the end of each response; a per-token KL penalty against the reference is baked into the reward.
- The PPO objective is a clipped surrogate loss on the log-prob ratio between the current and previous policy, plus MSE value loss, minus an optional entropy bonus. GAE provides the advantage.
- TRL's `PPOTrainer` (post-v0.9) is the standard implementation. Key knobs: `learning_rate` (1e-6 range for 7B full-parameter), `kl_coef` (0.02–0.1), `cliprange` (0.2), `response_length`, `temperature`, `num_ppo_epochs`, `total_episodes`.
- Reading the run: `objective/kl` is the load-bearing plot — steady growth is healthy, blowups are the KL-collapse failure (chapter 7). Also watch `objective/rlhf_reward`, `objective/entropy`, `policy/clipfrac`, and the value loss.
- Memory: a 7B full-parameter PPO run needs ~80–100 GB. LoRA on the policy (with the reference computed by disabling the adapter) is what makes this feasible on smaller hardware.
- Common bugs: wrong padding side (breaks generation), RM tokenizer mismatch (OOD reward), reference adapter not disabled (zero KL), reward scale drift across RMs, value head not warmed up.
- Reach for PPO when you have a reward but no fixed pairs, when the task is genuinely on-policy, when you can afford the compute, or when you want iterative RM refresh. Otherwise DPO (chapter 5) is the modern default.
