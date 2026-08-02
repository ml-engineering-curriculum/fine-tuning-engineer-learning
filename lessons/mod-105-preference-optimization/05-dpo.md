# Direct Preference Optimization

**Estimated read:** 50 minutes.
**You should be able to, by the end:** re-derive DPO's loss from the KL-constrained RL objective in a few lines, wire TRL's `DPOTrainer` on top of an SFT base (including the QLoRA + LoRA recipe used to reproduce Zephyr-7B), reason about `β` and the "implicit reward" the DPO loss defines, and know the two or three places DPO differs from PPO that you can *feel* at training time.

## The one-slide sales pitch

Preference optimization *without* a reward model, *without* on-policy rollouts, *without* PPO. Just:

- The SFT model as reference `π_ref` (frozen).
- The policy `π_θ` (initialized from `π_ref`, trained).
- A dataset of preference pairs.
- A loss that pushes `π_θ` to *raise* the log-probability of `y_w` and *lower* it for `y_l`, both measured relative to `π_ref`.

The training loop looks almost exactly like SFT. That is the whole point. Rafailov et al., 2023 ("Direct Preference Optimization: Your Language Model Is Secretly a Reward Model") named this, proved it was equivalent to the RLHF objective from chapter 1, and released the loss.

Since mid-2023, DPO has been the default preference-optimization method for most teams outside frontier labs. This chapter walks the derivation, the trainer, the hyperparameter, and the ways DPO differs from PPO in practice.

## The derivation

Start from the KL-constrained RL objective (chapter 1):

```
π* = arg max_π E_{x, y ∼ π}[ r*(x, y) ] − β · KL(π(·|x) || π_ref(·|x))
```

Standard result from stochastic control / reverse-KL-regularized maximization: the optimal policy is

```
π*(y | x) = (1 / Z(x)) · π_ref(y | x) · exp( r*(x, y) / β )      (*)
```

where `Z(x) = Σ_y π_ref(y | x) · exp(r*(x, y) / β)` is the partition function. This is a Gibbs / Boltzmann policy — the reference reweighted by the exponentiated reward, normalized.

Take the log of (*):

```
log π*(y | x) = log π_ref(y | x) + (1/β) r*(x, y) − log Z(x)
```

Solve for `r*`:

```
r*(x, y) = β · log[ π*(y | x) / π_ref(y | x) ] + β · log Z(x)      (**)
```

Now substitute (**) into the Bradley-Terry preference likelihood (chapter 1):

```
P(y_w > y_l | x) = σ( r*(x, y_w) − r*(x, y_l) )
                 = σ( β · [ log(π*(y_w|x)/π_ref(y_w|x)) − log(π*(y_l|x)/π_ref(y_l|x)) ] )
```

The `β · log Z(x)` term appears on both sides of the subtraction and cancels — the same prompt gives the same partition. So the preference probability is expressible purely in terms of **the ratio of the optimal policy to the reference policy**, with no reference to `r*` at all.

Maximum-likelihood training on preference pairs then gives the DPO loss:

```
L_DPO(π_θ; π_ref) = − E_{(x, y_w, y_l) ∼ D} log σ(
                        β · log(π_θ(y_w|x) / π_ref(y_w|x))
                      − β · log(π_θ(y_l|x) / π_ref(y_l|x))
                    )
```

Read this: for each pair, compute the *log-ratio* between the current policy and the reference on both `y_w` and `y_l`; scale by β; take the difference; push through log-sigmoid; average over the dataset. Minimize.

Two useful reframings of the same object:

- **As an implicit reward.** Define `r_θ(x, y) = β · log(π_θ(y|x) / π_ref(y|x))`. Then the loss is exactly the Bradley-Terry loss on this *implicit reward*. DPO is training an implicit reward model — that just happens to be the policy itself.
- **As a comparison of log-ratios.** The gradient of DPO (worked out in appendix A of the paper) up-weights the log-ratio of `y_w` and down-weights the log-ratio of `y_l` by an amount proportional to `σ(r_θ(x, y_l) − r_θ(x, y_w))` — i.e., proportional to how confidently wrong the current policy is on this pair. Same shape as the RM loss (chapter 3), same gradient behavior.

The result is a supervised loss on preference pairs. No sampling, no RM, no PPO. Just: forward the policy on `y_w` and `y_l`, forward the reference on both, compute log-probs, plug into the formula, backward.

## What the loss actually does per pair

For a single pair `(x, y_w, y_l)`, denote:

- `Δ_w = log π_θ(y_w|x) − log π_ref(y_w|x)` — how much the policy has increased log-prob of `y_w` relative to reference.
- `Δ_l = log π_θ(y_l|x) − log π_ref(y_l|x)` — same for `y_l`.

The DPO loss on this pair is `−log σ(β · (Δ_w − Δ_l))`. Minimizing it pushes `Δ_w − Δ_l` positive. That is: **the policy should raise the log-prob margin of chosen over rejected, relative to what the reference model would have said.**

Notice: DPO does not require the policy to *increase* the log-prob of `y_w` in absolute terms. It requires that the *margin* between chosen and rejected grow. In practice the loss does typically increase `log π_θ(y_w|x)` while decreasing `log π_θ(y_l|x)`, but the gradient can also legally push both downward (as long as `y_l` drops faster) or both upward (as long as `y_w` rises faster). The paper's appendix B and follow-up work (Pal et al., 2024; Rafailov et al.'s 2024 follow-up on "R-DPO") discuss cases where both drop and the training dynamics get pathological — worth reading.

## The `DPOTrainer` recipe

TRL's `DPOTrainer` has been stable across recent versions. The current setup:

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

SFT_ID = "./sft-checkpoint"    # or an off-the-shelf SFT, e.g. "HuggingFaceH4/zephyr-7b-alpha-sft"

tokenizer = AutoTokenizer.from_pretrained(SFT_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

policy = AutoModelForCausalLM.from_pretrained(
    SFT_ID, torch_dtype="bfloat16",
)

# LoRA for cheap fine-tuning. See mod-104 chapter 1 for the config choices.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")

training_args = DPOConfig(
    output_dir="./dpo-run",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,             # DPO with LoRA: 1e-6 to 5e-6; DPO full-param: 5e-7 to 1e-6
    num_train_epochs=1,             # DPO overfits fast; one epoch is often optimal
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
    bf16=True,
    beta=0.1,                        # THE key hyperparameter
    max_length=1024,
    max_prompt_length=512,
    loss_type="sigmoid",             # standard DPO; also "hinge", "ipo", "kto", etc.
)

trainer = DPOTrainer(
    model=policy,
    ref_model=None,                  # None + peft_config uses the disabled-adapter trick
    args=training_args,
    train_dataset=dataset["train_prefs"],
    eval_dataset=dataset["test_prefs"],
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.train()
trainer.save_model("./dpo-run/final")
```

Load-bearing details:

- **`ref_model=None` + `peft_config=lora_config`.** This is the memory-saving trick that makes DPO fit on a single consumer GPU. `DPOTrainer` sees the LoRA config and knows the "reference" is the same base model with the adapter *disabled* (via `peft.disable_adapter()`). Zero additional memory for the reference beyond the LoRA-wrapped base — a huge win over PPO's separately-loaded reference.
- **`beta=0.1`.** The KL-anchor strength. Higher `β` → less deviation from `π_ref` per pair → more conservative updates. Lower `β` → more aggressive divergence. 0.1 is the paper's default and the Zephyr-7B recipe's default. `β ∈ [0.01, 0.5]` is the working range; see the next section.
- **`learning_rate=5e-7`.** Much smaller than SFT LR. DPO is a fine-tune of a fine-tune — the base is already competent. Values above 5e-6 blow up loss and produce policy collapse.
- **`num_train_epochs=1`.** DPO overfits *fast*. Three epochs of DPO on the same preference set typically produce a worse model than one epoch (over-optimization on the implicit reward). Watch eval loss and stop early.
- **`loss_type="sigmoid"`.** Standard DPO. TRL's `DPOTrainer` supports other loss types (IPO, hinge, KTO-style, cDPO for label smoothing) via this argument — chapter 6 uses them.
- **`max_length` / `max_prompt_length`.** The trainer needs both because it separately truncates prompt and response to fit total length.

## The QLoRA + DPO recipe

For 24-GB-single-GPU DPO on a 7B, combine the above with QLoRA (mod-104 chapter 2). The additional pieces:

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

policy = AutoModelForCausalLM.from_pretrained(
    SFT_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)
# ... rest of the setup unchanged
```

The reference is *still* the same 4-bit base with the adapter disabled — you do not need to hold a second copy. Peak memory for DPO + QLoRA + LoRA on 7B lands in the 12–16 GB range on a single 24 GB card with `per_device_train_batch_size=1` and gradient accumulation. This is the recipe that put DPO into every mid-career engineer's reach.

## The `β` hyperparameter, in detail

`β` is the single most consequential DPO hyperparameter. Its role in the loss:

```
L = −log σ( β · (Δ_w − Δ_l) )
```

Higher `β` means larger effective magnitude on `(Δ_w − Δ_l)` inside the sigmoid — the gradient saturates faster, so the loss becomes small (near-zero-loss) as soon as the margin is meaningful. Lower `β` keeps the gradient signal alive further into training — the policy is pushed to make the margin bigger and bigger.

Interpretations that help:

- **`β` is inverse-temperature of the KL constraint.** Higher `β` = tighter KL constraint = policy stays closer to reference. Lower `β` = looser constraint = policy allowed to diverge further.
- **`β → ∞` gives a "just-copy-reference" policy** (the loss saturates immediately regardless of margin).
- **`β → 0` gives no anchoring at all** (the policy is trained to maximize the margin without bound; log-ratios can grow arbitrary large).

Empirical starting points:

- **`β = 0.1`.** The Zephyr / DPO-paper default. Works for most SFT-followed-by-DPO setups.
- **`β = 0.01`–`0.05`.** More aggressive; use when your preference signal is clean and the model needs to move significantly.
- **`β = 0.3`–`0.5`.** More conservative; use when you saw DPO destabilize the base at the default.

**How to tune.** Do a short (say 500-step) DPO run at `β ∈ {0.03, 0.1, 0.3}` and plot:

- Eval loss vs. training step.
- The mean `log(π_θ(y_w|x) / π_ref(y_w|x))` (the "chosen log-ratio") over training.
- The mean `log(π_θ(y_l|x) / π_ref(y_l|x))` (the "rejected log-ratio") over training.

A healthy run: chosen log-ratio grows positive and slowly (0 → ~1); rejected log-ratio drifts negative or stays near zero; eval loss decreases and plateaus. An unhealthy run at low β: both log-ratios drop steeply — the policy is decreasing the probability of *everything*, and the margin is preserved only because rejected drops faster.

## Reading a DPO run

TRL's `DPOTrainer` logs a handful of specific metrics that are meaningful only for DPO. What to watch:

### `train/loss`, `eval/loss`

Straight cross-entropy on the pairwise objective. Should decrease. Watch for eval loss plateauing / rising while train loss still drops — the DPO-specific overfitting pattern (see below).

### `rewards/chosen`, `rewards/rejected`

The *implicit rewards* `β · log(π_θ(y_w|x) / π_ref(y_w|x))` for chosen and rejected, averaged over the batch. In a healthy run, chosen grows positive, rejected shrinks (or drifts negative), the gap widens.

### `rewards/margin`

The difference `rewards/chosen − rewards/rejected`. This is what the loss is directly maximizing. Should grow steadily.

### `rewards/accuracy`

The fraction of the batch where `rewards/chosen > rewards/rejected`. Equivalent to "would this policy have picked the correct answer on this pair?" Should climb toward 1.0 as training progresses, but 1.0 on train is often an overfit signal.

### `logps/chosen`, `logps/rejected`

Raw log-probs of chosen and rejected under `π_θ` (not the log-ratio). Interesting mainly to catch the "everything is dropping" pathology: if both are decreasing steeply, the model is becoming less confident about everything and the DPO margin is a distraction.

### `logits/chosen`, `logits/rejected`

Similar but at logit level; less commonly watched.

## DPO's specific failure modes

Three failure modes are worth naming as DPO-specific. Chapter 7 covers them alongside the family-wide failures.

### DPO overfitting

DPO loss on the training set drops quickly and eval loss follows for a while, then eval loss reverses and starts climbing while training loss keeps falling. This happens within ~1 epoch on most standard datasets. The pattern is: the implicit reward is overfitting the specific pairs; the policy is over-optimizing the exact chosen-over-rejected margins in the training set and drifting off the general preference distribution.

**Fixes:** (a) fewer epochs (one is often enough), (b) higher `β` (tighter KL), (c) switch to IPO (chapter 6) which regularizes the loss to be less prone to this.

### "Log-prob deflation"

Both `logps/chosen` and `logps/rejected` drop steeply during training, and the model starts producing degenerate outputs (empty responses, EOS on step 1, or looping). The margin is preserved because rejected drops faster than chosen, but chosen is dropping too.

**Fixes:** (a) higher `β`, (b) add an SFT-loss auxiliary term (Pal et al., 2024's DPO+SFT recipe, sometimes called "DPO with reference-free control"), (c) reduce LR, (d) switch to KTO (chapter 6) which uses a different loss structure.

### Reference model drift under QLoRA

Because the QLoRA + DPO recipe computes the reference by disabling the LoRA adapter, any *state* that persists on the base beyond the adapter — e.g., LayerNorm parameters casted to FP32 by `prepare_model_for_kbit_training` and then subtly updated during training — can cause `π_ref` to drift. In pathological cases the KL to the "reference" grows without bound because the reference is not fixed.

**Fixes:** (a) freeze all non-LoRA parameters explicitly, (b) verify at start of training that `disable_adapter()` reproduces the original SFT model's outputs on a canned prompt, (c) if drift persists, cache reference log-probs once at the start of training (TRL supports `precompute_ref_log_probs=True`).

## When DPO underperforms

DPO does not always beat PPO — the literature has thoughtful counterexamples. Cases to watch for:

- **Small or noisy preference sets.** DPO's data-hungriness is real; PPO can leverage a strong RM trained on noisier data more effectively than DPO can leverage the pairs directly.
- **Multi-turn agent tasks.** DPO's fixed-dataset structure does not model the trajectory-level reward that agent tasks require; on-policy PPO / GRPO tends to be stronger.
- **Tasks with a natural verifiable reward.** RLVR / GRPO with a rule-based reward avoids the "learn a preference model" step entirely and is often strictly better than DPO on math / code / structured-output tasks.
- **When you can afford to run several rounds of on-policy training and iterative RM refresh.** DPO's fixed-set structure cannot benefit from that; PPO / iterative DPO can.

For most SFT-shaped chat preference tuning at 7B–70B scale, DPO wins on operational simplicity and reasonable quality. For everything else, check chapter 6.

## Practical checklist before you start a run

- SFT model chosen and pinned (revision hash).
- Preference dataset shipped with a dataset card (chapter 2's checklist).
- Tokenizer chat template applied and verified — tokenize 20 pairs, print the tokenized text, confirm chosen and rejected are structurally identical except at the response.
- LoRA config decided (typically all-linear r=16 for a 7B).
- `β` in the 0.03–0.3 range; default 0.1.
- `learning_rate` in 5e-7 to 5e-6; scale to LoRA vs full-parameter.
- One epoch scheduled; eval every 200 steps; early-stop signal identified (eval loss reversal).
- Reference-computation trick understood: `ref_model=None` + `peft_config` disables the adapter for reference.
- Reward histograms + implicit reward metrics logged.
- Model card template ready with all the run parameters.

Do this once and it becomes automatic. Skip it and you will lose a Sunday to "why is my DPO run producing empty strings" three weeks later.

## Exercise pointer

`exercises/exercise-04-dpo-run-and-comparison-to-ppo.md` walks you through a DPO run on the same SFT base and preference set you used in exercises 02–03, and asks you to produce a side-by-side comparison of DPO and PPO — including compute, wall-clock, memory, eval scores, and qualitative outputs.

## Summary

- DPO derives an equivalent form of the KL-constrained RL objective (chapter 1) by expressing the optimal policy as a Gibbs distribution over the reward, substituting into the Bradley-Terry preference likelihood, and canceling the partition function. The result is a supervised loss on preference pairs, no reward model or PPO required.
- The loss `L_DPO = −log σ(β · [Δ_w − Δ_l])` where `Δ = log(π_θ / π_ref)` maximizes the margin of chosen over rejected log-ratios. It is equivalent to Bradley-Terry on an implicit reward `r_θ = β · log(π_θ / π_ref)`.
- TRL's `DPOTrainer` handles the standard case. `ref_model=None` + `peft_config` uses the "disabled adapter" trick to share the reference with the LoRA-wrapped policy; combined with QLoRA it fits a 7B DPO run in ~14 GB.
- `β` is the KL-constraint strength; 0.1 is the working default, 0.03–0.3 the useful range. Higher `β` = tighter KL. Log the implicit-reward metrics (`rewards/chosen`, `rewards/rejected`, `rewards/margin`, `rewards/accuracy`) and watch for the specific DPO failures: overfitting, log-prob deflation, and reference drift under QLoRA.
- DPO wins on operational simplicity vs. PPO for most SFT-shaped preference tuning. Reach for PPO / GRPO / iterative RLHF when the task is agent-shaped, verifiable-reward, or genuinely on-policy.
