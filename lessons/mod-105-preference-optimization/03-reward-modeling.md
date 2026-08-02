# Reward Modeling

**Estimated read:** 45 minutes.
**You should be able to, by the end:** turn a preference-pair dataset into a scalar reward model via the Bradley-Terry loss, wire the model into TRL's `RewardTrainer`, and evaluate the resulting RM with pairwise accuracy, calibration, and length-conditional slices — enough to know whether it is fit to drive a PPO loop.

## Where the reward model sits

Chapter 1 established the RLHF pipeline: preference pairs → reward model → PPO. This chapter is that middle arrow.

A **reward model** (RM) is a function `r_φ(x, y) → R` that assigns a scalar to a `(prompt, response)` pair. Higher is better. In classical PPO-RLHF (chapter 4), the RM is called during rollouts to score generated responses; the resulting reward drives the policy update. In DPO and the reference-free family, the RM is analytically eliminated — you do not train one — which is why DPO is called "reference-based but reward-model-free." An RM is only necessary when you plan to run PPO / GRPO / any method that consumes an on-policy generated response and asks "is this good?"

That said: training an RM is worth doing even if you do not plan to run PPO. Reward models are:

- **Good pairwise evaluators.** RM accuracy on a held-out preference set is a cheap, useful proxy for downstream preference-tuning gains.
- **Necessary for many "best-of-N" inference strategies.** BoN and rejection-sampling recipes generate N candidates at serve time and pick the one with the highest reward.
- **Independently interpretable diagnostics.** An RM's failure modes (length bias, refusal bias, sycophancy) reveal the training set's failure modes and often the eventual policy's failure modes.

So even in a DPO-first pipeline, one production RM is often on the shelf.

## The Bradley-Terry loss, worked out

Recall chapter 1: under the Bradley-Terry assumption,

```
P(y_w > y_l | x) = σ( r*(x, y_w) − r*(x, y_l) )
```

Reward modeling replaces `r*` with a parametric approximation `r_φ` (a neural net). The maximum-likelihood loss over a preference dataset `D = {(x_i, y_w_i, y_l_i)}` is:

```
L_RM(φ) = − E_{(x, y_w, y_l) ∼ D} [ log σ( r_φ(x, y_w) − r_φ(x, y_l) ) ]
```

Read this: for each pair, compute the reward on both sides, take the *difference*, push it through a sigmoid, and minimize the negative-log-likelihood that chosen has higher reward. It is a pairwise binary logistic loss on the reward margin.

Two facts to internalize before you touch code:

- **Only the difference matters.** The loss is invariant to adding a constant to `r_φ` — the sigmoid of a difference cancels the constant. So the *absolute* value of a reward is meaningless; only ranking matters. Consequence: you cannot compare rewards across two different RMs directly, and even a single RM's reward scale drifts across runs.
- **The gradient shape.** ∂L/∂(r_w − r_l) = σ(r_l − r_w) — the loss is largest when the model is confidently *wrong* (r_l >> r_w), zero when it is confidently *right*. This is the same shape as binary cross-entropy on a single class label.

## The RM's architecture: a sequence classifier

An RM is almost always a language model with the LM head replaced (or supplemented) by a *scalar regression head*. The forward pass:

1. Tokenize `(x concatenated with y)` using the same chat template you will use downstream. Feed to the base transformer.
2. Take the hidden state at the **last non-padded token** (typically the EOS or the last assistant token).
3. Pass that hidden state through a linear layer with output dim 1. That scalar is `r_φ(x, y)`.

In HuggingFace terms, this is an `AutoModelForSequenceClassification` with `num_labels=1`. `RewardTrainer` in TRL wraps this pattern and takes care of the "last token" selection.

The base can be any decoder-only LM; the RM is typically initialized from an SFT model of similar capability, sometimes the same SFT model that will become the PPO policy. Starting from a strong SFT initialization is standard — an RM trained from a random init is weaker than one trained from a pretrained-and-instruction-tuned init at the same data size.

A common architectural note: many RM implementations *do not* initialize the scalar head to zero. They initialize it randomly, and the first few thousand steps of training move both the head and the shared backbone. That is fine — the loss handles it — but expect the loss to look ugly in the very first steps.

## The TRL `RewardTrainer` recipe

Since TRL ≥ 0.4 (and stable in the current line), the `RewardTrainer` handles the standard case. A working setup:

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments,
)
from trl import RewardTrainer, RewardConfig

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"  # or your SFT checkpoint

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=1,
    torch_dtype="bfloat16",
)
model.config.pad_token_id = tokenizer.pad_token_id

# Dataset must have "chosen" and "rejected" columns (either raw text or
# already-tokenized). TRL's RewardTrainer applies the chat template if you pass
# the tokenizer and the columns are message-lists.
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")

training_args = RewardConfig(
    output_dir="./reward-model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,          # RMs like small LRs
    num_train_epochs=1,          # one epoch is often enough
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
    bf16=True,
    max_length=1024,             # per-side sequence limit
    remove_unused_columns=False, # RewardTrainer needs the raw columns
    center_rewards_coefficient=0.01,   # optional L2 on mean reward, keeps scale sane
)

trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train_prefs"],
    eval_dataset=dataset["test_prefs"],
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model("./reward-model")
```

Line-by-line, what is load-bearing:

- **`num_labels=1`** — the scalar-head architecture. If you leave this at the default, you will accidentally train a 2-class classifier, which is not the same objective.
- **`center_rewards_coefficient`** — an optional L2 regularizer on the mean reward that keeps the reward scale from drifting. Not strictly required; makes the RM's absolute scores easier to interpret across training runs.
- **`learning_rate=5e-6`** — one-to-two orders of magnitude smaller than an SFT LR. RM training on top of a strong SFT init is a "finetune the head, gently adjust the backbone" job, not a fresh training job. Values of 1e-6 to 1e-5 are typical.
- **`num_train_epochs=1`** — RMs overfit fast on preference data. One epoch is often optimal; more than three is usually worse.
- **`max_length=1024`** — the trainer truncates both `chosen` and `rejected` to this length. Long preference pairs (agent trajectories, multi-turn) may want more; pay for it in memory.
- **`processing_class=tokenizer`** — since TRL 0.11 or so, `RewardTrainer` takes `processing_class` instead of `tokenizer`. Older tutorials use `tokenizer=`; check your version.

The trainer handles: per-example tokenization of chosen and rejected, chat-template application, last-token selection for the reward head, and the pairwise loss. Debugging is much easier when you understand each step, which is why the from-scratch alternative below is worth writing at least once.

## From-scratch reward loop (understand once)

```python
import torch
import torch.nn.functional as F

def reward_loss(model, batch):
    # batch has "chosen_input_ids", "chosen_attention_mask",
    # "rejected_input_ids", "rejected_attention_mask" — pre-tokenized.
    chosen_out = model(
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"],
    )
    rejected_out = model(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"],
    )

    # Reward = last-token logit (num_labels=1 model returns (B, seq, 1) or (B, 1))
    chosen_reward = chosen_out.logits.squeeze(-1)
    rejected_reward = rejected_out.logits.squeeze(-1)

    # If the model returns per-token logits, select the last non-pad token.
    if chosen_reward.dim() == 2:
        chosen_last_idx = batch["chosen_attention_mask"].sum(dim=1) - 1
        chosen_reward = chosen_reward.gather(1, chosen_last_idx.unsqueeze(1)).squeeze(1)
        rejected_last_idx = batch["rejected_attention_mask"].sum(dim=1) - 1
        rejected_reward = rejected_reward.gather(1, rejected_last_idx.unsqueeze(1)).squeeze(1)

    # Bradley-Terry loss.
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

    accuracy = (chosen_reward > rejected_reward).float().mean()
    return loss, {"acc": accuracy, "chosen_reward": chosen_reward.mean(),
                  "rejected_reward": rejected_reward.mean()}
```

You do not need to run this to train RMs in production — `RewardTrainer` is fine — but reading it once makes the loss's shape concrete. The specific detail worth noticing is the last-token selection: RM implementations that pool over the sequence (mean or max) exist, but the last-token convention is what TRL uses and what most papers assume.

## RM training with PEFT

An RM does not have to be fully fine-tuned. LoRA on top of a frozen SFT base is a common and cheap recipe:

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,   # not CAUSAL_LM
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["score"],    # DO save the classifier head fully
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)
```

Two things worth attention:

- `task_type=TaskType.SEQ_CLS`. If you leave this at the default `CAUSAL_LM`, `peft` will not correctly wire the loss.
- `modules_to_save=["score"]`. The scalar reward head *must* be fully trainable — you cannot LoRA a randomly-initialized output head and expect it to converge. `modules_to_save` tells `peft` to keep this module fully trainable rather than adapt it with LoRA. The exact module name (`score` vs `classifier` vs `v_head`) depends on the model class; check with `list(dict(model.named_modules()).keys())` and grep for the classifier head.

QLoRA over a 4-bit base works the same way; see mod-104 chapter 2 for the load pattern and combine with the `LoraConfig` above.

## Evaluating a reward model

RM eval is where most beginners underinvest. Four numbers you should report for every trained RM:

### 1. Pairwise accuracy

The fraction of held-out pairs where `r_φ(x, y_w) > r_φ(x, y_l)`. This is the direct proxy for "did the RM learn the preference relation."

Typical numbers on standard datasets (rough, dataset-dependent):

- Well-trained RM on UltraFeedback: 70–80% pairwise accuracy on the held-out split.
- Anthropic HH-RLHF: 65–75%.
- SHP: 60–70% (noisier labels).

Below 65% and the RM has not learned enough to drive PPO usefully. Above 90% and you should check for a leak (train/eval prompt overlap) or a shortcut (see length-bias below).

### 2. Calibration

For a good RM, the reward *margin* — `r_φ(x, y_w) − r_φ(x, y_l)` — should correlate with the *confidence* the label deserves. Bin the eval pairs by margin quantile (5 bins), compute per-bin accuracy, and plot.

A well-calibrated RM has monotonically increasing accuracy with margin: bin 1 (smallest margin) is around 55%, bin 5 (largest margin) is around 90%. A miscalibrated RM has a flat or non-monotone curve.

### 3. Length-conditional accuracy

Split the eval pairs into "chosen is longer than rejected" and "chosen is shorter than rejected" (drop ties). Compute pairwise accuracy on each subset separately.

If overall accuracy is 75% but "chosen longer" is 85% and "chosen shorter" is 55%, the RM has learned length. This is the diagnostic Singhal et al., 2023 popularized; it is the single most useful check to run before you take an RM into PPO.

The mitigation is either dataset-side (length-balance the training set — chapter 2) or objective-side (add a length penalty to the RM loss, or use a length-normalized method downstream).

### 4. Reward distribution shape

Plot `r_φ(x, y)` histograms for the chosen and rejected sides of the eval set. Overlaid, they should be two well-separated Gaussian-ish blobs — chosen slightly higher, rejected slightly lower, with meaningful overlap.

Failure signatures:

- **Both blobs are bimodal with the same peaks.** The RM is a two-cluster classifier (e.g., "did-generate vs did-refuse") not a preference model. Retrain with better data or reduce LR.
- **Both blobs are extremely narrow (variance << mean difference).** The RM is overconfident; PPO on it will chase the small delta hard and reward-hack quickly.
- **Both blobs have very different means but overlap almost completely.** The absolute reward scale drifted; check `center_rewards_coefficient`.

### 5. RewardBench and public leaderboards

RewardBench (Allen AI, 2024) is a standardized benchmark for reward models with subsets for chat, safety, reasoning, and code. Running a candidate RM through RewardBench (or a subset that matches your surface) gives you a comparable number against published RMs — useful when picking a starting point rather than training from scratch. See `resources.md`.

## RM failure modes

Beyond length bias (covered above), five failure modes worth naming:

### Sycophancy

The RM learns to prefer responses that agree with the *prompt's implicit stance*, even when the response is wrong. If the prompt is "Isn't it true that X?" the RM prefers "Yes, X is true." over the correct rebuttal.

**Detection:** hand-inspect eval pairs where the prompt contains a leading assumption. Look for the sycophantic response winning.

**Mitigation:** add adversarial preference pairs where the chosen response corrects a false prompt premise. Anthropic's constitutional AI and Sharma et al., 2023 ("Towards Understanding Sycophancy in Language Models") both discuss this in depth.

### Refusal-preference collapse

The RM learns "refusing is safe, so refusing is preferred." Every well-formed answer gets a lower score than a refusal. Downstream policy trained against this refuses everything.

**Detection:** if the RM systematically ranks a canned refusal string above real answers on prompts that do not warrant refusal, the collapse is happening.

**Mitigation:** balance your preference set with a substantial fraction of "chosen is a helpful answer, rejected is a refusal" pairs on prompts that clearly warrant a helpful answer.

### Format bias

The RM prefers markdown-formatted, bulleted, or code-blocked responses over prose responses, independent of content quality. Common when the LLM judge in your labeling pipeline was itself format-biased.

**Detection:** hold format constant across a subset of eval pairs (regenerate both sides with matching format) and see if the RM's preference collapses. If it does, the RM was distinguishing on format alone.

**Mitigation:** dataset-side balancing; or a two-step training where a "format-controlled" set is upweighted.

### Reward over-optimization / Goodhart

Chapter 7 covers this in detail. Preview: even a well-trained RM has an implicit ceiling — beyond a certain amount of PPO optimization against it, the policy starts producing responses that score high on the RM but *badly* on human preferences. Gao, Schulman, and Hilton (2023, "Scaling Laws for Reward Model Overoptimization") measured this precisely on a synthetic setup. There is no fix in the RM itself; the mitigation is in the PPO loop (early stopping on a held-out human eval, KL-throttling, or moving to DPO).

### Distribution shift from PPO rollouts

The RM was trained on preference pairs drawn from *some* distribution of responses. As PPO progresses, the policy's rollouts drift out of that distribution — the RM has never seen responses shaped the way the current policy generates. The RM's score on those OOD responses is essentially untrustworthy.

**Detection:** log the mean reward on rollouts over the PPO run. If it climbs monotonically while eval loss on the RM's own held-out set is flat, the RM is scoring OOD responses higher without gaining actual preference alignment. This is a proximate cause of reward hacking.

**Mitigation:** either train the RM on a broader distribution to start with, or refresh the RM mid-training on pairs drawn from the current policy's rollouts (iterative RLHF; Anthropic's "Constitutional AI" and Llama-2 both do variants of this).

## The one-page RM report

Every trained RM should ship with a one-page report analogous to the SFT run report from mod-103. Contents:

- **Base model + revision**, dataset name + revision, seed, config file.
- **Training curves.** Loss and pairwise accuracy on train and eval, at each eval step.
- **Final numbers.** Pairwise accuracy on eval (overall + length-conditional slices).
- **Calibration plot.** Margin bin → accuracy.
- **Reward histograms.** Chosen and rejected distributions overlaid.
- **Failure-mode diagnostics.** Sycophancy check (5 canned prompts), refusal check (5 prompts that should not be refused), format-bias check (10 prompts, both formats).
- **Compute + wall-clock.**

The report is what makes the RM shippable. Without it, the next person picking it up cannot tell whether it is safe to plug into PPO.

## Exercise pointer

`exercises/exercise-02-reward-model-training.md` walks you through training a reward model on your preference set from exercise 01 with TRL's `RewardTrainer`, running the four eval checks above, and producing the one-page RM report.

## Summary

- A reward model is a scalar function `r_φ(x, y)` trained on preference pairs via the Bradley-Terry loss: minimize `−log σ(r_φ(x, y_w) − r_φ(x, y_l))`. The reward is defined up to an additive constant.
- Architecturally, the RM is a decoder-only base with an `AutoModelForSequenceClassification` (`num_labels=1`) head that scores the last non-padded token.
- TRL's `RewardTrainer` is the standard recipe; `RewardConfig` sets the pairwise loss, chat-template handling, and evaluation. Small LR (1e-6 to 1e-5), one epoch, and `center_rewards_coefficient` for scale control are the defaults to start from.
- PEFT works but requires `task_type=SEQ_CLS` and `modules_to_save=["score"]` so the classifier head is fully trainable, not adapted.
- Eval: pairwise accuracy, calibration (margin → accuracy), length-conditional accuracy, reward distribution histograms, and a RewardBench-style external check.
- Failure modes: length bias (Singhal et al., 2023), sycophancy, refusal-preference collapse, format bias, and reward over-optimization / Goodhart drift during PPO. Chapter 7 goes deep on the last one.
- Ship every trained RM with a one-page report. Every downstream trainer assumes it exists.
