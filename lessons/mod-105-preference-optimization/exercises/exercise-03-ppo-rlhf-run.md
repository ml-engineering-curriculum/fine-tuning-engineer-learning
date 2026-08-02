# exercise-03: PPO-RLHF Run

**Estimated effort:** 4 hours (mostly wall-clock; PPO is slow)
**Companion chapters:** [`04-ppo-rlhf.md`](../04-ppo-rlhf.md), [`07-failure-modes.md`](../07-failure-modes.md)

## Objective

Run a classical PPO-RLHF loop that combines your SFT base, the reward model from exercise 02, and TRL's `PPOTrainer`, and diagnose the run from its KL / reward / value-loss curves. The point is not to produce a production-quality preference-tuned model in one shot — it is to feel the mechanics of PPO, watch the diagnostics move, and produce a run report that a colleague could read to understand what happened.

## Prerequisites

- The SFT model from mod-103 exercise 01 (or a public SFT model on the same base family).
- The reward model from exercise 02 (`outputs/reward-model/`).
- Prompts split from exercise 01 (`outputs/preference-set-v1/train/` — you only need the *prompts* here, not the pairs; PPO generates its own responses).
- TRL ≥ 0.9 with the modern `PPOTrainer` API. Older versions have a different arg surface — pin your version.
- A GPU with ≥ 40 GB VRAM for a full-parameter 1B–3B PPO run; ≥ 24 GB with LoRA on a 1B–3B; use QLoRA + LoRA if you want to try 7B on a single 24 GB card. PPO is memory-heavy — chapter 4 has the math.

## Requirements

### Part A — Assemble the four models

Load, in memory:

1. **Policy** — your SFT model, `AutoModelForCausalLM`, `torch_dtype=bfloat16`. Attach LoRA if you cannot fit full-parameter.
2. **Reference** — the same SFT model, frozen. If using LoRA/QLoRA, pass `ref_model=None` to `PPOTrainer` and it will use the disabled-adapter trick (verify in your `trl` version's docs).
3. **Reward model** — the trained RM from exercise 02, `AutoModelForSequenceClassification`, `num_labels=1`.
4. **Value model** — a separate SeqCls model (again `num_labels=1`), initialized from the SFT base. (Modern TRL `PPOTrainer` takes a separate `value_model` argument; older variants used a shared value head.)

Print `torch.cuda.max_memory_allocated()` right after loading all four. Log this number.

### Part B — Prepare the prompts dataset

The dataset for PPO is *prompts only* — the trainer generates its own responses. From `outputs/preference-set-v1/train/`, extract the `prompt` column and tokenize:

```python
def make_prompt_row(example):
    return {"input_ids": tokenizer(example["prompt"], truncation=True,
                                    max_length=512).input_ids}
train_prompts = dataset["train"].map(make_prompt_row,
                                     remove_columns=dataset["train"].column_names)
```

Set `tokenizer.padding_side = "left"` — generation *requires* left-padding. This is one of the top-3 PPO bugs; assert it explicitly.

### Part C — Configure `PPOConfig`

Follow the chapter-4 template. Explicit values to start from (adjust for your base size):

```python
config = PPOConfig(
    output_dir="./outputs/ppo-run",
    learning_rate=1e-6,           # 5e-6 with LoRA
    num_ppo_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    total_episodes=5_000,         # start small; extend if stable
    kl_coef=0.05,
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=0.1,
    response_length=200,
    temperature=0.7,
    bf16=True,
    logging_steps=1,
    save_steps=500,
)
```

Justify each non-default value in a `config_notes.md` — one sentence per non-default flag citing where in chapter 4 the value came from. This is not busywork; it forces you to know why you set what you set.

### Part D — Run and monitor

Kick off `trainer.train()`. Wall-clock will be dominated by rollout generation.

At every logging step, capture the following (TRL logs many of these by default; write them to a JSONL for later plotting):

- `objective/kl` (KL to reference)
- `objective/rlhf_reward` (reward after KL penalty)
- `objective/scores` (reward before KL penalty)
- `objective/entropy`
- `policy/clipfrac`
- `policy/approxkl`
- `loss/policy_avg`
- `loss/value_avg`
- `val/mean`, `val/var`
- Mean response length (compute from generated sequences)

Additionally, dump a **sample generation** every 100 steps. Pick 5 fixed prompts; generate a response with the current policy; append to `outputs/ppo-run/samples.jsonl`. You will want to eyeball how the model is drifting.

### Part E — Diagnose the run

Produce `outputs/ppo-run/diagnostics.md` with:

1. **The 6-panel plot.** One image with 6 subplots: KL, reward-after-KL, entropy, clipfrac, value loss, and mean response length. Same x-axis (training step) across all. `matplotlib` is fine.
2. **A one-paragraph narrative** of what the plots show. Did KL grow slowly and steadily? Did reward climb and then oscillate? Did the value loss converge? If entropy crashed, at what step?
3. **A verdict.** Was this run healthy or not? Which of the chapter-7 failure modes (if any) does it match? Cite the specific plot evidence.
4. **Sample generation trajectory.** Show the same prompt's generated response at steps 0, 500, 2500, and 5000. Comment on qualitative drift.
5. **Compute + wall-clock.** Total train time; per-iteration mean time; peak VRAM.

### Part F — Held-out eval

Independent from anything the trainer saw during training. On the `eval_benchmark` split from exercise 01:

- **Pairwise win-rate against SFT.** For each `(prompt, chosen, rejected)` in a random 200-sample subset of `eval_benchmark`, generate a response with (a) the SFT model and (b) the PPO-trained policy. Use a third-party judge (a stronger LLM if you have API access, or another RM different from the one you trained on) to compare `sft_response` vs `ppo_response`. Report the win rate for PPO.
- **RM reward on held-out prompts.** Score the SFT's and PPO's responses on those same prompts with your RM; report mean and std. If PPO's mean reward is much higher than SFT's *and* the third-party judge disagrees, you probably reward-hacked.
- **Response length delta.** Mean and 90th-percentile length for SFT vs PPO. Watch for the length-inflation failure mode.

Save to `outputs/ppo-run/eval.md`.

## Starter guidance

- **Start with a small `total_episodes`.** 5000 is a first-run number. Do NOT set `total_episodes=100000` on your first try — you will burn wall-clock on a run that is not learning.
- **KL blowup shows up in the first 500 steps or not at all.** If KL is well-behaved through step 500, the rest of the run will likely be stable at this LR / kl_coef.
- **Read your samples log.** Half a dozen generated responses at three different steps will teach you more about the run than any single scalar diagnostic.
- **`tokenizer.padding_side = "left"` is not optional.** Assert it before you call `trainer.train()`. Right-padding produces garbage generations, which score zero on the RM, which explode KL and confuse everyone.
- **If your RM has a length bias (exercise 02's diagnostic caught this), PPO WILL exploit it.** Response length will trend upward from the first hundred steps. Adding a length penalty to the reward — subtract `α · len(y)` inside the training loop — is the standard mitigation and worth trying.
- **Do not compare PPO's reward numbers to another RM's reward numbers.** Rewards are defined up to an additive constant; the scales are not comparable. Compare accuracy / win rate instead.
- **This is a slow run.** A 1B-class PPO on a single A100 at these settings takes on the order of 2–4 hours. Plan around it; do not baby-sit.

## Acceptance criteria

You have completed the exercise when:

- `outputs/ppo-run/` contains the trained policy weights (or LoRA adapter), the JSONL of per-step diagnostics, the JSONL of sample generations, `config_notes.md`, `diagnostics.md`, and `eval.md`.
- The 6-panel plot in `diagnostics.md` exists and covers the full run.
- Your narrative names which (if any) chapter-7 failure mode the run exhibited, with plot evidence.
- Pairwise win-rate against SFT is reported on ≥ 200 held-out prompts with a named judge.
- Mean response length is reported for SFT and PPO; delta is discussed.
- The run's total wall-clock and peak VRAM are documented.

## Stretch goals

- **Adaptive KL controller.** If your TRL version supports it (or if you implement a simple one), set a target KL (e.g., 6 nats per response) and let the coefficient adapt. Compare the resulting KL curve to your fixed-`kl_coef` baseline.
- **Length penalty.** Add a subtracted `α · response_length` term to the RM reward inside the trainer's reward-computation step. Rerun. Report the length delta vs the baseline PPO run.
- **RM ensembling.** If you did the stretch on exercise 02, plug both RMs into the PPO reward (average their scores). Compare stability.
- **Iterative RLHF preview.** After your first PPO run, sample 500 rollouts from the trained policy, label them (LLM judge or hand), retrain the RM, restart PPO from the policy. Report changes in both RM eval and downstream win-rate. This is a very small taste of what Anthropic / Meta do at scale.
- **Compare `PPOTrainer` vs the older `PPOTrainer` legacy path.** If your TRL version supports both, run the same setup through both and compare. The rewritten trainer is closer to InstructGPT's recipe; the legacy is simpler. Not a required comparison but a useful teaching diff.
- **Serve the trained policy.** Merge the LoRA adapter (mod-104 chapter 3) or export the full model, load in vLLM, and serve a chat endpoint. Try 20 real prompts and read the responses. This is what "shipping" the PPO output looks like.
