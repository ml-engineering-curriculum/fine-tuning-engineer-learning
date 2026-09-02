# exercise-04: DPO Run and Comparison to PPO

**Estimated effort:** 4 hours (mostly training wall-clock)
**Companion chapters:** [`05-dpo.md`](../05-dpo.md), [`07-failure-modes.md`](../07-failure-modes.md), background from [`04-ppo-rlhf.md`](../04-ppo-rlhf.md)

## Objective

Run DPO on the same SFT base and preference set you used in exercise 03, produce a side-by-side comparison to that PPO-RLHF run, and defend a choice of method for a specified deployment scenario. The point is not to declare a universal winner — chapter 6's decision matrix already argues DPO is the honest default for chat-style SFT tuning and PPO is the frontier / heavy-lift option. The point is to *feel* the difference in ergonomics, wall-clock, memory, and behavior on the same starting conditions, so the next time you have to pick, the pick is informed by a run you actually did.

## Prerequisites

- The SFT model from mod-103 exercise 01 (or the same public SFT you used in exercise 03).
- The preference dataset from exercise 01 (`outputs/preference-set-v1/`).
- The PPO artifacts from exercise 03 (`outputs/ppo-run/`) — you will compare against this run directly.
- `trl` ≥ 0.9 with a working `DPOTrainer` and `DPOConfig`. Pin the version.
- `peft`, `bitsandbytes` (for QLoRA), `accelerate`, `datasets`, `transformers`.
- A GPU with ≥ 24 GB VRAM for a QLoRA + LoRA DPO run on a 1B–7B base. Full-parameter DPO at 7B needs closer to 48 GB. Chapter 5 has the memory math.
- The reward model from exercise 02 (`outputs/reward-model/`) — used as one of the eval judges here, not for training.

## Requirements

### Part A — Configure the DPO run

Use the *same* SFT base as your exercise-03 PPO run so the comparison is honest. Wire up TRL's `DPOTrainer` following the chapter-5 recipe. Explicit values to start from (adjust for your base):

```python
config = DPOConfig(
    output_dir="./outputs/dpo-run",
    beta=0.1,                              # chapter 5's default; the load-bearing knob
    loss_type="sigmoid",                   # standard DPO; you will vary this in Part F
    learning_rate=5e-6,                    # 1e-4 range with LoRA
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    max_length=1024,
    max_prompt_length=512,
    bf16=True,
    eval_strategy="steps",
    eval_steps=200,
    logging_steps=25,
    save_steps=500,
    report_to=["wandb"],                   # or "tensorboard" — pick one and stick with it
)
```

If you are using QLoRA + LoRA (the recommended recipe for 24 GB GPUs), pass `ref_model=None` and let `DPOTrainer` use the disabled-adapter trick — chapter 5 details this and chapter 7 flags the reference-drift risk you must audit for.

Justify each non-default value in a `config_notes.md` — one sentence per non-default flag citing where in chapter 5 (or 7) the value came from. Same rule as exercise 03: not busywork, forces you to know why you set what you set.

### Part B — Train and log

Kick off `trainer.train()` on the `train` split of your preference set; use the `eval_loss` split as `eval_dataset`.

At every logging step, capture the following (TRL logs most by default; write them to a JSONL for later plotting):

- `train/loss` and `eval/loss`
- `train/rewards/chosen` and `train/rewards/rejected` — the *implicit* rewards (`β · log(π_θ / π_ref)`); their trajectories are the DPO analog to PPO's reward curve
- `train/rewards/accuracies` — fraction of pairs where chosen implicit reward > rejected
- `train/rewards/margins` — `chosen − rejected`
- `train/logps/chosen` and `train/logps/rejected` — raw log-probs, needed to detect log-prob deflation (chapter 5's failure)
- Effective KL against the reference — chapter 5 gives the formula (`(logps_chosen - ref_logps_chosen).mean()` on the eval split)

Additionally, dump a **sample generation** every 200 steps on the same 5 fixed prompts you used in exercise 03. Append to `outputs/dpo-run/samples.jsonl`. Reusing the exercise-03 prompts is required — the qualitative trajectory comparison in Part D depends on it.

Log peak VRAM (`torch.cuda.max_memory_allocated()`) after loading models and again at end of training. Log total wall-clock.

### Part C — Diagnose the run

Produce `outputs/dpo-run/diagnostics.md` with:

1. **The 4-panel plot.** One image with 4 subplots: (a) train/eval loss, (b) `rewards/chosen` and `rewards/rejected` on the same axes, (c) `rewards/margins` and `rewards/accuracies`, (d) `logps/chosen` and `logps/rejected` overlaid. Same x-axis (training step). `matplotlib` is fine.
2. **A one-paragraph narrative** of what the plots show. Did margins grow monotonically or oscillate? Did `logps/chosen` climb, drop, or stay flat? Did *both* logps drop (log-prob deflation, chapter 5)? Did eval loss reverse within the epoch (overfit, chapter 6's IPO motivation)?
3. **A verdict.** Is the run healthy, or does it match one of chapter 7's failure modes? Cite the specific plot evidence. Pass-through of "healthy" is a valid verdict if the numbers support it.
4. **Step-0 KL sanity check.** Assert and report: at training step 0, effective KL against the reference is zero (or within numerical noise). If not, you have a reference-model bug (chapter 7's "reference drift" section); fix before continuing.

### Part D — Held-out eval

Independent from anything the trainer saw. Reuse the *exact same* 200-sample subset of `eval_benchmark` you used in exercise 03 — pin the seed. Produce:

- **Pairwise win-rate against SFT.** For each prompt, generate a response with (a) the SFT model, (b) the DPO-trained policy. Judge with the same third-party judge you used in exercise 03. Report the win rate for DPO.
- **Pairwise win-rate against PPO.** For each prompt, generate with the PPO model from exercise 03 and the DPO policy; judge; report DPO's win rate vs PPO. This is the core comparison of this exercise.
- **RM reward on held-out prompts.** Score SFT / PPO / DPO responses with the reward model from exercise 02. Report mean and std per model. Big DPO−SFT gap with a small judge-win-rate gap is a reward-hack signal; note it.
- **Response length delta.** Mean and 90th-percentile length for SFT / PPO / DPO. Watch for the length-inflation failure mode across both trained models.

Save to `outputs/dpo-run/eval.md`.

### Part E — The side-by-side comparison

`outputs/comparison-ppo-vs-dpo.md`, ~1 page. This is the deliverable that separates a "ran two trainers" submission from a "understand the trade-off" one.

Cover, in a single table where possible:

| Dimension | PPO (exercise 03) | DPO (this exercise) |
|---|---|---|
| Trainer + version | `PPOTrainer`, TRL `X.Y.Z` | `DPOTrainer`, TRL `X.Y.Z` |
| Loaded models (policy / ref / RM / value) | 4 models | 2 models (or 1 with LoRA disabled-adapter) |
| Trainable parameters | | |
| Peak VRAM | | |
| Wall-clock (total; per-step median) | | |
| Learning-signal source | On-policy rollouts scored by RM | Frozen preference pairs |
| Training-time diagnostic that actually moved | reward-after-KL, KL, entropy | implicit rewards, margins, logps |
| Pairwise win-rate vs SFT (held-out judge) | | |
| RM mean reward on held-out | | |
| Median response length | | |
| Failure mode hit (if any) | | |

Below the table, write **three paragraphs**:

1. **What was easier about DPO.** Speak from the specific artifacts you produced. "PPO required loading four models and a value head; DPO required two. My VRAM peak was Xg vs Yg." Be concrete.
2. **What was easier about PPO.** Steelman it. "PPO's on-policy rollouts caught a failure mode DPO cannot see — specifically ..." or "PPO's explicit KL knob let me control drift directly." If you cannot find anything, say so explicitly — that itself is a finding.
3. **Which method actually produced better generations.** Not the numbers — you already put those in the table. Read 10 side-by-side sample outputs and write a paragraph on qualitative differences. If they are indistinguishable, say so.

### Part F — Defend a choice for a deployment scenario

Pick **one** of the following deployment scenarios, then write a ~300-word recommendation defending which of PPO or DPO you would ship for it, citing evidence from the run:

- **A.** Small startup, 2 engineers, single 24 GB consumer GPU. Weekly iteration on a chat assistant for a niche vertical. Preference data comes from user thumbs-up/down.
- **B.** Mid-size org, 8 engineers, 8× H100 node. Monthly release cadence for a general-purpose chat model. Preference data from a mix of human annotators (~5k pairs / month) and an LLM judge (~50k pairs / month).
- **C.** Frontier research lab, unlimited compute. Quarterly release cadence with a heavy safety-alignment focus. Preference data from a red-team pipeline + expert reviewers, refreshed continuously.
- **D.** Solo hobbyist, 12 GB consumer GPU, one-shot experiment on 5k preference pairs from an open source (UltraFeedback subset).

The recommendation should reference your comparison table numbers, chapter 6's decision matrix, and chapter 7's failure modes if relevant. A recommendation that could have been written before running either trainer is a failing recommendation.

### Part G — (Optional but strongly recommended) The `loss_type` sweep

DPO's `loss_type` is a single-arg change. On a subset of your training set (fastest: 2000 pairs), rerun with `loss_type="ipo"` (chapter 6). Log the same diagnostics. Compare margins and eval-loss trajectory to the sigmoid run. This is a preview of exercise 05's survey experiment and takes ~30 minutes of extra wall-clock.

## Starter guidance

- **Same base, same data, same seed for prompt sampling.** The whole point of the comparison is that the *only* varied axis is the trainer. Any other variable you vary makes the comparison harder to read.
- **`β = 0.1` is the community-consensus DPO default; don't reach for a different value on the first run.** Chapter 5 is explicit: the β-sweep belongs in a second-pass exercise, not the first. Note the value; move on.
- **Watch `logps/chosen` from the first checkpoint.** If it is dropping while `rewards/chosen` is climbing, you are seeing the "log-prob deflation" failure (chapter 5): the margin is growing because rejected drops faster, but chosen is being pulled *down* in absolute terms. This is the top DPO bug worth catching early.
- **Reference-model drift is a QLoRA-specific gotcha (chapter 7).** Assert `step-0 KL == 0` before any training happens. If it is not zero, `ref_model=None` is not actually giving you a clean reference — inspect the disabled-adapter path in your `trl` version.
- **DPO is fast.** A 1B-class DPO run on a single A100 at these settings should complete in well under an hour. If you find yourself waiting 3+ hours for one epoch, something is off (usually: LoRA is not attached and you are training all parameters, or `batch_size × grad_accum` produced a giant effective batch).
- **The judge matters more than the trainer.** If your third-party judge is weak (e.g., an old model or one with a strong length bias), the win-rate numbers in Part D will be dominated by the judge's biases, not the model's improvements. Use a modern strong judge (frontier LLM API access if you have it; a strong RM like `sfairXC/FsfairX-LLaMA3-RM-v0.1` if you do not) and record which judge you used.

## Acceptance criteria

You have completed the exercise when:

- `outputs/dpo-run/` contains the trained policy (adapter or full weights), the per-step diagnostics JSONL, the samples JSONL, `config_notes.md`, and `diagnostics.md`.
- The 4-panel plot in `diagnostics.md` exists and covers the full run.
- Your narrative names which (if any) chapter-5 or chapter-7 failure mode the run exhibited, with plot evidence.
- Step-0 KL is asserted zero and the value is recorded.
- `outputs/comparison-ppo-vs-dpo.md` exists with the full comparison table filled in *and* the three paragraphs. Empty rows or "TBD" cells are a fail.
- Held-out win-rate is reported for **both** DPO-vs-SFT and DPO-vs-PPO on ≥ 200 pinned prompts with a named judge.
- The deployment-scenario recommendation is written and cites at least two specific numbers from the comparison table.

## Stretch goals

- **β sweep.** Rerun DPO at `β ∈ {0.01, 0.05, 0.1, 0.3, 1.0}` on a subset. Plot margins and eval loss per β. Which β maximized held-out win-rate on your data? Does it match chapter 5's default (0.1)?
- **Reference-model swap.** Swap the reference from your SFT model to (a) the base pretrained model (no SFT), (b) an entirely different SFT checkpoint. Rerun DPO. Which reference gives better win-rate? Which drifts most?
- **Iterative DPO.** After the first DPO run, sample fresh responses from the trained policy on unseen prompts, re-label them (LLM judge or a stronger RM), add them to the training set, restart DPO from the current policy with the current policy as the new reference. This is Meta's "Self-Rewarding LM" pattern in miniature; the win-rate should climb further.
- **DPO on a length-controlled subset.** Filter your preference set so `|chosen| ≈ |rejected|` (within 20 tokens) and rerun DPO. Report the length delta in generated outputs vs the unfiltered run. This is the direct empirical test of chapter 7's length-bias thesis.
- **Merge and serve.** Merge the LoRA adapter (mod-104 chapter 3), load in vLLM, and serve. Run 20 real prompts through both the PPO-served endpoint and the DPO-served endpoint and read the outputs. This is what "shipping the winner of the comparison" looks like.
- **Reproduce Zephyr.** Follow the alignment-handbook DPO recipe end-to-end (SFT → DPO on UltraFeedback-Binarized) on a 7B base. Report MT-Bench score against the published Zephyr-7B-β numbers. Being within 2 points is a strong signal your pipeline is honest.
