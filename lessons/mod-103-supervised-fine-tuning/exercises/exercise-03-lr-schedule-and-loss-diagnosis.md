# exercise-03: LR Schedule and Loss Diagnosis

**Estimated effort:** 4 hours (mostly wall-clock — the compute is small)
**Companion chapters:** [`03-learning-rate-warmup-and-checkpoint-cadence.md`](../03-learning-rate-warmup-and-checkpoint-cadence.md), [`04-reading-loss-curves.md`](../04-reading-loss-curves.md)

## Objective

Sweep three learning rates and two schedules on a fixed dataset and base model. Produce loss-curve plots for each of the six runs. Then, for each run, write a one-paragraph diagnosis using chapter-4's shape guide and pick which run you would ship — defending the choice against the loss curves, not against intuition.

The point is not to identify the "best" LR. The point is that you can *look at* six curves and correctly say what each one is doing.

## Prerequisites

- Same environment as exercise 01.
- The base model + dataset combination from exercise 01. Do not switch here — half the point is that the *only* thing changing is LR × schedule.
- Enough compute to run six short SFT jobs. On a 1B base with a 5–10k row dataset and 1 epoch, that is roughly 6 × 20–60 minutes on a single 24 GB GPU.

## Requirements

### Part A — Fix everything except LR and schedule

Author `sweep.py` (or a runner script) that runs the six configurations below. Everything except `learning_rate` and `lr_scheduler_type` must be identical across runs — same dataset (same seed for shuffling), same effective batch size, same `max_length`, same `packing`, same `warmup_ratio`, same `weight_decay`, same `num_train_epochs`, same seed.

Six runs:

| run_id | learning_rate | lr_scheduler_type |
|--------|---------------|-------------------|
| lr-lo-cos | 5e-6  | cosine  |
| lr-mid-cos | 2e-5 | cosine  |
| lr-hi-cos | 1e-4  | cosine  |
| lr-lo-lin | 5e-6  | linear  |
| lr-mid-lin | 2e-5 | linear  |
| lr-hi-lin | 1e-4  | linear  |

The three LRs span the interesting range for a modern 1B–7B base under full-parameter SFT (chapter 3 anchors: full-param ≈ 1e-5 to 5e-5, so the mid value is centered and the extremes bracket "too low" and "too high"). If you are doing this sweep on a LoRA config (mod-104), shift the range one order of magnitude up (5e-5 / 2e-4 / 1e-3).

Emit each run's logs into `outputs/lr-XX-YY/` (its own directory) so runs are unambiguously separable.

### Part B — Plot the six curves

Produce a single figure (`sweep.png`) with:

- **Left column:** the three cosine runs, one axis, three colored lines for train loss + three lighter-colored lines for eval loss.
- **Right column:** the three linear runs, same layout.
- **X axis:** optimizer step. **Y axis:** loss, log scale.
- Legend that names each run by `run_id`, LR, and schedule.

Also produce three per-run plots (`outputs/lr-XX-YY/loss.png`) with the two-curve view for that run alone. Chapter 4 relies on the per-run view for diagnosis; the sweep-comparison view answers a different question.

### Part C — Diagnose each run

For each of the six runs, write a paragraph in `diagnosis.md`. Each paragraph:

1. **Names the shape** — healthy, flat, exploded, eval-diverging, plateau, suspiciously smooth, spiked, curve-fine-output-bad. Use chapter 4's vocabulary.
2. **Cites the numeric guideposts** — step-0 loss, end-of-warmup loss, final train loss, best eval loss and step.
3. **Cross-references the gradient norm** for any anomalies. If `grad_norm` spiked or diverged, say so.
4. **States what the run is telling you** in plain English. "LR is too high, the model has clearly overshot"; "LR is too low, warmup ate half the schedule and the model barely learned"; "healthy, ships."

### Part D — Ship-decision

Add a `pick.md` (one page) at the top of `outputs/` answering:

1. Which of the six runs would you actually ship, and why?
2. Which two runs are closest in numbers, and how do you break the tie? (Best eval loss, throughput, checkpoint hygiene, template smoke test — pick a policy.)
3. What would you change *next*? A finer LR sweep between the two closest? A different warmup ratio? More epochs?

Do not hedge. "It depends" is not a valid answer here — you are the person picking. Say what you would ship and why, and then say what you would try next.

### Part E — Optional: cross-reference with mid-run generations

For each run, at the final checkpoint, generate 3 responses to the same 3 canned prompts (temperature 0). Paste them into `diagnosis.md` under each run.

You will often find a run whose loss curve is fine but whose generations betray the failure mode — a template leak, over-confident refusal, or a tell-tale training-set phrase (chapter 5's overfitting case). This is *the* exercise where you learn that loss curves are necessary but not sufficient.

## Starter guidance

- Time-box each run. If you are on a small GPU, cap each of the six runs at 30 minutes wall-clock and use a `max_steps` limit rather than `num_train_epochs`. The relative shape of the curves is what you are comparing, and 500 steps is enough to show it.
- Do all six runs from the same script so they inherit the same seed and hyperparameters by construction. Do not run them from six copy-pasted scripts — the accidental divergence rate is 100%.
- Log to a single tracking backend (WandB, TensorBoard, JSONL — one of them) and use consistent run names. Diagnosis is much faster when six runs are on the same axes than six PNGs.
- Do not skip the `lr-hi-*` runs "because obviously they will diverge." Watching them diverge is how you learn to spot the shape in the future.
- If none of your six runs exhibit an unhealthy shape, your dataset is small or gentle enough that even the "high" LR is tolerable. Bump the high LR another 5× and re-run one to actually see the exploded shape.

## Acceptance criteria

You have completed the exercise when:

- All six runs produced logs, checkpoints, and per-run plots.
- `sweep.png` shows both columns (cosine / linear) side by side with train + eval loss for all three LRs.
- `diagnosis.md` contains six paragraphs, each naming the shape, citing guideposts, and explaining the run.
- `pick.md` unambiguously names the ship candidate with a defensible rationale, an explicit tiebreaker if two runs are close, and a next-step recommendation.
- A colleague reading only `pick.md` and `sweep.png` can independently arrive at the same ship decision, or can articulate exactly where they would disagree.

## Stretch goals

- **Add a constant-with-warmup baseline.** Run one more with `lr_scheduler_type="constant_with_warmup"` at the mid LR. Compare to the cosine mid run. When would constant-with-warmup be preferable?
- **Sweep warmup ratio.** Fix LR at the winner from Part D and sweep `warmup_ratio` ∈ {0.0, 0.03, 0.1, 0.3}. Which warmup fraction gives the cleanest early-training shape? Does the winner change if the run is longer or shorter?
- **Redo the sweep with LoRA.** Use the equivalent LR range for LoRA (10× higher). Diff the curves. LoRA runs are famously more forgiving of high LR — is that visible in your loss curves?
- **Test the "more warmup for very short runs" rule from chapter 3.** Run one at `warmup_ratio=0.03` and one at `warmup_ratio=0.1` on the same 200-step run. Does the higher-warmup one have a healthier shape? By how much?
