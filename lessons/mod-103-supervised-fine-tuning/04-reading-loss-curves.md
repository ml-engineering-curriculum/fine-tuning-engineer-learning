# Reading Loss Curves

**Estimated read:** 35 minutes.
**You should be able to, by the end:** given a training-loss + eval-loss + gradient-norm plot from a real SFT run, call it "healthy," "under-trained," "over-trained," "LR too high," "LR too low," "data broken," or "template broken" — and defend the call in a sentence.

## The two curves that matter

Every SFT run produces a lot of scalars. Only two are load-bearing:

- **Training loss** (`loss` in the trainer log) — cross-entropy on the loss-masked tokens of the current batch. Reported every `logging_steps`.
- **Eval loss** (`eval_loss`) — the same quantity, computed on a fixed held-out slice of your SFT dataset. Reported every `eval_steps`.

Everything else — gradient norm, learning rate, throughput, memory usage — is a diagnostic support scalar. Chapter 3 explained what the LR schedule does; this chapter explains how to read the loss curves and cross-reference them against the diagnostics.

## The healthy-run shape

A healthy full-parameter SFT run of a 1B–7B base on a well-curated mix produces a curve that looks like this in the log-y-axis plot:

```
loss
  │  ╲
  │    ╲
  │      ╲__
  │         ─___
  │              ─___                        ← smooth decay through cosine
  │                   ─__________            ← flattens near end-of-training
  │─────────────────────────────────  eval
                                              ← eval loss tracks train loss with a small +Δ
  step →
```

Numeric guideposts (order-of-magnitude, for a modern instruct base on English chat data at BF16):

- **Step 0** — loss in the 2–4 range for chat data on a modern base. If step 0 is > 8 or is NaN, your data or your tokenizer is broken; stop the run.
- **End of warmup** — loss should already be trending down. If it is flat at end of warmup, your LR is probably too low.
- **Mid-training** — smoothly decreasing. Small oscillations from batch to batch are normal; a monotone-looking curve is either eval loss (not train) or logged too infrequently.
- **End of training** — loss in the 0.5–1.5 range for well-formatted assistant responses. Numbers much lower than 0.5 for full-parameter SFT of a small dataset are a warning sign for overfitting (chapter 5).

**Do not memorize the exact numbers.** Memorize the *shape*. Any deviation from the shape is a signal.

## The seven common deviations, and what each one means

### 1. Flat train loss through warmup and after

```
loss ── ─ ─ ─ ─ ─ ─ ─ ─ ─
```

**Diagnosis:** either your LR is far too small, your data is broken (all `-100` labels, corrupted rows), or your masking is producing zero-loss batches.

**Verify:** decode one collated batch (chapter 2). If `sum(labels != -100) == 0` on many batches, your masking is broken. If the batch looks fine, raise LR by 3–10× and re-run a short (100-step) sanity job.

### 2. Loss explodes early (NaN or > initial value)

```
loss  │        ╱─╲
      │       ╱   ╲___
      │      ╱         NaN
      │     ╱
      │────╱                   ← warmup up
```

**Diagnosis:** LR peaked before warmup could tame the early gradients, or gradient norms are exploding.

**Verify:** check gradient-norm plot. If `grad_norm` spikes to hundreds/thousands, you either need more warmup or a lower peak LR. Also check `max_grad_norm=1.0` is actually on (it is a `SFTConfig` default; verify you did not disable it).

**Fix:** halve the LR, double the warmup ratio, and re-run.

### 3. Train loss decreases, eval loss increases mid-run

```
train  ─ ╲___
              ─___
                   ─__
eval   ─    _____________________
                                   ╱
                              ___╱
                        ___╱
                   ___╱                    ← eval bottomed and is rising
```

**Diagnosis:** classic overfitting. You have trained past the point of generalization on this dataset. See chapter 5 for full treatment.

**Verify:** confirm the eval slice is disjoint from the train slice (unbelievably common bug). If it is, the training gap is real.

**Fix:** roll back to the checkpoint at the eval-loss minimum (this is why `load_best_model_at_end=True` from chapter 3 exists), reduce epochs, or grow the dataset. **Never** just "keep training and hope."

### 4. Train and eval loss both plateau early

```
train  ─╲__________________________
eval   ─────╲___________________________
```

**Diagnosis:** under-training. The model is doing what it can with what you gave it and there is more to learn.

**Verify:** is your LR schedule almost done (cosine near its floor)? Are you within 1–2 epochs? Compare a generation from this checkpoint to a base-model generation — if they look nearly identical, you also under-trained.

**Fix:** more steps (more epochs, or a longer dataset), or a slightly higher peak LR.

### 5. Loss is *incredibly* smooth (suspicious)

```
train  ────────╲___________________
eval   ────────╲___________________
                                     ← lines lie exactly on top of each other
```

**Diagnosis:** likely a mask bug — every batch is producing (nearly) the same loss because the loss is being computed on almost no tokens per batch, or the same tokens per batch. Also possible: you accidentally set `eval_dataset = train_dataset`.

**Verify:** log `num_tokens_predicted` per step (a custom callback that sums `(labels != -100).sum()` in the batch). If this number is tiny or constant, your masking is not producing what you think.

**Fix:** re-verify the decode-and-print output from chapter 2. Fix the mask; do not proceed.

### 6. Loss drops, then abruptly spikes at a specific step

```
train  ─╲___
             ─___
                  ─__
                       ╲              ← sharp downward spike
                        │
                        ╱──────       ← recovers to slightly-worse baseline
```

**Diagnosis:** one bad batch. Data quality issue in a specific row (long garbage, wrong-language text, corrupted encoding), or a specific gradient-norm spike that grad clipping caught late.

**Verify:** cross-reference against `grad_norm` at that step. Log the batch index. Inspect the rows in that batch.

**Fix:** the row(s) responsible are usually obvious once you look — fix the dataset (mod-102 chapter 1's filter stack) or add a specific filter. Rerun.

### 7. Loss curve looks fine, model output is garbage

**Diagnosis:** the model is training on tokens it should not be training on (mask inverted), or your eval-time template does not match your training-time template (mod-102 chapter 4 covered the byte-diff assertion).

**Verify:** generate 5 responses from the trained model with `apply_chat_template(..., add_generation_prompt=True)`. If the responses contain user-turn-like text, the mask is inverted. If they emit stray `<|im_end|>` or `<|eot_id|>` tokens or fail to emit them at all, template drift.

**Fix:** re-run the decode-and-print. Fix the mask. Diff the rendered strings.

## Gradient norm: the diagnostic that saves runs

`grad_norm` (the L2 norm of the gradient across all parameters, pre-clipping) is what you look at when the loss curve is doing something weird. TRL logs it by default; if you configured a custom trainer, add a callback.

Sane behavior:

- **Warmup:** `grad_norm` starts high (fresh model, fresh distribution) and drops through warmup as the LR ramps and gradients stabilize.
- **Steady-state:** `grad_norm` around 0.1–2.0 for full-parameter SFT with sane data. LoRA / QLoRA may sit an order of magnitude higher because the trainable parameter set is small and updates are concentrated.
- **`max_grad_norm=1.0` clipping.** Pre-clip norms that regularly exceed 1.0 (log around 3–10) indicate clipping is firing every step; that is a signal your LR is too high or your data has occasional very-hard batches.

Two specific patterns to recognize:

- **`grad_norm` spikes correlated with loss spikes** → bad batch (case #6 above).
- **`grad_norm` climbing steadily over the run** → training is destabilizing; either LR is too high, warmup was too short, or numerical issues in mixed precision. Consider FP32 the affected matmul or switch to BF16 if you were on FP16.

## Throughput as a diagnostic

`train_samples_per_second` and `train_tokens_per_second` (log yourself as a callback if not present) should be roughly constant across a run. Sudden drops mean:

- **The dataloader can't keep up.** GPU utilization dropping and throughput falling in step. Increase `dataloader_num_workers`.
- **You crossed into eval or checkpoint save.** Not a bug; the trainer pauses for these. Verify by cross-referencing the step number against `eval_steps` / `save_steps`.
- **Memory fragmentation.** Rare at the 1B–8B scale, more common with FSDP on longer runs. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` sometimes helps.

Throughput is not a quality signal. But a run where throughput mysteriously halved is a run where something changed you should understand before trusting the checkpoint.

## The two-plot rule

When you finish a run, the first thing you look at is one plot with **two curves on it**:

- Training loss (with heavy log frequency during warmup, standard frequency after).
- Eval loss (at every `eval_steps`).

If those two curves show the healthy shape and the numeric guideposts, you can *tentatively* call the run good and move on to eval (mod-108). If they do not, no downstream eval will save the run — the failure is already visible in the plot.

That is the whole rule. Look at the two-curve plot before running expensive evals. Anything else is expensive procrastination on training-loop debugging.

## What to actually log

At a minimum, on every `logging_steps`:

- Step number.
- Loss.
- Learning rate.
- Gradient norm (pre-clipping).
- Throughput (samples/sec and tokens/sec).
- Number of tokens in the batch that contributed to loss (`(labels != -100).sum()`).

Every `eval_steps`:

- Eval loss.
- Any custom quality-proxy metric from chapter 3.

Every `save_steps`:

- Checkpoint path.
- Whatever `load_best_model_at_end` needs to make its choice.

Log to Weights & Biases, TensorBoard, `mlflow`, or a JSONL file — pick one and use it consistently across all runs so runs are comparable. The training report (chapter 6) is a rendering of these scalars into a page a human can read.

## Slicing loss by data attribute

Because your dataset rows carry the metadata envelope from mod-102 chapter 1 (`source`, `category`, `generator`, `language`), you can slice eval loss by any of those fields. Two versions of this that pay off:

- **Loss by source.** If OASST1 and Alpaca have very different eval losses, you have a per-source quality disparity that will show up in downstream evals as a category-specific regression.
- **Loss by category.** If reasoning-tagged examples have loss 2× the average, either the model is genuinely weak on reasoning (expected; that is why you added the data) or the reasoning examples are noisy (mod-102 problem).

Slicing this way requires that you emit per-row losses, not just batch means. Add a callback that, on eval, computes loss per example and joins it back to the metadata. A five-line callback that produces a five-line report is one of the highest-leverage additions you can make.

## Exercise pointer

`exercises/exercise-03-lr-schedule-and-loss-diagnosis.md` has you produce the two-plot view for three LRs × two schedules on the same dataset and write the one-paragraph diagnosis for each. The point is that you leave the exercise able to name what a curve is doing without hedging.

## Summary

- The two-curve (train + eval loss) plot is the primary artifact of an SFT run. Look at it before running downstream evals.
- The healthy shape is: warm up, smooth decrease, small train/eval gap, flatten near the end of the schedule. Numeric guideposts: 2–4 at step 0, 0.5–1.5 at end for well-formatted assistant data.
- Seven common deviations: flat, exploded, eval-diverging, plateau, suspiciously smooth, spiked-at-a-step, curve-fine-output-bad. Each maps to a specific class of fix.
- Gradient norm is the diagnostic that tells you *why* the loss curve is doing something weird. Log it every step; correlate spikes with loss spikes.
- Throughput is not a quality signal but it is a "something changed" signal. Investigate mysterious drops before trusting the checkpoint.
- Log: step, loss, LR, grad_norm, throughput, tokens-in-loss per micro-batch. Slice eval loss by dataset metadata to localize regressions.
