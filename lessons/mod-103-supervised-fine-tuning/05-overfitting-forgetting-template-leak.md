# Overfitting, Catastrophic Forgetting, and Template Leak

**Estimated read:** 40 minutes.
**You should be able to, by the end:** name three failure modes that specifically hit SFT runs, describe the loss / eval / hand-inspection signals that distinguish them, and pick a small ordered list of mitigations for each — knowing which one to try first and why.

These three failure modes account for most of the "we shipped an SFT model and it got worse" incidents in real teams. They look different from generic pretraining failure modes and require different tools. This chapter is a taxonomy plus a mitigation ladder.

## Failure mode 1: overfitting

**What it is.** The model has memorized the training set well enough that its behavior on unseen inputs — even inputs drawn from the same distribution — degrades. In SFT this most often looks like: near-zero training loss, rising or plateauing eval loss, and a model that repeats template phrases from training verbatim on new prompts.

### Signals

Loss:

- Chapter 4 pattern #3: train loss keeps falling while eval loss bottoms out and rises.
- Train loss under ~0.5 on full-parameter SFT of a small dataset is a warning sign; under ~0.2 is a strong signal.

Generation:

- Prompt the model with something clearly out-of-distribution but plausible; it responds with a phrase or format lifted from a training row.
- Ask for a task variation of a training example; the model gives the exact training-set response with mechanical substitutions.

Held-out eval:

- MMLU-style multiple choice regresses even though your SFT set has no MMLU-shaped data — a symptom of "the model has learned a specific answer format at the expense of general capability."

### Root causes

The three usual suspects, in likelihood order:

1. **Too many epochs.** Small dataset, three or more epochs, no early stopping. See chapter 3 on epoch counts.
2. **Learning rate too high for the dataset size.** Small dataset + full-parameter update magnitudes = memorization within a fraction of an epoch.
3. **Dataset is too narrow.** The model has no choice but to over-index on the narrow behavior your data teaches. Not really a training bug; a data bug.

### Mitigation ladder (try in this order)

1. **Roll back to the eval-loss-minimum checkpoint.** `load_best_model_at_end=True` from chapter 3 makes this a config change, not a code change. Costs nothing and often is enough.
2. **Reduce epochs.** If a 3-epoch run overfit at epoch 2, ship the 1-epoch run.
3. **Lower LR by a factor of 2–3.** Only after (1) and (2) — LR change is a full re-run.
4. **Grow the dataset.** Mod-102 chapter 6 covers mixing across capabilities; adding well-curated data from adjacent domains is the durable fix.
5. **Regularization (weight decay bump, dropout on the projection layers if the model supports it).** Diminishing returns in SFT; try only if the above did not work.

Note what is *not* on the ladder: "add noise to the input." That is a pretraining regularization technique and is essentially useless in SFT of a base model.

## Failure mode 2: catastrophic forgetting

**What it is.** The SFT run improved the model on the target behavior at the cost of degrading capabilities that were present in the base — general knowledge, non-target languages, code, reasoning. Catastrophic forgetting is well-documented in continual-learning literature (McCloskey & Cohen, 1989; French, 1999) and is a distinctive SFT failure because you are updating a model that was expensive to pretrain.

### Signals

Held-out eval (this is the signal that matters):

- Base model on MMLU: 45%. Your SFT'd model on MMLU: 41%. That is forgetting; the base did not know things the SFT model has now forgotten.
- Base on HumanEval: 22%. SFT'd model on HumanEval: 14%. Same story.
- Loss curves alone will NOT tell you this is happening. Your eval loss on the SFT eval slice will look fine because that slice does not test the forgotten capabilities.

Generation:

- The model answers your target-domain queries well and gives shorter, less accurate answers to general-knowledge questions than the base did.
- If your SFT was heavy on refusal data, the model over-refuses benign queries the base handled — this is over-refusal, a specific safety-tuning failure mode that mod-106 covers, but the mechanism is the same as generic forgetting.

### Root causes

1. **LR too high for the base model size.** Every full-parameter step moves every weight. Bigger LR × more steps × bigger updates = base capabilities decaying faster.
2. **SFT dataset too narrow relative to the base's original capabilities.** The gradient consistently points in one direction; weights that only mattered for the base's other capabilities are not reinforced.
3. **Too many epochs.** As above.
4. **Un-diverse chat template.** If every training example has exactly one system prompt, the model becomes rigidly conditioned on that string and generalizes poorly to inference-time system prompts (mod-102 chapter 4).

### Mitigation ladder

1. **Measure the delta first.** Never guess at forgetting. Run MMLU or a comparable well-known benchmark on the base and on your SFT'd model. Report the number. Exercise 04 makes you do this.
2. **Lower the LR.** For full-parameter runs this is the single most effective mitigation. Halve the LR, keep everything else equal, re-run. Compare.
3. **Switch to LoRA / QLoRA.** LoRA touches a tiny fraction of parameters; the base weights are literally unchanged, and catastrophic forgetting is dramatically reduced (Biderman et al., 2024, is a useful reference on this specific point). Mod-104 covers the mechanics.
4. **Add general capability data ("replay").** Mix a small fraction of general-domain data (Tülu 3's mix is a reference) into your SFT set specifically to reinforce the base's original capabilities. Even 10–20% is often enough.
5. **KL-regularized objectives.** DPO / IPO / KTO with a reference model built into the loss (mod-105) reduce forgetting relative to plain SFT. Not strictly an SFT mitigation, but the next stage in your pipeline.
6. **Reduce epochs.** Fewer passes = less drift.

The mitigation ladder is different from overfitting's: (1) *measure* is not optional here because the run's loss curve won't tell you.

### The specific case: SFT of an Instruct base

Fine-tuning an already-instruct model (e.g., you take `Llama-3.2-1B-Instruct` and SFT it on your customer-support data) is uniquely at risk of catastrophic forgetting, because the base already has the instruction-following capability you would otherwise be teaching and your SFT can only overwrite it. Recommendations:

- Use a lower LR than you would for the base (1e-5 or below).
- Prefer LoRA / QLoRA over full-parameter.
- Mix in a small amount of the base's own SFT-style data (or a public analog) to preserve general instruction-following.
- Always measure the delta on a general benchmark before / after.

## Failure mode 3: template leak

**What it is.** The model's outputs contain (or omit) template scaffolding in a way that reveals training-loop bugs and breaks the inference-time contract. Two symmetric versions:

- **Positive leak** — the model emits template tokens (`<|eot_id|>`, `<|im_end|>`, `<|start_header_id|>system<|end_header_id|>`, etc.) in the middle of its response text, where they should never appear.
- **Negative leak** — the model fails to emit its own turn terminator, so responses run on indefinitely or the serving stack cannot detect end-of-turn.

Either is a smoking gun that training-time and inference-time template handling disagree.

### Signals

Generation:

- Sample 20 responses at temperature 0 to varied prompts. Any occurrence of raw template tokens in the response text is positive leak. Any response that keeps generating past a natural stopping point (until `max_new_tokens`) is negative leak.
- Serving pipeline reports `finish_reason=length` (rather than `stop`) far more often than usual.

Loss / eval:

- Not visible in the loss curve. Chapter 4's "case #7 — loss fine, output garbage" is exactly this.

### Root causes

1. **Terminator not in the loss.** The turn-ender (`<|eot_id|>` etc.) is masked to `-100` in your training labels, so the model never learns to emit it. Chapter 2's decode-and-print catches this; run it.
2. **Training template ≠ inference template.** The two byte strings differ (a `\n`, a BOS, a whitespace) so the model has learned to emit its assistant response after one specific opener and the inference stack is asking it to complete after a slightly different opener. Model does the best it can, which is confused.
3. **Padding token confusion.** `pad_token = eos_token` (mod-101 chapter 2 footgun) plus a mask bug means the model has been rewarded for emitting EOS in every position, including mid-response.
4. **Un-normalized special tokens.** Tokenizer treats template tokens as normal subwords in one place and as special tokens in another. Serialization boundaries (save + reload the tokenizer) are a specific source of this.

### Mitigation ladder

1. **Decode a training batch and confirm the turn-terminator is in the loss.** Chapter 2 assertion. If it is not, fix your mask; do not proceed with any run.
2. **Diff the training-time and inference-time rendered templates on the same conversation.** Any byte difference is a bug (mod-102 chapter 4 covered this in detail).
3. **Confirm your inference-time template matches what the base model documentation prescribes.** Do not free-hand the inference template because you happen to have written the training template in the source.
4. **Add an end-to-end generation smoke test to CI.** Prompt the trained model with a canned prompt, generate to 200 tokens, assert the response ends in the expected terminator and contains no raw template tokens.
5. **If you must use `pad_token = eos_token`**, verify padding positions are masked in `labels` (they are, by every off-the-shelf collator, but a custom collator can silently break this).

### The specific case: multiple chat templates in one dataset

If your dataset mix includes rows from multiple sources with slightly different formatting (a common consequence of unifying OASST1 + ShareGPT + synthetic data without a canonical re-render), the model sees inconsistent scaffolding during training and its output at inference is a mixture of the templates it saw most often. This is a data problem, not a training problem — fix it in mod-102, in the dataset-build step: re-render every example through *one* canonical `apply_chat_template` call before writing the training file.

## The joint failure mode: quiet regression

The nastiest SFT failure is the run where all three of the above happen simultaneously at moderate severity: eval loss looks fine, MMLU dropped 3 points, a small percentage of responses have stray template tokens, and no single symptom rings the alarm loud enough to block release. This is the case for the discipline built into chapter 6:

- Every run reports a **before / after benchmark delta**, not just an SFT eval loss.
- Every run reports a **template smoke test** count of failing rows in an audit sample.
- Every run report is signed off by the person who ran it, so "we did not check" is not an acceptable answer.

The chapter-6 training report exists so that no run can quietly regress under discipline.

## A generic detection protocol

At the end of every SFT run, before you touch downstream evals, run this five-step protocol:

1. **The two-plot view** from chapter 4. If unhealthy, stop and diagnose.
2. **Decode-and-print** one batch of the trained model's inputs and confirm masking is still correct. (Templates can silently regress.)
3. **Generation smoke test:** 20 prompts, temperature 0. Read them. Count template-leak occurrences and format failures.
4. **Base-vs-SFT delta** on one small held-out benchmark (a MMLU 200-row slice is enough for a first read).
5. **Save the artifacts** (loss curves, batch print, generations, benchmark numbers). These are the raw material for chapter 6.

Five minutes of protocol has saved teams weeks of "why did we ship this?" post-mortems.

## Exercise pointer

`exercises/exercise-04-catastrophic-forgetting-investigation.md` walks you through measuring MMLU before and after an SFT run, quantifying the delta, and applying one mitigation from the ladder above to demonstrate whether it helped. The exercise is worth doing on the same checkpoint you produced in exercise 01 so you can see the specific behavior of your specific setup.

## Summary

- Three failure modes specifically hit SFT: overfitting (eval loss diverges from train), catastrophic forgetting (held-out benchmarks regress even though loss curves look fine), template leak (stray or missing template tokens in outputs).
- Overfitting mitigations, in order: roll back to best checkpoint, fewer epochs, lower LR, grow dataset, regularization.
- Catastrophic forgetting mitigations, in order: **measure first**, lower LR, switch to LoRA, replay general data, KL-regularized preference optimization, fewer epochs.
- Template leak mitigations, in order: decode-and-print to confirm terminator is in loss, byte-diff training vs. inference render, confirm inference template matches docs, add a CI generation smoke test.
- The joint failure mode is quiet regression — moderate hits from all three at once, invisible to the loss curve. Chapter 6's training report exists specifically to prevent it.
- Run the five-step end-of-run detection protocol on every SFT run before running expensive downstream evals.
