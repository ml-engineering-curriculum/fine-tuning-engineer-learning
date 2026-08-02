# exercise-04: PEFT Methods Comparison Matrix

**Estimated effort:** 4 hours (mostly wall-clock training)
**Companion chapter:** [`04-peft-methods-in-context.md`](../04-peft-methods-in-context.md)

## Objective

Run **three** PEFT methods on the *same base and dataset*, produce a comparison matrix (trainable params, VRAM, wall-clock, final eval, adapter file size), and use it to defend a recommendation for a specific serving scenario. The chapter's decision matrix is a starting bet; this exercise makes you produce your own with numbers you measured yourself.

## Prerequisites

- Same environment as exercises 01–03: `torch`, `transformers`, `trl`, `datasets`, `accelerate`, `peft`.
- A single GPU with ≥ 16 GB VRAM (24 GB comfortable). Smaller cards work if you drop the base to 1B–3B.
- Familiarity with LoRA (exercise 01) and QLoRA (exercise 02). This exercise reuses their patterns.

## Requirements

### Part A — Fix the base and the dataset

Do not move these across runs. Suggested:

- **Base:** `Qwen/Qwen2.5-1.5B` (fast to train, easy to compare) or `meta-llama/Llama-3.2-3B` (larger; sharper differences).
- **Dataset:** `HuggingFaceH4/no_robots` or a 5k-row slice of `HuggingFaceH4/ultrachat_200k`.
- **Eval:** the corresponding held-out split, or a canned 100-prompt set you author.

Pin base revision, dataset split, and seeds. Same LR schedule, warmup, weight decay, batch size, sequence length, and epoch count in every run. The *only* thing that varies across the three runs is the PEFT method.

### Part B — Run three PEFT methods

Pick three. **LoRA is required.** Pick two others from the list below. Reasonable trios:

- **LoRA + QLoRA + IA³** — the "modern efficient trio," most useful for a real production comparison.
- **LoRA + IA³ + Prompt Tuning** — the "spectrum of parameter budgets" trio, most illustrative for chapter 4's decision matrix.
- **LoRA + Prefix Tuning + Prompt Tuning** — the "additive vs. non-additive" trio.

Suggested `peft` configs (start here, then tune if you feel like it):

```python
# LoRA
LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj",
    "gate_proj","up_proj","down_proj"], lora_dropout=0.05, task_type="CAUSAL_LM")

# IA³
IA3Config(target_modules=["k_proj","v_proj","down_proj"],
          feedforward_modules=["down_proj"], task_type="CAUSAL_LM")

# Prefix tuning
PrefixTuningConfig(num_virtual_tokens=30, task_type="CAUSAL_LM")

# Prompt tuning
PromptTuningConfig(num_virtual_tokens=100,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Follow the user's instructions carefully.",
    tokenizer_name_or_path=BASE_ID,
    task_type="CAUSAL_LM")
```

For QLoRA if included: same `LoraConfig` as LoRA, plus the `BitsAndBytesConfig` load path from exercise 02.

For each method, capture during training:

- `print_trainable_parameters()` at model construction — trainable + total + %.
- `torch.cuda.max_memory_allocated()` at first optimizer step.
- Wall-clock training time (`time.perf_counter()` around `trainer.train()`).
- Final train loss and final eval loss.
- Adapter directory size on disk (`du -sh` or equivalent).

Save each run's outputs under `outputs/<method-name>/` with the adapter, logs, and a `metrics.json` capturing the numbers above.

### Part C — Build the comparison matrix

`matrix.md` (or `matrix.csv` if you prefer) with, at minimum, this shape:

| Method | Trainable params | % of base | Peak VRAM (train) | Wall-clock | Final train loss | Final eval loss | Adapter size on disk |
|---|---|---|---|---|---|---|---|
| LoRA | ... | ... | ... | ... | ... | ... | ... |
| Method 2 | ... | ... | ... | ... | ... | ... | ... |
| Method 3 | ... | ... | ... | ... | ... | ... | ... |

Add rows for anything else you measured (throughput tokens/s, per-request inference latency for a small canned batch, etc.).

### Part D — Qualitative comparison

Generate on 10–20 canned prompts with each trained model at greedy decoding. Include (a) a few clearly-in-distribution prompts, (b) a few base-capability prompts (things the base was already good at), and (c) 2–3 adversarial prompts (long context, unusual format, out-of-domain).

Produce `qualitative.md`:

- For 5 prompts, paste all three responses (and the base's un-adapted response for reference).
- One sentence per prompt: which method's output do you prefer, and why.
- Note any prompt where a method visibly regressed compared to the base — this is the catastrophic-forgetting or "adapter is doing nothing useful" signal.

### Part E — Recommendation for a specified scenario

Author `recommendation.md`. Pick ONE of these scenarios (or invent one and state it clearly):

- **Scenario 1: 24-GB consumer GPU, 7B production SFT for a single customer, weekly retrains, single deployed model.**
- **Scenario 2: Multi-tenant SaaS, 200+ tenants each with their own light style-tuning, 40 GB serving GPUs, sub-100ms latency.**
- **Scenario 3: One-shot research prototype on a large base (70B), single 80 GB GPU available, exploratory data of ~500 examples.**
- **Scenario 4: Air-gapped enterprise deploy, small base (< 3B), heavy domain adaptation, plenty of GPU budget.**

Write ~500 words:

1. State the scenario constraints in your own words.
2. Cite your matrix numbers to argue for one method.
3. Address the runner-up: what would change your mind, and what evaluation would you run to decide?
4. Name one operational risk that the matrix does not capture (deployment tooling, team familiarity, framework maturity, etc.).

## Starter guidance

- **Keep the schedule identical across runs.** LR that is optimal for LoRA is *not* optimal for prompt tuning; the chapter warns about this. Use the same LR / schedule so the comparison is on "at fixed hyperparameters, which method wins" — and then, in the write-up, honestly acknowledge that a tuned prompt-tuning run might close some of the gap.
- **VRAM measurements are noisier than you expect.** Run each measurement twice and report the higher; that is closer to what you would provision for.
- **If your prompt-tuning or prefix-tuning run diverges or fails to learn**, verify your initialization (`prompt_tuning_init_text` for prompt tuning) and check `num_virtual_tokens` is not so small it under-parameterizes. Chapter 4 gives typical ranges.
- **QLoRA's VRAM number is misleading if you also enabled gradient checkpointing.** Report both with-and-without gradient checkpointing if you turned it on for QLoRA only.
- **Adapter size on disk is a real, load-bearing number for multi-tenant scenarios** — a 100 MB LoRA vs. a 400 KB prompt-tuning delta is 250× on the wire. Do not skip this column.
- **You are allowed to be inconclusive.** If your matrix does not decisively pick one method, say so and explain what evaluation would decide it. That is a stronger recommendation than a false certainty.

## Acceptance criteria

You have completed the exercise when:

- Three PEFT methods (LoRA + two others) have been trained on the same base and dataset with the same seeds and schedule.
- `outputs/<method>/` for each method contains adapter files, logs, and `metrics.json`.
- `matrix.md` (or `.csv`) exists with all columns filled and every number pointing to the run that produced it.
- `qualitative.md` includes at least 5 prompts × 4 responses (three methods + base) with per-prompt commentary.
- `recommendation.md` picks a scenario, cites specific numbers from your matrix, and addresses the runner-up honestly.

## Stretch goals

- **Run a fourth method** (e.g., Houlsby-style adapter via `peft`'s `AdaptionPromptConfig` on a Llama base). Add it to the matrix. Where does it land vs. the modern trio?
- **Redo the matrix with a bigger base.** If you started at 1.5B, rerun on 7B (QLoRA if you must). Which rankings change? Chapter 4's "prompt tuning scales with base size" claim is directly testable here.
- **Add inference-latency measurements per method.** For a fixed sequence length, measure p50 / p95 latency of a single-prompt generation on each. Prefix tuning's per-request KV overhead is directly measurable this way.
- **Ties-merge two LoRA adapters vs. weighted-sum two LoRA adapters.** Not part of the required comparison, but a useful adjacency; document what you find.
- **Sweep `r` inside the LoRA row and `num_virtual_tokens` inside the prompt-tuning row.** Report the matrix as a function of the capacity knob for each method. This is the fair-comparison people usually skip.
