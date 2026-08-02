# exercise-01: LoRA from Scratch and Against `peft`

**Estimated effort:** 4 hours
**Companion chapters:** [`01-lora-math-and-implementation.md`](../01-lora-math-and-implementation.md), [`05-adapter-failure-modes.md`](../05-adapter-failure-modes.md)

## Objective

Implement `LoRALinear` from scratch in PyTorch, wire it into a small transformer for a real SFT run, then run the *same* fine-tune again using `peft`'s `LoraConfig` — and either demonstrate numeric equivalence or explain the differences you observe. The point is not to build a competitor to `peft`; the point is that when a `peft`-driven run misbehaves, you can walk into its internals without a map.

## Prerequisites

- The mod-103 exercise-01 environment (`torch`, `transformers`, `trl`, `datasets`, `accelerate`).
- `peft` installed at a recent version.
- A single GPU with ≥ 12 GB VRAM. CPU-only fallback works with a smaller base (see *Starter guidance*).
- Familiarity with the mod-103 SFT loop — this exercise assumes you can run a full-parameter fine-tune already.

## Requirements

### Part A — Implement `LoRALinear`

Author `lora.py` containing a `LoRALinear(nn.Module)` that matches the reference in chapter 1 in every load-bearing detail:

1. Wraps an existing `nn.Linear` and freezes its `weight` and `bias`.
2. Trainable parameters: `lora_A` of shape `(r, d_in)` initialized with Kaiming-uniform (`a=sqrt(5)`), and `lora_B` of shape `(d_out, r)` initialized to zeros.
3. Constructor accepts `r`, `alpha`, and `dropout`; stores `scaling = alpha / r`.
4. `forward(x)` returns `base(x) + scaling * dropout(x) @ A.T @ B.T`. Do not fuse into one matmul.

Also implement `add_lora(model, target_names, r, alpha, dropout)` that walks `model.named_modules()`, swaps matching `nn.Linear` layers for `LoRALinear`, and re-freezes / re-enables gradients as chapter 1 shows.

### Part B — Numeric sanity checks

Before you train anything, prove your implementation is correct with three assertions in a `test_lora.py`:

1. **Zero at init:** load a fresh base, wrap it, and confirm that `add_lora`-wrapped model produces bit-identical logits to the un-wrapped base for a canned input (tolerance 0 in FP32; ~1e-6 in BF16 due to matmul associativity).
2. **Non-zero after a step:** run one optimizer step on a canned loss and confirm the wrapped model's logits now differ from the base's.
3. **Trainable-params match:** confirm `sum(p.numel() for p in model.parameters() if p.requires_grad)` matches the parameter-count math from chapter 1's "Parameter-count math" section for your chosen `r` and target list.

### Part C — Train with your implementation

Wire your `LoRALinear` into a real SFT run. Suggested config:

- Base: `Qwen/Qwen2.5-1.5B` or `meta-llama/Llama-3.2-1B` (whichever you used in mod-103 exercise 01).
- Dataset: a small slice (500–2000 rows) of `HuggingFaceH4/no_robots` or `GAIR/lima`.
- Target modules: `["q_proj", "v_proj"]`.
- `r = 16`, `alpha = 32`.
- LR: `1e-4` (higher than full-FT because only the LoRA path trains).
- Schedule / epochs / batch size: same as your mod-103 exercise-01 run so curves are comparable.

Run it. Save `outputs/from-scratch/adapter/` (the `A` and `B` tensors as a `safetensors` file plus a JSON config), the loss curve, and the final eval loss.

### Part D — Train with `peft.LoraConfig` at the same settings

Recreate the same run using `peft`. Use identical `r`, `alpha`, `target_modules`, `lora_dropout`, learning rate, schedule, batch size, and seed. Save `outputs/peft/adapter/`.

### Part E — Compare

Produce `compare.md` with:

1. **Loss-curve overlay.** One PNG with both curves. Comment on whether they track each other.
2. **Final eval numbers.** From a held-out eval set, using the same generation seed. Report absolute numbers and the delta.
3. **Parameter-tensor comparison.** Load both adapters' `A` and `B` tensors and report, per adapted layer, the cosine similarity between the two implementations' `A` tensors and between their `B` tensors. Cosine ≈ 1 across most layers is a pass; systematic disagreement is a diagnostic — write down which layers disagreed and speculate on the cause.
4. **Wall-clock and peak VRAM.** From `torch.cuda.max_memory_allocated()` and wall-clock timings around `trainer.train()`.
5. **Health check.** Run the singular-value analysis from chapter 5's `lora_health` function against both adapters. Report effective rank per layer, side by side.

Where the two implementations diverge, name the cause. Common candidates: (i) different init RNG streams, (ii) `peft` fusing dropout / scaling differently, (iii) different treatment of gradient checkpointing, (iv) an actual bug in your from-scratch code that the comparison surfaced.

### Part F — Write the one-page report

`report.md` (one page, ~500 words):

1. Which implementation matched expectations, which surprised you, and why.
2. One thing about the `peft` internals you can now name that you could not before this exercise.
3. One config you would change on either run if you did it again, with a one-sentence justification.

## Starter guidance

- **CPU / small-GPU fallback.** If you cannot fit a 1B base with a full SFT, drop to `HuggingFaceTB/SmolLM2-360M` (or an even smaller base) and cut `max_length` to 512 and `per_device_train_batch_size` to 1. The comparison logic still holds; the numbers get noisier.
- **The zero-at-init check catches most from-scratch bugs.** If it fails, either your `B` init is not zero (typo), your dropout is not `Identity()` at `dropout=0`, or your `requires_grad_(False)` on the base did not stick.
- **Cosine similarity on `A` alone can be misleading** — `A` and `B` have a joint gauge freedom (multiply `A` by any invertible `R` and `B` by `R^{-1}` and the product `BA` is unchanged). Cosine on `BA` per layer is a stricter test; do that if `A`-only comparison surprises you.
- **Set `seed`, `data_seed`, and `torch.manual_seed` identically** in both runs. If you skip this, the comparison in Part E is almost meaningless.
- **Do not use `torch.compile` on the from-scratch model** for this exercise. Debugging is easier without it, and the compile pass can silently alter the operator order in ways that make the numeric comparison noisier than the effect you are trying to observe.

## Acceptance criteria

You have completed the exercise when:

- `lora.py`, `test_lora.py`, both training scripts, both adapter directories, `compare.md`, and `report.md` all exist and are self-consistent.
- The three assertions in `test_lora.py` pass on your code.
- `compare.md` includes the loss overlay, the eval delta, the per-layer cosine table (or `BA` comparison), the wall-clock / VRAM numbers, and the health-check output.
- Where the two runs disagree, `compare.md` names the specific cause — not "runs disagree because of randomness."
- `report.md` answers all three questions from Part F.

## Stretch goals

- **Add `rslora` scaling.** Modify your `LoRALinear` to accept `use_rslora=True` and replace `scaling = alpha / r` with `alpha / sqrt(r)`. Re-run the from-scratch training with `use_rslora=True` on both, and compare against `peft`'s `LoraConfig(use_rslora=True)`.
- **Compare an all-linear target list vs. `q, v` only.** Same base and hyperparameters; report the delta in trainable params, final eval, and per-layer effective rank.
- **Add a Kaiming-normal init variant.** Swap `kaiming_uniform_` for `kaiming_normal_` on `A`; retrain; compare. This is the kind of ablation `peft` PRs argue about; you now have the tools to do it yourself.
- **Ship a merged model.** Take one of your trained adapters, implement `merge_and_unload` yourself (`W += scaling * B @ A` for each wrapped `LoRALinear`, then unwrap), save the merged model, and verify inference matches the unmerged adapter's outputs at low temperature. This is the on-ramp for exercise 03.
