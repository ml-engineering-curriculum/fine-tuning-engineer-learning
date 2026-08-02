# QLoRA: 4-bit Base Weights on a Single GPU

**Estimated read:** 45 minutes.
**You should be able to, by the end:** load a 7B–13B base model in 4-bit NF4 with `bitsandbytes`, attach LoRA adapters on top, and reason about the resulting memory footprint on a 24 GB GPU. Explain what NF4, double quantization, and paged optimizers are contributing individually, and why QLoRA's quality is close to (but not identical to) full-precision LoRA.

## The problem QLoRA solves

Chapter 1 established that LoRA's memory win is on gradients and optimizer state — not on the base weights, which are still fully materialized on the GPU. For a 7B base at BF16 that is ~14 GB of frozen weight, plus adapter parameters, plus activations. A single 24 GB consumer GPU can handle it — barely — for a 7B. It cannot handle a 13B. It cannot handle a 7B with a long context or a generous micro-batch.

QLoRA — Quantized LoRA, from Dettmers et al., 2023 — cuts the base weight footprint by ~4× by storing the frozen weights in a 4-bit quantized format and dequantizing on the fly during each matmul. The trainable adapters remain in higher precision (BF16 or FP32). The net effect is that a 7B base fits in ~5 GB of weights instead of ~14, freeing budget for longer sequences, larger micro-batches, and 13B / 34B bases that would not otherwise fit.

The paper's headline claim was that a 65B model could be fine-tuned on a single 48 GB GPU with adapter quality "within a rounding error" of full-precision LoRA. That headline scales down: on a 24 GB consumer GPU, QLoRA is the difference between "cannot train a 13B" and "can train a 13B with room to spare."

## Three techniques stacked together

QLoRA is not a single trick. It is three techniques stacked, each named separately in the paper:

1. **NF4 (4-bit NormalFloat)** — the quantization datatype for the frozen weights.
2. **Double quantization** — quantize the quantization constants themselves for extra memory savings.
3. **Paged optimizers** — CPU-offloadable optimizer state pages to survive activation-memory spikes.

Learn them separately so you know which one is causing what when you tune a run.

### NF4: a 4-bit format matched to weight distributions

Pretrained transformer weights are approximately zero-mean Gaussian. NF4 is a 4-bit floating-point-like format whose 16 representable values are chosen to be information-theoretically optimal for a standard normal distribution — the levels are the quantiles of `N(0, 1)`, not uniformly spaced.

Concretely, each block of contiguous weight values (default block size 64) is:

1. Standardized: divide by an absolute-maximum scale to normalize into `[-1, 1]`.
2. Quantized: map each normalized weight to the nearest of the 16 NF4 codepoints.
3. Stored: 4 bits per weight + 1 scale (FP32 or BF16) per block of 64.

Dequantization at forward time reverses steps 2 and 1: look up the codepoint, multiply by the block scale, run the matmul in BF16.

The three quantization datatypes you will meet in `bitsandbytes` are:

- **FP4** — 4-bit floating point with a standard 1-2-1 sign-exponent-mantissa split. Adequate for weights but worse than NF4.
- **NF4** — the QLoRA-paper format above. Use this by default.
- **INT8** — legacy 8-bit path; different memory profile, closer to full precision, but eats 2× the weight memory of NF4. Useful when you specifically need INT8 (e.g., an inference pipeline your team already uses).

For weights, use NF4. FP4 exists mostly for comparison; INT8 is a fallback when NF4 breaks a specific architecture.

### Double quantization: quantizing the quantization constants

Each block of 64 weights has a scale factor (a full-precision float). At 32 bits per block, that scale contributes `32 / 64 = 0.5` bits per weight — non-trivial when the weights themselves are only 4 bits.

Double quantization takes those block scales and quantizes *them* (into 8-bit FP with a second-level scale of much lower frequency). The paper reports ~0.37 bits per weight saved by this — about 3–4% additional memory reduction on the base model. The name is more clever than the trick; conceptually it is "the scale table is also compressible."

You enable it with one flag (`bnb_4bit_use_double_quant=True`). It has essentially no runtime cost. Use it.

### Paged optimizers: surviving activation spikes

Training activations spike in size during backward pass (especially with long sequences). If the optimizer state + activations exceed VRAM, the run OOMs.

`bitsandbytes` provides *paged* variants of AdamW (`PagedAdamW32bit`, `PagedAdamW8bit`) that live in CPU pinned memory and use CUDA's unified memory paging to fault pages onto the GPU only when the optimizer step touches them. Activations get the GPU during forward/backward; optimizer state is on-CPU most of the time.

Costs:

- The optimizer step is slower because pages move over PCIe. Depending on your GPU / PCIe generation, expect 5–30% slower step time than a fully in-GPU AdamW.
- Requires a system that supports unified memory (CUDA 11+ on Linux; some Windows / driver combinations are flakier).

Benefit: you can keep training when a naive AdamW would OOM at the same batch size. On a 24 GB consumer card training a 13B QLoRA, `PagedAdamW32bit` is not optional.

## The QLoRA layer, mechanically

Combine the pieces and one adapted linear layer in QLoRA looks like this at forward time:

```
Frozen quantized weight  W_q ∈ NF4 (block-quantized)
                              │
                              ▼ dequantize per block, materialize BF16 slice
Input activation  x ─▶ (BF16 W) · x  ─▶ base output (BF16)
                                             +
                              (α/r) · B · A · x   ← LoRA path, all BF16, all trainable
                                             ↓
                                         layer output
```

Two subtleties:

- The dequantized BF16 view of `W` is a *transient* — it lives long enough for the matmul, then goes away. Peak dequantized memory equals the largest single matmul's weight tile, not the whole model.
- The LoRA parameters `A` and `B` are stored in BF16 (or FP32; check the trainer's `bf16` / `fp16` flag). Gradients flow through the dequantization *of `W`* only far enough to reach the LoRA path — because `W.requires_grad = False`, backprop does not need to differentiate the quantization step itself.

The paper is careful about a detail worth internalizing: **quantization introduces a small forward-pass error**, and the LoRA update learns partly to *compensate* for that error. A QLoRA run's `B, A` values are subtly different from a full-precision LoRA run's `B, A` values, even on identical data and hyperparameters. That is why you cannot cleanly de-quantize the base and drop in a QLoRA-trained adapter and expect bit-identical behavior to a full-precision LoRA run — the adapter learned in the presence of quantization noise.

## The loader recipe: bitsandbytes + transformers + peft

The load pattern for a QLoRA training script, on any recent `transformers`/`peft`/`bitsandbytes`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "meta-llama/Llama-3.1-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # NF4, not FP4
    bnb_4bit_use_double_quant=True,          # +0.37 bits/param savings
    bnb_4bit_compute_dtype=torch.bfloat16,   # matmul dtype after dequant
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",  # or "sdpa"
)

# Enables gradient checkpointing on a k-bit-loaded model and sets input_grad
# flags so gradients flow through the quantized weights correctly.
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, lora_config)
model.print_trainable_parameters()
```

Line-by-line, what each argument buys you:

- `load_in_4bit=True` triggers the `bitsandbytes` `Linear4bit` module swap in the `transformers` loader.
- `bnb_4bit_quant_type="nf4"` is the NF4-vs-FP4 switch. Default here.
- `bnb_4bit_use_double_quant=True` is the double-quant switch. Cheap. Turn it on.
- `bnb_4bit_compute_dtype=torch.bfloat16` is the dequantized compute dtype. BF16 on Ampere+ (RTX 3090, A10, A100, H100, RTX 40-series). FP16 on Turing (T4, RTX 20-series) — check your GPU.
- `prepare_model_for_kbit_training` does three things: (a) casts the LayerNorm parameters to FP32 for numerical stability, (b) casts the LM head output to FP32 for the same reason, (c) turns on `enable_input_require_grads()` so gradient checkpointing plays nicely with a frozen base. Do not skip it.
- `use_gradient_checkpointing=True` trades ~30% throughput for ~50% activation memory. On a 24 GB card training 7B QLoRA, you want it on.

## The memory budget, worked out for a 7B on 24 GB

Rough accounting (numbers are approximate; use them for order-of-magnitude reasoning, not for provisioning decisions):

| Component | 7B FP16/BF16 full FT | 7B BF16 LoRA | 7B QLoRA NF4 |
|---|---|---|---|
| Base weights | 14 GB | 14 GB | ~3.5 GB |
| Optimizer moments | 56 GB (AdamW 32-bit) | ~1 GB (adapters only) | ~0.7 GB (paged AdamW 8-bit) |
| Gradients | 14 GB | ~0.5 GB (adapters only) | ~0.5 GB |
| Activations (batch × seq × hidden × layers) | 3–10 GB (depends on gc) | 3–10 GB | 3–10 GB |
| **Estimated total** | ~87–94 GB | ~19–25 GB | **~8–14 GB** |

On a 24 GB card, QLoRA leaves you 10+ GB of headroom to use for longer contexts, larger micro-batches, or a 13B base instead of a 7B. That is the practical point of the technique.

Verify with `torch.cuda.max_memory_allocated()` at the top of your training loop; the numbers above should be within a couple of GB of what you observe. If they are not, either your architecture is atypical or you missed a flag (usually gradient checkpointing).

## Training dynamics: what changes vs. plain LoRA

QLoRA is not just LoRA in a smaller memory box. A few dynamics differ:

- **Slightly higher LR is often reasonable.** The paper's Guanaco recipe uses learning rates in the 2e-4 to 5e-4 range for QLoRA on 7B–65B Llama-family bases, versus 1e-4 to 3e-4 for full-precision LoRA. The LoRA path is "louder" relative to the (noisier) quantized base, and the network tolerates a somewhat more aggressive step.
- **Convergence is a touch noisier.** Loss curves have slightly more variance step-to-step because the dequantization introduces a fixed forward-pass error. Do not over-interpret single-step spikes.
- **Long runs may plateau slightly higher.** On some tasks, QLoRA final loss lands 0.01–0.05 above full-precision LoRA. This is the "quality tax" of quantization. On most SFT-style tasks it is negligible; on tight instruction-following evals it may show up.

Copy known-good QLoRA hyperparameters (Guanaco, Zephyr-QLoRA, the OpenAccess AI Collective recipes) when you are starting. Then vary from those, one knob at a time.

## What breaks in QLoRA that does not break in LoRA

- **Distributed training is harder.** `bitsandbytes` `Linear4bit` does not play cleanly with FSDP in older `transformers` versions. `bitsandbytes` + DDP works; `bitsandbytes` + FSDP works with a specific `transformers` / `accelerate` / `peft` version matrix that changes fast. If you need multi-GPU QLoRA, check the current compatibility matrix before you commit. Mod-107 covers distributed strategies in depth.
- **Some architectures are not supported.** The 4-bit path assumes the model is composed of standard `nn.Linear` layers. Architectures with fused / bespoke linears may not be swappable. Check `bitsandbytes` model-support docs before you pick a base.
- **Serving requires more thought.** Merging a QLoRA adapter into a 4-bit base is not straightforward (there is no BF16 base to merge into). Serving strategies are covered in chapter 3; the short version is that you either dequantize + merge into a BF16 base, or serve the base 4-bit with the adapter kept separate.
- **Numerical corner cases.** Very small models (< 1B) or very short sequences can show quantization artefacts at higher rates. QLoRA is designed for the 7B+ regime.

## When to reach for QLoRA vs. LoRA vs. full FT

The simplest decision rule:

- **Single 24 GB GPU, 7B–13B base, SFT-style task:** QLoRA. This is the QLoRA-shaped hole in the world.
- **A100 40/80 GB or multi-GPU, 7B–13B base:** LoRA. The extra memory buys you throughput and cleaner distributed behavior.
- **Base < 3B, or heavy domain shift, or long continued-pretraining:** full fine-tuning (mod-103). PEFT is not the right tool.

The rule of thumb across the industry: start with LoRA if you have the memory; drop to QLoRA when you do not; step up to full FT only when a rank-64 or rank-128 LoRA on all-linear targets is still plateauing above where you need to be.

## Exercise pointer

`exercises/exercise-02-qlora-on-a-7b-base-single-gpu.md` walks you through fitting a 7B-class base on a 24 GB GPU with NF4 + double quantization + paged AdamW + gradient checkpointing, and asks you to defend every flag against the memory table above.

## Summary

- QLoRA stacks three techniques on top of LoRA: NF4 4-bit weight quantization, double quantization, and paged optimizers. Together they cut the base-weight footprint ~4× at essentially no quality cost.
- NF4 is a 4-bit format whose 16 codepoints are the quantiles of `N(0, 1)`, matched to the empirical distribution of pretrained weights. Double quantization compresses the block scales further; paged optimizers offload optimizer state to CPU pinned memory to survive activation spikes.
- The `bitsandbytes` + `transformers` + `peft` loader recipe is ~15 lines and always the same shape. Know what `prepare_model_for_kbit_training` does before you skip it.
- A 7B QLoRA run fits comfortably in 8–14 GB of VRAM, freeing budget for longer contexts, larger micro-batches, or a 13B base on the same card.
- Quality is close to but not identical to full-precision LoRA. LR is often slightly higher; loss is a touch noisier; final loss may plateau 0.01–0.05 above full-precision LoRA on some tasks.
- Distributed training and adapter serving are the two places QLoRA imposes extra friction relative to LoRA — chapters 3 and mod-107 explain what to do about it.
