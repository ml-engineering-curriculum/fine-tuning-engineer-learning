# Gradient Checkpointing, BF16 / FP8, and the Memory Budget

**Estimated read:** 45 minutes.
**You should be able to, by the end:** compute an approximate per-rank memory budget for a fine-tuning run from first principles, choose the right mixed-precision policy for your hardware (BF16 default, FP8 where the hardware supports it), enable and tune activation checkpointing under FSDP / DeepSpeed, decide when FlashAttention-2/3 is worth the integration cost, and know which of these levers to pull first when a run OOMs.

## The six things that consume GPU memory

Chapter 1 listed them; this chapter goes into each. For a model with `N` parameters trained with AdamW in mixed precision, on `P` sharded-DP ranks:

1. **Base weights** — `2N` bytes in BF16, `N/2` in NF4 (QLoRA). Under ZeRO-3 / FSDP FULL_SHARD, `2N/P` per rank.
2. **Optimizer state** — AdamW keeps master weights + first moment + second moment in FP32: `12N` bytes total. Under ZeRO-1/2/3, `12N/P` per rank. `AdamW8bit` (bitsandbytes) drops this to ~4N.
3. **Gradients** — `2N` in BF16, or `2N/P` under ZeRO-2/3.
4. **Activations** — the tensor outputs of every forward-pass op that need to be retained for backward. Scales with batch size, sequence length, hidden size, number of layers, and attention implementation. Formula and full breakdown below.
5. **Communication and framework buffers** — NCCL communication buffers, FSDP all-gather scratch space, cuDNN workspace, PyTorch inductor compile caches, misc. Modest (~1–4 GB per rank) but non-zero.
6. **Fragmentation** — PyTorch's caching allocator leaves gaps between blocks. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (PyTorch 2.1+) helps materially; even with it, budget 5–15% headroom.

Every technique in this chapter moves one or more of these.

## Activation memory, the one term that surprises people

Weights, gradients, and optimizer state are easy: they scale with `N`. Activation memory is harder because it depends on batch shape.

The dominant terms per transformer layer, at micro-batch size `b`, sequence length `s`, hidden size `h`, and (for standard attention) `a` attention heads:

- **MLP intermediate**: ~`4 · b · s · h` bytes (BF16). If the MLP intermediate is `4h`, this is the main term.
- **Attention QKV output**: ~`6 · b · s · h` bytes across Q, K, V, output.
- **Attention scores** (with vanilla `O(seq²)` attention): `2 · b · a · s²` bytes. This is the term that dominates at long sequence lengths.
- **LayerNorm residuals**, small biases: `~2 · b · s · h` bytes.

Sum across `L` layers and add the embedding + output head activation. As a rough approximation for a Llama-family model at BF16 without activation checkpointing:

    activation_memory ≈ L · (34 · b · s · h + 2 · b · a · s²) bytes

For Llama-3.1-8B (`L=32`, `h=4096`, `a=32`, `N=8B`) at `b=4`, `s=2048`:

    ≈ 32 · (34 · 4 · 2048 · 4096 + 2 · 4 · 32 · 2048²) bytes
    ≈ 32 · (1.14 GB + 1.07 GB) ≈ 71 GB

That is *per rank*, unsharded across the DP dimension. The 71 GB is exactly why activation checkpointing is not optional on longer sequences without FlashAttention.

Two consequences to internalize:

- **Activation memory grows linearly in batch size and sequence length**, and — with vanilla attention — *quadratically* in sequence length via the attention score term. FlashAttention (below) removes the `s²` term.
- **Activation memory does not shrink with more DP ranks.** Sharded DP shards weights, gradients, and optimizer state; each rank still holds the activations for its own micro-batch. To reduce activations, reduce micro-batch or sequence length, add sequence parallelism, or use activation checkpointing.

## Gradient (activation) checkpointing

Chen et al. (2016, "Training Deep Nets with Sublinear Memory Cost") introduced the idea: instead of retaining every activation for the backward pass, discard some and recompute them on demand. In transformer training this typically means retaining only the input to each transformer block; when backward reaches that block, the forward is re-run to recompute intermediate activations.

- **Memory savings**: from `O(L · b · s · h)` down to `O(√L · b · s · h)` in the theoretical version; in practice, one checkpoint per transformer block cuts activation memory by ~4–8× depending on block internals.
- **Compute cost**: ~30% more FLOPs per step (one extra forward pass per block).
- **Throughput cost**: 20–35% wall-clock, roughly. Sometimes less if the run was memory-bound to begin with (larger batches now fit, amortizing the recompute).

Three ways to turn it on:

- **`model.gradient_checkpointing_enable()`** — the HF transformers path. Works with DDP, FSDP, DeepSpeed. Uses PyTorch's `torch.utils.checkpoint` under the hood.
- **`fsdp_activation_checkpointing: true`** in Accelerate FSDP config — applies `apply_activation_checkpointing` from `torch.distributed.algorithms._checkpoint.checkpoint_wrapper`. Composes correctly with FSDP's unshard / reshard flow. Preferred for FSDP.
- **`activation_checkpointing: true`** in DeepSpeed config (or `deepspeed.checkpointing.checkpoint(...)` in code). DeepSpeed's version supports partition-across-ranks checkpointing that's more memory-efficient than plain per-rank checkpointing.

For fine-tuning of 7B+ models, activation checkpointing is on by default. The only time to turn it off is when you have plentiful memory (e.g., a small model on H100 80 GB) and want peak throughput.

One recompute quirk: `use_reentrant=False` is the modern default (`use_reentrant=True` is deprecated). With `use_reentrant=True`, checkpointing can fail with LoRA + FSDP + certain PEFT configurations because the reentrant autograd hook interacts badly with parameter groups. Modern versions of Accelerate / transformers set `False` automatically; if you see cryptic `RuntimeError: Trying to backward through the graph a second time` errors, check this flag.

## Mixed precision: BF16, FP16, FP32, FP8

The dtype policy is a lever on all of memory (halves the size of the affected tensors), throughput (tensor cores are dramatically faster in lower precision), and numerical stability (small magnitudes underflow, large ones overflow).

### BF16 (bfloat16)

The default for fine-tuning on Ampere and newer GPUs.

- **16 bits** total: 1 sign, 8 exponent, 7 mantissa.
- **Same exponent range as FP32** — no underflow / overflow issues that plague FP16.
- **Half the memory** of FP32 for the affected tensor.
- **No loss scaling needed** (unlike FP16).
- Supported on A100, RTX 3090+, H100/H200, and any Ampere-or-later card.

Set `bf16=True` in TrainingArguments or `mixed_precision: bf16` in Accelerate. Under FSDP, this configures `MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)` by default. Some setups keep `reduce_dtype=torch.float32` for extra safety on gradient reduction; small memory cost, tiny throughput cost, more robust on the tail of runs where gradient magnitudes get small.

### FP16

Legacy for LLM fine-tuning on modern hardware. Same 16 bits total: 1 sign, 5 exponent, 10 mantissa. More mantissa precision than BF16 but a narrower exponent range that overflows and underflows in exactly the places training gradients live.

- Requires **loss scaling** — multiply the loss by a large constant before backward so gradients don't underflow. Dynamic loss scaling (grad scaler) auto-adjusts, but every once in a while a step's loss scale is too large and NaNs cascade.
- Historical need on pre-Ampere hardware (T4, V100) that lacks BF16.
- **Use FP16 only if you're on Turing/Volta.** On A100/H100/RTX-40, BF16 is universally the right choice.

### FP32

The reference precision. Used for master weights and optimizer state (momentum, variance) even in mixed-precision runs — accumulating small gradient updates into FP16 or BF16 loses precision to rounding.

- **AdamW's `12N` bytes** breakdown: `4N` for master weights + `4N` for momentum + `4N` for variance, all in FP32.
- **`fp32_reduce`** is the FSDP option to reduce gradients in FP32 while keeping other tensors in BF16 / FP16. Small extra memory; buys numerical robustness on the tail of long runs.
- Rare to want full FP32 forward/backward for fine-tuning; typical only for numerical debugging.

### FP8 (Hopper, via Transformer Engine)

The frontier of mixed-precision training. On H100 / H200 (Hopper) and Blackwell (B100/B200), tensor cores support FP8 in two subformats:

- **E4M3** (4 exponent, 3 mantissa) — higher precision, narrower range. Used for activations and weights in forward.
- **E5M2** (5 exponent, 2 mantissa) — wider range, less precision. Used for gradients in backward.

FP8 is **not** a drop-in config flag. You enable it by using NVIDIA's Transformer Engine library (`transformer_engine.pytorch`), which provides replacement modules (`te.Linear`, `te.LayerNormLinear`, `te.MultiheadAttention`) that internally quantize inputs to FP8, do the matmul in FP8, and dequantize the output. Some fine-tuning stacks integrate this transparently (NVIDIA's NeMo, Megatron-LM); the Hugging Face path is emerging but not fully turnkey as of this writing — check the current status of `transformers`' FP8 support and Accelerate's FP8 integration before committing.

Trade-offs:

- **~2× throughput** vs BF16 on H100 for compute-bound layers.
- **~2× memory savings** on activations and weights.
- **Numerical care required** — Micikevicius et al. (2022, "FP8 Formats for Deep Learning") describe scaling factors and calibration. The Transformer Engine handles most of this automatically; you still may need per-model tuning.
- **Convergence risk** for very small models or non-standard architectures. Well-validated on Llama / GPT-family transformers at 7B+.

For a fine-tuning engineer: **enable FP8 if your hardware is Hopper+ and your framework supports it out of the box**; otherwise stay on BF16. The complexity vs benefit trade-off shifts with each new release cycle; revisit when your cluster upgrades.

## FlashAttention-2 and FlashAttention-3

Dao et al. (2022, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"), Dao (2023, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"), Shah et al. (2024, "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"). Three generations of the same idea: compute attention block-wise in on-chip SRAM instead of materializing the full `[b, a, s, s]` score matrix in HBM.

- **Memory**: the `2 · b · a · s²` activation term disappears. Attention becomes `O(b · s · h)`.
- **Throughput**: 2–4× faster than vanilla attention on modern GPUs. FA3 adds further speedups on Hopper via async warps.
- **Correctness**: exact attention, not an approximation. Loss curves are identical (up to numeric precision).

Enable via `attn_implementation="flash_attention_2"` in `from_pretrained` (transformers 4.36+). For H100, prefer FA3 where supported (`attn_implementation="flash_attention_3"` in recent versions, or via the `flash-attn` package directly). Hardware compatibility: FA2 works on Ampere+ (A100, RTX 3090+); FA3 requires Hopper.

Alternatives when FA2/3 is not available: PyTorch's built-in `sdpa` (`attn_implementation="sdpa"`) uses efficient kernels where available; it is a good default fallback.

## The memory budget, worked out end-to-end

The prior chapters gave partial numbers; here is a defensible full budget for a Llama-3.1-8B full-parameter SFT run on 8× A100 80 GB with FSDP FULL_SHARD, BF16, gradient checkpointing on, FA2, and no offload:

| Component | Formula | Value per rank |
|---|---|---|
| Sharded weights (BF16) | `2N / P` | ~2 GB |
| Sharded optimizer state (FP32 AdamW) | `12N / P` | ~12 GB |
| Sharded gradients (BF16) | `2N / P` | ~2 GB |
| Activations (with gc + FA2, `b=4`, `s=2048`) | ~10–15 GB | ~12 GB |
| FSDP all-gather buffer (one layer's worth) | `2N/L` peak | ~0.5 GB |
| NCCL / cuDNN / framework buffers | | ~2 GB |
| Fragmentation headroom (~10%) | | ~3 GB |
| **Total per rank** | | **~33 GB** |

That leaves ~47 GB of headroom on the 80 GB card, which you spend on larger micro-batch, longer sequence, or the reference model in a DPO run. If you drop to 40 GB A100s, you're at 80% utilization — workable but no headroom for the reference model. If you drop to 24 GB consumer cards, it doesn't fit; QLoRA is the answer (mod-104 chapter 2).

The same model on 4× A100 80 GB would double the per-rank shard sizes to ~4 GB weights + ~24 GB optimizer + ~4 GB grads = ~44 GB fixed, leaving less activation headroom. Reducing micro-batch from 4 to 2 or shortening sequence from 2048 to 1024 halves the activation term. That is the shape of the memory-budget conversation.

## The recommended "start conservative, tune outward" recipe

For a new run on new hardware, the first-run config that almost always works:

1. **BF16** mixed precision. FP16 only on legacy hardware; FP8 only if the stack is proven.
2. **FSDP FULL_SHARD** (or DeepSpeed ZeRO-3 with `stage3_gather_16bit_weights_on_model_save: true`).
3. **Activation checkpointing on** (via the framework-specific flag).
4. **FlashAttention-2** (`attn_implementation="flash_attention_2"`) for any Ampere-or-later hardware.
5. **Micro-batch `b=1`, gradient accumulation to hit target effective batch.** Grow `b` after the run stabilizes; smaller micro-batch is more forgiving of activation memory spikes.
6. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** in the environment; cuts fragmentation losses.
7. **No offload initially.** Add optimizer offload only if OOM at `b=1`; add param offload only if optimizer offload isn't enough.
8. **Log `torch.cuda.max_memory_allocated()`** at the top of each epoch and after each config change.

If the first run OOMs, tune in this order:

1. Reduce micro-batch to 1 (if not already).
2. Reduce sequence length (if application allows).
3. Turn on `fsdp_forward_prefetch: false` and `fsdp_backward_prefetch: BACKWARD_POST` (chapter 2).
4. Add optimizer offload (DeepSpeed).
5. Add gradient / activation checkpointing (should already be on).
6. Add parameter offload (DeepSpeed) or full CPU offload (FSDP).
7. Reconsider whether the cluster is the right shape (chapter 7).

If the first run works but is slow, tune in the reverse direction — turn off checkpointing, grow micro-batch, enable forward prefetch, etc. — measuring MFU (chapter 5) as you go.

## The 8-bit optimizer path

`bitsandbytes` provides an `AdamW8bit` (and `PagedAdamW8bit`) that quantizes the optimizer state to 8 bits, cutting the `12N` cost to `~4N`. Under sharded DP, per-rank optimizer state drops from `12N/P` to `~4N/P`. A 7B on 8 ranks: from 12 GB to ~4 GB, freeing ~8 GB per rank.

- **Enable via** `optim="paged_adamw_8bit"` or `optim="adamw_bnb_8bit"` in `TrainingArguments`.
- **Convergence** is close to identical to 32-bit AdamW on most fine-tuning tasks (Dettmers et al., 2022, "8-bit Optimizers via Block-wise Quantization"). Very small models occasionally show a difference.
- **Composes with FSDP and DeepSpeed** on recent versions. Composes with QLoRA natively (this is where `PagedAdamW` originated).

For full-parameter fine-tuning of 7B+ on memory-tight setups, 8-bit AdamW is a low-risk win. For LoRA / QLoRA runs where the optimizer state is tiny to begin with, it's not necessary.

## Sequence / context parallelism, briefly

Chapter 1 named this as the emerging tool for long-context fine-tuning. When sequence length pushes past what activation checkpointing + FA2 + micro-batch=1 can absorb (typically 16K → 128K contexts on 80 GB cards), the next lever is to shard the sequence dimension itself.

- **DeepSpeed Ulysses** shards along the sequence dimension for attention (Jacobs et al. 2023). Composes with ZeRO-3.
- **Megatron sequence parallelism** shards LayerNorm and dropout along the sequence dimension inside TP groups.
- **Ring Attention** (Liu et al. 2023) uses a communication ring to compute long-context attention without materializing the full attention matrix on any single device.

These are less commonly-configured tools than FSDP / ZeRO, and each has framework-specific setup. If your fine-tuning task is long-context (32K+), start with the framework's documented recipe rather than assembling from parts. Chapter 7 notes when to escalate to the platform team.

## Diagnosing "why isn't this fitting?"

When a run OOMs, the useful move is to bisect *what* blew up. A quick sequence:

1. **Print `torch.cuda.max_memory_allocated()` at multiple checkpoints** in the loop: after model load, after first forward, after first backward, after first optimizer step. Whichever step spikes is your culprit's phase.
2. **Look at the OOM message.** PyTorch reports "Tried to allocate X GiB. Y GiB is free. ..." — the size of the failing allocation is often diagnostic (small attention tensor with a huge `s²` = attention scores, i.e., FA is not on; huge tensor mid-forward = big activation without checkpointing; huge tensor at optimizer step = optimizer state without sharding).
3. **`nvidia-smi` while the run is going**: watch for the memory spike location in the step. If it's inside the optimizer step, ZeRO / offload is the answer; if it's inside forward, activation memory is the answer.
4. **PyTorch memory snapshots** (`torch.cuda.memory._record_memory_history`, saved snapshot viewable with `torch.cuda.memory._save_segment_usage`) — visualizes where allocator time-blocks live. Best debugging tool for opaque OOMs. Chapter 6 has the full playbook.

## Exercise pointer

The exercises in this module all touch memory: exercise 01 asks you to compare memory footprints across sharding strategies, exercise 02 asks the same across DeepSpeed offload options, exercise 03 profiles per-op memory alongside throughput, and exercise 04 forces you to debug OOM at different steps of the training loop.

## Summary

- Six things consume GPU memory: **weights, optimizer state, gradients, activations, framework/comm buffers, fragmentation.** Every technique in this module moves one or more.
- **Activation memory** is the term that surprises people. Grows linearly in batch and sequence; grows quadratically in sequence with vanilla attention (FA2 removes the quadratic term).
- **Gradient (activation) checkpointing** trades ~30% compute for ~4–8× activation memory reduction. Almost always on for 7B+ full-FT. Use the framework-specific flag (`fsdp_activation_checkpointing` / DeepSpeed `activation_checkpointing`); prefer `use_reentrant=False`.
- **BF16 is the default mixed-precision policy** on Ampere+. FP16 only for pre-Ampere; FP8 only on Hopper+ with a proven Transformer Engine integration. FP32 stays for master weights and optimizer state.
- **FlashAttention-2 / 3** removes the `O(s²)` attention memory term and roughly 2–4× the attention throughput. Enable via `attn_implementation="flash_attention_2"` on any Ampere-or-newer card. FA3 on Hopper.
- **8-bit AdamW** cuts optimizer state from `12N` to `~4N`. Composes with FSDP / DeepSpeed / QLoRA. Low-risk win for memory-tight full-FT.
- **Long-context fine-tuning** (32K+) reaches for sequence parallelism (DeepSpeed Ulysses, Megatron SP, Ring Attention). Framework-specific; use the documented recipe.
- The recommended first-run config: BF16 + FULL_SHARD (or ZeRO-3) + gradient checkpointing + FlashAttention-2 + micro-batch 1 + gradient accumulation + no offload. Tune outward from there.
