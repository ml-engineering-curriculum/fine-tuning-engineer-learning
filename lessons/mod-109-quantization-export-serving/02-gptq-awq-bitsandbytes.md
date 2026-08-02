# GPTQ, AWQ, and bitsandbytes 8-bit / 4-bit

**Estimated read:** 45 minutes.
**You should be able to, by the end:** describe at a working level how GPTQ, AWQ, and `bitsandbytes` (`LLM.int8()` and NF4) each convert a BF16 weight matrix into a low-bit representation, run each of the three on a fine-tuned checkpoint with a defensible choice of calibration data and group size, spot the failure modes (bad calibration, wrong quantization target, tokenizer / dtype mismatch), and pick which method to reach for based on the base size, hardware, target runtime, and latency SLO.

## The problem all three methods solve

You have a BF16 checkpoint. You want a serving artifact that is 4× smaller (INT4) or 2× smaller (INT8) at the smallest possible quality delta on the mod-108 release gate. Post-training quantization does this in three moves:

1. **Load the model in BF16.**
2. **Push a calibration dataset (a few hundred prompts) through it** and record, for each weight matrix, some statistics about the activations that flow through it and about the weights themselves.
3. **Choose a low-bit representation for each weight matrix** — a set of integer codes plus scales — that minimizes some reconstruction error, either on the weights alone (naive) or on the *matmul output* (better).

The three methods differ in what they measure in step 2 and how they minimize error in step 3. Getting that difference straight is the point of this chapter.

## The math you need before we start

All three methods store a weight matrix `W` (shape `[out_features, in_features]`) as:

- an integer tensor `W_int` in b bits (b = 8 or 4), packed compactly on disk;
- a small set of `float16` / `float32` scale factors `s` (and optionally zero-points `z`);
- a grouping scheme that says which weights share which scale.

At inference:

```
w_reconstructed ≈ s * (W_int - z)             # per-group dequantization
y = x @ w_reconstructed.T                     # ordinary matmul in fp16/bf16
```

Three grouping schemes matter:

- **Per-tensor**: one scale for the whole matrix. Fastest, worst quality.
- **Per-channel** (per-row): one scale per output channel. Good balance for INT8.
- **Per-group** (block of `group_size` consecutive weights per row): one scale per group. Standard for INT4. Typical `group_size` values are 32, 64, or 128. Smaller groups = better quality, larger file, marginally slower.

The "activation-outlier" problem — a small number of activation channels have magnitudes 10–100× larger than the rest, and quantizing weights naively is dominated by errors on those channels — is the core reason all three methods exist. Each solves it differently.

## bitsandbytes: the "easiest thing that works"

`bitsandbytes` (Dettmers et al.) is a CUDA library that provides two production PTQ paths, both invoked directly through `transformers` at model-load time. No separate quantization step, no calibration data on disk — the quantization happens in memory as the weights load.

### `LLM.int8()` — 8-bit weight-only with outlier decomposition

The mechanism (Dettmers et al., 2022):

1. For each linear layer's input activation, identify channels whose magnitude exceeds a threshold (default: 6.0). Route them through a separate fp16 matmul against a fp16-preserved slice of the weight.
2. For the remaining "normal" activation channels, quantize the weights to INT8 with per-channel scales and run an INT8 matmul.
3. Sum the two matmul outputs.

The recipe:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
```

That's the whole thing. No calibration, no group size, no additional files on disk. The weight quantization happens on the fly.

Quality: near-lossless on 7B and larger models in the original paper's evals. Speed: the outlier decomposition adds overhead; `LLM.int8()` is generally slower per token than BF16 despite the memory saving. Use it when you need the memory saving cheaply and latency is not the constraint (e.g., fitting a bigger model on a smaller GPU for offline generation).

Save-on-disk caveat: `bitsandbytes`-quantized weights can be saved with `model.save_pretrained(path)` and reloaded with the same `BitsAndBytesConfig`, but the serving story is weaker than GPTQ / AWQ — many production inference servers (vLLM in particular) prefer GPTQ / AWQ as their 4-bit path. Chapter 4 covers what vLLM accepts.

### 4-bit NF4 — the QLoRA base format

Same API surface, different flag:

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)
```

The flags, one at a time:

- **`bnb_4bit_quant_type="nf4"`** — NormalFloat 4 (Dettmers et al., QLoRA, 2023). Uses 16 code points positioned at the quantiles of a standard normal distribution, so more precision near zero. The alternative, `"fp4"`, uses a floating-point 4-bit layout and is slightly worse for weight quantization. NF4 is the default and the right choice for LLM weights.
- **`bnb_4bit_use_double_quant=True`** — the per-group scales (which are fp16 by default) are themselves quantized to 8 bits. Saves ~0.4 bits per weight in exchange for negligible quality cost. On by default in modern QLoRA recipes; leave it on.
- **`bnb_4bit_compute_dtype=torch.bfloat16`** — the dtype in which the matmul actually runs. The 4-bit weights are dequantized to this dtype at matmul time. `bfloat16` matches the training-time dtype for most modern fine-tunes; use `torch.float16` on Ampere / Turing GPUs that do not support bf16 well.

The full recipe is the same one mod-104 chapter 2 introduced for QLoRA training. At serving time you use the same load, without a LoRA adapter attached (or with, if you are running multi-LoRA on top of an NF4 base — chapter 3).

### When bitsandbytes is the right pick at serving time

- The load-and-serve simplicity matters (research / offline / dev environments).
- You are already using QLoRA and want the base format to be identical between training and serving.
- Your target runtime is `transformers` directly or TGI with the `bitsandbytes` path.
- You want 8-bit and 4-bit both from the same tool without maintaining two pipelines.

### When it is not

- You care about inference throughput on modern GPUs — GPTQ and AWQ have kernels (ExLlamaV2, Marlin, ExLlama, GEMM-int4) that are meaningfully faster than the current `bitsandbytes` 4-bit path at inference.
- Your target is vLLM, which accepts `bitsandbytes` but is faster and better tuned on GPTQ / AWQ.

## GPTQ: the Frantar et al. 2022 method

GPTQ ("Generative Pre-trained Transformer Quantization," Frantar, Ashkboos, Hoefler, Alistarh, 2022) is the archetypal one-shot post-training 4-bit method for LLMs. Two things distinguish it:

**1. Layer-wise reconstruction, not weight matching.** Naive 4-bit quantization minimizes `||W - W_int * s||` — the difference between the original weights and their reconstruction. GPTQ instead minimizes the *matmul output* error: `||W · X - W_int · s · X||`, where `X` is the calibration-data activations for that layer. Two weight matrices with the same `L2` reconstruction error can produce very different outputs on real activations; GPTQ picks the one that matters.

**2. Approximate second-order optimization ("OBS-like").** The rounding is chosen using the inverse Hessian of the reconstruction loss with respect to the weights, computed on the calibration activations. When a weight is rounded, the *other* weights in the layer are adjusted to compensate — this is the "Optimal Brain Surgeon"-style update. It closes most of the quality gap between naive per-group INT4 and full precision.

Result: 4-bit GPTQ retains most of a model's quality (within 1–2 pp on academic evals for well-behaved 7B+ models), produces a file that GPU inference kernels (ExLlamaV2, Marlin, vLLM's GPTQ kernel) can load and run at high throughput, and takes ~30 min – 2 hours on one GPU for a 7B model.

### Recipe with `AutoGPTQ` / `gptqmodel`

The `AutoGPTQ` package (superseded in some ecosystems by `gptqmodel`, but the interface is essentially the same) is the reference implementation:

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = "./llama-3.1-8b-instruct-gptq-4bit"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,          # standard; 32 or 64 for higher quality at cost of file size
    desc_act=False,          # "act-order"; True is higher quality, slower, sometimes incompatible
    damp_percent=0.01,       # numeric stability of the OBS-like update; 0.01 is safe
)

model = AutoGPTQForCausalLM.from_pretrained(MODEL_ID, quantize_config)

# --- Calibration data ---
# 128–1024 samples of ~2048 tokens is a normal choice.
calibration = load_dataset(
    "allenai/c4", "en", split="train", streaming=True
)
calibration = [
    tokenizer(next(iter(calibration))["text"], return_tensors="pt", truncation=True, max_length=2048)
    for _ in range(256)
]

model.quantize(calibration)

model.save_quantized(OUT_DIR, use_safetensors=True)
tokenizer.save_pretrained(OUT_DIR)
```

### The flags that decide quality

- **`bits`** — 4 is the standard; 8-bit GPTQ exists but is rarely the right pick (SmoothQuant / bitsandbytes are easier at INT8).
- **`group_size`** — 128 is the default. 64 or 32 improve quality (typically 0.2–0.5 pp on academic evals) at ~1.5–3× the on-disk scale-table size. `-1` means per-channel and is generally worse than per-group at 4-bit.
- **`desc_act`** — activation-order quantization. Weights are quantized in order of the diagonal Hessian entries, so the most-important weights are quantized last and can adjust to compensate for the others. Higher quality (~0.5–1 pp), noticeably slower to quantize, and some inference kernels do not accept `desc_act=True` files. Check your target kernel before enabling.
- **`damp_percent`** — a stabilization term added to the Hessian diagonal. `0.01` is the recommended default; increase if you see numeric instability (NaN scales) during quantization.

### Calibration data — what to use

The single most impactful decision. Three reasonable defaults:

1. **C4 (allenai/c4) or the pile.** Generic pretraining-style text. Safe baseline. Works fine for a broad instruct model. This is what the original GPTQ paper used.
2. **A sample of your own SFT / DPO dataset.** Better for a heavily fine-tuned model where the distribution has shifted from generic web text. 256–512 rows of your training data is a strong default for a fine-tuned model.
3. **A sample of production traffic.** Best when you have logged traffic and it is representative. Requires care about PII and about not leaking eval-set contamination back into the calibration pool.

The rule: **calibrate on data that looks like what the model will see in production.** A fine-tuned code model calibrated on C4 will quantize worse than one calibrated on code prompts. A chat model calibrated on completions-only text (no chat template) will subtly degrade the model's instruction-following. Chapter 1's task-selective regression pattern is almost always a calibration-data mismatch.

<!-- needs-research: sample-size ablations vary across papers and models; the 128–1024 range is what the GPTQ paper and current tooling defaults span. If you have the budget, a small ablation (128 vs. 512 vs. 1024) on your task-specific held-out set is the honest way to pick. -->

## AWQ: the Lin et al. 2023 method

AWQ ("Activation-aware Weight Quantization," Lin, Tang, Tang, Yang, Cai, Zhu, Han, 2023) approaches the same problem from a different angle.

**The observation:** not all weights are equally important. A small fraction (~1%) of weights corresponds to activation channels with much larger magnitudes; preserving those weights at higher precision (or scaling them up before quantization so they land in the "middle" of the INT4 range) gives most of the quality back.

**The mechanism:** for each layer, AWQ searches for a per-channel scale that pre-scales the weights *and* inverse-scales the corresponding activation channel, such that the quantized-then-dequantized matmul minimizes activation-weighted reconstruction error. It does not need a Hessian, does not do OBS-style weight updates — just the scale search plus grouped-INT4 quantization.

Result: quality comparable to GPTQ (within a fraction of a pp on academic evals), typically faster to quantize (no Hessian inversion), and inference kernels are similar in performance. AWQ tends to do slightly better on some tasks and slightly worse on others; in practice the choice is often driven by which format your target runtime supports better, not by a quality gap.

### Recipe with `AutoAWQ`

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = "./llama-3.1-8b-instruct-awq"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoAWQForCausalLM.from_pretrained(MODEL_ID, safetensors=True)

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",   # or "GEMV" for smaller-batch use; check kernel compat.
}

# Calibration: same shape as GPTQ; typically 128–512 samples.
# AutoAWQ can pull a default calibration set if none is provided, but supplying your own is better.
model.quantize(tokenizer, quant_config=quant_config)

model.save_quantized(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
```

### AWQ knobs

- **`w_bit=4`** — 4-bit is the standard AWQ target. 3-bit exists in research but is not production-worthy.
- **`q_group_size=128`** — same trade-off as GPTQ's `group_size`; 128 is the safe default.
- **`zero_point=True`** — asymmetric quantization (an integer offset in addition to the scale). Recommended by AutoAWQ and usually a small quality win over symmetric.
- **`version="GEMM"` / `"GEMV"`** — kernel variants. GEMM is faster for larger batches (typical serving), GEMV is optimized for batch=1 (chat). Pick based on your traffic pattern; vLLM's AWQ kernel typically expects GEMM.

Calibration data guidance is the same as for GPTQ: match production distribution. AutoAWQ's default calibration set (Pile / MIT) is fine for generic instruct models; supply your own data for narrow use cases.

## GPTQ vs. AWQ: how to decide

The honest answer is that on well-behaved 7B–13B instruct models, well-calibrated 4-bit GPTQ and well-calibrated 4-bit AWQ produce nearly identical eval numbers. The decision is usually driven by ecosystem, not quality:

| Consideration | GPTQ | AWQ |
|---|---|---|
| Reference paper | Frantar et al., 2022 | Lin et al., 2023 |
| Tooling | `AutoGPTQ`, `gptqmodel`, `optimum` | `AutoAWQ`, `optimum` |
| vLLM support | ✓ (`--quantization gptq`) | ✓ (`--quantization awq`) |
| TGI support | ✓ | ✓ |
| Marlin fast kernel | ✓ (Marlin GPTQ) | ✓ (Marlin AWQ) |
| Quantization speed on 7B (rough) | 30–120 min | 15–60 min |
| Quality on generic instruct tasks | very close | very close |
| Quality with poor calibration | more sensitive (Hessian-based) | more sensitive (activation-scale search) |
| Notable strength | mature ecosystem, `desc_act` gives slight edge if enabled | slightly faster to quantize, no Hessian |
| Notable weakness | `desc_act=True` files may not run on some older kernels | somewhat newer; less battle-tested on very small (<1B) or very large (>70B) models |

Practical rule of thumb: **if your target runtime documents a preferred format for a given model family, use that one; otherwise try both, and pick the one whose release-gate delta is smaller on your specific held-out set.** The two methods are close enough that a real eval on your data beats any general recommendation.

## SmoothQuant: the W8A8 alternative worth naming

SmoothQuant (Xiao, Lin, Seznec, Wu, Demouth, Han, 2022) is the reference method for W8A8 — both weights AND activations quantized to INT8, matmul running on INT8 tensor cores. Not weight-only. The mechanism: migrate the difficulty of quantization from activations (which have outliers) to weights (which do not) by rescaling — activations become smoother (easier to quantize) at the cost of weights becoming harder (still doable). Then quantize both to INT8.

Why care: when your bottleneck is throughput on H100-class hardware and the INT8 tensor cores are underutilized because everything is FP16, W8A8 can unlock 2× or more throughput at a small quality cost. vLLM and TGI both support SmoothQuant-style W8A8 as a "compressed-tensors" or explicit quantization mode. Chapter 4 covers the vLLM flags.

For the fine-tuning engineer's decision tree, SmoothQuant lives at the intersection of "H100+ available" and "throughput matters more than the last 1 pp of quality." If both are true, benchmark it. If either is not, default to 4-bit GPTQ / AWQ.

## FP8 in 2026, briefly

Vendor-native FP8 formats (`e4m3`, `e5m2`) on H100 / H200 / Blackwell run on dedicated FP8 tensor cores. vLLM and TGI both accept FP8 checkpoints; quality is typically better than INT8 and near-BF16 on well-behaved models; throughput on FP8-capable hardware can be dramatically higher. Chapter 4 covers `--quantization fp8` in vLLM and the calibration tooling (Nvidia's `Model Optimizer` / `ammo` / `TensorRT-LLM`'s FP8 pipeline is common). If you have H100+ and are throughput-bound, FP8 is worth the experiment.

## The failure modes

The full list of "why does my quantized model produce garbage" causes:

**1. Calibration data does not match production.** Most common. The model quantizes fine, benchmarks fine on generic evals, then produces subtly worse outputs on production traffic. Fix: calibrate on data that looks like production.

**2. `desc_act=True` file loaded by a kernel that does not support it.** GPTQ-specific. Model loads, produces gibberish or fails silently. Fix: quantize with `desc_act=False`, or verify the target kernel's compatibility.

**3. Group-size / bits mismatch on load.** The file's metadata records `bits=4, group_size=128`; the load path defaults or explicit args set `bits=4, group_size=32`. Silent shape errors. Fix: always load with the config that matches the file's metadata (both `AutoGPTQForCausalLM.from_quantized` and `AutoAWQForCausalLM.from_quantized` read the metadata automatically — do not override).

**4. Tokenizer / chat-template mismatch after quantization.** You saved the quantized model but forgot to `tokenizer.save_pretrained(OUT_DIR)`. The serve path loads a stale tokenizer. Every downstream chat behavior is wrong. Fix: always save tokenizer next to the quantized weights, and prefer `tokenizer.save_pretrained(OUT_DIR)` from the same script.

**5. Merged the LoRA into an NF4 base before quantizing.** As discussed in mod-104 chapter 3, an NF4 base cannot be cleanly merged with a BF16 LoRA. If you tried and then quantized further, both operations are corrupted. Fix: merge the LoRA into a *BF16* base (dequantize or reload as BF16), then quantize the merged BF16 model with GPTQ or AWQ.

**6. Adapter's target modules include `lm_head` or other unquantized layers, and you quantized without excluding them.** Quality collapse from over-quantization of a critical layer. Fix: exclude the `lm_head` and embedding layers from quantization (`AutoGPTQ` `modules_to_not_convert=["lm_head"]`, or the equivalent in AWQ). The reference implementations do this by default for most model families.

**7. Wrong dtype at load.** The quantized weights are in INT4; the compute dtype must match training (usually BF16). Loading with `torch_dtype=torch.float32` or without specifying the compute dtype can regress quality on some paths. Fix: set the compute dtype explicitly.

## The smoke tests you run before the release gate

Before committing a quantized artifact to the release-gate suite (which takes hours), run a five-minute smoke test. If any of these fails, do not run the full gate — go back to step 1.

**Smoke test 1 — Load and generate.**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(OUT_DIR)
model = AutoModelForCausalLM.from_pretrained(OUT_DIR, device_map="auto")
inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=20)[0]))
```

Expected output: coherent completion mentioning Paris. If you get punctuation salad, the quantization broke.

**Smoke test 2 — Perplexity on a small held-out slice.**

```python
import torch
losses = []
for text in held_out_texts[:32]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    losses.append(out.loss.item())
ppl = float(torch.exp(torch.tensor(losses).mean()))
print(f"quantized perplexity: {ppl:.3f}")
```

Compare against the same number computed on the BF16 model. Expect a 1–5% relative increase in perplexity for a good INT4 quantization; a 20%+ increase is a failure.

**Smoke test 3 — A canonical short benchmark.**

Run 100 items of MMLU or your task-specific held-out set (`--limit 100` in `lm-evaluation-harness`). Should complete in a few minutes. Expect scores within a few pp of the full-precision numbers. If a task drops by 10+ pp, investigate before running the full suite.

Chapter 4's vLLM deploy and the exercise-01 comparison assume you have run these three before you commit hours to the full release gate.

## Serving-side considerations that leak back into quantization choice

The quantization decision is not made in isolation from serving. Three back-pressures from the runtime:

- **Kernel availability.** vLLM's Marlin kernel is the current fast path for both GPTQ and AWQ on Ampere / Hopper — but only for certain (bits, group_size) combinations. Off the fast path, the same quantized file runs 2–3× slower. Chapter 4 covers vLLM's `--quantization` flag and what it actually dispatches to.
- **Multi-LoRA compatibility.** vLLM's multi-LoRA path works with certain quantized-base combinations and not others; chapter 3 covers the current state. If you plan to serve multi-LoRA, quantize the base with a method the LoRA path supports.
- **FP8-only hardware.** If you are on H100 with a mixed cluster, an FP8-quantized artifact is faster and simpler than an INT4 one — but it will not run on the L4 / L40S / A100 boxes in the same cluster. Standardize the quantization to the least-common-denominator hardware, or maintain two artifacts.

## Exercise pointer

`exercises/exercise-01-gptq-awq-bnb-comparison.md` has you quantize a single fine-tuned checkpoint with all three methods (4-bit GPTQ, 4-bit AWQ, 4-bit NF4 via bitsandbytes) using calibration data drawn from your training set. You then run mod-108's fast-lane suite against each variant and produce a comparison matrix (quality, VRAM, latency, disk) with a defended recommendation for a specified serving scenario.

## Summary

- Post-training quantization takes a BF16 checkpoint plus a small calibration dataset and produces a low-bit weight file. This module covers the three current production methods: GPTQ, AWQ, and `bitsandbytes` (8-bit and 4-bit NF4).
- **`bitsandbytes`** is the easiest path — one flag at load time — and the right choice when simplicity or QLoRA-consistency matters more than inference throughput. `LLM.int8()` for 8-bit; NF4 for 4-bit.
- **GPTQ** (Frantar et al., 2022) does layer-wise reconstruction with an OBS-style weight update using the inverse Hessian of the calibration activations. Standard 4-bit choice; `AutoGPTQ` / `gptqmodel` are the tools; `bits=4, group_size=128` is the safe default.
- **AWQ** (Lin et al., 2023) does an activation-weighted per-channel scale search plus grouped-INT4 quantization. Comparable quality to GPTQ, typically faster to quantize; `AutoAWQ` is the tool; same defaults.
- The single most important choice is the **calibration data**. Calibrate on data that looks like production traffic — this is the difference between a "1 pp regression on the release gate" outcome and a "task-selective 5–10 pp regression" outcome.
- SmoothQuant is the reference W8A8 method (both weights and activations INT8) when you are on H100 and throughput matters. FP8 on H100+ is a strong alternative when hardware supports it.
- Run the three smoke tests (load-and-generate, perplexity delta, 100-item benchmark) before every full release-gate run. Most quantization failures show up on those in five minutes.
- The runtime you plan to serve on constrains the quantization method. Kernel support, multi-LoRA compatibility, and hardware FP8-availability all feed back into the choice; chapters 3–4 close the loop.
