# The Quantization Landscape for Fine-Tuned Models

**Estimated read:** 30 minutes.
**You should be able to, by the end:** state why we quantize a fine-tuned model at all, distinguish quantization-aware training from post-training quantization and say which one this module is about, name the numeric formats that show up in practice (INT8, INT4, NF4, FP8) and how they encode a weight, describe what "quality loss from quantization" actually means and how mod-108's eval suite measures it, and place the methods this module will cover (GPTQ, AWQ, bitsandbytes, GGUF k-quants) on that map.

## Why we quantize a served model

BF16 (2 bytes per parameter) is the format your fine-tuned checkpoint from mod-103, mod-104, mod-105, mod-106, or mod-107 is almost certainly saved in. It is fine for training. It is often the wrong format for serving. The forcing functions:

- **VRAM.** A 7B model in BF16 needs ~14 GB for weights alone. Add the KV cache for a realistic context length and batch size, activations, CUDA graphs, and framework overhead, and you are past 20 GB before the first request. Quantize the weights to 4 bits and they shrink to ~3.5 GB — the same GPU can now serve larger contexts, larger batches, a bigger model, or all three.
- **Cost.** VRAM headroom translates directly to throughput (you can batch more requests) and to hardware choice (a job that fit on 1× H100 now fits on 1× L40S, or on the consumer 24 GB card in a dev laptop). The cost per million tokens moves accordingly. Chapter 7 does the arithmetic.
- **Latency.** Weight-only quantization does not always help latency — the matmul still runs in a wider compute dtype internally — but it usually reduces memory bandwidth pressure, which for token-by-token decoding is the bottleneck on most GPUs. On CPUs (chapter 6, GGUF) the wins are large: quantized weights fit in cache; BF16 weights do not.
- **Deployability.** A GGUF Q4_K_M version of the model runs on a laptop CPU. A BF16 checkpoint does not. If "run on the customer's edge box" is a requirement, quantization is not a choice.

The cost of all this is quality. Quantization is a lossy compression on the weights; the model's outputs are not exactly the same afterwards. How much worse — measured against the release-gate suite from mod-108 — is what every method in this module is optimizing against. **"Free memory savings" does not exist. There is always a delta; the question is whether it is inside your tolerance.**

## Quantization-aware training vs. post-training quantization

Two families exist. This module is entirely about the second.

**Quantization-aware training (QAT).** The training loop inserts fake-quantization ops that simulate the low-precision arithmetic (round-to-INT8, dequantize) during forward and backward passes. The model learns to be robust to the quantization noise. Result: excellent quality retention at very low bit widths, at the cost of a training run. QAT is the right tool when you are training the model from scratch or through a full fine-tune specifically for a low-precision target. It is out of scope for this module — most fine-tuning teams do not have the training budget to add a QAT phase after the SFT / preference / safety pipeline is already done.

**Post-training quantization (PTQ).** The trained model is quantized after the fact, using a small calibration dataset (a few hundred prompts) to estimate the range and distribution of each weight matrix. No gradient updates. Runs in minutes to a couple of hours on one GPU. All of GPTQ, AWQ, bitsandbytes, and the llama.cpp `quantize` step are PTQ methods. This is what fine-tuning engineers actually do.

The QLoRA path from mod-104 is a hybrid: base weights are quantized (PTQ-ish, via `bitsandbytes` NF4) and then the LoRA adapters are trained on top. The base is never updated. For serving, the same NF4 base + LoRA can be reused, or the whole thing can be re-quantized with GPTQ / AWQ for a different runtime — chapter 2 covers when to do which.

## The numeric formats

Before any method, know the formats. The physical layout of the weight on disk matters because it determines which runtime can load it, whether the matmul stays in fp16 / bf16 or drops to int8 / int4, and how much memory bandwidth you actually save.

### FP16 / BF16 (the baseline)

Two bytes per weight. FP16 has 5 exponent + 10 mantissa bits; BF16 has 8 exponent + 7 mantissa bits (the wider range that matches FP32 exponent, at the cost of precision). Fine-tuning defaults to BF16 on H100 / A100 / L40; the served checkpoint you feed to the quantizer is almost always BF16. Chapter 2 assumes BF16 inputs unless otherwise noted.

### INT8 (`LLM.int8()` and friends)

One byte per weight. A weight matrix is stored as `int8` values plus a small set of `float16` / `float32` scales — one scale per row, per group of rows, or per channel, depending on the scheme. At matmul time, the runtime dequantizes on the fly (or uses INT8 tensor cores when available) to produce a fp16 / bf16 output.

The Dettmers `LLM.int8()` paper (2022) established that vector-wise INT8 quantization can be near-lossless for LLMs *if* the outlier channels are handled separately in fp16 — because a small number of activation channels have extreme magnitudes that a naive INT8 scheme cannot represent without either clipping them (quality collapse) or bumping every other channel's precision (defeats the purpose). This is the "mixed-precision decomposition" idea, and it is why `bitsandbytes`' 8-bit path exists.

### INT4 (GPTQ, AWQ, and the 4-bit path)

Half a byte per weight. Weights are packed two per byte, with a scale (and sometimes zero-point) per group of typically 32, 64, or 128 weights. The Frantar et al. GPTQ paper (2022) and the Lin et al. AWQ paper (2023) showed 4-bit weight-only quantization can retain most of a model's quality if the calibration and rounding are done well. Chapter 2 unpacks *how* they do that; here, the point is that INT4 is 4× smaller than BF16 with (in practice, for methods that work) a quality delta small enough to be inside most release-gate tolerances.

### NF4 (`bitsandbytes`)

Four bits per weight, but the codebook is not linear. NF4 (Normal Float 4) uses the fact that a neural network's weights are approximately normally distributed and picks 16 code points that are the *quantiles* of a standard normal — so more resolution is spent near zero (where most weights are) and less far from zero. Together with **double quantization** (the per-group scales are themselves quantized to 8 bits, then a per-block scale for those), NF4 approaches the effective precision of a well-tuned INT4 scheme with a simpler recipe. NF4 is the format QLoRA (mod-104 chapter 2) used and remains the `bitsandbytes` 4-bit default.

### FP8 (H100-era)

One byte per weight, with a floating-point layout — `e4m3` (4 exponent + 3 mantissa) or `e5m2` (5 exponent + 2 mantissa). Runs on FP8 tensor cores on Hopper (H100) and newer. Currently supported for serving in vLLM (Marlin FP8, w8a8) and TGI's FP8 path. Quality is generally between INT8 and BF16; the throughput on FP8 tensor cores can be dramatically higher than either. FP8 is the "if you have H100s, look here" option; chapters 4 and 5 cover the runtime flags.

### GGUF k-quants (`llama.cpp`)

Not a single format — a family. `Q8_0`, `Q6_K`, `Q5_K_M`, `Q5_K_S`, `Q4_K_M`, `Q4_K_S`, `Q3_K_M`, `Q2_K`, and more. The `_K_*` variants use a more sophisticated per-block quantization with mixed precision *within* the file (attention layers might be kept at higher precision than feed-forward layers). Chapter 6 unpacks the ladder; the takeaway now is that "GGUF" is a container format and the interesting choice is which quantization inside it.

### The bit-width vs. quality trade-off in one picture

The empirical shape, roughly (Dettmers & Zettlemoyer, "The case for 4-bit precision," 2023):

```
Quality (relative to BF16, higher = better)
   ^
1.0┤ ==========================
   │                          `..
   │                             `.        BF16
   │                               `.      FP8
   │                                 `.    INT8
   │                                   `.  NF4 / INT4 (GPTQ/AWQ)
0.9┤                                    `.
   │                                      `.
   │                                        `.
0.7┤                                          `. INT3
   │                                            `.
   │                                              `.
0.4┤                                                `. INT2
   +--------+--------+--------+--------+--------+----->  bits per weight
   16       8        6        5        4        3       2
```

The knee is at 4 bits. Above the knee, well-executed quantization costs almost nothing; below it, quality falls off sharply. This is why every "production" quantization method in this module lives between 4 and 8 bits and why aggressive 2–3 bit schemes are still research-grade.

<!-- needs-research: the exact position of the knee shifts with model size (bigger models are more robust to low-bit quantization) and family. Treat the sketch as directional, not calibrated. See Dettmers & Zettlemoyer 2023 for the empirical curves per model family. -->

## What "quality delta from quantization" actually means

The eval infrastructure you built in mod-108 is the measurement tool. Every quantized artifact in this module gets re-run through the release-gate suite; the delta between full-precision and quantized numbers is the primary quality signal.

Three shapes of delta show up:

**1. Uniform small regression.** Every task in the suite drops by roughly the same 0.5–1.5 pp. Reading: the quantization is mostly-lossless but adds a bit of numeric noise across the board. Common with a well-tuned INT8 or a well-calibrated INT4 GPTQ / AWQ run. Usually inside tolerance.

**2. Task-selective regression.** Most tasks are flat; one or two (typically math / code — GSM8K and HumanEval — or the task-specific held-out set) drop by 3–10 pp. Reading: the quantization is degrading a specific capability that requires numerical precision. Frequent with lower-bit schemes or with calibration data that does not cover the task's distribution. Chapter 2 addresses this via better calibration.

**3. Catastrophic collapse.** Aggregate numbers drop 20+ pp; generations show obvious quality failures (repetition, incoherence). Reading: something is wrong with the quantization pipeline — usually a group-size / scale mismatch, a botched merge before quantization, or a calibration dataset that is empty / mis-tokenized. Chapter 2's smoke tests catch this before the release gate ever runs.

The tolerance number is a decision, not a physical constant. A reasonable starting point (borrowed from the mod-108 chapter 7 handoff table):

| Precision | Typical academic-eval delta | Typical task-specific delta | Notes |
|---|---|---|---|
| BF16 (baseline) | 0 | 0 | — |
| FP8 | ≤ 0.5 pp | ≤ 1 pp | H100-only; quality near-lossless on well-behaved models |
| INT8 (`bitsandbytes` / SmoothQuant) | ≤ 1 pp | ≤ 1–2 pp | very safe on 7B+ models |
| INT4 GPTQ (well calibrated) | ≤ 1–2 pp | ≤ 2–4 pp | dominant production choice |
| INT4 AWQ (well calibrated) | ≤ 1–2 pp | ≤ 2–4 pp | activation-aware, similar quality |
| NF4 (`bitsandbytes` 4-bit) | ≤ 2–3 pp | ≤ 3–5 pp | easy to apply; slightly worse than GPTQ/AWQ at inference |
| GGUF Q4_K_M | ≤ 1–2 pp | ≤ 2–4 pp | CPU / edge; the k-quants punch above their bit width |
| GGUF Q3_K_M | 2–5 pp | 3–8 pp | edge only; test carefully |
| GGUF Q2_K | 5–15 pp | 5–20 pp | last resort, expect degradation |

<!-- needs-research: these ranges are drawn from the primary papers and llama.cpp / vLLM published benchmarks; they are order-of-magnitude anchors, not commitments. Verify against your own release-gate numbers on your specific model — the deltas depend heavily on base model, fine-tune style, and evaluation task. -->

The rule of thumb: **if the quantized number is inside your gate's per-task tolerance, ship it; if it is not, either re-quantize with better settings or accept a wider (bigger) format.** Chapter 2 covers the "better settings" knobs.

## Weight-only vs. weight-and-activation quantization

An orthogonal axis to bit width. When you say "INT4 quantization" you almost always mean weight-only: the weight matrices are stored in INT4, but activations flow through the network in fp16 / bf16. At each matmul, the runtime dequantizes the weights on the fly (or uses tensor cores that accept INT4 weight × fp16 activation directly) to produce an fp16 output.

The alternative — **weight-and-activation quantization** (W8A8, W4A8, W4A4) — stores both weights *and* activations at low precision. This can unlock much higher throughput on hardware with dedicated INT8 / INT4 tensor cores, at the cost of higher quality risk (activations are noisier than weights and outlier handling matters even more). SmoothQuant (Xiao et al., 2022), Marlin, and vLLM's W8A8 / W4A16 kernels operate in this regime. Chapter 2 mentions SmoothQuant; chapter 4 covers the vLLM engine flags that enable W8A8 for FP8 / INT8 paths on H100.

For the fine-tuning engineer, the practical rule: **default to weight-only 4-bit for VRAM wins; reach for W8A8 (FP8 / INT8) when your bottleneck is throughput on H100-class hardware and you can afford a small extra quality-validation pass.**

## Where each method lands on the map

The concrete choices this module covers, projected onto (bit width) × (runtime target):

|  | vLLM | TGI | `bitsandbytes` (transformers) | llama.cpp |
|---|---|---|---|---|
| BF16 | ✓ | ✓ | ✓ | ✗ |
| FP8 (E4M3 / E5M2) | ✓ (H100) | ✓ (H100) | — | ✗ |
| INT8 (`LLM.int8()`) | limited | limited | ✓ | ✗ |
| INT8 (SmoothQuant / W8A8) | ✓ | ✓ | — | ✗ |
| INT4 GPTQ | ✓ | ✓ | via `transformers` | ✗ |
| INT4 AWQ | ✓ | ✓ | via `transformers` | ✗ |
| INT4 NF4 (`bitsandbytes`) | limited | ✓ (bnb path) | ✓ | ✗ |
| GGUF (Q8 → Q2) | ✗ | ✗ | ✗ | ✓ |

The empty cells matter as much as the filled ones. GGUF is a `llama.cpp` format; it does not run on vLLM or TGI. GPTQ / AWQ are GPU inference formats; llama.cpp does not consume them (it has its own quantization pipeline). If your target runtime is vLLM, chapter 6's GGUF work does not help you serve — it is for edge / CPU deployments, which is a different product decision. Choose the runtime first, then the quantization method. Chapter 7's runbook makes this explicit.

## The two other quantization axes that matter

Beyond bit width and weight-vs-activation, two more axes decide whether a quantized model is any good:

**Calibration data.** GPTQ, AWQ, and SmoothQuant all consume a small calibration dataset (typically 128–1024 prompts, sometimes just tokens from C4 or the pile). The scales and grouping are chosen to minimize activation error on this data. If the calibration data does not match your production traffic, the quantized model can regress on production traffic more than eval numbers suggest. Chapter 2 covers what to calibrate on for a fine-tuned model.

**Group size.** Weights are quantized in groups (blocks of 32, 64, 128, or 256 weights sharing a scale). Smaller groups = better quality, larger file, slightly slower dequant. The default of 128 is a reasonable knee. Chapter 2 discusses when to move it.

## Where the KV cache fits

Weight quantization saves weight memory. It does not directly help the *activation* memory, and it does not touch the KV cache at all. The KV cache — the per-token stored key/value tensors used to skip recomputation during decoding — is a separate memory pool that grows linearly with context length × batch size, and it can dwarf the weights on long-context workloads.

Runtimes (vLLM, TGI) offer a separate `--kv-cache-dtype` knob to store the KV cache in fp8 / int8. This is a distinct quantization from weight quantization, with its own quality/throughput trade-off. Chapter 4 and chapter 7 cover it. Mention it here so you do not confuse "the model is INT4" with "everything is INT4" — the weights are, the KV cache is whatever the runtime is told to make it.

## The one-page mental model

Fitting the whole picture on one page before we dive in:

```
                        +----------------------------+
                        | Trained checkpoint (BF16)  |
                        | from mod-103/104/105/106/107 |
                        +--------------+-------------+
                                       |
       +-------------------------------+-------------------------------+
       |                               |                               |
       v                               v                               v
+------+-------+          +------------+-----------+       +-----------+----------+
| Adapter is    |          | GPU serving target     |       | CPU / edge target    |
| separate LoRA |          | (vLLM / TGI)           |       | (llama.cpp / Ollama) |
| (ch. 3)       |          |                        |       |                      |
+------+--------+          +------------+-----------+       +-----------+----------+
       |                                |                               |
       v                                v                               v
+---------------+            +----------+-----------+         +---------+----------+
| Merge into    |            | Choose quantization  |         | Convert to GGUF    |
| base OR keep  |            | method (ch. 2):       |         | (ch. 6):           |
| multi-LoRA    |            |  FP8, INT8, GPTQ,    |         |  Q8_0 → Q4_K_M →   |
| (ch. 3)       |            |  AWQ, bnb, W8A8     |         |  Q3_K_M → Q2_K     |
+------+--------+            +----------+-----------+         +---------+----------+
       |                                |                               |
       v                                v                               v
+---------------+            +----------+-----------+         +---------+----------+
| Deploy to     |            | Deploy to vLLM       |         | Deploy to          |
| vLLM / TGI    |            | (ch. 4) or TGI       |         | llama.cpp / Ollama |
| (ch. 4, 5)    |            | (ch. 5) with the     |         | / mobile runtime   |
|               |            | quantization flag    |         | (ch. 6)            |
+------+--------+            +----------+-----------+         +---------+----------+
       |                                |                               |
       +----------------+---------------+-------------------------------+
                        v
              +---------+---------+
              | Re-run mod-108's  |
              | release gate on   |
              | the served        |
              | artifact (ch. 1,  |
              | ch. 7)            |
              +---------+---------+
                        |
                        v
              +---------+---------+
              | Runbook + SLOs +  |
              | cost table        |
              | (ch. 7)           |
              +-------------------+
```

Every subsequent chapter fills in one of the boxes.

## What is in scope for this module

**In scope:**

- Applying GPTQ, AWQ, and `bitsandbytes` 8-bit / 4-bit to a fine-tuned checkpoint (chapter 2).
- Choosing between merging an adapter into the base vs. serving multi-LoRA (chapter 3).
- Deploying to vLLM and TGI with a structured serving config (chapters 4, 5).
- Exporting to GGUF for `llama.cpp` and verifying equivalence (chapter 6).
- Latency / throughput / cost trade-offs, load testing, and the serving runbook (chapter 7).
- Re-running mod-108's release gate against every quantized variant.

**Out of scope (handled elsewhere):**

- Quantization-aware training or full-precision distillation. Both would require a full training run; this module assumes the training pipeline is complete.
- The training-time QLoRA loop (chapter 2 references but does not re-derive it; mod-104 owns it).
- The eval methodology itself (mod-108 owns it; we consume the gate).
- Adapter registry, canary / rollback tooling, autoscaling, and the model card that ships alongside the artifact — all mod-111. Chapter 7's runbook is the handoff artifact.
- Managed / hosted inference (Anthropic, OpenAI, Bedrock) — mod-110.

## Summary

- Quantize to save VRAM, cut cost, sometimes cut latency, and unlock deployment targets that BF16 cannot reach — but every quantization step trades one axis for another, always with a quality delta measured against mod-108's suite.
- This module is entirely about **post-training quantization**: the trained model is quantized after the fact from a small calibration dataset. QAT is out of scope.
- The formats worth knowing are BF16 (baseline), FP8 (H100), INT8, INT4 (via GPTQ / AWQ), NF4 (`bitsandbytes`), and the GGUF k-quant family (`llama.cpp`). The empirical knee of quality vs. bit width is at 4 bits; above it is nearly free, below it is research-grade.
- Weight-only quantization saves weight memory; weight-and-activation quantization (W8A8, FP8) can unlock throughput on modern GPUs at higher quality risk. The KV cache is a separate pool with its own dtype knob.
- Choose the runtime first (vLLM / TGI for GPU, `llama.cpp` for CPU / edge), then the quantization method that runtime accepts. Formats do not cross runtimes.
- "Quality delta from quantization" means the gap between full-precision and quantized numbers on the release gate. Uniform ≤1 pp deltas are usual; task-selective 3–10 pp deltas mean a calibration problem; 20+ pp deltas mean a pipeline bug. Chapter 2 covers the knobs to move each.
