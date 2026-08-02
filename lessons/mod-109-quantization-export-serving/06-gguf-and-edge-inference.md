# GGUF Export and Edge Inference

**Estimated read:** 35 minutes.
**You should be able to, by the end:** convert a Hugging Face fine-tuned checkpoint to GGUF, choose a quantization level from the k-quant ladder (Q8_0 → Q6_K → Q5_K_M → Q4_K_M → Q3_K_M → Q2_K) with a defended trade-off, run inference with `llama.cpp` (`llama-cli` and `llama-server`) and Ollama, verify the GGUF file matches the source weights well enough to ship, and understand what changes when the target is a CPU, a laptop GPU, or a mobile / embedded device instead of a datacenter H100.

## Why a separate chapter for GGUF

Chapters 2, 4, and 5 covered GPU serving on modern datacenter hardware. The reality is that a lot of "the model" now needs to run somewhere else: on a laptop CPU because the customer is offline, on a M-series Mac because the developer needs local iteration, on a Raspberry Pi because the product is edge-deployed, on an Android or iPhone through an app-embedded runtime, on a k8s CPU pod because the workload is tiny and GPUs are not worth the price. GGUF + `llama.cpp` is the format and runtime that reaches all of those.

Two properties make it its own thing:

- **It is a `llama.cpp` format, not a Hugging Face format.** vLLM cannot load GGUF. TGI cannot load GGUF. `transformers` cannot load GGUF (natively — a bridge exists, with caveats). If your serving target is a datacenter GPU behind vLLM / TGI, this chapter is background reading. If your target is CPU / edge / desktop, this chapter is the deployment.
- **The quantization is applied at conversion time, not at inference.** GPTQ / AWQ produce an INT4 file with per-group scales; GGUF Q4_K_M produces a *different* INT4 file with a different layout that only `llama.cpp` and its ecosystem consume. You cannot copy a GPTQ file into `llama.cpp` and vice versa.

## What GGUF actually is

GGUF ("GGML Universal File") is a binary container format designed for `llama.cpp` inference. It packs:

- Model weights, quantized to one of ~20 supported schemes (see the ladder below).
- Tokenizer state (vocab, merges, special tokens, chat template).
- Model metadata (architecture, layer counts, hyperparameters).
- A KV metadata block for arbitrary extra fields.

Everything the runtime needs is in one file. No separate `config.json` / `tokenizer.json` / `special_tokens_map.json` — that is a `transformers` convention that GGUF collapses into one blob.

Successor to the older GGML format (from which the "G" survives). Every current `llama.cpp` release uses GGUF as its primary format.

## The k-quant ladder

The GGUF quantization suite is a family, not a single scheme. Understanding the family — and specifically the `_K_M` / `_K_S` variants — is the difference between choosing well and choosing "whatever a blog post said."

The ladder, high to low precision:

| Name | Bits per weight (approx.) | Notes |
|---|---|---|
| `F32` | 32 | Full precision. Rarely useful — memory-heavy, no speed win. |
| `F16` / `BF16` | 16 | Baseline. Loads as-is, no quantization. |
| `Q8_0` | 8.5 | INT8 with 32-weight blocks, one fp16 scale per block. Near-lossless. Reference for "quantization loss ≈ zero." |
| `Q6_K` | ~6.5 | The "K" family: mixed-precision within-file layout, per-block super-scaling. Very small quality delta from `Q8_0`. |
| `Q5_K_M` | ~5.5 | Slightly lossy from `Q6_K`, meaningfully smaller. Common workhorse for higher-quality edge deployments. |
| `Q5_K_S` | ~5.5 | "Small" variant: less internal mixed-precision, smaller file, slightly worse quality than `_M`. |
| `Q4_K_M` | ~4.8 | The **canonical edge default.** Small, fast, good quality on 7B+ models. |
| `Q4_K_S` | ~4.6 | Smaller than `_M`, worse quality. |
| `Q4_0` / `Q4_1` | ~4.5 | Older, uniform 4-bit schemes. Deprecated; use `Q4_K_M`. |
| `Q3_K_L` / `Q3_K_M` / `Q3_K_S` | ~3.5 | 3-bit K-family. Quality is workload-dependent — good for large bases (30B+), worse for small (7B). |
| `Q2_K` | ~2.6 | 2-bit. Last resort for extremely tight memory. Quality is fragile; test carefully. |
| `IQ*` (`IQ4_XS`, `IQ3_M`, `IQ2_XXS`, ...) | 2–4 | "importance-aware" quantization; uses per-tensor calibration ("imatrix") data. Better quality than K-quants at the same bit width when calibration is done well. |

The "M" and "S" suffixes on K-quants indicate mixed-precision layouts: some layers (attention output projection, feed-forward down-projection) are kept at higher precision than others because they matter more. The M ("medium") variant keeps more of these; the S ("small") variant fewer.

The empirical shape: **on a 7B model, Q4_K_M is the "one number to pick if you don't know."** It runs on modest CPUs at a few tokens/sec, on M-series Macs at 20–40 tokens/sec, on a laptop dGPU at 30–60 tokens/sec, and quality-degrades from BF16 by roughly what a well-tuned GPTQ / AWQ would (typically 1–3 pp on academic evals, more on math and code).

<!-- needs-research: exact quality deltas per k-quant depend on model and workload; the llama.cpp README and community perplexity tables are the primary references. Do not assume the numbers here beyond order-of-magnitude — measure on your own eval. -->

## Conversion from a Hugging Face checkpoint

The `convert_hf_to_gguf.py` script (in the `llama.cpp` repo) is the reference conversion tool. Two-stage pipeline:

**Stage 1 — Convert BF16 / F16 checkpoint to a GGUF at full precision.**

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
python -m pip install -r requirements/requirements-convert_hf_to_gguf.txt

python convert_hf_to_gguf.py \
  /models/support-v3-merged \
  --outfile /gguf/support-v3-f16.gguf \
  --outtype f16
```

The output is a plain F16 GGUF — same weights as the source, in `llama.cpp`'s container format. This is the "known-good, high-fidelity" version and the reference you compare quantized versions against for equivalence (see below).

**Stage 2 — Quantize the F16 GGUF to a lower-bit variant.**

Build `llama.cpp` (once):

```bash
cmake -B build
cmake --build build --config Release
```

Then quantize:

```bash
./build/bin/llama-quantize \
  /gguf/support-v3-f16.gguf \
  /gguf/support-v3-q4_k_m.gguf \
  q4_k_m
```

Repeat for each target quantization level you want to ship — typically Q4_K_M as the default, Q5_K_M or Q6_K if quality is critical, Q8_0 as a "reference near-BF16" for testing.

### Prerequisites for the source checkpoint

- The Hugging Face checkpoint must be a `transformers`-loadable format (safetensors preferred).
- Adapter must be merged first if it was trained as LoRA (mod-104 chapter 3, `merge_and_unload()`). GGUF conversion does not consume `.peft` adapters directly; the input must be a plain merged checkpoint.
- The tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`) must be present. The converter reads them and packs the tokenizer into the GGUF. Missing tokenizer state → the GGUF loads but tokenization is wrong at inference.
- The model architecture must be supported by `llama.cpp`. Coverage is broad (LLaMA, Mistral, Qwen, Gemma, Phi, DeepSeek, and many more) but not universal. Check the current `convert_hf_to_gguf.py` supported-architecture list.

### The imatrix step (for `IQ*` quants)

If you are going below Q3 or want to use the `IQ` importance-aware variants, generate a calibration `imatrix` first:

```bash
./build/bin/llama-imatrix \
  -m /gguf/support-v3-f16.gguf \
  -f /calibration/calibration_data.txt \
  -o /gguf/support-v3.imatrix
```

`calibration_data.txt` is a text file — typically a concatenation of a few hundred representative prompts / completions from your training or production distribution. Same calibration-data principles as GPTQ / AWQ (chapter 2): match the distribution of what the model will see.

Then quantize with the imatrix:

```bash
./build/bin/llama-quantize \
  --imatrix /gguf/support-v3.imatrix \
  /gguf/support-v3-f16.gguf \
  /gguf/support-v3-iq4_xs.gguf \
  iq4_xs
```

At Q4_K_M and above, the imatrix step is optional and gives a small quality bump. Below Q3, it is essentially required to keep the model coherent.

## Running with `llama.cpp`

Two frontends ship with `llama.cpp`:

### `llama-cli` — interactive CLI

```bash
./build/bin/llama-cli \
  -m /gguf/support-v3-q4_k_m.gguf \
  -p "Summarize: Fine-tuning is ..." \
  -n 128 \
  --temp 0.7 \
  --top-p 0.9
```

Good for smoke-testing conversion, iterating on prompts locally, and reproducing bugs on a specific input.

Chat mode (applies the tokenizer's chat template):

```bash
./build/bin/llama-cli \
  -m /gguf/support-v3-q4_k_m.gguf \
  --chat-template llama3 \
  -cnv
```

The `--chat-template` flag names a template baked into `llama.cpp` (LLaMA-3, ChatML, Mistral, Gemma, etc.) or falls back to what the GGUF's tokenizer metadata specifies. Wrong template = wrong behavior; verify on a known prompt.

### `llama-server` — HTTP server (OpenAI-compatible)

```bash
./build/bin/llama-server \
  -m /gguf/support-v3-q4_k_m.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 4096 \
  -np 4 \
  --chat-template llama3
```

Flags of note:

- `-c` — context size (KV-cache reservation). Same trade-off as vLLM's `--max-model-len`.
- `-np` — number of parallel sequence slots (concurrent requests). `llama.cpp`'s parallelism model is simpler than vLLM's; each slot is a decoding stream.
- `-ngl` — number of layers to offload to GPU (if you built with GPU support). `-1` offloads all layers; higher = faster and more VRAM.
- `--host` / `--port` — the HTTP surface.

The OpenAI-compatible `/v1/chat/completions` and `/v1/completions` endpoints work identically to vLLM's / TGI's. Any OpenAI client swaps by changing the base URL.

### `llama.cpp` on hardware backends

`llama.cpp` compiles to run on:

- **CPU** (any modern x86-64, ARM64, WebAssembly). Uses SIMD, AMX where available.
- **CUDA** (Nvidia GPUs). Similar throughput to `bitsandbytes` on 4-bit paths; less than vLLM's Marlin kernels on high-end hardware.
- **Metal** (M-series Macs). The reason `llama.cpp` is the default for Mac inference.
- **Vulkan** (cross-vendor GPU).
- **ROCm / HIP** (AMD GPUs).
- **CANN** (Huawei Ascend).
- **SYCL** (Intel GPUs).

Same GGUF file runs on all of them. The `-ngl` flag controls how many layers offload to accelerator.

## Ollama, LM Studio, and the ecosystem

GGUF + `llama.cpp` has spawned a family of higher-level tools:

- **Ollama** — a local CLI/daemon that wraps `llama.cpp`, handles model management, and exposes a REST API. `ollama run <model>` pulls a GGUF from the Ollama registry and runs it. For a local dev workflow, `ollama` is often the fastest path from "I have a GGUF" to "I have an endpoint."
- **LM Studio** — a desktop GUI on top of `llama.cpp` with model browser, chat UI, and local server mode.
- **KoboldCpp / Text Generation WebUI (`oobabooga`)** — hobbyist / power-user frontends.
- **Mobile runtimes** (MLC LLM, llama.cpp Android / iOS demos, ExecuTorch) — variants targeting phones and embedded devices; often consume GGUF or a derived format.

For a fine-tuning engineer shipping to non-datacenter targets, the two questions are: which runtime does the customer use, and what quantization level does that runtime prefer? Ollama's default catalog leans on Q4_K_M for most model sizes and that is a fine default for laptop deployments.

## Equivalence: does the GGUF match the source?

You quantized. You produced `support-v3-q4_k_m.gguf`. Before you ship it, prove it matches the source weights well enough. Three checks that scale with confidence needed:

**Check 1 — Load-and-generate parity (cheap).**

Run the same fixed prompt through the source `transformers` model (BF16) and the GGUF (any quantization) at *greedy decoding* (temperature 0, no sampling). Compare the generations.

```python
# transformers side
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("/models/support-v3-merged")
m = AutoModelForCausalLM.from_pretrained("/models/support-v3-merged", torch_dtype=torch.bfloat16, device_map="auto")
inputs = tok("The capital of France is", return_tensors="pt").to(m.device)
print(tok.decode(m.generate(**inputs, max_new_tokens=20, do_sample=False)[0]))
```

```bash
# llama.cpp side
./build/bin/llama-cli -m /gguf/support-v3-q4_k_m.gguf -p "The capital of France is" -n 20 --temp 0
```

At a well-behaved 4-bit quantization, the two greedy outputs should agree for the first several tokens and drift only slightly after that. Complete divergence from token 1 = something is broken in the conversion (tokenizer mismatch, wrong chat template, quantization catastrophically corrupted).

**Check 2 — Perplexity on a held-out slice (medium).**

`llama.cpp` ships a `llama-perplexity` tool that computes perplexity on a text corpus using the GGUF. Compare against the same perplexity computed on the source BF16 model. Expect a ~1–5% relative increase for Q4_K_M; more than ~10% is a problem.

```bash
./build/bin/llama-perplexity \
  -m /gguf/support-v3-q4_k_m.gguf \
  -f /data/held_out.txt \
  --chunks 32
```

Same recipe on the F16 GGUF (the reference) and any quantized variant. If Q4_K_M's perplexity is 5–10% above F16, that is the quantization delta on this text distribution.

**Check 3 — The release-gate suite (expensive but definitive).**

Use `lm-evaluation-harness` in its `local-chat-completions` backend pointed at `llama-server`. Run the mod-108 fast-lane subset (MMLU, ARC-Challenge, GSM8K, IFEval) and compare against the source model's numbers. This is the shippable equivalence check.

```bash
# start llama-server serving the GGUF, then:
lm_eval --model local-chat-completions \
  --tasks mmlu,arc_challenge,gsm8k,ifeval \
  --model_args base_url=http://localhost:8080/v1,model=support-v3 \
  --seed 42 \
  --output_path results/gguf-q4-k-m/
```

Expect deltas inside your release-gate tolerances (chapter 1's table). If not, either bump to a higher-precision quant (Q5_K_M, Q6_K) or investigate why (chat template? tokenizer? imatrix would have helped?).

## Fine-tuned + LoRA + GGUF

`llama.cpp` supports GGUF LoRA adapters — but only as a separate file loaded on top of a base GGUF, and support quality has varied across model families. Practical recipe for a fine-tuned model:

- **Preferred:** merge the LoRA into the base with `peft.merge_and_unload()` first, then convert the merged BF16 model to GGUF. One artifact, no LoRA-at-inference concerns.
- **Alternative:** convert the base to GGUF and the adapter to a GGUF LoRA (`convert_lora_to_gguf.py` in the `llama.cpp` repo), load at inference with `--lora`. Support is real but less mature than the merged path. If you take this path, do check 1 above with the adapter attached to make sure the LoRA loaded and applied.

For most edge deployments, the merged path is the answer.

## When Q4_K_M is not enough

Two common failure modes:

**1. Code / math regression at Q4_K_M.** GSM8K and HumanEval-style tasks are sensitive to numerical precision. If the release gate shows a 5+ pp regression on these while other tasks are fine, bump to Q5_K_M or Q6_K. Alternatively, generate an imatrix from code / math prompts and re-quantize.

**2. Small-model quality collapse.** Sub-3B models can degrade sharply below Q5. If you are shipping a Phi-2 or a Qwen-1.5B, prefer Q5_K_M or Q6_K over Q4_K_M as the default.

The correspondence rule for choosing across the ladder:

- Big base (13B+), general tasks: Q4_K_M is safe.
- Big base, code / math heavy: Q5_K_M.
- Medium base (7B), general: Q4_K_M is safe.
- Medium base, code / math heavy: Q5_K_M.
- Small base (<3B): Q5_K_M or Q6_K.
- "Must be near-BF16": Q8_0 is the closest quantized reference; the file is roughly half the size of F16.

## Deployment surface — CPU, laptop, edge

For each target, the practical shape:

### Datacenter CPU (`llama-server` on a k8s CPU pod)

Common when the workload is small (a few requests/second) and the team wants to avoid GPU cost or complexity. Q4_K_M on a modern x86 or Graviton CPU serves 5–20 tokens/sec for a 7B; parallel decoding via `-np` scales linearly to the point CPU cores are saturated.

Not competitive with GPU serving on latency or throughput. Right when the alternative is "no serving at all because GPUs are unavailable."

### M-series Mac (Metal backend)

The best per-dollar developer inference environment as of 2026. A 7B Q4_K_M runs at 30–60 tokens/sec on an M2 / M3 Pro; a 13B at 15–30 tokens/sec on an M2 / M3 Max. `llama.cpp` on Metal or Ollama on top of it are the standard tools.

Great for local model iteration during fine-tuning cycles. Not typically a production target.

### Consumer / gaming GPU (CUDA / Vulkan / ROCm)

A 24 GB card (RTX 3090 / 4090) runs a 7B at Q4_K_M at 70–150 tokens/sec via `llama.cpp`'s CUDA backend — comparable throughput to `bitsandbytes` on the same card, less than vLLM's Marlin path but simpler to deploy.

### Mobile / embedded (Android, iOS, edge devices)

Requires a smaller model (1B–3B range for phone-class SoCs) and typically Q4_0 or Q4_K_S for the memory budget. Latency is workload-dependent; 5–15 tokens/sec on modern phones is realistic. `llama.cpp` has Android / iOS build targets, and downstream projects (MLC LLM, ExecuTorch, MediaPipe LLM) offer more optimized paths.

Chapter 7's cost / latency framework applies here too, with the twist that "cost" is often battery / thermal budget rather than dollars per token.

## The failure modes

- **Tokenizer not packed into the GGUF.** Converter warned but you skipped it. Inference produces gibberish tokens. Fix: rerun the conversion with the tokenizer files present and no warnings.
- **Wrong chat template at inference.** The GGUF has one template baked in; `llama-cli --chat-template` names another. Behavior at inference is subtly wrong. Fix: verify the chat template with a known prompt; leave `--chat-template` unset unless you have a reason.
- **Adapter not merged before conversion.** The GGUF is of the base only. Fix: `merge_and_unload()` first, then convert.
- **`convert_hf_to_gguf.py` errors on an unsupported architecture.** Some custom modeling code is not in the converter's mapping. Fix: check the current supported-architecture list; if unsupported, either wait for support or (for research) hand-edit the converter's model registry.
- **Perplexity way above F16 baseline.** Bad calibration for imatrix quants, or the quant target is too aggressive for the model. Fix: bump to a higher-precision quant, or generate a proper imatrix.

## Exercise pointer

`exercises/exercise-04-gguf-export-and-edge-inference.md` has you convert the exercise-01 merged checkpoint to two quantization levels (Q4_K_M and one other of your choice — Q5_K_M, Q6_K, or an IQ variant), run all three equivalence checks against the source weights, benchmark tokens/sec on the CPU or laptop hardware available to you, and produce an "edge deployment brief" that a mobile or on-prem product team could execute.

## Summary

- GGUF is the `llama.cpp` container format for CPU / edge / desktop inference. Not consumable by vLLM or TGI. Choose the runtime target first; if it is `llama.cpp` / Ollama / a downstream mobile runtime, this is the format.
- The k-quant ladder (Q8_0 → Q6_K → Q5_K_M → Q4_K_M → Q3_K_M → Q2_K, plus the `IQ*` importance-aware variants) is a family, not a single choice. Q4_K_M is the safe default for 7B+ general instruct models; bump to Q5_K_M / Q6_K when code / math or small-model quality matters.
- Conversion is a two-stage pipeline: `convert_hf_to_gguf.py` produces an F16 GGUF from the merged Hugging Face checkpoint, then `llama-quantize` reduces it to the target k-quant. LoRA adapters must be merged before conversion for the safe path.
- Equivalence is proven by three checks — greedy generation parity (smoke), perplexity delta on a held-out slice (numeric), and the mod-108 release-gate fast lanes via `local-chat-completions` (definitive). Do not ship without at least the first two.
- Deployment targets — CPU pod, M-series Mac, consumer GPU, mobile / embedded — all consume the same GGUF file with different runtime backends. Throughput, latency, and cost all differ; chapter 7's framework applies but with battery / thermal budget as an additional axis on mobile.
- The most common failure modes are missing tokenizer packing, wrong chat template at inference, and unmerged adapters. Each is caught at check 1 (load-and-generate) in five minutes.
