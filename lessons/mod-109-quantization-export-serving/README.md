# mod-109-quantization-export-serving: Quantization, Export, and Serving Adapted Models

**Estimated effort:** 12 hours
**Position in the track:** ninth module. Consumes the checkpoints from mod-103 (SFT), mod-104 (LoRA / QLoRA), mod-105 (DPO / KTO / ORPO), mod-106 (safety), and mod-107 (distributed) together with the release-gate suite from mod-108. Turns those training-time artifacts into serving-time artifacts — a quantized weight file, an adapter registry, a vLLM or TGI process, or a GGUF for the edge — and hands the result to mod-111 (post-training systems design), which wraps them in a platform.

The training run is over. The eval gate said the model is good enough. Now something has to answer HTTP requests inside your latency budget, at your throughput target, on the hardware you actually have. Everything in this module is about the gap between "the checkpoint from mod-107" and "a running endpoint the product can call" — the gap where post-training quantization, adapter merging, LoRA-aware serving, GGUF export, and the serving runbook live.

The through-line: **the serving artifact is not the training artifact.** BF16 weights on disk are the raw material; the served artifact is what you built out of them for a specific runtime, a specific quality tolerance, a specific latency SLO, and a specific hardware budget. Every decision in this module — GPTQ vs. AWQ vs. bitsandbytes, merge vs. multi-LoRA, vLLM vs. TGI vs. llama.cpp, GGUF at Q4_K_M vs. Q5_K_M — trades one axis against another. This module is about making those trades on purpose, backed by numbers, and writing them into a runbook that the next on-call engineer can execute.

## Learning objectives

- Apply post-training quantization (GPTQ, AWQ, bitsandbytes 8-bit / 4-bit) and measure the quality / latency / VRAM impact of each
- Choose between adapter-merging and LoRA-aware serving (vLLM multi-LoRA) per use case
- Deploy a fine-tuned model behind vLLM and Text Generation Inference (TGI) with a structured serving config
- Export to GGUF for llama.cpp / CPU / edge inference and verify equivalence against the source weights
- Reason about latency / throughput / cost trade-offs and write a serving runbook

## Lecture chapters

1. [`01-quantization-landscape.md`](01-quantization-landscape.md) — The Quantization Landscape for Fine-Tuned Models. Why we quantize (VRAM, cost, latency), the difference between quantization-aware training and post-training quantization, the numeric formats in play (INT8, INT4, NF4, FP8), what "quality loss" means and how to measure it against mod-108's suite, and the map of methods this module actually covers.
2. [`02-gptq-awq-bitsandbytes.md`](02-gptq-awq-bitsandbytes.md) — GPTQ, AWQ, and bitsandbytes 8-bit / 4-bit. How each method works (calibration data, activation-aware weighting, per-channel scales, NF4), the concrete recipes for each in the current tooling (`auto-gptq` / `gptqmodel`, `AutoAWQ`, `bitsandbytes` via `transformers`), what runtimes accept which format, and the decision matrix over base size × hardware × latency SLO.
3. [`03-merge-vs-multi-lora-serving.md`](03-merge-vs-multi-lora-serving.md) — Merge vs. Multi-LoRA Serving. Extends mod-104 chapter 3 into a serving-side decision: when to fold the adapter into the base for a single-tenant deployment, when to keep N adapters hot on one base (vLLM multi-LoRA, TGI, LoRAX), the memory / latency / cold-start math, and the routing patterns that connect a request to an adapter.
4. [`04-vllm-deployment.md`](04-vllm-deployment.md) — Deploying with vLLM. Engine args that matter (`--tensor-parallel-size`, `--max-model-len`, `--gpu-memory-utilization`, `--kv-cache-dtype`, `--quantization`, `--enable-lora`), the OpenAI-compatible server, loading a GPTQ / AWQ / bitsandbytes model, wiring multi-LoRA, and the smoke tests that catch a bad deploy before it ships.
5. [`05-tgi-deployment.md`](05-tgi-deployment.md) — Deploying with Text Generation Inference (TGI). Launcher flags, sharding across GPUs, quantized weight loading (GPTQ / AWQ / bitsandbytes / EETQ / FP8), the LoRA-adapter path, the messages / generate / stream endpoints, and the operational differences vs. vLLM that decide which one to reach for.
6. [`06-gguf-and-edge-inference.md`](06-gguf-and-edge-inference.md) — GGUF Export and Edge Inference. What GGUF is, how to convert a Hugging Face checkpoint (`convert_hf_to_gguf.py`), the quantization ladder (Q8_0 → Q6_K → Q5_K_M → Q4_K_M → Q3_K_M → Q2_K), running with `llama.cpp` (`llama-cli`, `llama-server`) and Ollama, and the equivalence checks that prove the GGUF matches the source weights.
7. [`07-serving-runbook-and-cost.md`](07-serving-runbook-and-cost.md) — Latency, Throughput, Cost, and the Serving Runbook. TTFT vs. ITL vs. end-to-end latency, throughput under continuous batching, the KV-cache memory model, single-GPU cost per million tokens, load-testing patterns, the SLO / capacity table, and the runbook template that hands off to mod-111.

## Exercises

Each exercise is a hands-on companion to one or more chapters. They live in [`exercises/`](exercises/).

- [`exercise-01-gptq-awq-bnb-comparison.md`](exercises/exercise-01-gptq-awq-bnb-comparison.md) — quantize the same fine-tuned checkpoint with GPTQ 4-bit, AWQ 4-bit, and bitsandbytes 4-bit NF4, run mod-108's fast-lane eval on each variant, and produce a comparison matrix (quality delta, VRAM, latency, disk) with a defended recommendation.
- [`exercise-02-vllm-multi-lora-vs-merged-deployment.md`](exercises/exercise-02-vllm-multi-lora-vs-merged-deployment.md) — stand up vLLM in two configurations (one merged single-tenant, one multi-LoRA with 3+ adapters), load-test each with a shared workload, and defend the merge / multi-LoRA decision for a specified serving scenario.
- [`exercise-03-tgi-deployment-and-load-test.md`](exercises/exercise-03-tgi-deployment-and-load-test.md) — deploy the same quantized checkpoint behind TGI, exercise the messages / stream / generate endpoints, run a controlled load test against vLLM as a baseline, and write the operational comparison.
- [`exercise-04-gguf-export-and-edge-inference.md`](exercises/exercise-04-gguf-export-and-edge-inference.md) — convert the checkpoint to GGUF at two quantization levels, verify equivalence against the source weights on a fixed prompt set, benchmark tokens/sec on CPU / edge hardware, and produce an edge deployment brief.

## Labs and quizzes

Long-form labs land in [`labs/`](labs/) and the knowledge-check quiz in [`quizzes/`](quizzes/) on subsequent authoring cycles.

## Resources

See [`resources.md`](resources.md) for primary papers (Frantar et al. — GPTQ; Lin et al. — AWQ; Dettmers et al. — LLM.int8() and QLoRA / NF4; Xiao et al. — SmoothQuant; Sheng et al. — S-LoRA; Kwon et al. — vLLM / PagedAttention), official tooling docs (`auto-gptq` / `gptqmodel`, `AutoAWQ`, `bitsandbytes`, vLLM, TGI, `llama.cpp`, Ollama), and the runbook / SRE references cited across the chapters.

## What comes before

- **mod-104 chapter 3** introduced adapter merging, multi-adapter routing, and the first look at vLLM / TGI / LoRAX. This module extends that from "the API surface" to "the serving decision, the deploy, and the runbook."
- **mod-107 chapter 2** specified the consolidated `FULL_STATE_DICT` format that serving pipelines expect. Every quantization and export step in this module assumes that input format.
- **mod-108 chapter 6** built the release-gate eval suite. Every quantized artifact in this module gets re-run through that gate; the "quality delta" number every table in these chapters reports comes out of it.

## What comes next

- **mod-110 — Managed Fine-Tuning APIs** answers the same product question — "how does a fine-tune reach production?" — with someone else's serving stack. The cost / control trade-offs from chapter 7 are the axis on which "run vLLM myself" vs. "call a managed endpoint" gets decided.
- **mod-111 — Post-Training Systems Design** consumes the runbook template from chapter 7 as one input to the platform. Adapter registry, canary / rollback tooling, quantization-artifact lineage, and the SLO dashboard all live at that altitude; the serving artifact and its runbook are what mod-111 wraps.
