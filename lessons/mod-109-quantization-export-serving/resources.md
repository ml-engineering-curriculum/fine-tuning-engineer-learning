# Resources for mod-109-quantization-export-serving

Curated primary references. Prefer these — the paper of record, the official docs, the current tool README — over blog-post summaries when you go deep. Quantization tooling, kernel support, and runtime flags change quickly between releases; verify against the version-pinned docs for your install before designing around any specific flag or feature.

## Chapter 1 — The quantization landscape for fine-tuned models

### Framing / meta-references

- **Dettmers & Zettlemoyer, 2023** — "The case for 4-bit precision: k-bit Inference Scaling Laws." The paper behind the "knee at 4 bits" mental model. https://arxiv.org/abs/2212.09720
- **Gholami et al., 2021** — "A Survey of Quantization Methods for Efficient Neural Network Inference." Broad taxonomy of PTQ vs. QAT, weight-only vs. weight-and-activation, symmetric vs. asymmetric. https://arxiv.org/abs/2103.13630
- **Nagel et al., 2021** — "A White Paper on Neural Network Quantization." Qualcomm's applied reference on rounding schemes, per-channel vs. per-tensor, calibration, and the QAT / PTQ split. https://arxiv.org/abs/2106.08295

### Numeric formats

- **IEEE 754** — the standard behind FP32 / FP16. https://ieeexplore.ieee.org/document/8766229
- **Google Brain BF16 blog** — the original write-up of the bfloat16 format on TPUs. https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
- **Micikevicius et al., 2022** — "FP8 Formats for Deep Learning." The Nvidia / Arm / Intel joint spec introducing E4M3 and E5M2. https://arxiv.org/abs/2209.05433
- **Nvidia FP8 Primer (Transformer Engine docs)** — the applied reference for FP8 on Hopper / Blackwell. https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html

## Chapter 2 — GPTQ, AWQ, and bitsandbytes

### Core papers

- **Frantar, Ashkboos, Hoefler, Alistarh, 2022** — "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." https://arxiv.org/abs/2210.17323
- **Lin, Tang, Tang, Yang, Cai, Zhu, Han, 2023** — "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." https://arxiv.org/abs/2306.00978
- **Dettmers, Lewis, Belkada, Zettlemoyer, 2022** — "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." https://arxiv.org/abs/2208.07339
- **Dettmers, Pagnoni, Holtzman, Zettlemoyer, 2023** — "QLoRA: Efficient Finetuning of Quantized LLMs." The paper that introduced NF4 and double-quantization; the reference used by `bitsandbytes` at load time. https://arxiv.org/abs/2305.14314
- **Xiao, Lin, Seznec, Wu, Demouth, Han, 2022** — "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." The reference W8A8 method. https://arxiv.org/abs/2211.10438
- **Frantar & Alistarh, 2023** — "Marlin: An Efficient Mixed-Precision Matrix Multiplication Kernel." The kernel behind vLLM's fast GPTQ / AWQ path on Ampere / Hopper. https://arxiv.org/abs/2408.11743

### Official tooling

- `AutoGPTQ` — https://github.com/AutoGPTQ/AutoGPTQ
- `gptqmodel` (the ecosystem's current maintenance fork of `AutoGPTQ`) — https://github.com/ModelCloud/GPTQModel
- `AutoAWQ` — https://github.com/casper-hansen/AutoAWQ
- `bitsandbytes` — https://github.com/bitsandbytes-foundation/bitsandbytes
- Hugging Face `optimum` (unified interface to GPTQ / AWQ / bitsandbytes / ONNX / TensorRT-LLM) — https://huggingface.co/docs/optimum/
- Hugging Face `transformers` quantization guide — https://huggingface.co/docs/transformers/main/en/quantization/overview
- Neural Magic `compressed-tensors` (unified format for W8A8 / INT4 / FP8) — https://github.com/neuralmagic/compressed-tensors
- Nvidia TensorRT-LLM quantization docs — https://nvidia.github.io/TensorRT-LLM/reference/precision.html
- Nvidia `Model Optimizer` (FP8 / INT4 AWQ calibration on Nvidia hardware) — https://github.com/NVIDIA/TensorRT-Model-Optimizer

### Calibration-data references

- **Raffel et al., 2019** — "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5 / C4). The C4 corpus that most quantization tooling defaults to for calibration. https://arxiv.org/abs/1910.10683
- Hugging Face `datasets` — https://huggingface.co/docs/datasets/
- C4 dataset card (`allenai/c4`) — https://huggingface.co/datasets/allenai/c4

## Chapter 3 — Merge vs. multi-LoRA serving

### Core papers

- **Hu et al., 2021** — "LoRA: Low-Rank Adaptation of Large Language Models." The base LoRA paper the merge / attach discussion assumes. https://arxiv.org/abs/2106.09685
- **Sheng et al., 2023** — "S-LoRA: Serving Thousands of Concurrent LoRA Adapters." The paper of record for LoRA-aware serving at scale; vLLM's multi-LoRA path follows this design. https://arxiv.org/abs/2311.03285
- **Chen et al., 2023** — "Punica: Multi-Tenant LoRA Serving." An adjacent multi-tenant LoRA serving design worth reading alongside S-LoRA. https://arxiv.org/abs/2310.18547
- **Zhao et al., 2024** — "LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4, A Technical Report." Predibase's report on LoRAX and how many adapters one base can practically host. https://arxiv.org/abs/2405.00732

### Official tooling

- Hugging Face `peft` — https://github.com/huggingface/peft
- `peft.merge_and_unload` reference — https://huggingface.co/docs/peft/main/en/package_reference/peft_model#peft.PeftModel.merge_and_unload
- vLLM multi-LoRA guide — https://docs.vllm.ai/en/latest/features/lora.html
- TGI LoRA adapters guide — https://huggingface.co/docs/text-generation-inference/en/conceptual/lora
- LoRAX (Predibase) — https://github.com/predibase/lorax

## Chapter 4 — Deploying with vLLM

### Core papers

- **Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica, 2023** — "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM). https://arxiv.org/abs/2309.06180
- **Yu et al., 2022** — "Orca: A Distributed Serving System for Transformer-Based Generative Models." The paper of record for continuous batching; the design vLLM operationalizes. https://www.usenix.org/conference/osdi22/presentation/yu
- **Dao et al., 2022** — "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." The kernel modern serving stacks build on. https://arxiv.org/abs/2205.14135
- **Dao, 2023** — "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." https://arxiv.org/abs/2307.08691
- **Shah et al., 2024** — "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." The Hopper-era kernel used on H100 by modern vLLM. https://arxiv.org/abs/2407.08608

### Official docs and repositories

- vLLM main docs — https://docs.vllm.ai/
- vLLM repository — https://github.com/vllm-project/vllm
- vLLM engine arguments reference — https://docs.vllm.ai/en/latest/serving/engine_args.html
- vLLM OpenAI-compatible server — https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- vLLM quantization overview — https://docs.vllm.ai/en/latest/features/quantization/index.html
- vLLM structured outputs (`guided_json` / `guided_regex` / `guided_grammar`) — https://docs.vllm.ai/en/latest/features/structured_outputs.html
- vLLM prefix caching — https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html
- vLLM metrics reference — https://docs.vllm.ai/en/latest/serving/metrics.html
- `outlines` (grammar-constrained decoding used by vLLM / TGI) — https://github.com/dottxt-ai/outlines
- `xgrammar` (grammar backend) — https://github.com/mlc-ai/xgrammar

### OpenAI API reference (what the compatible server implements)

- OpenAI Chat Completions API — https://platform.openai.com/docs/api-reference/chat
- OpenAI Completions API — https://platform.openai.com/docs/api-reference/completions
- OpenAI Server-Sent Events streaming — https://platform.openai.com/docs/api-reference/streaming

## Chapter 5 — Deploying with Text Generation Inference (TGI)

### Official docs and repository

- TGI main docs — https://huggingface.co/docs/text-generation-inference/
- TGI repository — https://github.com/huggingface/text-generation-inference
- TGI launcher CLI reference — https://huggingface.co/docs/text-generation-inference/en/reference/launcher
- TGI quantization guide — https://huggingface.co/docs/text-generation-inference/en/conceptual/quantization
- TGI LoRA-adapter guide — https://huggingface.co/docs/text-generation-inference/en/conceptual/lora
- TGI messages API (OpenAI-compatible) — https://huggingface.co/docs/text-generation-inference/en/messages_api
- TGI metrics reference — https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/monitoring
- Hugging Face Inference Endpoints (managed TGI) — https://huggingface.co/docs/inference-endpoints/
- Nvidia EETQ (INT8 kernel referenced by TGI's `--quantize eetq`) — https://github.com/NetEase-FuXi/EETQ

## Chapter 6 — GGUF and edge inference

### Core references

- `llama.cpp` repository — https://github.com/ggml-org/llama.cpp
- GGUF file-format specification (in-tree) — https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- `convert_hf_to_gguf.py` (conversion tool) — https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py
- `llama-quantize` reference (in the `llama.cpp` repo `tools/quantize` docs) — https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize
- `llama-perplexity` reference — https://github.com/ggml-org/llama.cpp/tree/master/tools/perplexity
- `llama-server` reference (OpenAI-compatible HTTP server) — https://github.com/ggml-org/llama.cpp/tree/master/tools/server
- k-quantization discussion (llama.cpp PR #1684 and issues) — https://github.com/ggml-org/llama.cpp/pull/1684
- Importance-matrix ("imatrix") discussion (llama.cpp PR #4861 and follow-ups) — https://github.com/ggml-org/llama.cpp/pull/4861

### Downstream runtimes / higher-level tools

- Ollama — https://ollama.com/ and the repo https://github.com/ollama/ollama
- Ollama Modelfile reference — https://github.com/ollama/ollama/blob/main/docs/modelfile.md
- LM Studio — https://lmstudio.ai/
- KoboldCpp — https://github.com/LostRuins/koboldcpp
- text-generation-webui (oobabooga) — https://github.com/oobabooga/text-generation-webui
- MLC LLM (mobile / cross-platform runtime) — https://llm.mlc.ai/
- ExecuTorch (PyTorch on-device runtime) — https://pytorch.org/executorch/
- MediaPipe LLM Inference (Google on-device runtime) — https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference

## Chapter 7 — Latency, throughput, cost, and the serving runbook

### Latency / throughput / continuous batching

- **Agrawal et al., 2024** — "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve." Chunked-prefill and the TTFT / ITL trade-off. https://arxiv.org/abs/2403.02310
- **Zhong et al., 2024** — "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving." Split-role serving; useful vocabulary for prefill / decode. https://arxiv.org/abs/2401.09670
- **Patel et al., 2023** — "Splitwise: Efficient Generative LLM Inference Using Phase Splitting." Same theme as DistServe. https://arxiv.org/abs/2311.18677
- **Ainslie et al., 2023** — "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." The GQA kv-cache-shrinking discussion this chapter references. https://arxiv.org/abs/2305.13245

### Load-testing tools

- vLLM benchmark suite (`vllm bench serve`) — https://docs.vllm.ai/en/latest/getting_started/quickstart.html and https://github.com/vllm-project/vllm/tree/main/benchmarks
- Anyscale `llmperf` — https://github.com/ray-project/llmperf
- `k6` (general-purpose HTTP load test) — https://k6.io/docs/
- `locust` — https://docs.locust.io/
- Grafana `k6` LLM examples — https://grafana.com/docs/k6/latest/examples/

### Cost / capacity references

- Nvidia GPU product briefs (H100, H200, L40S, L4, B100/B200) — https://www.nvidia.com/en-us/data-center/products/
- AWS EC2 accelerated instance pricing — https://aws.amazon.com/ec2/instance-types/
- Google Cloud A3 / G2 pricing — https://cloud.google.com/compute/gpus-pricing
- Azure ND-series pricing — https://azure.microsoft.com/en-us/pricing/details/machines/gpu/
- OpenAI pricing — https://openai.com/api/pricing/
- Anthropic pricing — https://www.anthropic.com/pricing
- Google Vertex AI generative-model pricing — https://cloud.google.com/vertex-ai/generative-ai/pricing

<!-- needs-research: verify current per-GPU-hour and per-million-token pricing before quoting any specific figure in a runbook — cloud and API prices move quarterly. -->

### SRE / runbook practice

- Google SRE Book, Chapter 5 (Eliminating Toil) and Chapter 8 (Release Engineering) — https://sre.google/sre-book/table-of-contents/
- Google SRE Workbook, Chapter 8 (On-Call) — https://sre.google/workbook/on-call/
- Site Reliability Workbook, Chapter 2 (Implementing SLOs) — https://sre.google/workbook/implementing-slos/
- Nygard, "Release It!" (2nd ed.) — patterns for stability, rollback, and the on-call handoff. https://pragprog.com/titles/mnee2/release-it-second-edition/
- OpenTelemetry semantic conventions for GenAI — https://opentelemetry.io/docs/specs/semconv/gen-ai/

## Broader background referenced across chapters

### Transformer / attention foundations

- **Vaswani et al., 2017** — "Attention Is All You Need." https://arxiv.org/abs/1706.03762
- **Shazeer, 2019** — "Fast Transformer Decoding: One Write-Head is All You Need" (MQA). https://arxiv.org/abs/1911.02150

### Model families this module references

- Llama 3 model card — https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md
- Llama 3 technical report — https://arxiv.org/abs/2407.21783
- Mistral 7B paper — https://arxiv.org/abs/2310.06825
- Mixtral of Experts paper — https://arxiv.org/abs/2401.04088
- Qwen 2.5 technical report — https://arxiv.org/abs/2412.15115
- Gemma 2 technical report — https://arxiv.org/abs/2408.00118
- Phi-3 technical report — https://arxiv.org/abs/2404.14219
- DeepSeek-V3 technical report — https://arxiv.org/abs/2412.19437

## Suggested reading order for this module

1. Skim Dettmers & Zettlemoyer (2023) "The case for 4-bit precision" and the vLLM PagedAttention paper's introduction. Together they set the "what and why" mental model this module builds on.
2. Read chapter 1, then chapter 2 alongside the GPTQ and AWQ paper abstracts and the `bitsandbytes` README. Run exercise 01 immediately after — the tool ergonomics are the fastest route to intuition.
3. Read chapter 3 alongside the S-LoRA paper and vLLM's multi-LoRA doc. Exercise 02 is the payoff.
4. Read chapter 4 alongside the vLLM engine-arguments doc and the OpenAI Chat Completions reference. Read chapter 5 alongside the TGI launcher reference for the head-to-head.
5. Read chapter 6 alongside the `llama.cpp` README (particularly the quantization ladder table) and the GGUF spec. Exercise 04 exports and equivalence-checks against the source.
6. Read chapter 7 last of the core chapters and pair it with the Sarathi-Serve paper and the SRE workbook's SLO chapter. Exercise 03 (TGI + vLLM head-to-head load test) exercises the load-test pattern this chapter formalizes.

<!-- needs-research: on the next authoring cycle, add version pins for the `vllm`, `text-generation-inference`, `llama.cpp`, `AutoGPTQ` / `gptqmodel`, `AutoAWQ`, `bitsandbytes`, `optimum`, and `transformers` links so future readers land on the version this module was authored against. Managed-API pricing pages (OpenAI, Anthropic, Vertex) drift and should be re-verified before being cited in any runbook cost table. -->
