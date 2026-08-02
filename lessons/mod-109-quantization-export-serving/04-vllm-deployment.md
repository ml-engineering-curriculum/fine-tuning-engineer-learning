# Deploying with vLLM

**Estimated read:** 40 minutes.
**You should be able to, by the end:** stand up a vLLM server for a fine-tuned model — merged or with LoRA adapters attached, BF16 or quantized (GPTQ / AWQ / bitsandbytes / FP8) — using engine args you can defend, verify the deploy via the OpenAI-compatible endpoint, and reason about the flags that decide latency (`--kv-cache-dtype`, `--max-model-len`, `--gpu-memory-utilization`, `--tensor-parallel-size`) and throughput (`--max-num-seqs`, batching).

## Why vLLM

vLLM (Kwon et al., 2023) is the current field-standard open-source inference server for LLMs. Two design decisions make it dominant:

- **PagedAttention.** The KV cache is stored in fixed-size pages rather than contiguous per-request blocks, so requests can share the KV-cache pool efficiently, memory fragmentation is bounded, and throughput on mixed-length workloads is high.
- **Continuous batching.** Rather than a fixed-size batch that waits for the slowest request, vLLM schedules token generation at each step across whatever requests are in flight, adding and removing requests from the batch dynamically. Throughput on realistic bursty traffic can be several times higher than static batching.

The OpenAI-compatible server on top means any client that speaks the OpenAI API can point at vLLM without code changes. That is the "just switch the base URL" migration path most teams take.

## Installing and the smallest possible smoke test

vLLM ships as a Python package and a CLI. Recent versions favor Docker images for reproducibility; the `vllm/vllm-openai` image is the canonical one.

Local Python install:

```bash
pip install "vllm==<pinned-version>"
```

The two-command smoke test:

```bash
# 1. Start the server on port 8000.
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096

# 2. From another shell, hit the OpenAI-compatible endpoint.
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","prompt":"The capital of France is","max_tokens":16}'
```

If the second command returns a completion mentioning Paris in under ten seconds, your install is working. If not, the two most common causes are (a) VRAM (drop `--gpu-memory-utilization` to 0.85, drop `--max-model-len`), and (b) mismatch between the CUDA / PyTorch / vLLM versions (`nvidia-smi` and `python -c "import torch; print(torch.__version__, torch.version.cuda)"` are your first debug commands).

## The engine args that matter

vLLM's `serve` and `LLM(...)` accept many flags; the ones that materially affect quality, latency, throughput, or whether the server starts at all:

### `--model` and `--served-model-name`

`--model` is the checkpoint path or Hugging Face repo id. `--served-model-name` is the name clients use in the `model` field of a request. They can differ — useful for aliasing internal repo names to product-facing names, or for serving multiple adapters under different "model" names in the OpenAI API.

```bash
vllm serve /models/support-adapter-merged \
  --served-model-name support-v3
```

Clients then send `"model": "support-v3"`.

### `--dtype`

`auto` reads the checkpoint's dtype (usually `bfloat16`). Override to `float16` on Ampere / Turing GPUs that struggle with bf16, or to `bfloat16` explicitly to force. For quantized checkpoints, `--dtype` sets the *compute dtype*, not the weight dtype. Do not set `--dtype float32` unless you are debugging; it will exhaust VRAM instantly.

### `--quantization`

Tells vLLM the checkpoint is quantized and which format:

| Flag value | Format | Notes |
|---|---|---|
| `awq` / `awq_marlin` | AWQ 4-bit | Marlin kernel is the current fast path on Ampere / Hopper. |
| `gptq` / `gptq_marlin` | GPTQ 4-bit | Same story; Marlin fast path. |
| `bitsandbytes` | bitsandbytes NF4 / INT8 | Works; historically slower than GPTQ / AWQ. |
| `fp8` | FP8 (e4m3 / e5m2) | H100+ only. |
| `compressed-tensors` | The unified `compressed-tensors` format Neural Magic uses | Covers W8A8, INT4, FP8 variants under one loader. |
| (unset) | Auto-detect from `quantization_config` in `config.json` | Most modern quantized checkpoints record this; do not need to set the flag explicitly. |

The most common bug: setting `--quantization awq` on a checkpoint that was saved as GPTQ (or vice versa). vLLM errors on start. Fix by omitting the flag and letting auto-detect do its job, or by reading the checkpoint's `config.json` `quantization_config` block.

### `--kv-cache-dtype`

Quantizes the KV cache separately from the weights. Options in current vLLM include `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2`. `fp8` roughly halves KV-cache memory at a small quality cost; on long-context or high-batch workloads, this is the single biggest memory win on the runtime side.

Trade-offs:

- KV cache in fp8 requires supported hardware (Hopper for FP8 tensor cores; Ampere works but is slower).
- Some tasks are more sensitive to KV-cache precision than others; benchmark on your task-specific eval before committing.

### `--max-model-len`

Maximum context length the server will accept. Two forces to balance:

- Larger `--max-model-len` = more KV-cache reserved *per request* = fewer concurrent requests fit at peak. On a 7B model at 128k context, one request's KV cache alone can be tens of GB.
- Smaller `--max-model-len` = requests longer than the cap are rejected; if that includes real production traffic, you lose it.

The rule: set `--max-model-len` to `p99(prompt + expected completion) + a margin`, not to the model's architectural maximum. If you set it to the max (e.g., 128k for Llama-3.1) and your traffic is 4k, you waste enormous VRAM on the KV-cache reservation.

### `--gpu-memory-utilization`

Fraction of GPU memory vLLM is allowed to use. Default is typically `0.90`. The rest is left for CUDA-runtime overhead, activation buffers, and safety margin.

Too high (`0.98`) → occasional OOM, especially with long prompts.
Too low (`0.60`) → wasted GPU capacity; throughput suffers.

A safe default: `0.90`. Reduce to `0.85` on shared GPUs or when co-located with other workloads. Reduce further if you see OOMs during load spikes.

### `--tensor-parallel-size` and `--pipeline-parallel-size`

For models larger than a single GPU can hold, or when you want to shard for throughput.

- `--tensor-parallel-size N` splits each matmul across N GPUs. Standard for large-model serving. Requires NVLink or equivalent for good performance.
- `--pipeline-parallel-size N` splits the model layer-wise across N stages. Better for cross-node deployments where inter-node bandwidth is lower.

For a 7B model on one H100, both are `1` and can be omitted. For a 70B on 4× H100, `--tensor-parallel-size 4` is the standard.

### `--max-num-seqs` and `--max-num-batched-tokens`

The batch-scheduling knobs.

- `--max-num-seqs` — max concurrent sequences in the batch. Higher = better throughput on bursty traffic, more VRAM in flight.
- `--max-num-batched-tokens` — cap on total tokens (prompt + generated) in one batch step. Prevents a single long-prompt request from dominating.

Defaults are usually fine for a starting point. Tune upward for maximum throughput once you know your workload's mix of prompt lengths.

### `--enable-prefix-caching`

Reuses the KV cache across requests that share a prompt prefix (system prompt, few-shot exemplars, RAG context). Massive throughput win when prefix reuse is high; negligible cost when not. Turn it on unless you have a specific reason not to.

## The OpenAI-compatible server

The endpoint surface vLLM exposes matches the OpenAI API for the parts most clients use:

- `POST /v1/completions` — legacy completions endpoint. Prompt in, tokens out.
- `POST /v1/chat/completions` — chat endpoint. Takes messages, applies the model's chat template server-side, returns a message.
- `GET /v1/models` — lists served models (base + any registered LoRA adapters).
- `POST /v1/embeddings` — for models that expose embeddings.

The `/v1/chat/completions` endpoint applies the tokenizer's chat template automatically. This is convenient — the client sends messages, the server handles the template — but it also means chat-template bugs in the tokenizer surface here. If your fine-tuned model uses a custom chat template, verify it is present in `tokenizer_config.json` before deploying (mod-108 chapter 7 covers the mod-102-to-serving handoff for template consistency).

### Streaming

Add `"stream": true` to the request body; the response is Server-Sent Events. Every client library that talks to OpenAI can consume this.

### Request-level sampling parameters

Standard: `temperature`, `top_p`, `top_k`, `max_tokens`, `frequency_penalty`, `presence_penalty`, `stop`, `n`, `logprobs`. vLLM-specific extensions (via `extra_body` in the OpenAI client): `top_k`, `min_p`, `repetition_penalty`, `use_beam_search`, `best_of`, `guided_json` / `guided_regex` / `guided_grammar` for structured outputs.

## Loading a quantized checkpoint end to end

Concrete recipe for a GPTQ-4bit merged model:

```bash
vllm serve /models/support-v3-gptq-4bit \
  --served-model-name support-v3 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype auto \
  --enable-prefix-caching
```

The `quantization` flag can be omitted if the checkpoint's `config.json` has a `quantization_config` block naming `"gptq"` — vLLM auto-detects. If auto-detect fails, add `--quantization gptq_marlin` (Marlin is faster on Ampere / Hopper; falls back to `gptq` on unsupported hardware).

For AWQ:

```bash
vllm serve /models/support-v3-awq \
  --served-model-name support-v3 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```

For `bitsandbytes` NF4:

```bash
vllm serve /models/support-v3-bnb-nf4 \
  --served-model-name support-v3 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --max-model-len 4096
```

For FP8 on H100:

```bash
vllm serve /models/support-v3-fp8 \
  --served-model-name support-v3 \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 8192
```

<!-- needs-research: quantization flag names and the exact set of supported `--quantization` values evolve between vLLM versions. Verify against the version-pinned vLLM docs listed in `resources.md` for your install. -->

## Multi-LoRA with vLLM

Extending chapter 3's multi-LoRA config to a full server:

```bash
vllm serve meta-llama/Llama-3.1-8B \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --enable-lora \
  --max-loras 8 \
  --max-lora-rank 32 \
  --max-cpu-loras 64 \
  --lora-modules \
    support=/adapters/support \
    sales=/adapters/sales \
    code=/adapters/code
```

Each registered adapter shows up as its own model in `GET /v1/models`. Clients call the adapter by its name in the `model` field:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "support",
    "messages":[{"role":"user","content":"How do I reset my password?"}],
    "max_tokens": 128
  }'
```

New adapters can be registered at runtime via the `POST /v1/load_lora_adapter` endpoint (URL and payload subject to vLLM version — check the current docs):

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name":"legal","lora_path":"/adapters/legal"}'
```

And unloaded via `POST /v1/unload_lora_adapter`. This is what the adapter-warehouse pattern from chapter 3 hooks into.

### Pre-warming

At startup, issue one dummy request per adapter so the first real request does not pay the cold-load cost. A five-line startup script that iterates registered adapters and issues a one-token generation to each is standard.

## Structured outputs

vLLM supports constrained decoding via the `guided_*` request fields. Three flavors:

- **`guided_json` + JSON Schema.** Output must conform to the schema. Uses `outlines` / `xgrammar` under the hood. Cost: some throughput; token generation is constrained to the schema-legal continuations.
- **`guided_regex`.** Output must match the regex. Simpler; fine for structured extraction tasks.
- **`guided_grammar`.** Full context-free grammar (Lark-style). Rare in practice; regex or JSON schema covers most needs.

For a fine-tuned model that has been trained to emit JSON, structured outputs give you a hard guarantee on top of the model's soft preference. Especially valuable when the downstream consumer (a parser, a tool call) will fail on malformed JSON. Use them wherever the schema is known.

## The five smoke tests you run on every deploy

Before declaring a vLLM deploy "shipped," run these five checks. They take about ten minutes and catch the most common issues.

**1. `/v1/models` shows the expected model(s).** If a LoRA adapter is missing here, it was not registered.

```bash
curl -s http://localhost:8000/v1/models | jq '.data[].id'
```

**2. A completion round-trips with a coherent response.** Not just "200 OK" — actually read the output.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"...","messages":[{"role":"user","content":"Name the capital of France."}]}' \
  | jq -r '.choices[0].message.content'
```

Expected: coherent answer mentioning Paris. Punctuation salad means the chat template is wrong, or the wrong tokenizer got loaded, or the quantization is broken.

**3. Chat-template alignment matches expectation.** The best way is a "hidden phrase" test — a prompt the fine-tune was trained to respond to distinctively. If the response is wrong or matches the base model's response, the served checkpoint is not the checkpoint you think.

**4. `/metrics` (Prometheus) is scraping.** vLLM exposes queue length, active requests, GPU cache utilization, and TTFT / ITL histograms. If your monitoring is not consuming these, you are running blind.

```bash
curl -s http://localhost:8000/metrics | head -50
```

**5. A load-test at 2× your expected p95 concurrency does not degrade.** Chapter 7's load-test pattern. Failure signature: TTFT climbs to seconds, then requests start timing out or the server hits OOM.

## Observability

Wire the `/metrics` endpoint to Prometheus. The metrics worth watching in every deploy:

- **`vllm:num_requests_running`** and **`vllm:num_requests_waiting`** — queue depth. If waiting climbs, you are capacity-bound.
- **`vllm:gpu_cache_usage_perc`** — KV cache utilization. If pegged at ~100%, `--max-model-len` or `--max-num-seqs` is too high for the incoming traffic; the scheduler will start evicting.
- **`vllm:time_to_first_token_seconds`** — TTFT histogram. p95 climbing means saturation.
- **`vllm:time_per_output_token_seconds`** — inter-token latency. Should be nearly constant once the batch is filled; if it moves, the batch composition is changing (e.g., long prompts joining the batch).
- **`vllm:e2e_request_latency_seconds`** — end-to-end request latency; the SLO number.

Chapter 7 wires these into an SLO dashboard and the on-call runbook.

## Docker + Kubernetes shape

For a Docker deploy:

```bash
docker run --gpus all \
  -v /models:/models \
  -p 8000:8000 \
  vllm/vllm-openai:<pinned> \
  --model /models/support-v3-gptq-4bit \
  --served-model-name support-v3 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```

For Kubernetes, the standard pattern: one pod per replica, NodePool with GPUs, an HPA on `vllm:num_requests_waiting` or CPU (a proxy for saturation), init containers to pre-warm adapters if applicable. Mod-111 covers the platform-shaped version; here it is enough to know the container runs identically to the local invocation.

## Known-good starting configs by model size

Illustrative one-GPU starting points; adjust for your traffic.

| Model | GPU | Quantization | `--max-model-len` | `--gpu-memory-utilization` | Notes |
|---|---|---|---|---|---|
| 1B / 1.5B | Any 16 GB+ | BF16 | 8192 | 0.85 | Small enough to skip quantization. |
| 7B | 1× A100 40 GB | BF16 | 4096 | 0.85 | Fine but tight; consider quantizing. |
| 7B | 1× L40S / L4 | GPTQ 4-bit | 4096 | 0.85 | Comfortable. |
| 7B | 1× H100 80 GB | BF16 | 8192 | 0.90 | Comfortable; try `--kv-cache-dtype fp8` for longer context. |
| 13B | 1× A100 80 GB | BF16 | 4096 | 0.85 | Tight. |
| 13B | 1× H100 80 GB | GPTQ 4-bit | 8192 | 0.90 | Comfortable. |
| 70B | 4× A100 80 GB | BF16 | 4096 | 0.85 | `--tensor-parallel-size 4`. |
| 70B | 4× H100 80 GB | AWQ / GPTQ 4-bit | 8192 | 0.90 | `--tensor-parallel-size 4`; fp8 KV cache is worth trying. |

## Exercise pointer

`exercises/exercise-02-vllm-multi-lora-vs-merged-deployment.md` has you stand up vLLM in two configurations — one merged, one multi-LoRA — under a shared workload, and write the decision-with-numbers.

`exercises/exercise-01-gptq-awq-bnb-comparison.md` runs each quantization method against the release gate; the served-artifact step is a vLLM deploy per variant.

## Summary

- vLLM is the current field-standard open-source inference server for LLMs: PagedAttention for efficient KV-cache memory, continuous batching for high throughput, and an OpenAI-compatible API for zero-friction client migration.
- The engine args that materially decide behavior: `--dtype`, `--quantization`, `--kv-cache-dtype`, `--max-model-len`, `--gpu-memory-utilization`, `--tensor-parallel-size`, `--enable-lora` / `--max-loras`, `--enable-prefix-caching`. Every serving runbook records the values of each and the reason.
- Quantized checkpoints load via `--quantization <format>` (or auto-detect from `config.json`); the runtime chooses the fastest available kernel (Marlin on modern GPUs for GPTQ / AWQ; FP8 tensor cores on H100+).
- Multi-LoRA is a first-class feature: `--enable-lora --max-loras N --max-lora-rank R --max-cpu-loras M --lora-modules name=path ...`, plus `POST /v1/load_lora_adapter` for runtime registration. Each adapter becomes a served-model name in the OpenAI-compatible API.
- Structured outputs via `guided_json` / `guided_regex` / `guided_grammar` add hard guarantees on top of a fine-tuned model's soft preferences; use them where a downstream consumer would fail on malformed output.
- Wire `/metrics` (Prometheus) at deploy time and run the five smoke tests before declaring success. Chapter 7 turns those metrics into an SLO dashboard and the on-call runbook.
