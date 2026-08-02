# Deploying with Text Generation Inference (TGI)

**Estimated read:** 35 minutes.
**You should be able to, by the end:** deploy a fine-tuned model with Hugging Face TGI at either BF16 or a quantized weight format (GPTQ / AWQ / bitsandbytes / EETQ / FP8), understand the launcher flags that trade latency for throughput and correctness (`--num-shard`, `--max-input-length`, `--max-total-tokens`, `--max-batch-prefill-tokens`, `--quantize`), attach one or more LoRA adapters, exercise the messages / generate / stream endpoints, and articulate the operational differences from vLLM (chapter 4) that would push a team to pick one or the other.

## Where TGI fits

Text Generation Inference (TGI) is Hugging Face's open-source, production-oriented inference server. It grew up alongside vLLM but with different priorities: tight integration with the HF Hub, well-defined defaults per model family, first-class support for a wide set of quantization backends (bitsandbytes, GPTQ, AWQ, EETQ, FP8, Marlin), and a mature Rust-based launcher / router pair that keeps operational surprises small.

You will meet TGI in three situations:

- The team already runs Hugging Face services (Inference Endpoints, Spaces, or self-hosted) and wants continuity.
- vLLM does not yet support the exact combination of (model, quantization, feature) you need but TGI does — or vice versa. Both stacks race each other; the coverage matrix shifts.
- The deployment target is a Hugging Face-managed environment (Inference Endpoints, Sagemaker HF containers, Vertex HF containers), where TGI is the default runtime.

For the fine-tuning engineer, TGI and vLLM occupy the same slot in the runbook: "GPU inference server behind a REST API." The choice is usually pragmatic — feature parity for your specific quantized-adapter combination and the team's operational familiarity — rather than a clean quality difference.

## The two processes

TGI runs a launcher process (Rust) that supervises one or more shards (Python + CUDA). The launcher exposes the HTTP surface; the shards do the tensor work. This is transparent from the outside but matters when debugging: crashes on the shard side surface as launcher error logs, and shard-per-GPU (tensor-parallel) is set by `--num-shard`.

## Installing and the smallest possible smoke test

The reference deployment is the Docker image `ghcr.io/huggingface/text-generation-inference:<version>`.

```bash
docker run --gpus all --shm-size 1g \
  -p 8080:80 \
  -v /models:/data \
  ghcr.io/huggingface/text-generation-inference:<pinned> \
  --model-id /data/support-v3-merged \
  --max-input-length 3072 \
  --max-total-tokens 4096
```

The two-command smoke test:

```bash
# 1. Server up (foreground per above, or background it).
# 2. From another shell:
curl http://localhost:8080/generate \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"inputs":"The capital of France is","parameters":{"max_new_tokens":16}}'
```

Response is a JSON with `generated_text`. If you get "Paris" or a variant, you're up. If not, common causes: (a) VRAM (drop `--max-total-tokens`), (b) the checkpoint's tokenizer config is missing `chat_template` for the `/v1/chat/completions` endpoint (fine for the raw `/generate` path but breaks messages).

## The launcher args that matter

TGI's launcher exposes a large flag surface; the load-bearing subset:

### `--model-id`

Either a Hugging Face Hub id (`meta-llama/Llama-3.1-8B-Instruct`) or a local path (`/data/support-v3`). Local paths avoid a Hub download on startup and are the norm for production.

### `--revision`

Pin to a commit hash on the Hub. Chapter 3's base-revision safety story applies: pin at deploy time so a Hub-side revision bump does not silently change the served model.

### `--num-shard`

Tensor-parallel degree. For a 7B on one GPU, `1` (default). For a 70B on 4 GPUs, `4`. TGI shards the model across `--num-shard` GPUs; the launcher supervises all shards as one deployment.

### `--dtype`

`float16`, `bfloat16`, or leave to auto. Match the training dtype for correctness (see the mod-108 chapter 7 chat-template / dtype consistency story).

### `--quantize`

Tells TGI to load a quantized checkpoint or to quantize at load time:

| Flag value | Behavior |
|---|---|
| `bitsandbytes` | Loads via `bitsandbytes` in 8-bit (`LLM.int8()`). |
| `bitsandbytes-nf4` | Loads in 4-bit NF4. |
| `bitsandbytes-fp4` | Loads in 4-bit FP4. |
| `gptq` | Loads a pre-quantized GPTQ checkpoint. |
| `awq` | Loads a pre-quantized AWQ checkpoint. |
| `eetq` | Loads via Nvidia EETQ (8-bit); available on H100 / A100. |
| `fp8` | Loads via FP8 (Hopper+). |
| `marlin` | Uses the Marlin kernel for INT4 / INT8 (typically activated via GPTQ / AWQ paths automatically on supported hardware). |

The pattern: if the checkpoint is *already* quantized (GPTQ / AWQ / FP8 files on disk), pass the matching `--quantize` value so TGI uses the right loader. If the checkpoint is BF16 and you want on-the-fly quantization, `--quantize bitsandbytes` / `bitsandbytes-nf4` does it at load. Note the on-the-fly path is convenient for dev but slower to load and less flexible than pre-quantizing with GPTQ / AWQ.

<!-- needs-research: TGI's `--quantize` value set and per-value support (which model families work with which flag) drift version to version. Verify against the version-pinned TGI launcher docs for your install. -->

### `--max-input-length`

Cap on prompt length. Requests longer than this are rejected.

### `--max-total-tokens`

Cap on prompt + generation. Similar role to vLLM's `--max-model-len`. Same trade-off: too high wastes VRAM; too low rejects legitimate traffic.

### `--max-batch-prefill-tokens`

Maximum total prompt tokens in the *prefill* phase of a batch. Prefill (processing the input prompt) is the expensive per-step operation; capping the prefill batch limits worst-case latency for the requests already in flight. Common default: `2 * max-input-length` or thereabouts. If prefill dominates your latency (long-prompt workloads), tune this down.

### `--max-batch-total-tokens`

Cap on total tokens (prefill + decode) across all requests in one batch. Higher = better throughput, more VRAM used. Usually left at the launcher's default unless you are tuning aggressively.

### `--waiting-served-ratio`

TGI's continuous-batching scheduler admits new requests based on this ratio. Higher = prefers admitting new requests over decoding faster on in-flight ones. Lower = prefers finishing current requests before starting new ones. The default is a reasonable balance; tune only after observing queue behavior in production.

### `--enable-prefix-caching`

Reuses KV state across requests that share a prefix (system prompts, RAG contexts). Same win as vLLM's flag. Turn on by default.

### `--trust-remote-code`

Required for some model architectures whose modeling code is not in `transformers` core. Standard hazard: enabling this executes code from the Hub or your local path. Only enable for trusted sources.

## The API surface

TGI exposes three endpoint families:

### `POST /generate` and `POST /generate_stream`

TGI-native. Non-streaming and streaming variants.

```bash
curl http://localhost:8080/generate \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "inputs":"Summarize: Fine-tuning is ...",
    "parameters":{
      "max_new_tokens": 128,
      "temperature": 0.7,
      "top_p": 0.9,
      "stop": ["\n\n"]
    }
  }'
```

Streaming: `/generate_stream` emits SSE frames of `{token, index, ...}` until the final `{generated_text, ...}`.

### `POST /v1/chat/completions` and `POST /v1/completions`

OpenAI-compatible. Same shape as vLLM's and OpenAI's own. TGI applies the tokenizer's `chat_template` server-side for `/v1/chat/completions`. Streaming via `"stream": true`.

Migration convenience: OpenAI clients pointed at TGI work without code changes for chat / completions traffic.

### `GET /info` and `GET /metrics`

`/info` returns a snapshot of the served model, quantization mode, dtype, and launcher settings.

`/metrics` is a Prometheus endpoint with the operational counters: queue depth, batch fill, KV utilization, TTFT / ITL histograms. Chapter 7 wires these into the SLO dashboard.

## Loading a quantized checkpoint end to end

For a GPTQ 4-bit checkpoint:

```bash
docker run --gpus all --shm-size 1g \
  -p 8080:80 \
  -v /models:/data \
  ghcr.io/huggingface/text-generation-inference:<pinned> \
  --model-id /data/support-v3-gptq-4bit \
  --quantize gptq \
  --max-input-length 3072 \
  --max-total-tokens 4096 \
  --dtype float16
```

For AWQ:

```bash
--model-id /data/support-v3-awq \
--quantize awq
```

For bitsandbytes NF4 (loaded on the fly from a BF16 checkpoint):

```bash
--model-id /data/support-v3-merged  \
--quantize bitsandbytes-nf4
```

For FP8 on H100:

```bash
--model-id /data/support-v3-fp8 \
--quantize fp8
```

The `--dtype` flag sets the compute dtype for the dequantized matmuls; `float16` and `bfloat16` are both common. Match the fine-tuning dtype for correctness.

## LoRA adapters on TGI

TGI supports LoRA adapters via the `--lora-adapters` launcher flag and per-request adapter selection through the `adapter_id` parameter (or `model` field on the OpenAI-compatible endpoint, if adapters are registered as separate models).

```bash
docker run --gpus all --shm-size 1g \
  -p 8080:80 \
  -v /models:/data -v /adapters:/adapters \
  ghcr.io/huggingface/text-generation-inference:<pinned> \
  --model-id meta-llama/Llama-3.1-8B \
  --dtype bfloat16 \
  --max-input-length 3072 \
  --max-total-tokens 4096 \
  --lora-adapters support=/adapters/support sales=/adapters/sales code=/adapters/code
```

Per-request selection:

```bash
curl http://localhost:8080/generate \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "inputs":"How do I reset my password?",
    "parameters":{"adapter_id":"support","max_new_tokens":128}
  }'
```

On the OpenAI-compatible path, each adapter is served under its own model name:

```bash
curl http://localhost:8080/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "model":"support",
    "messages":[{"role":"user","content":"How do I reset my password?"}]
  }'
```

Same routing patterns from chapter 3 apply — header-based, classifier-based, canary, fallback.

<!-- needs-research: TGI's LoRA-adapter flags and endpoints have changed across versions and are not always feature-identical with vLLM's multi-LoRA. Verify LoRA-over-quantized-base support against the version-pinned docs before designing around it. -->

## The five smoke tests you run on every deploy

Same shape as the vLLM smoke tests (chapter 4), adapted to TGI's endpoints:

**1. `/info` reports the expected model, quantization mode, and dtype.**

```bash
curl -s http://localhost:8080/info | jq
```

**2. `/generate` round-trips a coherent completion.**

**3. Chat-template alignment via `/v1/chat/completions`.** If your fine-tune has a custom template, confirm the server-applied template matches the training template.

**4. `/metrics` is scraping and reports non-zero counters after warmup requests.**

**5. A load-test at 2× expected p95 concurrency does not degrade.** Chapter 7's pattern.

## Observability

The Prometheus metrics worth watching on TGI:

- **`tgi_queue_size`** — pending requests waiting for admission.
- **`tgi_batch_current_size`** — sequences currently in the batch.
- **`tgi_request_success_rate` / `tgi_request_failure_rate`** — request health.
- **`tgi_request_input_length` / `tgi_request_generated_tokens`** — histograms of workload shape.
- **`tgi_request_inference_duration` / `tgi_request_queue_duration`** — the parts of latency that add to the SLO.
- **`tgi_batch_inference_duration`** — batch-step latency; if it climbs, throughput will degrade.

Naming may differ across versions; verify against the version's `/metrics` output.

## TGI vs. vLLM — the decision matrix

The two stacks solve the same problem. Neither is universally better. The axes that actually decide:

| Axis | Reach for vLLM when | Reach for TGI when |
|---|---|---|
| Peak throughput per GPU | On modern GPUs with popular models, vLLM often leads on continuous-batching throughput. | For some model families or quantization combos, TGI's default tuning is closer to optimal out of the box. |
| Operational familiarity | Team already runs vLLM elsewhere; OpenAI-compatible API is the primary interface. | Team is on Hugging Face Endpoints / Inference / Sagemaker HF containers already. |
| Quantization coverage | GPTQ / AWQ / FP8 / compressed-tensors are the primary targets; multi-LoRA over quantized base is required. | bitsandbytes-nf4 / EETQ / on-the-fly quantization are useful; FP8 with the Nvidia toolchain is straightforward. |
| LoRA-over-quantized combinations | The specific combination is supported and battle-tested. | Ditto — verify each version. |
| Structured outputs | vLLM's `guided_json` / `guided_regex` are mature. | TGI supports grammars via `outlines` / `xgrammar`; check the version. |
| Streaming and SSE | Standard. | Standard, plus `/generate_stream` if TGI-native clients matter. |
| Multi-node inference (>4 GPUs) | vLLM has more mature pipeline-parallel + Ray integration in recent versions. | Single-node up to `--num-shard=8` is the common ceiling. |
| Ecosystem for HF-managed hosting | You self-host. | You want to run the same image on Inference Endpoints without changes. |

The honest recommendation for a new team building a serving stack from scratch: **spin up both against the exact same quantized checkpoint and adapter set, run chapter 7's load test on each, and pick based on the numbers.** The chapters 4 / 5 knowledge is what lets you configure both fairly. Do not delegate the choice to a benchmark blog post — the version and the model matter more than either stack's marketing.

## The handoff to the runbook

Both TGI and vLLM produce three deploy-time outputs that go into the runbook (chapter 7):

- **The launch command with every flag pinned.** No implicit defaults.
- **The `/info` (or `/v1/models`) response captured at first successful boot.** Proves the served artifact is the intended one.
- **A first-hour metrics snapshot.** Baseline for future regressions.

Same discipline as pinning mod-108's eval harness (chapter 6): if the config drifts, the artifact drifts.

## Exercise pointer

`exercises/exercise-03-tgi-deployment-and-load-test.md` has you deploy the quantized checkpoint from exercise 01 behind TGI, exercise `/generate`, `/generate_stream`, and `/v1/chat/completions`, and run a controlled load test against a vLLM deployment of the same artifact as the baseline. The write-up is an operational comparison — TTFT / ITL / p95 / throughput / VRAM / config surface — with a defended runtime recommendation.

## Summary

- Text Generation Inference (TGI) is Hugging Face's production-oriented inference server: a Rust launcher supervising Python / CUDA shards, an OpenAI-compatible API alongside a TGI-native `/generate` path, and first-class support for a wide quantization matrix (bitsandbytes / GPTQ / AWQ / EETQ / FP8 / Marlin).
- Load-bearing launcher flags: `--model-id`, `--revision`, `--num-shard`, `--dtype`, `--quantize`, `--max-input-length`, `--max-total-tokens`, `--max-batch-prefill-tokens`, `--enable-prefix-caching`, and `--lora-adapters`. Pin every one in the runbook.
- Quantized checkpoints on disk load via the matching `--quantize` flag; on-the-fly `bitsandbytes` quantization from a BF16 checkpoint is convenient for dev and slower for production. For production 4-bit, pre-quantize with GPTQ / AWQ (chapter 2) and load with `--quantize gptq` / `awq`.
- LoRA adapters register via `--lora-adapters name=path` and route per-request via `adapter_id` (or the OpenAI `model` field). The chapter-3 routing patterns apply unchanged.
- TGI vs. vLLM is a pragmatic choice: run both against the same artifact, load-test with chapter 7's pattern, pick on numbers. Neither is universally better; the coverage matrix for (model, quantization, feature) shifts version to version.
- Wire `/metrics` on deploy, capture `/info` and the launch command in the runbook, and re-run the mod-108 release gate against the served artifact — not the training checkpoint — to confirm the quantized model still meets the bar.
