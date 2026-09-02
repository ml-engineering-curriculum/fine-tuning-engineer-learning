# exercise-03: TGI Deployment and Load Test (vs. vLLM Baseline)

**Estimated effort:** 3 hours
**Companion chapters:** [`04-vllm-deployment.md`](../04-vllm-deployment.md), [`05-tgi-deployment.md`](../05-tgi-deployment.md), [`07-serving-runbook-and-cost.md`](../07-serving-runbook-and-cost.md)

## Objective

Deploy the same quantized fine-tuned checkpoint behind **Text Generation Inference (TGI)**, exercise its three endpoint families (`/generate`, `/generate_stream`, `/v1/chat/completions`), run a controlled load test against a matched **vLLM** deployment as the baseline, and produce the operational comparison that decides which runtime to reach for. The write-up recommends one runtime for a specified serving scenario, backed by your own latency, throughput, VRAM, config-surface, and observability numbers.

The point of this exercise is that "vLLM or TGI?" is not a taste question. Chapter 5 named the axes; the load test is where the axes turn into numbers. You should finish with a defended answer for your specific model × quantization × workload combination — and a runbook draft either runtime's on-call could execute.

## Prerequisites

- One quantized checkpoint that loads cleanly on both runtimes. The natural pick is the GPTQ-4bit or AWQ-4bit artifact from exercise 01 (`artifacts/gptq-4bit` or `artifacts/awq-4bit`); alternatively, a publicly available quantized fine-tune of a permissively-licensed base.
- Both runtimes installable on your hardware. The reference: TGI via Docker (`ghcr.io/huggingface/text-generation-inference:<pinned>`), vLLM via Docker (`vllm/vllm-openai:<pinned>`) or pip. Docker is preferred for reproducibility here; both `docker` and `--gpus all` support required on the host.
- 1 GPU with enough headroom to run one instance at a time (the exercise runs them sequentially against the same GPU to control for hardware). An A100 40 GB, H100 80 GB, or L40S 48 GB is comfortable for a 7B GPTQ-4bit.
- A load-test tool that speaks both APIs. `llmperf` (cross-runtime), `vllm bench serve` (needs OpenAI-compatible surface — both runtimes expose it), or `k6` / `locust` with a custom client. `llmperf` is the reference because it treats both runtimes symmetrically.
- Python 3.10+, `httpx` or `requests`, `openai` Python client for smoke tests, `curl` / `jq` for API-shape checks.

## Requirements

### Part A — Pin the environment

Author `requirements.txt` and `pins.md` capturing:

- TGI image tag (`ghcr.io/huggingface/text-generation-inference:<pinned>`) and the launcher flags it accepts (`docker run ... --help`, save to `pins.md`).
- vLLM version (image tag if Docker, package version if pip) and its `serve --help` output.
- Model id + revision + local path + `config.json` `quantization_config` block (paste it into `pins.md` — chapter 4 called this out as a common source of `--quantization` mismatch bugs).
- Load-tool version (`llmperf --version` or the git commit).
- Hardware spec block: GPU model, VRAM, CUDA / driver, container runtime version.

### Part B — Deployment 1: TGI

Launch TGI against the quantized artifact:

```bash
docker run --gpus all --shm-size 1g \
  -p 8080:80 \
  -v $(pwd)/artifacts:/data \
  ghcr.io/huggingface/text-generation-inference:<pinned> \
  --model-id /data/gptq-4bit \
  --quantize gptq \
  --dtype float16 \
  --max-input-length 3072 \
  --max-total-tokens 4096 \
  --max-batch-prefill-tokens 6144 \
  --waiting-served-ratio 1.2
```

Substitute `awq` / `bitsandbytes-nf4` / `fp8` as appropriate for your artifact. Capture the launch command verbatim in `deploy/tgi.launch.sh`. Record the boot-time output (`/info` response, launcher log lines showing the shard-up sequence) in `deploy/tgi.boot.md`.

Run the five TGI smoke tests from chapter 5:

1. `GET /info` — reports the expected model, quantize mode, dtype, and shard count.
2. `POST /generate` — coherent completion on a fixed prompt.
3. `POST /v1/chat/completions` — chat-template alignment (use a "hidden phrase" prompt that only the fine-tune should answer distinctively).
4. `GET /metrics` — non-zero counters after warmup requests. Save `curl -s http://localhost:8080/metrics | head -80` to `deploy/tgi.metrics_head.txt`.
5. `POST /generate_stream` — SSE frames arrive incrementally, final frame carries `generated_text`.

Save the outputs to `deploy/tgi.smoke.md`.

### Part C — Deployment 2: vLLM (the baseline)

Stop the TGI container. Launch vLLM against the *identical* artifact:

```bash
docker run --gpus all \
  -p 8000:8000 \
  -v $(pwd)/artifacts:/models \
  vllm/vllm-openai:<pinned> \
  --model /models/gptq-4bit \
  --served-model-name gptq-4bit \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

Capture in `deploy/vllm.launch.sh`. Boot output in `deploy/vllm.boot.md`. Run the five vLLM smoke tests (chapter 4) and save to `deploy/vllm.smoke.md`.

**Fairness rules for the head-to-head:** identical GPU, identical artifact on disk, identical `max-model-len` / `max-total-tokens`, identical dtype (both `float16` or both `bfloat16` — do not mix), identical prefix-caching state (either both on or both off), identical `--enable-prefix-caching` semantics if applicable. Document any unavoidable asymmetry in `pins.md` under a "known differences" heading.

### Part D — Endpoint parity check

TGI's `/generate` and vLLM's `/v1/completions` are close cousins but not identical. Before the load test, run parity checks so any latency difference is attributed to the runtime and not to a semantics gap.

Author `parity_check.py` that, for a fixed set of 20 prompts + fixed `max_new_tokens=64` + `temperature=0.0`:

- Hits TGI `/generate`, TGI `/v1/chat/completions`, vLLM `/v1/completions`, and vLLM `/v1/chat/completions`.
- Records the response text for each.
- Compares TGI-chat vs. vLLM-chat and TGI-generate vs. vLLM-completion at greedy decoding. First-N-token divergence beyond the third token warrants investigation — usually a chat-template or `add_generation_prompt` mismatch.

Save the diff table to `parity/parity_table.md`. If parity is off, halt and diagnose before load-testing — you would otherwise be timing two different generations.

### Part E — The shared workload

Define one workload used against both deployments. Fixed:

- **Prompt corpus.** ≥500 prompts drawn from ShareGPT, your held-out set, or a mix. Use identical `workload.jsonl` (verify by `sha256sum`).
- **Load profile.** Concurrency ramp: 1 → 2 → 4 → 8 → 16 → 32 → 64 concurrent. Hold each level for 60 seconds after a 30-second warm-up. If your GPU saturates below 64, stop the ramp at the last stable level.
- **Generation params.** `max_new_tokens=256`, `temperature=0.7`, `top_p=0.9`, `seed=42`. Identical across runtimes and clients.
- **Streaming mode.** Run the ramp twice: once with `stream=false`, once with `stream=true`. Streaming changes the effective ITL number; both matter.

Save `workload/workload.jsonl` and `workload/README.md`.

### Part F — Run the load test

Run against each deployment sequentially (only one runtime is up at a time). Capture per concurrency level:

- Throughput: req/s and tok/s (input, output, total).
- TTFT: p50, p95, p99.
- ITL / TPOT: p50, p95, p99 (streaming mode).
- E2E latency: p50, p95, p99 (non-streaming mode).
- Success rate (2xx / total) and error breakdown (5xx, 4xx, timeouts).
- GPU memory utilization at steady state (from `nvidia-smi dmon` sampled every second during the run — save to `results/*/gpu.log`).
- Runtime-native queue metric (`tgi_queue_size` for TGI, `vllm:num_requests_waiting` for vLLM) at steady state per level.

Save raw output to `results/tgi/` and `results/vllm/`, one subdirectory per (concurrency × streaming) combination. Include the load-tool invocation in each results directory's `README.md`.

### Part G — Build the comparison matrices

Author `build_matrices.py` that reads both result trees and produces:

**`latency_matrix.md`** — the head-to-head:

| Concurrency | Stream | TGI tok/s | vLLM tok/s | TGI TTFT p95 (ms) | vLLM TTFT p95 (ms) | TGI ITL p95 (ms) | vLLM ITL p95 (ms) | TGI E2E p95 (ms) | vLLM E2E p95 (ms) | Winner (per metric) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | false | | | | | — | — | | | |
| 1 | true | | | | | | | — | — | |
| 2 | ... | | | | | | | | | |

**`operational_matrix.md`** — the ops surface, one column per runtime:

| Axis | TGI | vLLM |
|---|---|---|
| Launcher flag count in the pinned config | | |
| Image size (`docker images` line) | | |
| Boot time (from `docker run` to first successful smoke test) | | |
| Peak VRAM under load | | |
| Prometheus metric names used | | |
| API surface pinned (which endpoints your workload uses) | | |
| Structured-output support (grammar / JSON schema) | | |
| LoRA-adapter surface (from chapter 3) | | |
| Multi-node support | | |
| Documentation depth (subjective, one paragraph in report) | | |
| Failure signature under overload | | |

**`saturation_analysis.md`** — for each runtime, at what concurrency does p95 TTFT cross 500 ms? At what concurrency does throughput plateau? Draw the two curves (throughput-vs-concurrency and TTFT-p95-vs-concurrency) as ASCII plots or link to a rendered PNG.

### Part H — The write-up

Author `report.md` with the following sections:

1. **Setup.** Model, quantization, hardware, both runtime versions + launch commands, workload description, parity-check result.
2. **Head-to-head latency and throughput.** From `latency_matrix.md`. Where does each runtime win? Do the numbers land where chapter 5's decision matrix predicts, or does your case diverge?
3. **Saturation and capacity.** From `saturation_analysis.md`. Which runtime keeps its p95 SLO longer as concurrency climbs?
4. **Operational and observability surface.** From `operational_matrix.md`. Which is easier to operate: flag surface, metrics naming, log discoverability, failure signature under overload?
5. **API-surface differences that would leak to callers.** From Part D's parity check. Does either runtime's `/generate` or `/v1/chat/completions` handle stop-sequences, `n`, `logprobs`, or streaming SSE framing in a way that would require client-side changes to swap runtimes?
6. **Recommendation.** For **this specified serving scenario**, which runtime do you ship?
   > "A single-tenant enterprise deployment of a support chatbot. Target GPU: 1× L40S (48 GB VRAM) per replica. Traffic mix: 200 tokens in / 200 tokens out, 8 concurrent requests p95 steady-state, peaks to 32. Latency SLO: p95 TTFT < 400 ms streaming; p95 E2E < 6 s non-streaming for the 5% of long-response requests. Team runs standard Prometheus + Grafana. No requirement for multi-LoRA in the current release; roadmap includes it in the next quarter."
7. **Runbook stub.** A one-page draft that the on-call for the chosen runtime would use: launch command, health check, five smoke tests, key metrics, SLOs from Part 6, rollback command, common failure signatures with first-response mitigations. This slots into chapter 7's full runbook template.
8. **What surprised you.** One paragraph.
9. **Reproducibility notes.** What is pinned, what would need to change to rerun on a newer version of either runtime.

### Part I — Directory layout

```
outputs/exercise-03/
├── requirements.txt
├── pins.md
├── artifacts/                       # symlink or copy of the quantized checkpoint
├── deploy/
│   ├── tgi.launch.sh
│   ├── tgi.boot.md
│   ├── tgi.smoke.md
│   ├── tgi.metrics_head.txt
│   ├── vllm.launch.sh
│   ├── vllm.boot.md
│   └── vllm.smoke.md
├── parity/
│   ├── parity_check.py
│   └── parity_table.md
├── workload/
│   ├── workload.jsonl
│   └── README.md
├── run_loadtest.sh
├── results/
│   ├── tgi/
│   │   ├── c1-stream-false/
│   │   ├── c1-stream-true/
│   │   ├── ...
│   │   └── gpu.log
│   └── vllm/
│       └── (parallel layout)
├── build_matrices.py
├── latency_matrix.md
├── operational_matrix.md
├── saturation_analysis.md
└── report.md
```

## Starter guidance

- **Same GPU, sequential runs.** Do not try to co-locate both runtimes; use `docker stop` between them. If you had to use two different GPUs, note it prominently — the comparison is confounded.
- **Fix the dtype up front.** TGI's `--dtype` and vLLM's `--dtype` both default to auto-detection, but the auto-detected dtype can differ between runtimes for the same GPTQ file. Pin both explicitly (`float16` or `bfloat16`) so the compute path is comparable.
- **Chat template correctness is a common trap.** TGI's `/v1/chat/completions` and vLLM's `/v1/chat/completions` both apply the tokenizer's `chat_template`. If the tokenizer files bundled with the artifact are missing or stale, one runtime may fall back differently. Parity check (Part D) catches this before it becomes a "vLLM is 100 ms faster" false conclusion.
- **`--max-total-tokens` and `--max-model-len` mean the same thing but count differently in edge cases.** TGI's `--max-total-tokens` bounds input + output; vLLM's `--max-model-len` is the model's context window. Set both so a 3072-token input + 256-token output fits comfortably in both, and cap the workload's prompt distribution below that.
- **Streaming SSE framing differs.** TGI's `/generate_stream` emits `{token, ...}` frames; vLLM's `/v1/chat/completions?stream=true` emits OpenAI-style `data: {"choices":[{"delta":{"content":"..."}}]}` frames. `llmperf`'s abstraction hides most of this, but a k6 script has to handle both. Use the OpenAI-compatible endpoint on both runtimes for the head-to-head to keep the framing identical.
- **Prefix-caching state must match.** vLLM's `--enable-prefix-caching` and TGI's `--enable-prefix-caching` are similar but not bit-identical. Turn both on for a "realistic production" comparison, or turn both off for a "pure kernel comparison" — do not mix.
- **Watch queue metrics during the ramp.** If `tgi_queue_size` or `vllm:num_requests_waiting` is climbing without bound at concurrency 32, throughput / latency measurements from that level are the saturation dynamics, not steady-state performance. Report the saturation point; do not treat pathological numbers as the "real" comparison.
- **Look at the `/info` and `/v1/models` outputs before every timing run.** If the runtime silently reloaded a different artifact after an error, you will spend hours debugging a latency delta that was really a "you were serving the base model" bug.
- **Version-pin, then read the release notes.** Both runtimes ship weekly-ish. A finding from six months ago may already be moot in the current version. Pin your version and cite the release notes in `pins.md`.

## Acceptance criteria

You have completed the exercise when:

- `pins.md` records exact versions for both runtimes, the model + revision, and the load-test tool.
- Both `deploy/tgi.smoke.md` and `deploy/vllm.smoke.md` show all five smoke tests passing.
- `parity/parity_table.md` shows greedy-decoding agreement between the two runtimes on the parity prompts (or documents the divergence and its root cause).
- `workload/workload.jsonl` is identical between runs (`sha256sum` matches in both `results/*/README.md`).
- `results/tgi/` and `results/vllm/` contain complete load-test output for every concurrency × streaming combination.
- `latency_matrix.md`, `operational_matrix.md`, `saturation_analysis.md` are populated with real numbers.
- `report.md` has all nine sections; the recommendation is defended by concrete numbers and the runbook stub is executable — not aspirational.

## Stretch goals

- **Add a third runtime.** Repeat the head-to-head with SGLang, or with TensorRT-LLM if you have the Nvidia stack. Add a column to `latency_matrix.md`. Chapter 5's "run both and pick on numbers" advice generalizes to three.
- **Test the multi-LoRA path.** Attach the three adapters from exercise 02 to both TGI (`--lora-adapters`) and vLLM (`--enable-lora`). Rerun the ramp with the workload's adapter distribution. This is the "which runtime does multi-LoRA better today" question in numbers.
- **Test structured outputs.** For a JSON-schema-constrained generation workload (grammar-guided decoding), rerun the ramp with `guided_json` on vLLM and TGI's grammar option. Report tokens/sec penalty on each.
- **Chunked-prefill and Sarathi-style scheduling.** If your vLLM version supports chunked prefill (`--enable-chunked-prefill`), rerun the long-context slice of the workload with it on and off. Report the TTFT vs. ITL trade-off. Cite the Sarathi-Serve paper in the write-up.
- **Simulate a rolling deploy.** Run one workload continuously; halfway through, `docker stop` the runtime and start the other. Measure how long errors persist and what the queue backlog does. Feeds into the runbook's rollback story.
- **KV-cache dtype ablation.** For the runtime that supports it, rerun with `--kv-cache-dtype fp8`. Report throughput, VRAM, and quality delta (via a mod-108 fast lane against the endpoint).
- **Re-run the release gate.** Point `lm-evaluation-harness`'s `local-chat-completions` backend at each endpoint and run mod-108's fast lanes. Confirm both runtimes reproduce the training-time quality numbers on the served artifact; disagreement means one runtime is subtly generating differently.
- **Cost projection.** Convert the throughput number into dollars-per-million-tokens using the price of the GPU you tested on. Compare against a managed-API baseline for the same tier. Chapter 7's cost model is the reference.
