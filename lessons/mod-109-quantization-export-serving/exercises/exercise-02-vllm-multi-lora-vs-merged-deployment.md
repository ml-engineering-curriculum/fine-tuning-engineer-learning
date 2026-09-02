# exercise-02: vLLM Multi-LoRA vs. Merged Deployment

**Estimated effort:** 3 hours
**Companion chapters:** [`03-merge-vs-multi-lora-serving.md`](../03-merge-vs-multi-lora-serving.md), [`04-vllm-deployment.md`](../04-vllm-deployment.md)

## Objective

Stand up vLLM in **two configurations** — one **merged single-tenant** deployment and one **multi-LoRA** deployment with 3+ adapters attached to a shared base — put both under an identical mixed workload, and produce the numbers that defend a merge / multi-LoRA decision for a specified serving scenario. The write-up must include a comparison matrix (per-request latency, throughput, VRAM, cold-start, config surface), the routing pattern chosen, and a defended recommendation that a platform team could execute.

The point of this exercise is the trade-off in chapter 3 turned into evidence. Merging is simpler and slightly faster per request; multi-LoRA is the only path if N > 3 adapters or if tenants churn. The exercise is where you feel the actual latency delta from LoRA-at-inference vs. merged, and the actual VRAM savings from one base + N adapters vs. N merged models.

## Prerequisites

- Three or more LoRA adapters on the **same base model** at the same revision. Options:
  - Your own — take intermediate checkpoints from your mod-104 / mod-105 / mod-106 runs (SFT adapter, DPO adapter, safety adapter) if they share a base.
  - Train three cheap 100-step LoRAs on different subsets of your SFT data (e.g., "code", "support", "chat"). Rank 8 or 16 is plenty for this exercise.
  - Public — three permissively-licensed adapters over the same base (e.g., a Llama-3.1-8B-Instruct base with community adapters). Verify each `adapter_config.json`'s `base_model_name_or_path` matches.
- The merged BF16 checkpoint from exercise 01 (`artifacts/bf16`) *or* the specific merged model for the adapter you will treat as "workhorse."
- A vLLM install that supports multi-LoRA on your base. A recent vLLM (pinned in `pins.md`) is the reference; verify with the current release notes that your (base × quantization) combination supports `--enable-lora`.
- 1 GPU with enough VRAM to hold the base at BF16 (or your target quantization) plus the KV cache plus the adapter slots. One A100 40 GB, one H100 80 GB, or one L40S 48 GB is comfortable for a 7B base.
- A load-test tool — `vllm bench serve`, `llmperf`, or `k6` with a small custom LLM script. The reference workload uses `vllm bench serve` because it is already tuned to the OpenAI-compatible surface.
- Python 3.10+, `openai` Python client for smoke tests, `curl` / `jq` for the API-shape checks.

## Requirements

### Part A — Pin the environment

Author `requirements.txt` and `pins.md` capturing at minimum:

- `vllm` (record the exact version and the CUDA build).
- `torch`, `transformers`, `peft`, `accelerate`.
- The base model id + revision hash used across both deployments.
- Each adapter's path + `adapter_config.json` hash + trained-against base revision.
- Hardware spec block: GPU model, VRAM, CUDA / driver versions, host NVMe throughput (matters for adapter cold-start numbers).

Verify the vLLM install with `vllm serve --help | head` and record the flag surface — chapters 4's flag names shift version to version, and the runbook will need to reference the exact ones your install accepts.

### Part B — Prepare and register the adapters

Confirm each adapter is loadable and its `base_model_name_or_path` matches the base you plan to serve. Author `verify_adapters.py` that:

1. Loads each adapter via `PeftModel.from_pretrained(base, adapter_path)`.
2. Runs one greedy generation on the same fixed prompt.
3. Confirms the output differs from the base (else the adapter is not actually applying).
4. Records `adapter_id → sha256(adapter_model.safetensors)` in `adapter_registry.md`.

If any adapter fails the base-revision check, halt and fix — this is exactly the silent-quality-drift failure mode chapter 3 warned about.

### Part C — Deployment 1: merged single-tenant

Merge your **workhorse adapter** into the base and serve the merged model. Options for the merged artifact:

- Reuse `artifacts/bf16` from exercise 01, or
- Do `peft.merge_and_unload()` on the workhorse adapter and save to `artifacts/merged_workhorse/`.

Launch vLLM:

```bash
vllm serve artifacts/merged_workhorse \
  --served-model-name workhorse \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --port 8000
```

Capture the launch command verbatim in `deploy/merged.launch.sh`. Run the five smoke tests from chapter 4 (`/v1/models`, coherent completion, chat-template alignment, `/metrics` scraping, 2× expected concurrency does not degrade). Save the outputs to `deploy/merged.smoke.md`.

### Part D — Deployment 2: multi-LoRA on the shared base

Launch a second vLLM instance (on the same GPU if VRAM permits — otherwise on a second GPU, and note the swap in `pins.md`) with the same base + N adapters attached:

```bash
vllm serve <BASE_ID> \
  --revision <BASE_REV> \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank <MAX_R> \
  --max-cpu-loras 16 \
  --lora-modules \
    workhorse=adapters/workhorse \
    tenant-a=adapters/tenant-a \
    tenant-b=adapters/tenant-b \
  --port 8001
```

`<MAX_R>` is the maximum rank across your adapters. Adjust `--max-loras` to at least the number of adapters you plan to have concurrently active in one batch step.

Capture the launch command in `deploy/multi_lora.launch.sh`. Run the same five smoke tests plus a sixth: **per-adapter routing works.** For each registered adapter, send a `/v1/chat/completions` request with `"model": "<adapter_name>"` and confirm the output differs from the base and matches what you saw in Part B's verification. Save to `deploy/multi_lora.smoke.md`.

### Part E — Pre-warm the multi-LoRA deployment

At startup, adapters are cold. Chapter 3 covered why pre-warming matters. Author `prewarm.sh` that, before the load test, issues one one-token generation per registered adapter against the multi-LoRA deployment. Log each request's TTFT — the first request per adapter should be noticeably slower (~100–500 ms extra) than steady-state; that is the number the "pre-warm on deploy" runbook step is defending against.

Include the pre-warm output in `deploy/multi_lora.prewarm.md`.

### Part F — The shared workload

Define one workload used against both deployments. Concrete shape:

- **Prompt corpus.** 500+ prompts drawn from ShareGPT (`anon8231489123/ShareGPT_Vicuna_unfiltered`), your held-out set, or a mix. The prompts must be identical between the two runs — a run-to-run mix change is a workload change, not a config change, and will invalidate the comparison.
- **Distribution over adapters** (multi-LoRA only). Choose a plausible split: 60% workhorse, 20% tenant-a, 20% tenant-b, or whatever matches your scenario. For the merged deployment, every request just goes to the merged model.
- **Load profile.** Concurrency ramp: 1 → 2 → 4 → 8 → 16 → 32 concurrent in-flight requests. Hold each level for 60 seconds after a 30-second warm-up.
- **Output length target.** `max_tokens=256` with `temperature=0.0` for reproducibility.

Author `workload/workload.jsonl` (the prompt set + adapter routing per row), `workload/README.md` (source, filters, adapter distribution), and `run_loadtest.sh` (calls into `vllm bench serve` or `llmperf` with the pinned parameters).

### Part G — Run the load test

Run against each deployment. Capture:

- Per-concurrency-level: throughput (req/s and tok/s), p50 / p95 / p99 TTFT, p50 / p95 / p99 ITL, p50 / p95 / p99 E2E, GPU memory utilization at steady state.
- Per-adapter (multi-LoRA only): the same metrics stratified by routed adapter. Adapter-specific TTFT p95 is what surfaces adapter cold-start regressions and per-adapter batching effects.
- Any failed requests (5xx, timeouts). Zero is the target; investigate anything non-zero before treating the run as valid.

Save the raw output under `results/merged/` and `results/multi_lora/`. Include the tool version + invocation in each results directory's `README.md`.

### Part H — Compute the comparison matrices

Author `build_matrices.py` that reads both result trees and produces:

**`latency_matrix.md`** — per-concurrency-level latency and throughput:

| Concurrency | Merged tok/s | Multi-LoRA tok/s | Merged TTFT p95 (ms) | Multi-LoRA TTFT p95 (ms) | Merged ITL p95 (ms) | Multi-LoRA ITL p95 (ms) | Multi-LoRA overhead (%) |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |
| 2 | | | | | | | |
| ... | | | | | | | |

**`system_matrix.md`** — resource footprint:

| Metric | Merged | Multi-LoRA |
|---|---|---|
| Base VRAM at boot (GB) | | |
| Peak VRAM under load (GB) | | |
| KV-cache pool available (GB) | | |
| Disk footprint (GB) | | (base + N adapters) |
| Config-file surface (line count in launch.sh) | | |
| Adapter cold-start p95 (ms, first request per adapter) | N/A | |
| Steady-state adapter TTFT p95 (ms) | | |

**`scaling_projection.md`** — extrapolate to N adapters. Two rows: (a) memory required if you merged N models; (b) memory required if you served one base + N adapters. Include the break-even N for your target GPU.

### Part I — The write-up

Author `report.md` with the following sections:

1. **Setup.** Base + revision, adapters + revisions, hardware, vLLM version, workload description, adapter distribution.
2. **Latency and throughput comparison.** From `latency_matrix.md`. Where does multi-LoRA's overhead show up — TTFT, ITL, or throughput plateau? Is the overhead consistent with the chapter 3 "5–15%" band, or does your workload land outside it? If outside, hypothesize why.
3. **VRAM and disk comparison.** From `system_matrix.md`. Which deployment costs more in absolute terms, and which scales better as N grows?
4. **Cold-start observations.** From Part E. How big was the first-request tax, and did pre-warming eliminate it?
5. **Routing design.** Which of the four chapter-3 routing patterns (header-based, classifier-based, fallback, canary) fits the specified scenario? Sketch the request path.
6. **Recommendation.** For **this specified serving scenario**, which configuration do you ship?
   > "A B2B SaaS product where each customer gets their own small fine-tune on top of a shared 7B base. Today there are 12 customers; the product team expects 30–60 within a year. Latency SLO: p95 TTFT < 500 ms at expected steady-state load of 4 concurrent requests per replica. Hardware budget: one L40S (48 GB VRAM) per replica; the platform team is comfortable running 4–8 replicas."
7. **What breaks the recommendation.** One paragraph. What change in the scenario (customer count, SLO, hardware, base size) would flip your answer?
8. **Reproducibility notes.** What is pinned, what is not, and what would need to change to rerun this exercise on a different vLLM version.

### Part J — Directory layout

```
outputs/exercise-02/
├── requirements.txt
├── pins.md
├── adapter_registry.md
├── verify_adapters.py
├── artifacts/
│   ├── merged_workhorse/
│   └── (adapters symlinked or copied into adapters/)
├── adapters/
│   ├── workhorse/
│   ├── tenant-a/
│   └── tenant-b/
├── deploy/
│   ├── merged.launch.sh
│   ├── merged.smoke.md
│   ├── multi_lora.launch.sh
│   ├── multi_lora.smoke.md
│   └── multi_lora.prewarm.md
├── prewarm.sh
├── workload/
│   ├── workload.jsonl
│   └── README.md
├── run_loadtest.sh
├── results/
│   ├── merged/
│   └── multi_lora/
├── build_matrices.py
├── latency_matrix.md
├── system_matrix.md
├── scaling_projection.md
└── report.md
```

## Starter guidance

- **One base, one revision, three adapters.** The most common way this exercise goes wrong is subtle base-revision drift: adapter A was trained against `Llama-3.1-8B abc123`, adapter B against `def456`, and the multi-LoRA server pins one or the other. Serving is technically legal (`W + BA` computes) but quality drifts silently. Verify in Part B; refuse to proceed on mismatch.
- **Same workload file between runs.** Do not regenerate the prompt corpus between the merged and multi-LoRA runs. A different prompt distribution swamps the config difference you are trying to measure. `sha256(workload.jsonl)` should be identical in both `results/*/README.md` outputs.
- **`--max-loras` matters.** If your workload has all three adapters active in the same batch step and `--max-loras=2`, one adapter gets queued out and its TTFT looks catastrophic. Set `--max-loras` to at least the concurrent-adapter count you expect.
- **Prefix caching is a hidden lever.** If your workload has a shared system prompt (common for chat), `--enable-prefix-caching` gives both deployments a large TTFT win — but the win concentrates on the merged deployment because the multi-LoRA path historically has more caveats around per-adapter cache invalidation. Verify both deployments actually benefit before over-crediting one.
- **`vllm bench serve` needs the OpenAI-compatible surface.** Point it at `http://localhost:8000/v1` (merged) and `http://localhost:8001/v1` (multi-LoRA), with the adapter name in the `--model` argument for multi-LoRA runs.
- **Pre-warm before every timing run.** A cold `/v1/load_lora_adapter` blows up p99 TTFT for the first burst; do not accidentally measure that as steady-state latency.
- **Check `vllm:num_requests_waiting` at every concurrency level.** If it is climbing without bound, the deploy is saturating and later measurements are meaningless. Reduce the load or resize the GPU before continuing.
- **Do not compare across GPUs.** If you had to run merged on one GPU and multi-LoRA on another, note it prominently in the report — the comparison is now confounded by hardware.

## Acceptance criteria

You have completed the exercise when:

- `pins.md` records the exact vLLM version, base model + revision, and each adapter's path + hash.
- `verify_adapters.py` output shows all adapters load, all applied, all match the intended base revision.
- Both deployments boot cleanly and pass the five (merged) or six (multi-LoRA) smoke tests.
- `prewarm.sh` output shows the first-request-per-adapter tax quantified and eliminated on subsequent requests.
- `workload/workload.jsonl` is identical (by hash) between both runs and has ≥500 prompts.
- `results/merged/` and `results/multi_lora/` contain full load-test output at every concurrency level.
- `latency_matrix.md`, `system_matrix.md`, `scaling_projection.md` are populated with real numbers (no TODO cells).
- `report.md` has all eight sections; the recommendation is defended by concrete numbers from your own runs and names the flip conditions.

## Stretch goals

- **Add a fourth adapter mid-run.** Use `POST /v1/load_lora_adapter` to register a new adapter to the running multi-LoRA server, then route a burst of traffic to it. Measure: adapter-add latency, first-request TTFT, and whether the other adapters' latencies degraded during the load. This exercises the adapter-warehouse pattern from chapter 3.
- **Quantized base + multi-LoRA.** Replace the BF16 base with the GPTQ-4bit artifact from exercise 01 in the multi-LoRA deployment. Verify vLLM's current-version support for LoRA-over-GPTQ on your base — if unsupported, the finding itself is the point. Rerun the load test and report the memory savings and any quality delta.
- **Canary routing.** Register two versions of the workhorse adapter (`workhorse-v1`, `workhorse-v2`). Route 95%/5% via a proxy in front of vLLM. Measure the tail latency during the canary and confirm both adapters receive their share.
- **Adapter eviction stress test.** Set `--max-loras=2` and `--max-cpu-loras=3` with 5 adapters registered. Send a workload that touches all 5 adapters in random order. Report how often adapters evict, the tail-latency impact, and whether raising `--max-cpu-loras` fixes it.
- **Head-to-head with TGI multi-LoRA.** Run the same workload through TGI's `--lora-adapters` path (chapter 5). Add a column to `latency_matrix.md` for TGI. This is a natural pre-work for exercise 03.
- **Cost projection.** Extend the recommendation with dollars-per-million-tokens under the specified traffic mix and compare against a merged-per-tenant baseline (N replicas × per-replica cost). Chapter 7's cost model is the reference.
- **Re-run the release gate.** Point mod-108's fast lanes at each deployed endpoint (via `lm-evaluation-harness`'s `local-chat-completions` backend) and confirm the served-artifact quality matches the training-time numbers. This closes the loop chapter 4 opened: the served artifact is not the training artifact.
