# Latency, Throughput, Cost, and the Serving Runbook

**Estimated read:** 40 minutes.
**You should be able to, by the end:** distinguish the three latency numbers your product cares about (TTFT, ITL, end-to-end) and where each comes from; reason about throughput under continuous batching and the KV-cache memory that bounds it; compute a rough dollars-per-million-token number for a self-hosted serving deployment; run a controlled load test against your artifact and read the results; and write a serving runbook — SLOs, capacity, deploy steps, rollback, on-call playbook — that the next engineer can execute cold.

## Why this chapter matters

The previous six chapters produced a serving artifact: a quantized fine-tuned model behind vLLM, TGI, or `llama.cpp`, possibly with LoRA adapters attached. That artifact has to answer HTTP requests at a specific latency, at a specific throughput, at a specific cost, and it has to keep doing that while on-call gets paged for other things.

This chapter is where the fine-tuning engineer's work meets the SRE's. You do not need to own the entire platform — mod-111 covers that altitude — but you do need to hand your quantized artifact off with enough documentation that the next person can run it. This chapter is the shape of that documentation.

## The three latency numbers

Product SLOs for a chat model are almost always written as one number ("p95 latency < 400 ms"), but that number is a composition of three:

**Time to first token (TTFT).** Wall clock from request submitted to the first output token. Includes queue wait, prefill (running the model over the full prompt), and any preamble the server needs to do. On a healthy vLLM / TGI deploy for a 7B model at moderate load, this is tens to a couple hundred milliseconds; on saturated deploys it can climb into seconds.

**Inter-token latency (ITL) / time per output token (TPOT).** Wall clock between successive output tokens. Depends on batch composition (a bigger batch is slower per token, faster per request) and model size. Usually more stable than TTFT and closer to a physical constant per hardware.

**End-to-end (E2E) latency.** TTFT + (ITL × output_tokens). This is what the user actually experiences on a non-streaming request. On a streaming request, TTFT is what they perceive as "responsiveness" and ITL is what they perceive as "the text appears smoothly."

The product SLO usually lives on one of these three:

- Chatbot with streaming: TTFT is the SLO (users tolerate slower generation once tokens start).
- Chatbot without streaming: E2E is the SLO (they see nothing until the whole response arrives).
- Batch generation (offline evaluation, dataset generation): throughput matters more than any latency; measure tokens/sec.
- Structured output for tool calls: E2E on typical output length is the SLO (a tool call is short but blocking).

Write the runbook to whichever number the product is measuring on. Do not report all three when the product only cares about one — it dilutes attention on the metric that matters.

## The throughput model — continuous batching, prefix caching, KV cache

Modern inference stacks (vLLM, TGI) do not batch by fixed slots. Continuous batching admits new requests each decoding step, evicts finished ones, and keeps the GPU busy. Two effects:

- **Throughput scales with concurrent load** up to the point the KV cache saturates.
- **Latency degrades gracefully** as saturation approaches — TTFT climbs, ITL rises modestly — until saturation, at which point either the queue grows without bound or requests get rejected.

The bounding resource in almost every deployment is the **KV cache**, not compute. For each token in a request's context, the model stores a `[num_layers, num_kv_heads, head_dim]` tensor pair (K and V) in cache. Rough memory:

```
KV_bytes_per_token ≈ 2 (K + V) × num_layers × num_kv_heads × head_dim × dtype_bytes
```

For Llama-3.1-8B at BF16 KV, this is about **131 KB per token**. For a 4096-token context, one request holds ~512 MB of KV cache. Multiply by concurrent-request count.

Reducing KV-cache pressure is what unlocks throughput:

- **`--kv-cache-dtype fp8`** roughly halves it, at a small quality cost.
- **Grouped-query attention (GQA)** in the model architecture reduces `num_kv_heads` at the source; most modern instruct models use it.
- **Prefix caching** (`--enable-prefix-caching`) reuses KV state across requests with the same prompt prefix — enormous win for RAG or system-prompt-heavy workloads.
- **Smaller `--max-model-len`** caps the reservation per request. Set to real p99 context, not architectural max.

**The runbook should include the number of concurrent requests your KV budget supports** at the chosen `--max-model-len`. This is the capacity number product planners will ask for.

## The load-test pattern

Every deploy gets load-tested before it takes production traffic. The pattern that fits vLLM / TGI:

**Tool.** `vllm bench serve` (ships with vLLM), `llmperf` (Anyscale, cross-runtime), or `k6` / `locust` with a custom LLM client. Chapter 4 / 5 exercises use one of these.

**Workload shape.** Not a synthetic "one prompt, N times" run — that measures nothing useful. Use a realistic mix drawn from:

- Real production traffic samples (best), or
- Your task-specific held-out set from mod-108 (good), or
- ShareGPT / a public conversation corpus (adequate baseline).

Specify prompt length distribution and expected output length distribution. A 7B chatbot with mean 200 tokens in / 200 tokens out behaves very differently from a summarizer with 3000 tokens in / 200 tokens out, and both differ from a code model with 500 tokens in / 800 tokens out.

**Load ramp.** Start at 1 concurrent request; ramp to 2, 4, 8, 16, 32, ... until either TTFT breaks your SLO or throughput plateaus. Two curves come out of this:

- Throughput vs. concurrency (should plateau; the plateau is your capacity).
- p95 TTFT vs. concurrency (should stay flat, then bend upward; the bend is your knee).

**Reporting.** For each concurrency level: throughput (req/s and tok/s), p50 / p95 / p99 TTFT, p50 / p95 / p99 ITL, p50 / p95 / p99 E2E, GPU memory utilization at steady state.

The runbook's capacity number is: **"one replica handles up to N concurrent requests before p95 TTFT exceeds M ms."** Everything else at deploy time is sized off this.

## The cost model

Dollars per million tokens is the vocabulary product managers use to compare self-hosted vs. managed APIs. Rough shape for a self-hosted deployment:

```
$/hour of GPU × replica count × 3600
  ÷ (throughput_tokens_per_second × 3600 × 1_000_000)
=  ($/hour) / (throughput_tokens_per_second × 3.6)
```

Concrete: one H100 at cloud pricing (~$3–8/hour depending on provider and commitment) serving Llama-3.1-8B at (say) 2000 tokens/second aggregate throughput:

- $5/hr ÷ (2000 × 3.6) ≈ **$0.69 per million tokens.**

Compare against a managed API (as of 2026, order-of-magnitude only — do not treat this as pricing advice):

- OpenAI GPT-4o-mini class: ~$0.15–0.60 per million tokens for input, similar or more for output.
- Anthropic Claude Haiku class: comparable range.
- Anthropic Claude Sonnet class: 5–10× higher.

**Self-hosting wins on cost per token when utilization is high.** A GPU running at 5% utilization is a very expensive per-token cost. A GPU running at 70% utilization is competitive with or cheaper than managed APIs for the same tier of model. Utilization — which the load test measures — is the parameter that decides.

Other costs that the runbook should quantify:

- **Cold storage** of model artifacts (S3 / GCS): pennies per GB per month, non-trivial for a 70B model deployment across many quantization variants.
- **Data egress** if the serving pod pulls model weights across a region boundary.
- **On-call time.** Not a dollar line item, but a real budget. Every operational quirk in the deploy is a fraction of an engineer's week; the runbook is how you reduce that.

<!-- needs-research: cloud GPU pricing changes constantly and varies enormously by provider, commitment, and market conditions. The numbers here are directional; the runbook should cite the current per-provider quote and the date of the quote. -->

## Latency vs. throughput vs. cost — the trade-off

The three axes trade against each other:

- **Bigger batch size** → higher throughput, higher ITL (per-token slower), lower cost per token.
- **Smaller batch size** → lower ITL, lower throughput, higher cost per token.
- **More concurrent requests** → higher throughput up to KV-cache saturation, higher TTFT (queue wait grows).
- **Larger context reservation** (`--max-model-len` high) → more capacity for long-context requests, fewer concurrent requests fit.
- **More aggressive quantization** → smaller model = more fits in VRAM = higher throughput and cheaper, at some quality cost.
- **More GPUs (tensor-parallel)** → lower latency on very large models, higher cost, more operational complexity.

There is no free optimum. The right operating point is the one where the SLO is met and cost per served token is acceptable. Chapter 7's runbook makes this explicit — SLO + cost + capacity, in that order.

## Autoscaling and capacity planning

Once you know one replica's capacity (from the load test), autoscaling is straightforward: scale replicas to meet forecasted request-per-second load, with headroom for burst.

Two common autoscaling signals for LLM serving:

- **Queue depth** (`vllm:num_requests_waiting` / `tgi_queue_size`). Direct signal of saturation. Scale up when it climbs, scale down when it stays at zero.
- **CPU / GPU utilization.** Indirect signal; useful as a fallback. GPU utilization above 80% for a sustained window = need more capacity.

Avoid autoscaling on request-per-second alone — a low RPS of very long-prompt requests can saturate a replica faster than a high RPS of short ones.

Scale-up latency matters: a replica takes 30–120 seconds to become ready (image pull + model load + first-request warmup). If your traffic can double in 30 seconds, static overprovisioning during peak hours is often more responsive than autoscaling.

## The runbook

The artifact this chapter is teaching you to produce. Structure it into the following sections, one page each:

### 1. Identity

- Model repo + revision hash.
- Fine-tune lineage (which SFT / DPO / safety adapter, from which training run).
- Quantization method + version + config hash.
- Serving artifact hash (safetensors / GGUF SHAs).
- Tokenizer version, chat-template version.

If a downstream engineer cannot uniquely identify the artifact from this section, add fields until they can.

### 2. Deploy command

The exact invocation of `vllm serve` / `docker run tgi` / `llama-server`. Every flag. No implicit defaults. A copy-paste that reproduces the deploy.

### 3. SLOs

- Which latency number (TTFT / ITL / E2E) is the SLO.
- The target (e.g., "p95 TTFT < 250 ms").
- The acceptable degradation window (e.g., "p99 TTFT < 800 ms; > 800 for 5 minutes fires a page").
- Product-side context: which product surface consumes this endpoint.

### 4. Capacity

- Per-replica: concurrent requests supported at the SLO knee.
- Per-replica: KV-cache budget in tokens.
- Cluster: number of replicas, headroom for burst.
- Autoscaling policy (metric, target, min/max replicas, cooldown).

### 5. Quality contract

- Release-gate delta from full-precision (per task from mod-108's suite).
- Tolerance thresholds — the numbers that decide the artifact is still shippable.
- Link to the mod-108 release-gate report for this artifact.

### 6. Observability

- Prometheus scrape URL + the metric names that matter.
- Log destination + fields worth searching on (request id, model name / adapter, TTFT bucket, error class).
- Dashboards (URLs) with the SLO plot and the throughput plot.
- Alert rules (queue depth, p99 TTFT, GPU OOM, replica restart rate).

### 7. Failure playbook

For each of the top-N failure modes seen or plausible, a numbered response:

- Replica OOM at peak → check `--gpu-memory-utilization`, `--max-model-len`, KV-cache dtype; roll back to prior config if a recent change caused it.
- p99 TTFT climbing → check queue depth; scale up replicas or reduce concurrency admission.
- Adapter not loading → check adapter registry auth; check `adapter_config.json` base-revision match.
- Structured-output failures → check `guided_json` schema; check model produces JSON at all.
- Silent quality regression → re-run the mod-108 fast-lane suite against the deployed artifact; compare to shipped baseline.

Every entry names the exact command to check status and the exact command to remediate. No prose. The person reading this at 3 AM has ninety seconds.

### 8. Rollback

- The previous artifact's identity (hash, tag).
- The exact command to point traffic at the previous artifact.
- How to verify the rollback took effect (smoke test, dashboard signal).
- The window in which rollback is safe (e.g., "prior artifact is retained for 30 days").

### 9. Change log

Every material change to the deploy, dated: quantization method changed, `--max-model-len` bumped, adapter added, GPU class changed, autoscaling threshold moved. This is where "why does the throughput look different since Tuesday" gets answered.

### 10. Handoff to mod-111

What this runbook expects the platform layer (mod-111) to own:

- Adapter registry integration (how new adapters get pushed and warmed).
- CI-triggered redeploy on new release-gate pass.
- Model-card lineage (which SFT run → which quantization → which deploy).
- Governance sign-off on quality contract changes.

If the platform owns these, this runbook has less content; if the platform does not, this runbook explicitly documents what is manual.

## The KV cache math worked through

Concrete example that ties throughput to physical parameters. Llama-3.1-8B specifications:

- 32 layers.
- 8 KV heads (GQA).
- 128 head dim.
- 2 bytes per activation at BF16 KV.

Per-token KV: `2 × 32 × 8 × 128 × 2 = 131 072 bytes` = 128 KB.

On an H100 with 80 GB HBM, model weights take:

- BF16: ~14 GB.
- AWQ / GPTQ 4-bit: ~4 GB.

At `--gpu-memory-utilization 0.90` (72 GB usable) and framework overhead (say 6 GB), the KV budget:

- BF16 base: `72 − 14 − 6 = 52 GB` KV cache = ~407k tokens of context capacity.
- 4-bit base: `72 − 4 − 6 = 62 GB` KV cache = ~485k tokens.

At `--max-model-len 4096`, each request reserves worst-case 4096 × 128 KB = 512 MB. Concurrent-request ceiling:

- BF16: `52 GB / 512 MB` ≈ 104 concurrent requests.
- 4-bit: `62 GB / 512 MB` ≈ 124 concurrent requests.

Reality is less than that: prefix caching adds requests without full KV reservation; short prompts free capacity; long-context bursts collapse it. The formula is the ceiling, not the operating point. **The load test measures the operating point.**

If you want to double context to 8192 without changing hardware:

- Concurrent-request ceiling roughly halves.
- Or: turn on `--kv-cache-dtype fp8`, which roughly halves the per-token cost, and restore the concurrent-request budget.

This is the math the runbook's capacity section defends.

## The SLO — quality contract linkage

The runbook has two SLOs, and they interact:

**Latency SLO** — "p95 TTFT < 250 ms." Measured continuously. If it breaks, the fix is capacity (add replicas, tune batching).

**Quality SLO** — "release-gate suite (mod-108) within tolerance of full-precision baseline." Measured at deploy time and periodically re-verified. If it breaks, the fix is not capacity — it is either a quantization / adapter / config issue or an actual model regression.

The trap: latency-first thinking treats the quality SLO as a static one-time check ("we ran the eval when we quantized"). In fact, quantized artifacts can silently regress on production traffic patterns that differ from the calibration data. Re-run the fast-lane suite against the served endpoint on a schedule (weekly is a starting point; nightly is better) — chapter 6 of mod-108's release-gate manifest is what makes this cheap.

## Handoff to mod-111

Chapter 7's runbook is one artifact. Mod-111 wraps it in the platform: the runbook template becomes the deploy CRD, the release-gate manifest becomes the CI gate, the adapter registry becomes a service, the change log becomes the deploy history, and the on-call rotation is where the failure playbook is exercised.

What mod-111 needs from you: the runbook, structured as above, versioned alongside the artifact. The platform can automate almost everything if the underlying deploy is documented well. It cannot automate anything if the deploy is a shell script in someone's home directory.

## Summary

- The three latency numbers a serving runbook has to distinguish: TTFT (queue + prefill), ITL / TPOT (per-token decode), and E2E (TTFT + ITL × output). The product SLO usually lives on one of these; write to that one.
- Throughput on modern inference stacks is bounded by KV-cache memory, not compute. Reduce KV-cache pressure with `--kv-cache-dtype fp8`, grouped-query attention, prefix caching, or a tighter `--max-model-len` — each is a knob the runbook explains.
- Load-test with realistic workload shape (real traffic sample or task-specific set from mod-108, not "one prompt N times"), ramp concurrency until the SLO bends, and record throughput / TTFT / ITL / E2E at every step. The knee is your capacity number.
- Dollars per million tokens is competitive with managed APIs when utilization is high (~$0.5–1 per million on a well-utilized H100 for a 7B; more expensive when GPUs sit idle). Utilization is the parameter that decides self-host vs. managed.
- The serving runbook is a ten-section artifact: identity, deploy command, latency SLOs, capacity, quality contract, observability, failure playbook, rollback, change log, handoff. This is what you hand to mod-111 and to on-call.
- Quality SLO (release-gate delta from full-precision) and latency SLO are separate; re-verify quality on the served endpoint periodically, not just at deploy. Production traffic can quietly diverge from calibration data.
- Every knob in this module — quantization method, merge vs. multi-LoRA, vLLM vs. TGI, GGUF ladder position, KV-cache dtype — ends up in the runbook as a config value with a documented reason. That is what makes the deploy operable by the next person, and that is the handoff to mod-111.
