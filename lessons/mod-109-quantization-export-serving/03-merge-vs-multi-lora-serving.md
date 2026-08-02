# Merge vs. Multi-LoRA Serving

**Estimated read:** 30 minutes.
**You should be able to, by the end:** decide, for a given serving scenario, whether to merge a LoRA adapter into the base and serve a plain quantized model or to serve the base with N adapters attached; describe the memory, latency, cold-start, and quality trade-offs that drive that decision; and explain how vLLM's multi-LoRA (and TGI's / LoRAX's equivalents) actually deliver on the "one base, many adapters" promise at production throughput.

## The question

You finished a LoRA fine-tune in mod-104. The adapter is 50–100 MB; the base is 3–14 GB depending on quantization. At serving time you have two options:

1. **Merge the adapter into the base** (`W ← W + (α/r) · B · A`), quantize the merged weights, and serve a plain quantized model. Every request goes through one adapter's worth of tuning, permanently.
2. **Keep the adapter separate** and serve base + adapter(s) at inference time using a LoRA-aware runtime (vLLM multi-LoRA, TGI, LoRAX). Requests can route to different adapters. Adapters can be added and removed without redeploying the base.

Mod-104 chapter 3 introduced the mechanics of both. This chapter is about the *serving decision*: given real request patterns, real hardware, and a real quantization target, which one do you actually pick?

## The recap in one paragraph

Merging is `W_merged = W_base + (α/r) · B · A` for each linear that has an adapter, then `peft.merge_and_unload()` returns a plain `transformers` model with no `peft` wrapping. Multi-LoRA keeps the base loaded once and attaches N small adapters, each request specifies which adapter to route through, and the runtime handles per-request adapter selection in the same batch. Both are covered in detail in mod-104 chapter 3; this chapter builds on that.

## The four scenarios that decide the answer

| Scenario | Right pick | Why |
|---|---|---|
| A. Single-tenant, one adapter, high QPS | **Merge** | No LoRA overhead per request; simpler artifact; the runtime does not need multi-LoRA support at all. |
| B. Single-tenant, one adapter, low QPS, latency SLO tight | **Merge** | Same as A; the LoRA overhead is small but non-zero. |
| C. Multi-tenant, N adapters (5+), N is stable, adapters per tenant | **Multi-LoRA** | Merging would produce N full merged models — N× the memory. Multi-LoRA amortizes the base. |
| D. Rapidly-added adapters (new tenants per day), adapters churn | **Multi-LoRA + adapter warehouse** | Cold-start cost matters; merging requires re-quantizing and redeploying per new tenant. |

Real deployments are usually A / C hybrids: a "workhorse" adapter serves 80–95% of traffic (candidate for merging into the base and treating as the default), and a long tail of per-tenant adapters is served via multi-LoRA on top. Chapter 7's runbook and exercise 02 walk through picking the split.

## The memory model

### Merged path

For each adapter you serve, you need one full base's worth of VRAM. Quantization reduces the base size but the multiplier is unchanged.

| Base | BF16 | INT8 | INT4 (GPTQ / AWQ / NF4) |
|---|---|---|---|
| 1B | ~2 GB | ~1 GB | ~0.5 GB |
| 7B | ~14 GB | ~7 GB | ~3.5 GB |
| 13B | ~26 GB | ~13 GB | ~6.5 GB |
| 34B | ~68 GB | ~34 GB | ~17 GB |
| 70B | ~140 GB | ~70 GB | ~35 GB |

Serving 20 merged 7B-INT4 tenants means 70 GB just for weights — before any KV cache. On a single H100 (80 GB HBM), you cannot fit them all. On four H100s, you can, but you are paying 4× the hardware for a workload that could plausibly run on one.

### Multi-LoRA path

One base + N adapters. Adapter memory:

`M_adapter ≈ 2 · r · (d_in + d_out) · num_target_layers · dtype_bytes`

For a 7B, LoRA rank 16, all-linear target modules, BF16 adapter dtype: ~80–100 MB per adapter. Concrete example: base 7B-INT4 (3.5 GB) + 50 adapters (~4 GB total). One GPU serves all 50 tenants. This is the reason multi-LoRA exists.

vLLM's implementation (following the S-LoRA design, Sheng et al., 2023) keeps a bounded set of adapters "hot" on GPU (up to `--max-loras`, e.g., 8) and pages the rest to CPU or disk. Continuous batching preserves throughput: multiple adapters can be active in the same batch step, with each request in the batch routed through its own adapter's `A` / `B`.

## The latency model

Per request, the merged path is one matmul per layer. The multi-LoRA path is one base matmul + one LoRA matmul per adapted layer. On paper the LoRA path is strictly more work — but the overhead in a well-tuned inference kernel is small.

Rough numbers (illustrative — actual depends on model, hardware, and batch size):

| Pattern | Overhead vs. merged base | Notes |
|---|---|---|
| Merged base, INT4 | baseline | No LoRA at inference. |
| Multi-LoRA, adapter hot on GPU, vLLM | ~5–15% | Punctuated GEMM in the same CUDA graph. |
| Multi-LoRA, adapter cold (paged from CPU) | +100–500 ms *first request* | Only affects the request that triggers the load; subsequent requests to that adapter are hot. |
| Adapter evicted then loaded from disk | +1–5 s *first request* | Rare in a well-sized `--max-cpu-loras`. |
| Naive `set_adapter` + generate loop (no batching) | +30–50% | Hand-rolled loops without fused kernels. Avoid. |

<!-- needs-research: exact numbers depend heavily on model size, adapter rank, batch size, and vLLM version. Treat the ~5–15% band as the operating point on modern vLLM; measure your workload. -->

The rule: **on a LoRA-aware runtime (vLLM / TGI / LoRAX), the per-request LoRA overhead is small enough that most workloads should not merge for latency reasons alone.** Reasons to still merge are simplicity, multi-adapter routing not being needed, or a runtime that does not support LoRA at all.

## The cold-start model

Merged deployments cold-start with the base weights: seconds to load 3.5 GB of INT4 weights from local NVMe, tens of seconds if pulling from remote object storage.

Multi-LoRA deployments have two cold-start events:

1. **Base cold-start** — same as merged. One-time per process.
2. **Adapter cold-start** — the first request per adapter loads that adapter into GPU (~50–500 ms for a typical 7B adapter, depending on where it lives). Subsequent requests are hot until the adapter is evicted.

For adapter churn (new tenants added continuously), pre-warming matters: at deploy time, force-load every adapter that will see traffic in the first minutes to avoid the first-request tax. The `LoRARequest` API in vLLM lets you make a warmup request per adapter at startup; do it.

## When merging is the right call at serving time

- **Single adapter, no plans for tenant-specific variants.** No memory saving to gain from multi-LoRA; simpler artifact wins.
- **Target runtime does not support LoRA at inference.** Some inference stacks (llama.cpp / GGUF path, some embedded runtimes, some enterprise gateways) accept a plain model but not adapters. Merge.
- **The base is 4-bit quantized with GPTQ / AWQ.** Whether multi-LoRA is supported on top of GPTQ / AWQ bases is runtime- and version-specific and has historically lagged BF16 multi-LoRA support. Check current vLLM / TGI docs. If your target combo does not support LoRA-over-INT4, merge the LoRA into a BF16 base first, then quantize the merged BF16 to INT4 with GPTQ or AWQ.
- **You want the smallest possible operational surface.** No adapter versioning, no multi-adapter routing bugs, no cold-start-per-adapter latency. Merged is the operationally simpler artifact and worth the memory cost for small deployments.

## When multi-LoRA is the right call

- **N adapters over one base, N > ~3.** Every additional merged variant costs a full base's worth of GPU memory; every additional attached adapter costs 50–100 MB.
- **Tenants can be added / removed continuously.** Merging + re-quantizing + redeploying per new tenant is slow and expensive; loading a new adapter into a running multi-LoRA server is a few hundred milliseconds.
- **You need A/B testing across adapters.** Route 5% of traffic to adapter v2 and 95% to v1 in the same server, without a second deployment. Multi-LoRA makes this a request-header change.
- **Adapter combinations (`add_weighted_adapter`) are part of the product.** Live-blending personas is a multi-LoRA-native pattern.

## The "quantized base + LoRA" question

The specific combination "INT4 base + BF16 LoRA at inference time" is the most-asked question in this chapter. The short answer:

- `bitsandbytes` NF4 base + BF16 LoRA works at *inference* time via `transformers` + `peft` — this is the QLoRA training-time pattern reused at serve time. Latency is not competitive with vLLM / TGI paths; use it for research or offline generation.
- vLLM has historically supported multi-LoRA over BF16 bases better than over INT4 bases. Support for LoRA over GPTQ / AWQ bases has been improving. Check the current vLLM release notes for your model family; if unsupported, either serve BF16 base + LoRA (fits less on GPU but works today) or merge into BF16, quantize, and serve merged.
- TGI has its own LoRA-adapter path; the LoRA-over-quantized story is similarly evolving. Verify against the current docs before designing around it.
- LoRAX (Predibase) is explicitly designed around this combination and is often the fastest path if "multi-LoRA over a quantized base" is the product.

<!-- needs-research: LoRA-over-quantized-base support in vLLM and TGI changes rapidly. The exercise for this module includes a smoke-test step that verifies the current-version behavior for the reader's chosen (base, quantization, adapter) tuple before designing around it. Do not commit to a serving architecture on the strength of a blog post; run the smoke test. -->

## Routing patterns at serve time

Mod-104 chapter 3 introduced the routing patterns; they matter more here because at serving scale the router is on the request path for every call. The four patterns worth pinning:

**1. Header-based routing (multi-tenant).** The tenant ID from the auth header maps to an adapter name; the request routes there. The pattern most SaaS PEFT deployments end up with. In vLLM: build a small proxy that translates `tenant_id → LoRARequest` before calling `llm.generate`. In an OpenAI-compatible server, use the `model` field in the request body (vLLM lets each LoRA adapter be registered with its own served model name).

**2. Classifier-based routing (single tenant, many skills).** A small classifier picks an adapter based on prompt content — "code" for code questions, "sales" for CRM prompts, "support" for tickets. Cheaper than MoE; interpretable; fails when the classifier is wrong. Log the classifier's decision and confidence with every request so you can debug misroutes.

**3. Fallback-chain routing.** Try adapter A; if the response fails a policy check (safety classifier, format validator, schema check), retry with adapter B or the base. Common with a small safety-tuned adapter as A and the general SFT adapter as B. Doubles cost for the small fraction of requests that fall through; keep the fallback rate observable.

**4. Canary / rollout routing.** A new adapter is deployed globally but only 5% of traffic sees it initially; the rest goes to the previous adapter or the base. Multi-LoRA is what makes this a one-line config change rather than a second deployment.

## The disk / registry layer

A production multi-LoRA server needs an adapter registry: a place where adapter files live, from which the runtime can load them at first request. Options:

- **Local disk.** Cheapest, lowest latency. Adapters synced to the pod at deploy time. Fine for small stable adapter sets.
- **Shared filesystem / NFS.** Adapters live once, all replicas can load. First load is slower than local.
- **Object storage (S3 / GCS / Azure Blob) via signed URL / passthrough.** Adapters can be added post-deploy without touching the pods. Cold-start is dominated by the object-storage read. Standard for large tenant sets.
- **A dedicated adapter service.** A microservice that fronts the registry, handles auth, does staged rollouts, and hands adapters to the inference server. Mod-111 covers this altitude.

For this chapter's altitude, the practical rule: **local disk for small stable sets; object storage with pre-warming for anything that churns.**

## Base-revision safety

The failure mode from mod-104 chapter 3 that gets more dangerous at serving scale: an adapter trained against `Llama-3.1-8B` revision `abc123` gets served against revision `def456` because someone pinned the base to a different revision in the serving config. LoRA math does not error — it just computes `W_new + BA` on the new base. Quality can drop silently.

Concrete mitigations in the serving config:

- Pin `revision=<hash>` in the base model config; log the pin at server startup.
- `adapter_config.json` records `base_model_name_or_path`; verify at load time and refuse to serve on mismatch unless explicitly overridden with an `--allow-base-revision-mismatch` flag.
- If you must swap the base (e.g., a security patch on the base), plan to re-tune every adapter on the new revision — do not assume they transfer.

## The vLLM multi-LoRA config in practice

Chapter 4 has the full engine args; here is the multi-LoRA-relevant subset with the trade-offs each flag controls:

```bash
vllm serve meta-llama/Llama-3.1-8B \
  --enable-lora \
  --max-loras 8 \
  --max-lora-rank 32 \
  --max-cpu-loras 64 \
  --lora-modules \
    support=/adapters/support \
    sales=/adapters/sales \
    code=/adapters/code
```

- `--enable-lora` — turns on the multi-LoRA path.
- `--max-loras N` — maximum adapters concurrently active in a batch. Adapters beyond N in a batch step wait for a slot. Set to your expected concurrent-adapter fan-out.
- `--max-lora-rank R` — hard cap on adapter rank. Requesting an adapter with rank > R fails. Set to the largest rank you plan to serve.
- `--max-cpu-loras M` — LRU cache size on CPU. Adapters evicted from GPU live here; eviction from this cache means the next request goes back to disk / object storage.
- `--lora-modules name=path ...` — adapters registered at startup, each with a name that becomes a served model in the OpenAI-compatible API. Additional adapters can be registered at runtime via the `POST /v1/load_lora_adapter` endpoint.

Two decisions the flags encode:

**How many concurrent adapters?** `--max-loras=8` is a common default. If your workload has a small hot set (say, 3–4 workhorse adapters + long-tail tenant adapters), 8 is comfortable. If every request is a different adapter, `--max-loras` becomes a batch-size cap — raise it, at the cost of some throughput per request.

**Rank cap?** Set to the maximum rank across adapters you plan to serve. Setting it higher than needed costs a bit of memory; setting it lower rejects legitimate requests.

## Exercise pointer

`exercises/exercise-02-vllm-multi-lora-vs-merged-deployment.md` has you stand up vLLM in two configurations — one merged single-tenant, one multi-LoRA with 3+ adapters — load-test each with a shared workload, and defend the merge / multi-LoRA decision for a specified serving scenario with your own numbers backing it up.

## Summary

- The serving-time decision between merging an adapter and serving multi-LoRA is not a taste question; it is driven by (number of adapters), (adapter churn rate), (target runtime's LoRA support), and (single-tenant / multi-tenant).
- Merge for single-adapter serving, tight artifacts, or runtimes without LoRA support. The main cost is one full quantized base per adapter you serve; the main benefit is operational simplicity.
- Multi-LoRA for N > ~3 adapters or churning tenants: one base + N small adapters, with per-request routing handled by the runtime. vLLM / TGI / LoRAX all implement the pattern at production throughput; the per-request overhead is ~5–15% vs. a merged base on modern kernels.
- The "quantized base + LoRA at inference" combination works but the supported combinations (which quantization method × which runtime × which LoRA path) evolve version to version. Smoke-test on your stack before designing around it.
- Base-revision drift is the silent killer: pin the base by hash at both train and serve time, verify against `adapter_config.json`'s `base_model_name_or_path` at load, and refuse to serve on mismatch. LoRA math will not error; quality will just drop.
- Chapter 4 covers the vLLM engine args in full; chapter 5 covers the TGI equivalents; chapter 7's runbook is where the merge / multi-LoRA decision gets written down alongside the SLOs it is defending.
