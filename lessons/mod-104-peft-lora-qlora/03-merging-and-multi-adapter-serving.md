# Merging Adapters and Multi-Adapter Serving

**Estimated read:** 40 minutes.
**You should be able to, by the end:** merge a LoRA adapter into base weights and explain when this is a good idea and when it is a bad one, keep multiple adapters loaded on a single base and route requests between them at inference time, and reason about the trade-offs a serving framework (vLLM, TGI, LoRAX) is making when it advertises "LoRA-aware serving."

## The three ways an adapter can live at serve time

An adapter is trained in one place and served in another. The serving side has three options:

1. **Merged.** The adapter is folded into the base weights (`W ← W + BA · alpha/r`) and the resulting model is served as an ordinary dense model. No LoRA at inference time.
2. **Attached statically.** The base loads once; the adapter loads once on top and stays there for the process's lifetime. Every request routes through the same base + adapter.
3. **Attached dynamically (multi-adapter).** The base loads once; N adapters load alongside it; each request picks which adapter to activate. This is the "one base, many personas" pattern that PEFT enables.

Which one you choose changes almost every downstream detail — request latency, memory, deployment simplicity, ability to swap tenants. This chapter walks all three.

## Merging: the math and the code

Merging is arithmetically simple. For each linear that has a LoRA adapter:

```
W_merged = W_base + (alpha / r) · B · A
```

The result is a plain `nn.Linear` with the same shape as `W_base`; the adapter parameters are gone from the module tree. Forward pass is now one matmul instead of two, and `peft` is out of the loop.

With `peft`, one line:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, ADAPTER_PATH)

merged = model.merge_and_unload()
merged.save_pretrained("./merged-model")
```

After `merge_and_unload()`:

- The returned object is a plain `transformers` model with no `peft` wrapping. It loads with `AutoModelForCausalLM.from_pretrained` afterwards.
- The adapter files (`adapter_model.safetensors`, `adapter_config.json`) are no longer needed at load time — they have been baked into the base weights on disk.

### When merging is the right call

- **Single-adapter production serving.** If the deployment only ever runs one adapter on top of one base, merging removes a forward-pass matmul and eliminates a whole class of "wait, is the adapter applied?" bugs.
- **Shipping to an inference stack that does not speak LoRA.** vLLM speaks LoRA at inference (see later section); a bespoke inference server might not. Merge and hand off a plain checkpoint.
- **Sharing a "final" model publicly.** Model consumers expect a dense model. Merge before pushing to the Hub or handing to a downstream team, unless they explicitly want the adapter separately.
- **Distilling further downstream.** If you are going to full-parameter fine-tune on top of the merged model, or quantize it aggressively for serving, merging first gives you a clean starting point.

### When merging is the wrong call

- **You want to serve multiple adapters over the same base.** Merging locks in one adapter; the memory benefit of "one base, many adapters" evaporates because each merged variant is now the size of the whole base.
- **The base is 4-bit quantized (QLoRA).** You cannot merge a BF16 LoRA cleanly into an NF4 base — there is no BF16 base weight to merge into. Options:
  - Dequantize the base to BF16, merge, keep the BF16 merged model as your serving artifact. Loses the QLoRA memory win at serving time. Fine if serving memory is not your constraint.
  - Merge into a *separately loaded* BF16 base — same effect. The 4-bit base was a training-time artifact.
  - Do not merge; keep the 4-bit base and adapter separate at serving time. Most LoRA-aware inference stacks support this.
- **You care about being able to *unmerge* later.** Merging is one-way; the adapter is folded into the base and cannot be recovered as an adapter from the merged weights alone. Keep the adapter files around even after you merge, always.
- **You have not yet finished evaluating.** A merged model is the model you serve; iterating means re-merging. Keep the adapter separate until you have a final adapter you want to ship.

### The double-merge trap

`merge_and_unload()` is safe to call once. Calling it a second time on an already-merged model is a no-op (there are no adapters left to merge) but the pattern where a script auto-merges every time it runs has burned people: the disk artifact ends up being a merged base, and subsequent runs cannot find the adapter to load. Whenever you merge, also archive the original `adapter_model.safetensors` + `adapter_config.json` under a versioned path. If you cannot reproduce the merge, you cannot reproduce the model.

## Multi-adapter routing on one base

The interesting PEFT-serving pattern is: one base model loaded, many small adapters loaded next to it, each request picks an adapter to route through.

Memory math for the pattern:

- Base: `M_base` (e.g., 14 GB BF16 for 7B, or 3.5 GB NF4).
- Each adapter: `M_adapter ≈ 2 · r · (d_in + d_out) · num_target_layers · 2 bytes` (BF16 params for both `A` and `B`). For a 7B all-linear r=16 adapter, this is ~85 MB.

Loading 50 adapters alongside a 7B BF16 base adds ~4.3 GB to a 14 GB base — you serve 50 fine-tunes for the price of ~1.3 bases. That is the pattern's whole point.

### The `peft` primitive: named adapters

`PeftModel` supports multiple adapters keyed by name:

```python
from peft import PeftModel

model = PeftModel.from_pretrained(base, "./adapter-support", adapter_name="support")
model.load_adapter("./adapter-sales", adapter_name="sales")
model.load_adapter("./adapter-code",  adapter_name="code")

# At inference time, pick one:
model.set_adapter("support")
outputs = model.generate(**inputs)

model.set_adapter("code")
outputs = model.generate(**inputs)

# To go back to just the base — no adapter:
with model.disable_adapter():
    outputs = model.generate(**inputs)
```

`set_adapter` is (in current `peft`) O(number of adapted layers) because it flips which adapter's `A`/`B` is active in each `LoraLayer`. It does not move memory; it re-points pointers. In a serving loop, calling `set_adapter` per request is cheap enough for typical serving loads (thousands of RPS with same-tenant routing; hundreds of RPS with mixed tenants).

### Adapter combination: weighted sum

`peft` also supports combining adapters — sum, weighted sum, cat, ties-merging — for A/B tests or persona blends:

```python
model.add_weighted_adapter(
    adapters=["support", "sales"],
    weights=[0.6, 0.4],
    adapter_name="support_sales_blend",
    combination_type="linear",
)
model.set_adapter("support_sales_blend")
```

Under the hood this pre-computes a new `A`, `B` pair equal to the weighted sum. It is a real merge — not a soft mixture. Set expectations accordingly: the blended adapter is not `0.6 · f_support + 0.4 · f_sales` in output space; it is the model that results from applying the blended weight matrix. Sometimes that behaves as you hoped. Sometimes it does not. Always eval the combination separately.

## LoRA-aware serving stacks

Modern high-throughput inference stacks now handle multi-adapter LoRA directly.

### vLLM LoRA support

vLLM (Kwon et al., 2023 — PagedAttention) added LoRA support in 2024. The core pattern:

```python
from vllm import LLM
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_loras=8,           # up to 8 concurrently active adapters in the batch
    max_lora_rank=32,      # cap; all adapters must be ≤ this rank
    max_cpu_loras=64,      # LRU cache size on CPU
)

sampling_params = ...

lora_req_support = LoRARequest(
    lora_name="support",
    lora_int_id=1,
    lora_path="./adapter-support",
)

# Route this specific request through the "support" adapter:
outputs = llm.generate(
    prompts=["Customer is angry about a delayed shipment. Draft a response."],
    sampling_params=sampling_params,
    lora_request=lora_req_support,
)
```

Under the hood, vLLM keeps a bounded set of adapters "hot" on GPU (up to `max_loras`) and pages the rest to CPU on demand. Continuous batching is preserved — multiple adapters can be active in the same batch step, with each request in the batch routed through its own adapter's `A`/`B`.

Key trade-offs:

- `max_loras` sets the maximum *concurrent* adapters. Adapter switches within that set are ~free; switching to an adapter that has been evicted to CPU incurs a load.
- `max_lora_rank` is a hard cap. Serve a rank-64 adapter against a `max_lora_rank=32` server and requests to that adapter fail. Pick the largest rank you plan to serve and set the cap there.
- Throughput impact per request is small (single-digit percent overhead vs. the merged base at the same rank), but the memory saved by *not* materializing dozens of merged bases dwarfs it.

### Text Generation Inference (TGI)

Hugging Face TGI added LoRA-adapter support with a similar shape: load an adapter, name it, route to it via a request header. Refer to the current TGI docs (linked in `resources.md`) for the exact request-header format; the pattern is the same as vLLM's.

### LoRAX

LoRAX (Predibase) is an inference stack explicitly designed around multi-adapter serving — the pattern is the product, not a feature. Similar API surface to vLLM's LoRA path, additional emphasis on cold-start and adapter warehouse patterns.

Whichever stack you pick, the mental model is the same: pinned base, hot adapters, cold adapters. Understand what your latency SLO is for each of those transitions.

## Load-time vs. request-time: latency implications

For each pattern, the request-time latency looks like:

| Pattern | Cold start (add adapter) | Warm request |
|---|---|---|
| Merged base | ~0 (adapter is baked in) | Base matmul only |
| Static single-adapter | ~seconds (load once) | Base + 2 LoRA matmuls per adapted layer |
| Multi-adapter, hot | O(ms) (`set_adapter`) | Base + 2 LoRA matmuls per adapted layer |
| Multi-adapter, cold | Seconds to load; sub-second if paged from CPU | Same as hot after load |
| vLLM LoRA batching | O(µs) per request (dispatched into the CUDA graph) | Base matmul + fused LoRA matmul in the same kernel |

Two request-time numbers worth pinning to memory for a well-tuned 7B on an A100:

- Base-only forward: on the order of tens of milliseconds for a short prompt.
- Adapted forward: ~5–15% slower than base-only for a typical r=16, all-linear-targets adapter, in the LoRA-aware inference stacks. On a hand-rolled loop that does not fuse, it can be 30–50% slower.

If your latency budget is tight, benchmark. Do not assume LoRA is free at inference.

## Routing patterns worth knowing

### Routing by header / auth claim (multi-tenant)

The user identity in the auth header maps to a tenant ID; the tenant ID maps to an adapter name; the request routes to that adapter. Simple, robust, the pattern most SaaS PEFT deployments end up with. Adapter loading is a lazy step at first request per tenant.

### Routing by task classifier (single tenant, many skills)

A cheap classifier (small model or rules) inspects the prompt and picks an adapter — "code" for code questions, "sales" for CRM-style prompts, "support" for tickets. Faster than a MoE; interpretable. Fails when the classifier is wrong.

### Routing by policy (fallback chain)

Try adapter A; if the response fails a policy check (safety classifier, format validator), retry with adapter B or the base. Common with a small policy-check adapter as A and the general SFT adapter as B.

### Never routing (canary + gradual rollout)

A single adapter is deployed globally but only 5% of traffic sees it initially; the rest goes to the base or the previous adapter. Multi-adapter loading is not exercised for routing here — it is used for the fallback / previous-generation adapter. This pattern is what a "roll back" button in your deploy tool actually needs.

## Adapters against different base revisions

The most surprising failure mode in adapter serving: an adapter trained against `Llama-3.1-8B` revision `abc123` gets loaded against revision `def456` in production, because someone bumped the base's pin. LoRA math does not error — it computes `W_new + BA` on the new base. But the geometry the adapter was tuned against is different. Quality can drop silently.

Concrete mitigations:

- Pin the base by revision hash (`from_pretrained(..., revision="abc123")`) both at train time and serve time. Log the pin at inference startup.
- Store the base's revision in `adapter_config.json` (`peft` does this automatically as `base_model_name_or_path`). Verify at serve-time load; refuse to serve on mismatch unless explicitly overridden.
- If you must swap the base (e.g., a security patch), re-tune the adapter on top of the new revision — do not assume it transfers.

Similarly for QLoRA: an NF4-quantized base uses a different set of numeric constants than another base loaded with `bnb_4bit_use_double_quant=False`. The adapter learned in the presence of the training-time quantization. Serve under the same quantization config, or expect drift.

## The disk artifact — what is actually saved

A LoRA adapter directory typically contains:

```
adapter-support/
├── adapter_config.json      # LoraConfig-as-JSON: r, alpha, target_modules, base model ref
├── adapter_model.safetensors # A, B tensors for every adapted layer
└── README.md                 # optional model card
```

Two properties you should verify:

- The file size is roughly what your parameter math predicts. A 7B all-linear r=16 BF16 adapter is ~85 MB on disk. If yours is a lot larger, either you saved unnecessary state (optimizer moments, base weights) or a config bug turned target_modules into "all-linear including the LM head" without you noticing.
- Loading with `PeftModel.from_pretrained` matches the base architecture. `adapter_config.json`'s `base_model_name_or_path` is checked; if it does not match your loaded base, you get a warning. Do not suppress the warning.

## Exercise pointer

`exercises/exercise-03-adapter-merging-and-multi-adapter-routing.md` has you train two small adapters on the same base, load them side-by-side, route requests between them, and produce a merged variant of one of them for direct comparison. The write-up asks you to defend the choice of "merge vs. attach" for a fictional serving scenario.

## Summary

- Three adapter-at-serve patterns: merged (baked into base weights), statically attached (one base + one adapter), and dynamically attached (one base + many adapters, per-request routing).
- Merging is `W ← W + (alpha/r) · B · A` per adapted layer; `peft.merge_and_unload()` implements it. Merge for single-adapter production serving or shipping a "final" model; do not merge when you want multi-adapter routing, when the base is 4-bit quantized, or when you might need to iterate on the adapter later.
- Multi-adapter routing is the main serving win of PEFT: one base + N small adapters uses far less memory than N merged bases. `peft`'s named-adapter API (`set_adapter`, `add_weighted_adapter`, `disable_adapter`) is the reference primitive; vLLM, TGI, and LoRAX implement the same pattern at production scale with continuous batching.
- Watch base-revision mismatch. An adapter trained against a specific base revision should be served against the same revision under the same quantization; drift can silently degrade quality.
- Latency at inference: LoRA-aware stacks add ~5–15% over a merged base; a hand-rolled loop that does not fuse can add 30–50%. Benchmark before you commit.
