# exercise-03: Adapter Merging and Multi-Adapter Routing

**Estimated effort:** 3 hours
**Companion chapter:** [`03-merging-and-multi-adapter-serving.md`](../03-merging-and-multi-adapter-serving.md)

## Objective

Train two small LoRA adapters on the same base for two visibly different tasks, load them side by side, route requests between them at inference, and produce a merged variant of one of them for direct comparison against the attached adapter. Write up the trade-offs for a specified serving scenario — "merge and ship" vs. "keep attached and route" — with concrete numbers backing the recommendation.

## Prerequisites

- The trained adapter from `exercise-02-qlora-on-a-7b-base-single-gpu.md`, OR any LoRA adapter you have on hand (a smaller-base one is fine — a 1B / 1.5B trained adapter from exercise-01 works).
- A GPU with enough VRAM to hold your base + two small adapters simultaneously (a single 24 GB card fits a 7B BF16 base + several adapters; a 12 GB card handles a 1B–3B base with room to spare).
- Same Python environment as exercises 01 and 02.

## Requirements

### Part A — Train two adapters on distinct tasks

Pick one base and train two adapters. Two clearly distinguishable tasks make the routing behavior visible:

- **Adapter A: "chatty helpful assistant."** Train on `HuggingFaceH4/no_robots` or a similar generic SFT slice. Style: warm, verbose, helpful.
- **Adapter B: "structured JSON output."** Train on a small custom slice (100–500 rows is enough) of `{"prompt": "...", "completion": "{...}"}` where every completion is a valid JSON object. Style: terse, structured, no prose.

Alternatively pick any two tasks that produce visibly different outputs on the same prompt (e.g., "answer as a pirate" vs. "answer in bullet points"). The point of the exercise is that the difference is obvious in a five-prompt eval, so pick tasks with high signal.

Keep both LoRA configs identical except for the training data — same `r`, `alpha`, `target_modules`, `learning_rate`. Same base revision (pin the hash). This makes the comparison in later parts less noisy.

Save `adapters/adapter-A/` and `adapters/adapter-B/`.

### Part B — Load both adapters side by side and route

Author `route.py` that:

1. Loads the base once at BF16 (or NF4 + `PeftModel` if you are continuing from exercise 02).
2. Loads adapter A as `adapter_name="chatty"`.
3. Loads adapter B as `adapter_name="json"` via `model.load_adapter(...)`.
4. Defines a small router: given a prompt, pick `"chatty"` or `"json"` (via a simple heuristic — e.g., `"json" in prompt.lower()` or a hard-coded routing table for 10 canned prompts).
5. For each of 10 canned prompts (some clearly-chatty, some clearly-json, some ambiguous), calls the router, calls `model.set_adapter(selected)`, and generates a response.
6. Also generates a response with `model.disable_adapter()` (base only) for each prompt, for reference.
7. Emits `routing_log.jsonl` with per-prompt: prompt, chosen adapter, generation from chosen adapter, generation from base, wall-clock latency for each.

### Part C — Verify the routing actually swapped

Add three assertions to `route.py` (or a separate `test_route.py`):

1. **Same-prompt / different-adapter outputs are non-identical.** For at least 3 prompts, `generate(prompt, adapter="chatty") != generate(prompt, adapter="json")` at greedy decoding. If they are identical, `set_adapter` never actually toggled — investigate.
2. **Base-only output differs from at least one adapter's output.** If it does not, either your adapters failed to train or your `disable_adapter` context is not doing what you think.
3. **`model.active_adapter` after `set_adapter("json")` equals `"json"`.** Catches the "silently loaded but never activated" bug from chapter 3.

### Part D — Merge adapter A and compare

Merge adapter A into a fresh copy of the base:

```python
from peft import PeftModel
base_fresh = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.bfloat16, revision=BASE_REV)
model_a = PeftModel.from_pretrained(base_fresh, "adapters/adapter-A")
merged_a = model_a.merge_and_unload()
merged_a.save_pretrained("merged/adapter-A-merged")
```

Then load the merged model and, for the same 10 canned prompts, compare outputs against the attached-adapter version. At low / zero temperature, they should be effectively identical (allow small numeric differences from matmul order — cosine similarity on next-token logits > 0.999 is a pass). Any prompts where they diverge meaningfully are diagnostic; log them.

### Part E — The trade-off write-up

Author `serving.md` (~1 page) that answers a specific serving scenario:

> "We serve a customer-support product. 90% of traffic goes to one 'general helpful' adapter (adapter A). The remaining 10% is split across ~20 small tenant-specific adapters we train per week. Latency budget: p95 < 400 ms per response on our current inference hardware."

Write a recommendation:

1. Merge adapter A or keep it attached? Argue with reference to per-request latency (from your Part B / D numbers), the operational cost of the two options, and the "cannot unmerge" trap from chapter 3.
2. Serve the 20 tenant adapters as attached / hot-swappable, or merged (i.e., 20 separate merged models)? Argue with reference to disk / VRAM / cold-start numbers.
3. Which serving stack (vLLM LoRA, TGI LoRA, LoRAX, or "roll our own") would you reach for and why? Chapter 3 covers the landscape; extend with any references you dig up.

There is no single right answer. The point is that your numbers back your recommendation.

### Part F — Weighted-adapter experiment (short)

As a small bonus and quick learning check, use `add_weighted_adapter` to create a `"blend"` adapter that is `0.7 * chatty + 0.3 * json`. Generate on your 10 prompts with the blend. Write two sentences in `serving.md` describing what you got — was the output a coherent middle ground, or something weirder? This is a common "intern first-week" question and the answer is instructive either way.

## Starter guidance

- **Pick tasks where the outputs are visibly different.** If your two adapters produce outputs that are 90% overlapping in style, the routing exercise is boring and Part D's low-temperature comparison hides real bugs. Pirate voice vs. JSON is a good working choice.
- **Pin the base revision in both adapter runs and at serving time.** The chapter-3 "adapter against wrong base revision" failure mode is the kind of bug that shows up two months later; do not create it now.
- **`set_adapter` in a request loop is cheap.** Do not build a queue-per-adapter architecture on the assumption that swapping is expensive. Measure first.
- **If merged vs. attached outputs diverge significantly at greedy decoding**, either (a) your base and adapter are on different dtypes (`bfloat16` vs. `float16` mismatches surface here), (b) you are comparing across different `revision` hashes, or (c) `merge_and_unload` did not actually run — check `PeftModel.is_prompt_learning` and the returned type.
- **Adapter B does not need to be well-trained.** For the routing / merging comparison to be meaningful, it just needs to *produce different output* than A. A 100-example, 50-step "JSON only" run is enough for this exercise.

## Acceptance criteria

You have completed the exercise when:

- `adapters/adapter-A` and `adapters/adapter-B` exist, both are ≤ ~100 MB, and both load cleanly with `PeftModel.from_pretrained`.
- `route.py` runs end to end, produces `routing_log.jsonl` with all 10 prompts and all three (chatty / json / base-only) generations each, and the three assertions in Part C pass.
- `merged/adapter-A-merged/` exists and loads via `AutoModelForCausalLM.from_pretrained` (no `peft` needed).
- The merged-vs-attached comparison shows > 99.9% cosine on next-token logits at low temperature, or you explain the divergence with a specific cause.
- `serving.md` makes a concrete recommendation for the customer-support scenario, backed by your own latency and memory numbers.
- The Part F weighted-adapter blend generation is included with two-sentence commentary.

## Stretch goals

- **Route with a small classifier instead of a heuristic.** Train a 100-example logistic regression on prompt embeddings that picks "chatty" vs. "json"; report its accuracy on a held-out set of 30 prompts and note the failure cases.
- **vLLM adapter serving smoke test.** Set up vLLM locally (or in a Colab), enable `enable_lora=True`, load both adapters, and route requests via `LoRARequest`. Report p50 / p95 latency compared to your `route.py` numbers. Chapter 3 covers the config.
- **Add a third adapter and try a 3-way blend.** Repeat Part F with three weights summing to 1. See if the output is coherent when your two adapters had very different objectives.
- **Fault-inject a base-revision mismatch.** Change `BASE_REV` at serving time to a different pin than training time. Log whether outputs subtly degrade. Add a startup assertion that catches the mismatch.
- **Package the merged model for handoff.** Write a `README.md` and a `model_card.md` that would make sense to a downstream team receiving `merged/adapter-A-merged/` cold. This is the artifact real handoffs need; almost no one produces it.
