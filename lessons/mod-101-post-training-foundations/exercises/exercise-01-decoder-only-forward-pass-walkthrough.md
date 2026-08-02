# exercise-01: Decoder-Only Forward Pass Walkthrough

**Estimated effort:** 3 hours
**Companion chapter:** [`01-decoder-only-forward-pass.md`](../01-decoder-only-forward-pass.md)

## Objective

Prove — with code — that you can point at every meaningful tensor in a real decoder-only forward pass and name what happens to it. By the end you will have (a) a labeled shape trace of every intermediate activation through one transformer block of a real 1B-class model, and (b) a written argument for which `nn.Linear` layers a default QLoRA config would target, cited by module path.

The point of this exercise is not to write elegant code. It is to make sure that the words "residual stream," "q_proj," "SwiGLU," and "post_attention_layernorm" refer to specific tensors and modules in your head, not to blog-post abstractions.

## Prerequisites

- Python 3.10+ with `torch`, `transformers`, and optionally `peft` installed.
- Enough compute to load a 1B-class model in half precision on CPU or a small GPU (~4 GB VRAM is plenty).
- Read of the companion chapter.

Model choice: pick any one open Llama-family model in the 1B–3B range that you can load. `meta-llama/Llama-3.2-1B` is a good default (requires HF auth acceptance of the Llama license); `Qwen/Qwen2.5-1.5B` or `microsoft/Phi-3-mini-4k-instruct` are drop-in alternatives if you cannot obtain Llama access. Do NOT use a quantized checkpoint for this exercise — you want to see the un-modified architecture.

## Requirements

### Part A — Shape trace of one block

Write a script that:

1. Loads the tokenizer and model in `torch.float16` (or `bfloat16`).
2. Prints the model's high-level structure — `print(model)` is a fine start.
3. Encodes a short prompt (say, `"The capital of France is"`) to `input_ids` of shape `[1, seq]`.
4. Runs the token embedding manually: `x = model.model.embed_tokens(input_ids)`.
5. Attaches PyTorch **forward hooks** to the first transformer block (`model.model.layers[0]`) that capture and print the shape and dtype of every intermediate tensor produced by:
   - `input_layernorm`
   - the self-attention sublayer's `q_proj`, `k_proj`, `v_proj`, `o_proj`
   - the residual after the attention sublayer
   - `post_attention_layernorm`
   - the MLP's `gate_proj`, `up_proj`, `down_proj`
   - the residual after the MLP sublayer
6. Runs the forward pass through *just that one block* (call `layer0(x)` directly, or run the full model and rely on the hooks).
7. Produces a clean text output that is a numbered trace: "1. input_layernorm output: [1, 6, 2048] fp16" and so on.

Format the trace so a colleague could hand you a printout and, without your code in front of them, follow which tensor is which. This is a communication exercise as much as a coding exercise.

### Part B — Where would QLoRA hook in?

Without running any training, answer these in a short markdown writeup (aim for ~300 words, not a wall of text):

1. If you ran `peft.LoraConfig(target_modules=["q_proj", "v_proj"])` against your chosen model, list — by full module path (e.g. `model.model.layers[7].self_attn.q_proj`) — every `nn.Linear` that would have a LoRA adapter added. You do not have to list all 32 layers by hand; describe the pattern and count them.
2. Now expand to `target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` — the "all linear" QLoRA recipe. Same question. How many trainable adapters would you have if `r=16`? (Compute the total trainable parameter count — sum of `A` and `B` matrix sizes across all target modules — and compare it to the base model's total parameter count.)
3. Would either config touch the LM head, the token embedding, or any RMSNorm? Why or why not?
4. Point at *one* module in the base model that a naive "just target all linear layers" search would sweep up and that you would explicitly *exclude* from a LoRA target list, and explain in one sentence why. (Hint: check `named_modules()` for anything named like `lm_head` or containing a projection outside the transformer blocks.)

### Part C — Sanity check with the KV cache

Extend your script to run `model.generate(input_ids, max_new_tokens=5, use_cache=True)` and then re-run with `use_cache=False`. Report the wall-clock time difference on your machine and, in one sentence, explain what the KV cache is caching and why the second run is slower.

## Starter guidance

- `named_modules()` on the base model is your friend. You can filter for `isinstance(m, nn.Linear)` to enumerate every linear in the stack.
- PyTorch forward hooks: `handle = module.register_forward_hook(lambda mod, inp, out: print(...))`. Remember to `handle.remove()` when done, or your test runs will leak hooks.
- The Llama attention module in `transformers` returns a tuple; the first element is the hidden state you want to trace. Read `LlamaAttention.forward` in the `transformers` source if you get confused.
- Some models split the SwiGLU MLP's `gate_proj` and `up_proj` into a single fused linear for efficiency (search for `mlp.gate_up_proj`). If your chosen model does that, note it — this is a real fine-tuning gotcha.
- For part B counting: LoRA adds `A: [in_features, r]` and `B: [r, out_features]` per target linear. Total trainable = `r * (in + out)`.

## Acceptance criteria

You have completed the exercise when:

- Your shape trace is present, correct, and cleanly formatted; every step is labeled with a module name and tensor shape.
- Your writeup names — in the module-path notation of the actual chosen model — every layer a default and an "all-linear" QLoRA config would target, with counts.
- Your parameter-count comparison for the `r=16` all-linear QLoRA config is arithmetically correct (double-check by comparing against `peft.get_peft_model(...).print_trainable_parameters()` at the end).
- Your KV-cache section reports actual wall-clock numbers from your machine, not asserted-from-memory ones.
- The writeup demonstrates that you can read a config file and a `named_modules()` listing side by side and reason about them.

## Stretch goals

- Add a **residual stream inspection**: at each block boundary, log the L2 norm of the hidden state. Watch it grow or shrink through the stack. Explain what you see.
- Do the same trace on an ALiBi-based model (`mosaicml/mpt-1b-*` or a Falcon-1B checkpoint if available). What is different in the module tree? Where does positional information enter?
- Compare `LlamaForCausalLM` with `GPT2LMHeadModel`. Enumerate the naming differences (`c_attn` and `c_proj` vs `q_proj`/`k_proj`/`v_proj`/`o_proj`). What LoRA target-module list would you use for a GPT-2 base? (This becomes relevant if you ever fine-tune older checkpoints.)
- If you have `peft` installed: attach a real LoRA to the model with the config from part B question 2, and confirm your parameter count matches `print_trainable_parameters`.
