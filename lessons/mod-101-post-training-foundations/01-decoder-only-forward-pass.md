# The Decoder-Only Forward Pass (and Where PEFT Hooks In)

**Estimated read:** 45 minutes.
**You should be able to, by the end:** name every tensor operation between an input token and an output logit, sketch a Llama-style transformer block on paper, and point at the exact `nn.Linear` layers that LoRA, IA3, and prefix tuning modify.

## Why a fine-tuner needs this

Fine-tuning is not "shipping a `train.py` at Hugging Face and hoping." A fine-tuner picks a base model, a PEFT method, a rank, and a set of target modules — every one of those choices routes back to a specific tensor inside the forward pass. If you cannot answer "which `nn.Linear` does `q_proj` refer to?" or "why does LoRA target the attention projections and not the LayerNorms?", you will make choices by copy-pasting other people's `peft_config` and never diagnose why your adapter underperforms.

This chapter walks the forward pass at the granularity a fine-tuner actually needs, then labels which layers each PEFT family touches.

## The architecture we care about

Almost every open base model you will fine-tune in 2025-era work is **decoder-only** — the "GPT branch" of the family tree from the original *Attention Is All You Need* encoder-decoder (Vaswani et al., 2017). Concretely: Llama 2 / Llama 3, Mistral / Mixtral, Qwen, Gemma, Phi, DeepSeek, Falcon. They share a common recipe:

1. **Token embedding** — a lookup table of shape `[vocab_size, d_model]`.
2. **Positional information** — either an additive embedding, or (much more commonly now) RoPE applied inside attention. Covered in chapter 3.
3. **N transformer blocks** — the stack. `N` is 32 for Llama 3 8B, 80 for Llama 3 70B.
4. **Final norm** — a single RMSNorm (or LayerNorm) applied to the last hidden state.
5. **LM head** — a linear projection back to `[vocab_size]` to produce logits. Often weight-tied to the input embedding on smaller models; separate on larger ones.

The training objective in pretraining is **next-token prediction with a causal mask**: at position `t`, the model may attend only to positions `≤ t`, and the loss is cross-entropy between the model's logits at position `t` and the true token at position `t+1`. Every SFT / DPO / RLHF variant you will learn later is a modification of *what tokens contribute to the loss* and *how the loss is shaped* — the forward pass itself is unchanged.

## Inside one block (Llama-style, pre-norm)

Modern decoder-only blocks use **pre-normalization**: the norm happens *before* the sublayer, and a residual carries the input around. Compared to the original post-norm design, pre-norm is what makes very deep stacks trainable without warmup gymnastics. Concretely, one block does:

```
# pseudo-code — this is the Llama / Mistral / Qwen family. Not GPT-2's exact layout.
def block(x, past_kv=None):
    # --- Self-attention sublayer ---
    h = rms_norm(x, weight=input_layernorm_weight)
    attn_out, new_kv = self_attention(h, past_kv)   # RoPE lives inside here
    x = x + attn_out                                # residual

    # --- Feed-forward sublayer ---
    h = rms_norm(x, weight=post_attention_layernorm_weight)
    ffn_out = swiglu_mlp(h)
    x = x + ffn_out                                 # residual

    return x, new_kv
```

Two norm layers, two residuals, one attention, one MLP. Everything else you will read about (grouped-query attention, sliding-window attention, MoE experts) is a variation on the attention or MLP block — the outer skeleton is stable.

### Self-attention, one head at a time

Inside `self_attention`, three projections take the normalized hidden state `h` of shape `[batch, seq, d_model]` and produce **queries, keys, values**:

```
q = h @ W_q     # W_q shape [d_model, n_heads * d_head]      -> q_proj
k = h @ W_k     # W_k shape [d_model, n_kv_heads * d_head]   -> k_proj
v = h @ W_v     # W_v shape [d_model, n_kv_heads * d_head]   -> v_proj
```

For a "vanilla" multi-head attention (MHA), `n_kv_heads == n_heads`. For **grouped-query attention (GQA)** — used in Llama 2 70B, Llama 3, Mistral, Qwen — several query heads share one K/V head, so `n_kv_heads < n_heads` and the K/V projections are proportionally narrower. This matters practically because it changes the shape (and therefore rank cost) of any LoRA you put on `k_proj` / `v_proj`.

The heads are then reshaped to `[batch, n_heads, seq, d_head]`, RoPE rotates `q` and `k` (chapter 3), and attention runs per head:

```
scores = q @ k.transpose(-2, -1) / sqrt(d_head)     # [b, n_heads, seq, seq]
scores = scores.masked_fill(~causal_mask, -inf)     # future positions blocked
weights = softmax(scores, dim=-1)
context = weights @ v                                # [b, n_heads, seq, d_head]
```

Heads are concatenated back to `[batch, seq, d_model]` and passed through the **output projection**:

```
attn_out = context @ W_o    # W_o shape [n_heads * d_head, d_model] -> o_proj
```

So a single attention sublayer contains exactly four `nn.Linear`s per Hugging Face module name: `q_proj`, `k_proj`, `v_proj`, `o_proj`. These are the four canonical LoRA targets, and Hugging Face `peft` recognizes those exact names for Llama-family models.

### The MLP (SwiGLU on Llama-family)

The FFN sublayer in Llama, Mistral, Qwen, and Gemma is a **SwiGLU** MLP with three linear layers (not the two of the classic Transformer):

```
gate = h @ W_gate     # gate_proj -> [d_model, d_ff]
up   = h @ W_up       #   up_proj -> [d_model, d_ff]
ffn_out = (silu(gate) * up) @ W_down   # down_proj -> [d_ff, d_model]
```

`d_ff` is typically ~2.7 × `d_model` for SwiGLU (to match the parameter count of a 4× GeLU MLP, since we now have three matrices instead of two). Older GPT-2 / GPT-J style models use a plain two-linear GeLU MLP with `fc_in` / `fc_out` — same idea, fewer target names.

### The two norms

Each block has two normalization layers. Modern decoder-only models use **RMSNorm** (Zhang & Sennrich, 2019) rather than LayerNorm — it drops the mean-subtraction and the bias term, and normalizes only by the root mean square of the activations. In Hugging Face `LlamaForCausalLM` these live at `.input_layernorm` and `.post_attention_layernorm` on each layer, plus one final `.norm` before the LM head. They contain a single learnable scale vector per layer; no bias.

## The stack, end-to-end

Putting it together, the forward pass through `LlamaForCausalLM` on an input of `input_ids` shape `[batch, seq]` is:

1. `x = embed_tokens(input_ids)` → `[batch, seq, d_model]`
2. For layer `i` in `0..N-1`: `x, kv_cache[i] = layers[i](x, kv_cache[i])`
3. `x = norm(x)` → final RMSNorm
4. `logits = lm_head(x)` → `[batch, seq, vocab_size]`
5. `loss = cross_entropy(logits[:, :-1], input_ids[:, 1:])` during training (with any masking rules from mod-103 applied)

At inference, step 2 uses the **KV cache**: for each new token, `q` is computed only for the current position, but `k` and `v` for every past position are read from the cache instead of recomputed. This is why serving stacks (mod-109) obsess about KV-cache memory — for long contexts it dominates the resident VRAM.

## Where PEFT hooks in

Now the payoff. PEFT ("parameter-efficient fine-tuning") is a family of techniques that freezes almost all of the base weights and inserts a small number of trainable parameters somewhere in the forward pass. **"Where"** is the question — and the answer is almost always inside the six `nn.Linear` layers per block that we just named.

| Method | Inserts / modifies | Typical Hugging Face `target_modules` on Llama |
|---|---|---|
| **LoRA** (Hu et al., 2021) | Adds a trainable low-rank update `ΔW = BA` alongside a frozen `W` on each targeted `nn.Linear` | `["q_proj", "v_proj"]` for a minimal setup; `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` for the full "all-linear" recipe used in QLoRA |
| **QLoRA** (Dettmers et al., 2023) | Same shape as LoRA, but the frozen `W` is quantized to 4-bit NF4 and dequantized on the fly | Same target modules as LoRA — the quantization is orthogonal to the adapter placement |
| **IA3** (Liu et al., 2022) | Learns per-vector rescaling on the K activations, V activations, and one FFN activation | `["k_proj", "v_proj", "down_proj"]` per the paper's recipe |
| **Prefix tuning** (Li & Liang, 2021) | Prepends `n` trainable K/V vectors *inside every attention layer*, not as input tokens | Modifies all attention layers; no targeting choice |
| **Prompt tuning** (Lester et al., 2021) | Prepends `n` trainable input embeddings at the embedding layer only | Modifies embedding input only; the transformer stack is untouched |
| **BitFit** (Ben Zaken et al., 2022) | Trains only bias parameters | Note: Llama-family models have no biases on their linears; BitFit is a poor fit for these bases |
| **Adapter modules** (Houlsby et al., 2019) | Inserts small bottleneck FFNs *inside each block* between the sublayer output and residual | Modifies the block layout, not just weights |

Two observations that trip up new fine-tuners:

**LoRA targeting is a knob, not a fact.** The 2021 paper originally targeted only `q_proj` and `v_proj` because that was cheapest-per-quality on GPT-3. Modern QLoRA practice on 7B–70B open models routinely targets *all seven* linear layers per block (attention + MLP) because the extra rank is nearly free compared to the frozen base and can materially improve downstream quality. There is no universal right answer — mod-104 makes you measure this.

**Norms and the LM head are almost never targeted.** RMSNorms are cheap and tuning them is a known way to break base capabilities. The LM head is huge (`d_model × vocab_size`) and shared with (or tightly coupled to) the embedding table, so training it usually harms more than it helps. If you see a config touching `lm_head` or `norm`, treat it as an intentional choice that needs a rationale, not a default.

## Reading a real config

Here is an abridged `LlamaConfig` for Llama 3 8B (fields you will actually reference during fine-tuning):

```jsonc
{
  "hidden_size": 4096,                 // d_model
  "intermediate_size": 14336,          // d_ff for SwiGLU
  "num_hidden_layers": 32,             // N blocks
  "num_attention_heads": 32,           // n_heads
  "num_key_value_heads": 8,            // n_kv_heads -> GQA with 4:1 ratio
  "vocab_size": 128256,                // matters for LM-head param count and tokenizer chapter
  "max_position_embeddings": 8192,     // pretrained context length (chapter 3)
  "rope_theta": 500000.0,              // RoPE base — you will touch this to extend context
  "rms_norm_eps": 1e-5
}
```

Every value here shows up somewhere in the forward pass we just described. Get in the habit of opening `config.json` for any base you fine-tune and reading it before you write a single line of training code — half the "why won't my run work" questions on Discord dissolve at this step.

## Exercise pointer

The chapter's hands-on companion is `exercises/exercise-01-decoder-only-forward-pass-walkthrough.md`, where you instrument a real forward pass on a 1B-class Llama-family model and produce a labeled shape trace of every intermediate tensor, then identify by name every `nn.Linear` that a default QLoRA config would target.

## Summary

- Decoder-only transformers are a token embedding, N identical pre-norm blocks (attention + MLP with two RMSNorms and two residuals), a final norm, and an LM head.
- Each block contains six `nn.Linear`s of note: `q_proj`, `k_proj`, `v_proj`, `o_proj` in attention; `gate_proj`, `up_proj`, `down_proj` in the SwiGLU MLP.
- KV cache is a **read** of past `k` and `v` during inference; it does not exist during training.
- PEFT methods insert small trainable parameters against those linears (LoRA, IA3), inside the attention layer's K/V stream (prefix tuning), or at the input embedding (prompt tuning). Norms and the LM head are almost never targeted.
- Every choice you will make in `peft_config` — target modules, rank, alpha, whether to quantize the base — attaches to a specific tensor in the forward pass. Read `config.json` before you write training code.
