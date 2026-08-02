# Positional Encodings: Absolute, RoPE, and ALiBi

**Estimated read:** 40 minutes.
**You should be able to, by the end:** explain why attention needs positional information at all, describe how absolute, RoPE, and ALiBi encode it differently, and name the specific config knobs a fine-tuner turns to extend context length.

## Why attention needs a position channel

Take the attention formula from chapter 1:

```
scores = softmax(q @ k.T / sqrt(d_head)) @ v
```

Nothing in that expression depends on the order of the tokens. If you shuffle the sequence, `q`, `k`, and `v` all permute together, and the output is exactly the permutation of the original output. Attention is **permutation-equivariant** by construction. That is useful for sets but catastrophic for language, where "the dog bit the man" and "the man bit the dog" are different sentences.

The fix is to inject positional information somewhere in the pipeline. Every transformer architecture ever shipped does this, and the *how* has evolved through three regimes: absolute embeddings, rotary embeddings (RoPE), and additive linear biases (ALiBi). Which one your base model uses determines what you can do about context length at fine-tuning time.

## Absolute positional embeddings (the original recipe)

The original *Attention Is All You Need* (Vaswani et al., 2017) used **fixed sinusoidal** positional embeddings — deterministic vectors of shape `[max_seq, d_model]` computed as sines and cosines of the position with geometrically spaced frequencies — added to the token embeddings before the first block. GPT, GPT-2, and BERT swapped this for **learned absolute** positional embeddings — the same `[max_seq, d_model]` tensor, but trained by gradient descent alongside the rest of the model.

Structurally both variants behave the same way: position `t` gets a fixed vector `p_t` that is added to the token embedding at position `t`. This has three properties a fine-tuner needs to internalize:

1. **Position information lives at the input, not inside attention.** It flows to every layer through the residual stream but is never re-applied.
2. **The maximum context length is baked into the tensor shape.** If the learned embedding table has 2048 rows, positions ≥ 2048 have no embedding at all — the model literally cannot process them without either extending the table (and retraining those new rows from scratch) or interpolating between existing rows.
3. **Absolute position is arbitrary.** The model has no built-in notion that positions 500 and 600 are "closer" than positions 500 and 1000; that has to be learned.

Absolute embeddings dominated the encoder-heavy era (BERT, T5's absolute-like relative bias aside, early GPTs). They have almost entirely fallen out of decoder-only pretraining after 2022 because they extrapolate poorly and complicate context-length extension.

## RoPE: rotary position embeddings

Introduced by Su et al. (2021) in the RoFormer paper and adopted by essentially every modern decoder-only base model — Llama 1/2/3, Mistral, Qwen, Gemma, DeepSeek — **Rotary Position Embeddings (RoPE)** encode position by *rotating* the query and key vectors, per position, inside the attention layer.

### The mechanic

Pair up adjacent dimensions of `q` (and separately `k`) into 2-D subvectors. For each pair `(x_{2i}, x_{2i+1})` at position `m`, apply a 2-D rotation by angle `mθ_i`, where the frequencies `θ_i` are geometrically spaced:

```
θ_i = base ** (-2i / d_head)     with base typically 10000 (Llama 1/2) or 500000 (Llama 3)
```

The transformed pair becomes

```
[ cos(mθ_i)  -sin(mθ_i) ] [x_{2i}    ]
[ sin(mθ_i)   cos(mθ_i) ] [x_{2i+1}  ]
```

After applying RoPE to both `q` (at query position `m`) and `k` (at key position `n`), the inner product `q_m · k_n` depends on `(m − n)` — the **relative** position — not on `m` and `n` individually. RoPE encodes absolute positions in Q and K but the attention scores themselves are position-relative. This is the whole point of the design.

### Why RoPE won

- **Applied inside every layer**, on Q and K only. The residual stream and the values `v` never see it directly. Adding a new layer does not "dilute" positional information.
- **Encodes relative distance naturally.** Two tokens 5 apart get the same attention bias whether they are at positions (10, 15) or (1000, 1005) — up to the point where the rotation frequency wraps around.
- **Deterministic and parameter-free**, once the base `θ` is chosen. There is no learned table to grow when you extend context.

### The knob a fine-tuner cares about: `rope_theta` (and its scaling)

Because the highest-frequency rotations wrap around within the first hundred tokens or so, RoPE effectively has a "wavelength budget." A model pretrained at 4k context with `rope_theta = 10000` sees rotations from a specific frequency spectrum. Push it to 32k without adjustment and the high-frequency rotations become unrecognizable to the trained weights, and quality collapses.

Four families of context-length extension have been developed on top of RoPE, and you will meet all of them in mod-107 and mod-111:

- **Position Interpolation (PI)** — Chen et al. (2023). Divide every position by a scaling factor `s` before applying RoPE. Positions in `[0, s · L_pretrain]` now land inside the pretrained rotation regime. Requires a short SFT pass on long-context data to recover quality. Simple, cheap, and the baseline for everything below.
- **NTK-aware scaling** — proposed on the r/LocalLLaMA subreddit and rapidly adopted. Instead of scaling positions uniformly, scale the RoPE base `θ` to redistribute frequency loss across dimensions. Can extend context substantially *without any fine-tuning* on some models, though usually still improved by a short adaptation pass.
- **Dynamic NTK** — apply NTK scaling only past the pretrained length, keeping the un-extended regime unchanged.
- **YaRN** — Peng et al. (2023). Combines NTK-by-parts scaling, a temperature adjustment on attention softmax, and a short SFT budget. Currently the best-quality standard extension method; Qwen and several other open models ship YaRN'd variants directly.

Practically: **read `config.json` for `rope_theta` and `rope_scaling`** before you commit to a context-length target. Llama 3.1 8B, for example, ships with a `rope_scaling` block:

```jsonc
"rope_scaling": {
  "factor": 8.0,
  "low_freq_factor": 1.0,
  "high_freq_factor": 4.0,
  "original_max_position_embeddings": 8192,
  "rope_type": "llama3"
}
```

This lets a single set of pretrained weights be presented as either 8k-context (ignore the block) or 128k-context (apply the scaling). The trainer must be told which one you are targeting; get it wrong and long-context batches will attend nonsensically.

## ALiBi: attention with linear biases

Press, Smith, and Lewis (2022) introduced **ALiBi** — Attention with Linear Biases — as an alternative that eliminates positional embeddings entirely. Instead, ALiBi adds a fixed, non-learned bias to the raw attention scores that penalizes attention to distant tokens:

```
scores[i, j] += -m_h · |i - j|   for head h, where i ≥ j (causal)
```

`m_h` is a fixed **head-specific slope** chosen from a geometric sequence — smaller heads get gentler slopes, so different heads "see" different effective context ranges. There is no rotation, no positional embedding table, and no parameter that grows with sequence length.

ALiBi was adopted by MPT (MosaicML) and BLOOM, and by Falcon in some variants. Its signature property is **strong length extrapolation**: an ALiBi model trained to 2k context can generate coherently at 8k or beyond with no fine-tuning, because the linear-bias mechanism is defined for any distance. RoPE without scaling degrades at extrapolation; ALiBi does not.

The tradeoff is that ALiBi's inductive bias — "closer tokens matter more, monotonically" — is stricter than RoPE's, and empirical results on many benchmarks favor RoPE at fixed short-to-medium context. The industry has largely settled on RoPE + one of the extension schemes above rather than ALiBi, but a fine-tuner still meets ALiBi in MPT / BLOOM-based bases, in some long-context research checkpoints, and in older-generation open models a client may still be running in production.

## Why this matters for context-length fine-tuning

Every one of these mechanics constrains what "fine-tune to 32k context" actually means:

- **Absolute-embedding model?** You cannot simply extend context. The position embedding table has hard shape limits; adding rows means training random-init'd vectors from scratch, which is not fine-tuning in any useful sense. Practical answer: don't. Pick a different base.

- **RoPE model?** You have the four scaling levers above. Even the cheapest of them (PI or NTK) usually needs a short SFT run on long-context data to be quality-competitive — a few hundred million tokens is a common budget. mod-107 gives you the operator-altitude recipe.

- **ALiBi model?** Extrapolation is nearly free, but you may still need a short adaptation pass to teach the model to *use* the extended range (i.e., to attend usefully to tokens beyond its pretraining exposure — different from being able to score them at all).

Cross-cutting: **the KV cache scales linearly in context length.** A 70B model at 128k context can spend tens of GB on the KV cache alone. Any positional-encoding trick that unlocks context length is a distributed-serving problem waiting to happen; mod-109 covers what serving stacks (vLLM, TGI) actually do about it.

### One concrete workflow

Suppose you take Llama 3 8B (pretrained at 8k context) and want to fine-tune to a 32k effective context on legal document data.

1. **Check `config.json`** for `max_position_embeddings` (8192) and `rope_theta` (500000).
2. **Choose an extension recipe.** For 4× extension (8k → 32k), PI is a defensible cheap baseline; YaRN is the higher-quality choice with a modest additional SFT budget.
3. **Configure the model.** In Hugging Face this looks like passing a `rope_scaling` dict at `AutoModelForCausalLM.from_pretrained(..., rope_scaling={"type": "linear", "factor": 4.0})` — or picking Llama 3.1's built-in scaling if you use that release.
4. **Fine-tune on genuinely long-context data.** Short training samples do not teach the model to use long attention windows; a data mix heavy on documents at or near the new target length is required.
5. **Evaluate long-context quality specifically** — needle-in-a-haystack, RULER, LongBench. mod-108 covers this in depth.

## Exercise pointer

`exercises/exercise-03-rope-vs-alibi-context-length-experiment.md` has you run inference on a RoPE model beyond its trained context length with and without scaling, and then repeat the experiment on an ALiBi base, to feel the difference viscerally.

## Summary

- Attention is permutation-invariant; positional information must be injected somewhere.
- **Absolute embeddings** (added at the input) are limited by a fixed table size and are effectively legacy for new decoder-only pretraining.
- **RoPE** rotates Q and K inside every attention layer, encoding relative distance naturally, and is the near-universal default for modern open bases.
- **ALiBi** replaces embeddings with a fixed linear bias on attention scores per head, and extrapolates well but has been less commonly adopted at scale.
- Extending context on a RoPE model at fine-tuning time uses PI, NTK-aware scaling, dynamic NTK, or YaRN — plus a targeted SFT pass on long-context data. Extending context on an absolute-embedding model is impractical; on an ALiBi model, mostly free.
- Always read `rope_theta` and `rope_scaling` in `config.json` before you commit to a target context length. The wrong setting is silent and wrecks long-context quality.
