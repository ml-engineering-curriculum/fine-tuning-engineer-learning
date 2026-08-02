# PEFT Methods in Context: LoRA, QLoRA, IA³, Prefix Tuning, Prompt Tuning

**Estimated read:** 45 minutes.
**You should be able to, by the end:** describe what each of the five PEFT methods actually adds to a frozen base model, explain why LoRA / QLoRA have become the default and where the others still fit, and pick one of the five for a given use case using a decision matrix rather than a copy-paste.

## The shared move

Every PEFT method freezes the pretrained weights and adds a small set of trainable parameters. The methods differ along three axes:

1. **Where the extra parameters live.** Inside every linear? At the sequence level? As virtual tokens? As per-channel scalars?
2. **How they combine with the frozen weights.** Additive (LoRA), multiplicative (IA³), attention-mask extension (prefix / prompt tuning).
3. **How many parameters they cost.** Fractions of a percent of the base, in most cases; but the fractions differ.

Everything else — the training loop, the loss, the dataset, the base — is unchanged from mod-103. That is what makes PEFT drop-in.

The rest of this chapter tours the four other family members (LoRA + QLoRA were chapters 1 and 2) and lands on a decision matrix.

## IA³ — Infused Adapter by Inhibiting and Amplifying Inner Activations

**One-line summary:** learn a per-channel scaling vector applied to keys, values, and the MLP hidden activation. Multiplicative, not additive.

Liu et al., 2022 ("Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning") introduced IA³ as a further reduction on top of LoRA's parameter count. Instead of a low-rank matrix, IA³ learns a single **vector** per adapted place: multiply the frozen key by a learned diagonal, multiply the frozen value by another, multiply the MLP hidden state by a third.

For attention:

```
K_adapted = K · diag(l_k)      # l_k ∈ R^(d_head)
V_adapted = V · diag(l_v)      # l_v ∈ R^(d_head)
```

For the MLP:

```
hidden = act(W_up · x) · l_ff  # l_ff ∈ R^(d_intermediate)
```

At initialization all `l_*` vectors are all-1s, so the adapted model equals the base model. The training signal moves each element of each vector up or down (typically stays close to 1).

Parameter count for a Llama-3-8B model:

- One layer: `d_head · num_heads · 2 (K and V) + d_intermediate (MLP) ≈ 4096 · 2 + 14336 = 22,528`.
- 32 layers: ~720K parameters. Roughly **60× fewer** than a rank-16 LoRA on the same targets.

Why it can work: the multiplicative gates learn to *inhibit or amplify* directions that already exist in the frozen weights. If the base already knows how to answer support tickets and just needs to answer with fewer exclamation marks, gently attenuating certain feature channels can carry that. If the base does not know how to write SQL and you are teaching it, per-channel scaling on existing channels cannot invent new ones.

**Where IA³ shines:**

- Few-shot / very-low-data regimes (the paper's setting). Tens to low-thousands of training examples.
- Extremely constrained inference budgets where even LoRA's tens of MB per adapter is too much (multi-tenant systems with thousands of adapters).
- Serving-side style tweaks that are close to the base's existing behavior.

**Where IA³ falls off:**

- Larger data regimes (10k+ examples) — LoRA's expressive capacity almost always wins.
- Domain adaptation that requires new capabilities. Multiplicative gates cannot conjure directions.
- Any task where a rank-16 LoRA is already at the ceiling of what you want — IA³ will plateau lower.

`peft` supports IA³ via `IA3Config`:

```python
from peft import IA3Config, get_peft_model

config = IA3Config(
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],
)
model = get_peft_model(base, config)
```

Note the split between `target_modules` (both attention and MLP) and `feedforward_modules` (the subset that gets the MLP-style rescale rather than the attention-style rescale). `peft`'s docs list the correct settings per model family.

## Prefix Tuning

**One-line summary:** learn a set of "virtual token" key/value pairs prepended to every attention layer, keep everything else frozen. Extends the *attention state*, not the *weight matrices*.

Li and Liang, 2021 ("Prefix-Tuning: Optimizing Continuous Prompts for Generation") predates LoRA. The core observation: a transformer's attention layer looks up a K/V cache from past tokens. If you inject learnable K/V pairs at the *front* of the cache — as if there were `p` extra "prefix tokens" at positions 0 through p-1 — you can steer every attention head's behavior without touching the base weights.

Mechanically, for each attention layer `l`, prefix tuning maintains two learnable tensors:

```
P_K[l] ∈ R^(p × num_heads × d_head)
P_V[l] ∈ R^(p × num_heads × d_head)
```

At forward time, they are concatenated onto the K/V cache before attention: `K_full = concat(P_K, K)`, `V_full = concat(P_V, V)`. The current token attends over both. The base weights, tokenizer, and vocabulary are untouched.

`p` (prefix length) is the main hyperparameter. Typical values: 10–100. Parameter count per layer is `2 · p · num_heads · d_head`; total across a 32-layer 8B model with `p=32` is ~4M params.

**Where prefix tuning shines:**

- Generation tasks where you want a light, sequence-level steering (summarization tone, style transfer).
- Multi-tenant scenarios where the "adapter" needs to be small in absolute bytes.

**Where it falls off:**

- Long-context tasks. Every request pays for the prefix's K/V being computed and stored per-layer. At long contexts this is not free.
- Tasks that require weight-level adaptation. Prefix tuning cannot change what the model *knows*, only what it *pays attention to*.
- Modern architectures with GQA / grouped-KV. Prefix K/V has to match the base's K/V head structure exactly; misalignments in K-head count are a real footgun.
- Instruction-following SFT at scale. Empirically, LoRA outperforms prefix tuning on most SFT tasks reported in the literature; the paper's original setting was much smaller.

`peft` supports it via `PrefixTuningConfig` and works with the same trainer scaffolding you use for LoRA.

## Prompt Tuning

**One-line summary:** learn a set of continuous "soft" input embeddings prepended to every input. Same shape as regular token embeddings; distinct value space (not tied to any vocabulary word).

Lester et al., 2021 ("The Power of Scale for Parameter-Efficient Prompt Tuning") is the simplest PEFT method to describe. The trainable state is a single tensor:

```
P ∈ R^(p × d_hidden)
```

At forward time, `P` is prepended to the base's input embedding matrix for each example. The base's transformer stack sees `p` "soft prompt" embeddings followed by the tokenized input. Only `P` is trained.

Parameter count is `p · d_hidden`. For `p=100` on Llama-3-8B (hidden 4096), that is 409,600 parameters. Roughly **the size of a small classifier's weights**.

The paper's central finding is in the title: **prompt tuning's effectiveness scales with base model size**. At smaller models it lags full fine-tuning noticeably; at larger models (T5-XXL, 10B+ base) it approaches full-fine-tuning quality on many benchmarks. Interpretation: bigger bases have enough latent capability that a small learned prompt is sufficient to steer them into it.

**Where prompt tuning shines:**

- Very large bases (10B+) where you want the smallest possible per-tenant footprint.
- Tasks that are essentially "unlock a capability the base already has" — classification-shaped tasks, prompt-response formatting.
- Cases where you *want* to be able to inspect / interpret the trained state as a coherent addition to a vocabulary embedding.

**Where it falls off:**

- Small bases (< 3B). Prompt tuning underperforms LoRA by a wide margin.
- Multi-turn instruction following. A soft prompt at position 0 has to steer arbitrarily-long conversations from far away; it does not.
- Tasks that require weight-level adaptation. Same limitation as prefix tuning.

`peft` supports it via `PromptTuningConfig`. Initialization matters — the paper recommends initializing `P` from the embeddings of a text prompt ("Classify the following sentence:") rather than random init. `peft` supports both via `prompt_tuning_init="TEXT"` vs `"RANDOM"`.

## P-Tuning v2

Worth mentioning briefly: Liu et al., 2022 ("P-Tuning v2") is essentially prefix tuning generalized to work well at all model scales (v1 was a smaller-scale variant). Practically, prefer LoRA / QLoRA for new work; prefer prefix tuning or P-Tuning v2 only when the situation-specific reasoning above says so.

## Adapter modules (Houlsby-style, historical)

The original "adapter" concept (Houlsby et al., 2019) inserted small trainable bottleneck MLPs *between* transformer sub-layers rather than modifying existing linears. Serial-adapter and parallel-adapter variants exist. These predate LoRA and have been largely displaced by it — LoRA achieves similar or better quality at similar parameter counts, with a simpler mental model (the update is an additive delta on an existing weight, not a new module in the graph).

You will meet Houlsby-style adapters in papers from 2019–2021 and in some `peft` config paths (`AdaptionPromptConfig` for Llama-Adapter, for instance). Practically, LoRA / QLoRA is the modern default; Houlsby adapters are useful history.

## Decision matrix

The right way to pick a PEFT method is to have four numbers in mind about the deployment:

1. **Base size.** Small (< 3B), medium (3B–13B), large (13B+), very large (70B+).
2. **Training data volume.** Very low (< 1k), low (1k–10k), medium (10k–100k), large (100k+).
3. **Serving-side constraint.** How many adapters per tenant / task; how tight the memory / latency budget is.
4. **Task shape.** Style transfer / light steering vs. new-capability domain adaptation.

The matrix:

| Situation | Recommended | Second choice | Notes |
|---|---|---|---|
| Small base (< 3B), medium+ data | Full FT (mod-103) | LoRA r=32 all-linear | PEFT savings small at this scale |
| 7B–13B, single 24 GB GPU, medium data | **QLoRA** | LoRA if you have 48 GB | The default single-GPU SFT recipe |
| 7B–13B, A100 40/80 GB, medium data | **LoRA** | QLoRA | Cleaner distributed, no quant tax |
| 13B–70B, single node multi-GPU | LoRA | QLoRA (if FSDP support works) | Mod-107 covers distributed strategies |
| Very-large base (70B+), light steering | Prompt tuning or LoRA r=8 | QLoRA r=16 | Prompt tuning's scale advantage shows up here |
| Multi-tenant serving, thousands of tenants | LoRA (small r) or IA³ | Prompt tuning | Per-adapter footprint dominates memory math |
| Few-shot / very-low-data | IA³ or LoRA r=4 | Prompt tuning | Small tunable state prevents overfit |
| Heavy domain adaptation (new capabilities) | Full FT | LoRA r=64 all-linear | PEFT constrains updates — sometimes too much |
| Long continued-pretraining | Full FT | Large-rank LoRA (loses most PEFT savings) | Not the PEFT-shaped problem |

The matrix is a *starting bet*. Every deployment has to run its own eval and confirm.

## The "which is best" question, honestly

The empirical picture across 2023–2025 papers, in one paragraph: LoRA (and QLoRA on tight GPU budgets) has become the default because it is the only method with a good story across the four axes at once — parameter efficient, quality close to full FT for most SFT-style tasks, plays well with modern architectures, and has broad tooling support (`peft`, `bitsandbytes`, vLLM, TGI, LoRAX, Axolotl, Unsloth). IA³ is a real win on very-low-data or extreme parameter-budget scenarios. Prefix and prompt tuning are worth understanding for historical and specific-use-case reasons but rarely the right first choice on modern SFT. Anyone claiming a universal PEFT winner is either selling something or working in a narrow slice of the design space.

## Exercise pointer

`exercises/exercise-04-peft-methods-comparison-matrix.md` has you run three of the methods on the same base and dataset, produce a small comparison matrix of your own (parameters, VRAM, wall-clock, final eval), and defend a choice for a specified scenario.

## Summary

- Every PEFT method freezes the base and adds a small trainable state; they differ in *where* it lives and *how* it combines.
- **LoRA / QLoRA:** additive low-rank update on existing linears. The default; strongest across most axes.
- **IA³:** per-channel multiplicative rescaling on K, V, and MLP hidden. 60× fewer params than LoRA; best in few-shot / extreme-budget cases.
- **Prefix tuning:** learnable K/V pairs prepended to every attention layer's cache. Steers attention without touching weights; awkward with GQA and long contexts.
- **Prompt tuning:** learnable soft input embeddings prepended to every input. Simplest method; effective only at very large base scales.
- The decision matrix depends on base size, data volume, serving constraints, and task shape. Copy it, then eval your specific setup — the matrix is a starting bet, not a proof.
