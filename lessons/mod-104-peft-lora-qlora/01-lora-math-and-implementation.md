# LoRA: The Math, the Layer, and the Trade-offs

**Estimated read:** 50 minutes.
**You should be able to, by the end:** derive the low-rank update `ΔW = BA` from the Hu et al. paper, implement a working `LoRALinear` module in fewer than 40 lines of PyTorch, and explain what changing `r`, `alpha`, and the target-module list does to (a) trainable parameter count, (b) update magnitude at initialization, and (c) which parts of the model actually adapt.

## Why we need PEFT at all

Mod-103 walked through full-parameter SFT: every weight in the base model is updated, and every gradient / optimizer moment for every weight lives in GPU memory. At BF16 with AdamW, a 7B base sits around 84 GB *before* activations (chapter 2 in mod-103 has the arithmetic). One 24 GB consumer GPU does not fit that. Neither does a single A100 40 GB. This is the memory wall that Parameter-Efficient Fine-Tuning (PEFT) exists to route around.

The PEFT family's shared move is: **freeze the pretrained weights and train a small set of extra parameters that live alongside them.** Different PEFT methods differ in where those extra parameters sit and how they combine with the frozen weights (chapter 4 tours the family). LoRA — Low-Rank Adaptation, from Hu et al., 2021 — is the method that has become the default, and this chapter is about the math and the layer.

## The core claim

For any linear layer with weight matrix `W ∈ R^(d_out × d_in)`, LoRA replaces the fine-tuning update `W → W + ΔW` with

```
ΔW = (alpha / r) · B · A
```

where `A ∈ R^(r × d_in)` and `B ∈ R^(d_out × r)`, and `r ≪ min(d_in, d_out)`. `A` and `B` are trained; `W` stays frozen. `alpha` is a scalar; `r` is the rank — the low-rank hyperparameter.

The forward pass through the adapted layer, for an input `x ∈ R^(d_in)`, becomes:

```
y = W x + (alpha / r) · B · A · x
     └────┘   └───────────────┘
      frozen   trainable update
```

At initialization, `A ∼ N(0, σ²)` (Kaiming-uniform in the reference implementation) and `B = 0`. Because `B = 0`, `ΔW = 0` at step 0, so the adapted model is *exactly* the base model on the first forward pass. Training then moves `B` (and `A`) away from that initialization.

Two facts fall out immediately, and they are worth stating explicitly:

1. **The trainable parameter count for one adapted layer is `r · (d_in + d_out)`,** not `d_in · d_out`. For a 4096-dim attention projection in a 7B model, `r = 8` gives `8 · (4096 + 4096) = 65,536` parameters instead of `4096 · 4096 = 16,777,216`. That is a 256× reduction, per layer.
2. **The optimizer state and gradients live only on the trainable parameters.** AdamW's 8-byte-per-param moments are on the LoRA parameters only, not the frozen base. This is where the memory savings actually come from — the base weights are ballast, not budget.

## Why low rank works at all

Hu et al. observed empirically that the *update* a fine-tune produces on top of a well-pretrained base has approximately low **intrinsic** rank — meaning the change to each weight matrix over the course of fine-tuning is well-approximated by a rank-r matrix for small r. This is not a claim about the pretrained weights themselves (which are close to full rank), but about `ΔW`. Think of it as: the base model already knows how to do most of what a fine-tuned task needs; the delta to specialize it is small in an information-theoretic sense.

The paper's ablations show r as small as 1 or 2 recovering full-fine-tuning quality on some GLUE-scale tasks; on generative SFT of larger bases, r in the range 8–64 is the common working zone (see chapter 4 for how the numbers change per method and per task). This is empirical and task-dependent; treat "low rank works" as a starting bet, not a theorem.

## A working `LoRALinear` in PyTorch

Read this once. Type it out yourself before you use `peft`. If you cannot re-derive it, you will not be able to debug the `peft` version when it misbehaves.

```python
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A drop-in replacement for nn.Linear that adds a low-rank trainable update."""

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        d_in = base_linear.in_features
        d_out = base_linear.out_features

        # A: (r, d_in). Kaiming-uniform init, as in the reference implementation.
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: (d_out, r). Zero-init — this is what makes ΔW = 0 at step 0.
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen base pathway.
        base_out = self.base(x)

        # Trainable low-rank pathway.  x: (..., d_in) -> Ax: (..., r) -> BAx: (..., d_out)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * lora_out
```

Things worth pausing on:

- `self.base.weight.requires_grad_(False)` is not optional. If you forget it, the optimizer will update the base weights and you have accidentally done full fine-tuning with an extra low-rank appendix. The `peft` library does this for you; a hand-rolled implementation must do it explicitly.
- The forward pass is written as two matmuls (`x @ A.T @ B.T`) rather than one (`x @ (B @ A).T`). At training time the two-matmul form is faster because you never materialize the `d_out × d_in` product. `B @ A` only shows up when you *merge* the adapter into the base — chapter 3.
- The scaling factor is `alpha / r`. Doubling `r` while keeping `alpha` fixed halves the effective update magnitude — you will feel this in later sections. Some implementations use `alpha` directly instead of `alpha / r`; check yours before you compare loss curves against a paper's reported settings.
- Dropout is applied to the *input* of the LoRA path, not to the LoRA output. This is the reference-implementation choice; it prevents the trained low-rank update from over-specializing to individual training examples.

To wrap a real transformer, walk the base model's modules and swap `nn.Linear` layers you want to adapt:

```python
def add_lora(model: nn.Module, target_names: list[str], r: int, alpha: int, dropout: float):
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(t) for t in target_names):
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        child_name = name.rsplit(".", 1)[1]
        setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
    # Freeze everything else.
    for p in model.parameters():
        p.requires_grad_(False)
    # Re-enable requires_grad on LoRA params only.
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad_(True)
    return model
```

The `target_names` argument is where you pick which weight matrices adapt — the topic of the next section.

## `r`, `alpha`, and target modules — the three knobs

### Rank `r`

`r` sets **capacity**. Each rank-1 component `b_i · a_i^T` (where `b_i`, `a_i` are columns/rows of `B`, `A`) is a rank-1 outer product; the update `ΔW` is the sum of `r` of them.

Practical ranges seen in the literature and in real recipes:

- `r = 4` to `r = 8` — style transfer, small behavioral changes, chat-tone SFT. Common for LoRA on 7B–13B bases.
- `r = 16` to `r = 32` — moderate task adaptation, e.g., a domain SFT that changes what the model knows about, not just how it talks. Common for QLoRA runs.
- `r = 64` to `r = 128` — larger behavioral shifts, or when you know your task is at the ceiling of low-rank capacity. Diminishing returns beyond this on most SFT-style workloads.

Two failure modes on the rank axis (chapter 5 catalogs them properly):

- **Rank too small.** The training loss stalls above where full fine-tuning would land, and no amount of learning-rate tuning fixes it. Diagnostic: retrain at 2× rank; if loss drops meaningfully, you were rank-bound.
- **Rank too large.** Trainable-param count grows linearly with `r` so wall-clock and memory grow, and — on small datasets — you overfit sooner. Diagnostic: eval loss diverges from train loss faster than a smaller-rank run.

### Alpha `alpha`

`alpha` sets the **effective learning rate on the low-rank update**. Because `ΔW = (alpha / r) · B · A`, doubling alpha is (approximately) equivalent to doubling the learning rate on the LoRA parameters.

The literature convention is `alpha = 2 · r` (so the scaling is `2`), or `alpha = r` (scaling 1). "LoRA+" recipes and QLoRA papers sometimes use different ratios; the QLoRA paper commonly sets `alpha = 16` at `r = 64` (scaling `0.25`). What matters is the *product* `alpha / r`, and that different projects report the two numbers separately even when they mean the same effective scaling.

Rule of thumb when you are copying a config: if you change `r`, either scale `alpha` proportionally to keep `alpha / r` constant, or accept that you are also changing the effective learning rate.

### Target modules — which weights adapt

The set of `nn.Linear` layers you wrap with `LoRALinear` determines *where in the model* the adaptation happens. In a decoder-only transformer block there are typically:

- **Attention projections** — `q_proj`, `k_proj`, `v_proj`, `o_proj` (Llama-style naming).
- **MLP projections** — `gate_proj`, `up_proj`, `down_proj` (Llama / Mistral SwiGLU).
- **Embeddings** — the input token embedding and the LM head (often tied to the input embedding).

Three common target-module choices, from cheapest to most capable:

1. **`q_proj` and `v_proj` only.** The original LoRA paper's default for GPT-style models. Smallest parameter budget. Works well for style-transfer SFT on small models.
2. **All four attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`).** The QLoRA paper's finding for 7B–65B Llama-family models was that attention-only adaptation captures most of the fine-tuning value on general SFT.
3. **All linear layers (attention + MLP).** Highest capacity, roughly 3× the trainable parameters of attention-only on modern MLP-heavy architectures. Preferred when you have the compute and the task genuinely needs more expressive change. This is the QLoRA paper's actual recommendation for their setups.

You will meet the `peft` convention:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 6,780,357,632 || trainable%: 0.6186
```

**Warning about `target_modules` and architecture.** The names `q_proj`, `k_proj`, etc. are model-family conventions. Mistral / Llama use these; older GPT-2-family models have `c_attn` (a fused QKV linear); some architectures use `Wqkv`. `LoraConfig(target_modules="all-linear")` (in modern `peft`) or the explicit list generated by `[n for n, m in model.named_modules() if isinstance(m, nn.Linear)]` is safer than a hard-coded string list you copied from a different model family.

**Do not adapt the LM head or the embeddings by default.** They are large and full fine-tuning usually treats them specially; hitting them with LoRA changes what "token identity" means and often hurts. `peft` explicitly excludes them unless you opt in with `modules_to_save`.

## Whom the update actually reaches

The mental model of "the adapter changes attention" needs a small correction. When you LoRA `q_proj`, you are not changing the base query weight in place — you are adding a low-rank *additive* update that only affects the output of that specific linear at forward time. The rest of attention (softmax, causal mask, RoPE, output projection) sees a *slightly perturbed* query and does its normal thing with it.

The consequence: LoRA on `q_proj` and `v_proj` shapes what queries and values the attention head is asking / carrying, but the attention *pattern* (softmax over queries · keys) is only weakly steered because `k_proj` was not touched. If you want to change attention *patterns*, LoRA `q_proj` *and* `k_proj`.

Symmetric argument for the MLP block: LoRA on `up_proj` and `down_proj` reshapes the feature computation happening inside each MLP; LoRA on `gate_proj` steers the gating (in SwiGLU / GEGLU architectures) that decides which features fire.

You are not obligated to reason about it at this depth for every run. You are obligated to notice when the target-module choice does not match what you were trying to change.

## Parameter-count math you should be able to do in your head

For a Llama-3-8B model (4096 hidden, ~32 layers, MLP intermediate 14336):

- One attention linear (`q`, `k`, `v`, `o`): 4096 × 4096 = 16.78M params. LoRA r=16: 16 · (4096 + 4096) = 131,072 params. 128× reduction.
- One MLP up-projection (`up_proj` or `gate_proj`): 4096 × 14336 = 58.7M params. LoRA r=16: 16 · (4096 + 14336) = 294,912 params. 199× reduction.
- One MLP down-projection (`down_proj`): 14336 × 4096 = 58.7M params. Same LoRA cost as up. 199× reduction.

For 32 layers and *all* linear targets at r=16 (7 linears per block: q, k, v, o, gate, up, down):

```
per-layer LoRA params = 4 · 131,072 (attn) + 2 · 294,912 (up/gate) + 1 · 294,912 (down)
                      = 524,288 + 589,824 + 294,912 = 1,409,024
total LoRA params     = 32 · 1,409,024 ≈ 45M
```

Against 8B base parameters, that is ~0.56% trainable. This is what `print_trainable_parameters()` will confirm. Being able to do this math is what lets you catch config bugs (wrong `r`, forgotten module) from the trainer's startup log alone.

## When LoRA does *not* work

Three settings where LoRA is a poor fit and you should reach for full fine-tuning (mod-103) or another PEFT method (chapter 4):

- **Very small bases (< 500M).** The base has too little capacity to leave room for a low-rank *addition* to matter — full fine-tuning is cheap enough that PEFT does not buy you much.
- **Very heavy domain shifts.** Adapting a code base model to a legal-writing SFT is a large distribution shift; the low-rank assumption may not hold. Empirical signal: LoRA loss curves plateau higher than a full fine-tune, regardless of rank / alpha / target-module choices.
- **Long continued-pretraining.** If your run is closer to "continue pretraining on 100B tokens of new domain text" than to "SFT on 50k conversations," LoRA constrains the update in a way that hurts. Full fine-tuning or a large-rank LoRA over the whole model at long horizons converges more like full FT — but at that point you have given up most of the PEFT savings.

## Summary

- LoRA replaces the fine-tuning update `ΔW` with a low-rank factorization `(alpha / r) · B · A`, where `A` and `B` are trained and the base weight `W` stays frozen.
- At initialization `B = 0` so the adapted model equals the base model on step 0. Only `A` and `B` receive gradients and optimizer state, which is where the memory savings come from.
- A working `LoRALinear` module is ~40 lines of PyTorch. Type it out. The two-matmul forward, the zero-init on `B`, and the `requires_grad_(False)` on the base weight are the load-bearing details.
- `r` sets capacity (4–8 for style; 16–32 for moderate adaptation; 64–128 for larger shifts). `alpha` sets update-magnitude scaling; the product `alpha / r` is what matters — copy configs with that ratio held constant. Target modules pick *which* layers adapt (attention-only vs attention + MLP is the primary axis).
- LoRA is a poor fit for very small bases, very heavy domain shifts, and long continued-pretraining. In those regimes, either accept the plateau or step out to full fine-tuning.
