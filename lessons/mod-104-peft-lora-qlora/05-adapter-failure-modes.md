# Adapter Failure Modes: Rank Collapse, Target Mis-Selection, Alpha Mis-Scaling

**Estimated read:** 40 minutes.
**You should be able to, by the end:** recognize the loss-curve, eval-delta, and singular-value signatures of the three most common LoRA / QLoRA failure modes, and know which config change to try first for each.

## The general shape of an adapter-specific failure

Mod-103 chapter 5 covered the SFT-general failure catalog: overfitting, catastrophic forgetting, template leak. Those hit LoRA runs too — the training objective is the same and the mask is the same. This chapter is about the three additional failure modes that only exist because the update is low-rank and gated by scalars you chose.

The specifics:

1. **Rank collapse.** The effective rank of the learned update is smaller than the nominal `r`. Some singular values of `BA` are zero (or numerically zero) throughout training. The adapter is silently wasting parameter budget.
2. **Target-module mis-selection.** The set of layers you adapted does not match what the task requires. Either you picked too few (loss stalls high) or you picked the wrong ones (loss drops but eval regresses).
3. **Alpha mis-scaling.** The effective LR on the LoRA path (`alpha / r`) is too high or too low. Symptoms overlap with generic LR mis-tuning but the fix is often to change the ratio, not the base LR.

Each has a distinctive fingerprint. Each has a first thing to try.

## Rank collapse

### What it is

At any point in training, the trained update for a layer is `ΔW = B A ∈ R^(d_out × d_in)`. Its rank is at most `min(r, d_in, d_out) = r` (assuming `r < d_in, d_out`). Ideally the update *is* rank-`r` — every one of `r` singular values is non-trivially non-zero.

Rank collapse happens when some singular values of `BA` are essentially zero throughout training. If your r=16 adapter has only 4 non-zero singular values, you paid for 16 ranks of capacity and used 4. That is fine if 4 was enough; it is a design problem if it was not, because you cannot tell from loss alone.

### The diagnostic

Compute the singular values of `B @ A` for each adapted layer at end of training (or during training, at each eval step):

```python
import torch

for name, module in model.named_modules():
    # peft's LoraLayer holds .lora_A[adapter], .lora_B[adapter]
    if not hasattr(module, "lora_A"):
        continue
    A = module.lora_A["default"].weight   # (r, d_in)
    B = module.lora_B["default"].weight   # (d_out, r)
    delta = B @ A                          # (d_out, d_in)
    s = torch.linalg.svdvals(delta)        # (min(d_out, d_in),)
    print(name, s[:module.r].tolist())
```

Read the first `r` singular values. A healthy adapter's singular values decay smoothly (largest to smallest is typically a factor of 3–10× across the `r` non-trivial ranks). A collapsed adapter has a hard cliff — the first `k < r` values are non-trivial and the rest are numerically zero.

An alternate cheap proxy is the *effective rank*:

```
eff_rank = (sum(s) ** 2) / sum(s ** 2)   # Roy & Vetterli's stable rank
```

Report this per-layer. A healthy r=16 adapter usually has effective rank in the 8–14 range; below 4 is a signal.

### What causes it

Three common causes, roughly in order of frequency:

1. **`r` was too small for the task.** The optimization saturates the available directions early and further training just refines them.
2. **Too little training data.** With few examples the gradient does not push into all `r` dimensions.
3. **`alpha / r` too low.** The update magnitude stays small, so only the strongest gradient directions accumulate; the rest never move.

### The fix, in order

1. **Try `r = 2r`.** If the effective rank was 4 out of 16, and doubling `r` to 32 shifts you to 8 out of 32, `r` is probably not the bottleneck (the task genuinely wants low rank). If effective rank stays at 4 out of 32 *and* training loss did not drop, you were already saturated.
2. **Increase `alpha` to keep `alpha / r` the same.** Same scaling, more expressive capacity.
3. **Increase learning rate on the LoRA path.** For QLoRA especially; the paper's LR of 2e-4 is aggressive for a reason.
4. **Add more or more-diverse training data.** If the run is data-limited, the effective rank will not grow no matter what.

Rank collapse is often less of a bug and more of a "you overpaid for rank." Detecting it lets you drop `r` and save memory / wall-clock without losing quality.

## Target-module mis-selection

### What it is

You picked `target_modules` that either do not cover the layers the task actually needs to change, or cover too many, or cover the wrong ones (LM head when you did not mean to).

Three specific mis-selections:

- **Attention-only when the task needs MLP.** Attention adapts *what* the model attends to; MLP adapts *what it computes* per position. Domain adaptation with new terminology / new patterns typically wants MLP; style transfer often does not.
- **All-linear including the LM head or embeddings.** `target_modules="all-linear"` on some `peft` versions has swept the LM head into the target set. This often improves training loss because the LM head is a big lever; it often regresses eval because you have partially retrained token-identity.
- **Wrong architecture strings.** You pass `["q_proj", "v_proj"]` for a model where the linear is fused as `Wqkv`. `peft` will warn and adapt nothing. The trainer runs; loss barely decreases; you think the run was unlucky.

### The diagnostics

The three signals to check, in order:

1. **`print_trainable_parameters()`** at model construction. The number should be within a small factor of your parameter-count math (chapter 1). Off by 10× means you missed a target family; off by 2× means the LM head is in or out unexpectedly.
2. **Grep the model for the target strings**:

```python
target_hits = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and any(n.endswith(t) for t in ["q_proj", "k_proj", "v_proj", "o_proj"])]
print(len(target_hits), "matched linears")
```

Zero matches means you have the wrong strings for this base. Compare to `[n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]` to see the actual names.

3. **Compare `set(model.named_parameters(...))` before and after `get_peft_model`.** Any weight you did *not* expect to become trainable is a bug. Any weight you *did* expect to become trainable but did not is a bug.

### What each mis-selection looks like

- **Too few targets (`q`, `v` only for a domain task).** Training loss drops early, then plateaus higher than a run with all-linear at the same `r`. Eval matches training — the model is just not learning enough.
- **Too many targets (all-linear + embeddings).** Training loss looks great. Eval on held-out prompts regresses on tasks the base was already good at (base capabilities are being unlearned by hitting the embedding / LM head). Catastrophic forgetting via LoRA.
- **Wrong string entirely.** `print_trainable_parameters()` reports a suspiciously small number (or zero if nothing matched); loss basically flat.

### The fix

- If you got zero matches: dump the model's linear names and pick actual strings. `"all-linear"` (in modern `peft`) is safe if you also verify the LM head is not swept in.
- If eval regressed: remove the LM head / embeddings from the target set. Retrain from scratch (not from the regressed adapter).
- If you were attention-only and stalled: extend to MLP (`gate_proj`, `up_proj`, `down_proj`). Almost always a strict improvement on modern SwiGLU architectures at similar r.

## Alpha mis-scaling

### What it is

Recall: the effective update magnitude scales as `alpha / r`. Doubling alpha is (approximately) doubling the learning rate on the LoRA path. Two mis-scaling patterns:

- **Alpha too high.** Adapter behaves like a very-high-LR full-parameter run. Loss is noisy, may not converge, may diverge at long horizons.
- **Alpha too low.** Adapter under-trains — loss decreases slowly and plateaus higher than it should.

### The trap: copying alpha across ranks

The most common alpha-mis-scaling bug is copying a config's `alpha` between different `r` values. A paper that used `r=64, alpha=16` (ratio 0.25) is not the same recipe at `r=8, alpha=16` (ratio 2.0). You changed the effective LR by 8×.

Rule: **when you sweep `r`, keep `alpha / r` constant.** Explicitly: `alpha = k · r` for some k you picked (typical: k = 1 or k = 2).

### The diagnostic

Two signals distinguish alpha mis-scaling from generic LR mis-tuning:

1. **The pattern is consistent across bases.** If moving from Llama-3-8B (alpha_ratio X) to Mistral-7B (same alpha_ratio X) reproduces the same shape, and both diverge from a paper's reported curve, the culprit is likely alpha_ratio, not per-model LR sensitivity.
2. **A `torch.linalg.norm(module.lora_B @ module.lora_A)` sweep** across training tells you the actual update magnitude over time. If that norm grows unusually fast, your alpha ratio is high; unusually slow, your ratio is low. Log this once per eval step for a few representative layers.

### The fix

The three-step fix, in order:

1. Divide `alpha` by 2 (or double it) and re-run for 10% of the original steps. See which direction changed the loss curve toward what you wanted.
2. Once you know the direction, sweep `alpha ∈ {r/4, r/2, r, 2r, 4r}` at fixed `r`. This is 5 short runs; do them in parallel or serially depending on the compute.
3. Pick the smallest `alpha` that gets you to the target loss. Smaller alpha = more stable and closer to the base at each intermediate checkpoint.

### `rslora`: a scaling-invariant alternative

The `rslora` (rank-stabilized LoRA) variant, from Kalajdzievski, 2023 ("A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA"), replaces the scaling factor with `alpha / sqrt(r)` instead of `alpha / r`. This makes the effective update magnitude invariant to `r` — you can sweep rank without re-tuning alpha. Supported by `peft` via `LoraConfig(use_rslora=True)`.

If you find yourself sweeping `r` a lot and forgetting to re-scale `alpha`, `rslora` removes the footgun. It is not a strict quality improvement — some tasks prefer classic LoRA scaling — but it is one config flag and worth trying if `r`-sweep pain is your bottleneck.

## The failure catalog, at a glance

| Fingerprint | Likely cause | First fix |
|---|---|---|
| Loss plateaus high; effective rank saturates well below `r` | Rank ceiling; `alpha / r` too low | Increase `alpha / r`; only then increase `r` |
| Loss looks fine; eval regresses on held-out base tasks | LM head / embedding in target set | Remove from targets, retrain |
| Trainable params << expected | Wrong target-module strings for the base | Grep model for the correct names |
| Loss noisy / diverges | `alpha / r` too high | Halve `alpha` |
| Loss decreases painfully slowly | `alpha / r` too low OR LR too low | Halve `alpha`, then double base LR, in that order |
| Effective rank ~ half of `r`, training loss OK | Overpaid for rank | Cut `r` in half, keep `alpha / r` |
| QLoRA loss noisier than the paper's reported curve | Compute dtype mis-set or double-quant off | Confirm `bnb_4bit_compute_dtype=torch.bfloat16` and `bnb_4bit_use_double_quant=True` |
| Adapter loads but produces base-model outputs | `set_adapter` never called OR adapter files are placeholder / empty | Verify `A` has non-zero rows; verify `set_adapter` was called |

The last row is a specifically nasty variant of the "wrong strings" bug at *serving time*: the adapter loads without errors, the pipeline works, and outputs match the base because the adapter path never activates. Add an assertion at deploy time that `PeftModel.active_adapter` matches the expected name.

## The one-shot LoRA health check

A short script to run against a trained adapter before you call it done:

```python
def lora_health(model, adapter_name="default"):
    reports = []
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        A = module.lora_A[adapter_name].weight.float()
        B = module.lora_B[adapter_name].weight.float()
        r = module.r[adapter_name] if isinstance(module.r, dict) else module.r
        delta = B @ A
        s = torch.linalg.svdvals(delta)
        eff_rank = (s.sum() ** 2 / (s ** 2).sum()).item()
        max_sv = s.max().item()
        min_sv = s[:r].min().item()
        reports.append({
            "layer": name,
            "r": r,
            "eff_rank": eff_rank,
            "max_sv": max_sv,
            "min_sv": min_sv,
            "sv_ratio": max_sv / max(min_sv, 1e-12),
        })
    return reports
```

Flag any layer whose `eff_rank < r / 3` or whose `sv_ratio > 1e6`. Both signal a layer that is not carrying its share.

## Interaction with the SFT failure catalog

A few overlaps with mod-103 chapter 5 worth naming:

- **Overfitting under LoRA looks like `train_loss / eval_loss` divergence, same as full FT.** The fix ranking is different: reduce `r` (lowers capacity), *then* add dropout, *then* early-stop. Reducing LR / adding weight decay helps less than in full FT.
- **Catastrophic forgetting under LoRA is usually target-module driven.** LoRA on attention-only tends to forget less than all-linear + LM head, because the LM head is where "which token to emit" lives. If you regress on base-capability evals, remove the LM head from targets first.
- **Template leak under LoRA is not different from template leak under full FT.** The mask is the loss's problem, not the adapter's problem.

## Exercise pointer

`exercises/exercise-01-lora-from-scratch-and-against-peft.md` walks through the from-scratch implementation and asks you to reproduce the health-check script above; `exercises/exercise-02-qlora-on-a-7b-base-single-gpu.md` asks you to run the health check on a real QLoRA run and defend the resulting numbers.

## Summary

- LoRA-specific failure modes: **rank collapse** (effective rank << nominal r), **target-module mis-selection** (too few, too many, or wrong strings), and **alpha mis-scaling** (`alpha / r` too high or too low).
- Diagnose rank collapse by computing singular values of `BA`; healthy adapters use most of their `r` non-trivially.
- Diagnose target mis-selection by (a) checking `print_trainable_parameters()` against your math, (b) grepping model linears for the target strings, and (c) watching eval regressions on held-out base tasks.
- Diagnose alpha mis-scaling by tracking `‖BA‖` over training and by remembering that `alpha / r` — not `alpha` alone — is what controls the LoRA path's effective step size.
- `rslora` (via `LoraConfig(use_rslora=True)`) makes the update magnitude invariant to `r` and removes the "forgot to re-tune alpha when I changed rank" footgun.
- Adapter-specific failure modes overlap with SFT-general failure modes (mod-103 chapter 5) but the fix priority is different — reduce `r` before adding weight decay when overfitting; remove the LM head from targets before adding replay data when forgetting.
