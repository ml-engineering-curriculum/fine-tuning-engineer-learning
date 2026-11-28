# lab-03: Reading `rope_scaling` Across Real Releases

**Estimated duration:** 45 minutes.
**Companion chapter:** [`03-positional-encodings.md`](../03-positional-encodings.md).

## Objective

Teach your eyes to read the `rope_scaling` block in a real `config.json` and immediately know *which* context-extension recipe a release is using, what its pretrained context length was, and what you would need to change if you wanted to fine-tune it to a different length. By the end you will have inspected three real configs and produced a small table that maps each one to the correct recipe from chapter 3.

This lab is reading-and-reasoning, no GPU required. It does not duplicate [`exercise-03`](../exercises/exercise-03-rope-vs-alibi-context-length-experiment.md), which runs inference past the pretrained context length to *measure* quality — here you are learning to read the shipped metadata that drives those runs.

## Prerequisites

- Python 3.10+ with `transformers` installed (only for the convenience of `AutoConfig`; curl works too).
- A Hugging Face Hub login with any required license acceptances for the models you pick.
- You have read chapter 3 end-to-end, including the "The knob a fine-tuner cares about: `rope_theta`" section and the Llama 3.1 `rope_scaling` example.

## Steps

### Step 1 — Fetch three configs

Pick three models that represent distinct points on the "RoPE extension landscape":

| Model | What it should illustrate |
|---|---|
| `meta-llama/Meta-Llama-3-8B` | Base Llama 3, pretrained at 8k, no `rope_scaling` block. |
| `meta-llama/Meta-Llama-3.1-8B` | Same architecture, same weights lineage, but shipped with a `rope_scaling` block that re-presents it as 128k-context. |
| `Qwen/Qwen2.5-7B-Instruct` | A separate vendor's choice on vocab, `rope_theta`, and how (or whether) to bake in long-context scaling. |

If you cannot obtain Llama licenses, substitute any `Qwen/Qwen2.5-*`, `mistralai/Mistral-*`, and a `*YaRN*` community checkpoint — the goal is three configs with meaningfully different `rope_*` settings.

```python
from transformers import AutoConfig

for model_id in [
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3.1-8B",
    "Qwen/Qwen2.5-7B-Instruct",
]:
    c = AutoConfig.from_pretrained(model_id)
    print(f"--- {model_id} ---")
    print("max_position_embeddings:", c.max_position_embeddings)
    print("rope_theta             :", getattr(c, "rope_theta", "n/a"))
    print("rope_scaling           :", getattr(c, "rope_scaling", None))
    print()
```

**Expected output (shape, not exact numbers):** Llama 3 8B shows `max_position_embeddings=8192`, `rope_theta=500000`, and `rope_scaling=None`. Llama 3.1 8B shows `max_position_embeddings=131072` with a populated `rope_scaling` dict naming `rope_type` (e.g. `"llama3"`) and a `factor`. Qwen 2.5 7B shows its own choices — your job in later steps is to read them.

### Step 2 — Match each model to a recipe from chapter 3

For each of the three configs, decide which chapter-3 recipe best describes what is going on. The candidates:

- **No scaling** — the model is used at its pretrained `max_position_embeddings`.
- **Position Interpolation (PI)** — `rope_scaling` names `"linear"` or `"linear interpolation"`, with a single `factor`.
- **NTK-aware / dynamic NTK** — `rope_scaling` names `"dynamic"` or an NTK variant.
- **YaRN** — `rope_scaling` names `"yarn"` with temperature/attn-factor fields.
- **Llama 3's custom scheme** — `rope_scaling` names `"llama3"` with `low_freq_factor` and `high_freq_factor` keys and an `original_max_position_embeddings`.

Fill in a table in your notes:

| Model | `rope_theta` | pretrained max | shipped max | `rope_type` or scaling scheme |
|---|---|---|---|---|
| Llama 3 8B | | | | |
| Llama 3.1 8B | | | | |
| Qwen 2.5 7B | | | | |

### Step 3 — Explain one field per model in your own words

For each row, write one sentence explaining what the key field is doing. Target phrasings:

- *"Llama 3 8B's `rope_theta=500000` is the base frequency — much higher than Llama 2's 10000 — which (per chapter 3) gives more headroom for context extension even before any scaling block is added."*
- *"Llama 3.1 8B's `rope_scaling` block with `rope_type=\"llama3\"`, `factor=8.0`, and `original_max_position_embeddings=8192` means the weights are the 8k-pretrained Llama 3 weights, presented to the loader as having a 64k-ish effective range via Meta's custom scaling scheme."*

If a field name in a given config does not match any recipe in chapter 3, leave a `<!-- needs-research: ... -->` marker rather than guessing — e.g. `<!-- needs-research: Qwen2.5-7B-Instruct rope_scaling key "rope_type":"X" not covered in chapter 3 -->`.

### Step 4 — Write the "what would I change" note

Pick one of your three models and write a short (3–5 sentences) note answering:

- Suppose you were asked to fine-tune this model at **4× its pretrained context length**.
- Which `rope_scaling` setting would you use as a starting point? (Cite the recipe from chapter 3 by name — PI, NTK, YaRN, or the model's own scheme.)
- Besides the `rope_scaling` setting itself, what two concerns from the pipeline map (chapter 4) would you need to address? Candidates: data mix tilted toward long documents, KV-cache memory budget, eval harness that actually measures long-context quality, batch shape changes.

This note is the deliverable from this lab. Keep it under 150 words.

### Step 5 — Sanity check against the chapter's example

Chapter 3 quotes Llama 3.1's `rope_scaling` block:

```jsonc
"rope_scaling": {
  "factor": 8.0,
  "low_freq_factor": 1.0,
  "high_freq_factor": 4.0,
  "original_max_position_embeddings": 8192,
  "rope_type": "llama3"
}
```

Compare this against what you actually loaded for `meta-llama/Meta-Llama-3.1-8B` in step 1. If there is any divergence — different keys, different values — figure out why before moving on. Possibilities: a newer release pinned to a different `rope_type`, a `transformers` version that renames fields, or a checkpoint revision with a different scaling factor.

**Expected output:** either a confirmation that your loaded config matches the chapter's example, or a crisp one-liner explaining the divergence (e.g. *"Checkpoint revision `abc123` was updated 2024-11; the chapter quote is from the initial 3.1 release"*).

## Checkpoint

You are done with this lab when you can, without rereading the chapter:

- Name the four RoPE extension recipes from chapter 3 (PI, NTK-aware, dynamic NTK, YaRN) and give a one-sentence description of each.
- Point at the `rope_scaling` field in any of the three configs and say which recipe it implements.
- State, in one sentence, why a shipped `rope_scaling` block means you still need to pass compatible arguments at `AutoModelForCausalLM.from_pretrained` time — "the trainer must be told which [context length] you are targeting; get it wrong and long-context batches will attend nonsensically."

## Hand-off

Keep the table from step 2 and the note from step 4. When you reach [`mod-107`](../../mod-107-distributed-fine-tuning/), the recipe choice you drafted here will be stress-tested against real cluster constraints — this lab sets the vocabulary you will use in that conversation.
