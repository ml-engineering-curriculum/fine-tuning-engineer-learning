# lab-01: Reading a Base Model's `config.json` Field by Field

**Estimated duration:** 45 minutes.
**Companion chapters:** [`01-decoder-only-forward-pass.md`](../01-decoder-only-forward-pass.md), [`03-positional-encodings.md`](../03-positional-encodings.md).

## Objective

Open the `config.json` of a real open base model and, for every field listed in chapter 1's abridged `LlamaConfig`, point at the tensor or operation inside the forward pass that it controls. You should leave this lab able to read a strange base model's config and immediately know which fields will constrain your fine-tuning plan (context length, VRAM budget, PEFT target shapes), without opening any tutorial.

This is guided practice. It does not duplicate [`exercise-01`](../exercises/exercise-01-decoder-only-forward-pass-walkthrough.md), which instruments the forward pass through actual tensors — here you are reading only the configuration file and relating each field to the chapter's diagrams.

## Prerequisites

- Python 3.10+ with `transformers` installed.
- A Hugging Face Hub login (`huggingface-cli login`) with any required license acceptances for the base you pick. The default `meta-llama/Meta-Llama-3-8B` requires license acceptance; `Qwen/Qwen2.5-7B` and `mistralai/Mistral-7B-v0.3` are license-lighter alternatives.
- You do not need GPU or CPU to run inference — we only download the config file.
- You have read chapter 1 and skimmed chapter 3.

## Steps

### Step 1 — Download the config file only

Pick one base model and download only its config. We do not need weights.

```bash
pip install -q transformers
python - <<'PY'
from huggingface_hub import hf_hub_download
import json, shutil, pathlib

MODEL_ID = "meta-llama/Meta-Llama-3-8B"   # or "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.3"
path = hf_hub_download(repo_id=MODEL_ID, filename="config.json")
dest = pathlib.Path("config.json")
shutil.copy(path, dest)
print(json.dumps(json.load(open(dest)), indent=2))
PY
```

**Expected output:** the JSON contents of the model's `config.json` printed to stdout and saved locally. If you see a 401 or 403 from the Hub, your account has not accepted the model's license — either accept it on the model page or switch to a license-lighter base.

### Step 2 — Map each field to its role in the forward pass

Open the saved `config.json` in your editor. For every field in the table below, write a one-line note in a scratchpad linking the field to the specific tensor or operation from chapter 1's forward pass. The chapter sections are listed so you can re-read quickly if a field escapes you.

| Field | Chapter section to re-read | Your one-liner |
|---|---|---|
| `hidden_size` | "Inside one block" (chapter 1) | |
| `intermediate_size` | "The MLP (SwiGLU on Llama-family)" | |
| `num_hidden_layers` | "The stack, end-to-end" | |
| `num_attention_heads` | "Self-attention, one head at a time" | |
| `num_key_value_heads` | same section — GQA | |
| `vocab_size` | "Reading a real config" | |
| `max_position_embeddings` | "Absolute positional embeddings" (chapter 3) | |
| `rope_theta` | "The knob a fine-tuner cares about" (chapter 3) | |
| `rope_scaling` (if present) | same section | |
| `rms_norm_eps` | "The two norms" | |
| `tie_word_embeddings` | "The architecture we care about," point 5 | |

**Expected output:** a short note per row, in your own words. For example, for `num_key_value_heads`: *"Count of K/V heads under GQA. On Llama 3 8B this is 8 vs. 32 Q heads, so `k_proj` and `v_proj` output tensors are 4× narrower than `q_proj`, which changes how expensive a LoRA on those two linears is."*

### Step 3 — Compute three derived quantities

Using only the config values, compute:

1. **Head dimension:** `d_head = hidden_size / num_attention_heads`.
2. **Approximate LM head parameter count:** `hidden_size × vocab_size`. Note whether `tie_word_embeddings` is true (in which case this table is shared with the input embedding and only counted once).
3. **A back-of-envelope "what would an all-linear LoRA cost at r=16" count,** following the formula from chapter 1's exercise pointer — `r × (in_features + out_features)` summed across all seven target linears per block, times `num_hidden_layers`. You do not need an exact number; one significant figure is plenty.

```python
import json
c = json.load(open("config.json"))
h = c["hidden_size"]
ffn = c["intermediate_size"]
n_layers = c["num_hidden_layers"]
n_q = c["num_attention_heads"]
n_kv = c["num_key_value_heads"]
vocab = c["vocab_size"]
d_head = h // n_q
kv_dim = n_kv * d_head
r = 16

per_block = r * (
    (h + h) +          # q_proj
    (h + kv_dim) +     # k_proj
    (h + kv_dim) +     # v_proj
    (h + h) +          # o_proj
    (h + ffn) +        # gate_proj
    (h + ffn) +        # up_proj
    (ffn + h)          # down_proj
)
total_lora = per_block * n_layers
print(f"d_head = {d_head}")
print(f"LM head params (approx) = {h*vocab:,}")
print(f"all-linear LoRA (r=16) params = {total_lora:,}")
```

**Expected output on Llama 3 8B** (`hidden_size=4096`, `intermediate_size=14336`, `num_hidden_layers=32`, 32 Q heads, 8 KV heads, vocab 128256):

```
d_head = 128
LM head params (approx) = 525,336,576
all-linear LoRA (r=16) params = 20,971,520
```

If your numbers are within ~5% of these values for Llama 3 8B, you have read the config correctly. For a different base, match the computation to its own numbers.

### Step 4 — Spot the context-length levers

Still in `config.json`, find:

- `max_position_embeddings` — the context length the config advertises.
- `rope_theta` — chapter 3's `base` (10000 on Llama 1/2, 500000 on Llama 3).
- `rope_scaling` — may be absent, or a dict like the Llama 3.1 example in chapter 3.

Answer, in one or two sentences each:

1. If you wanted to fine-tune this base to a context length *larger* than `max_position_embeddings`, which chapter 3 lever(s) would you reach for first, and why?
2. If the base already ships with a `rope_scaling` block, what does that tell you about how the provider wants you to load it?

## Checkpoint

You are done with this lab when you can, without looking at the chapter:

- Explain what every field in the table above controls, in one sentence each.
- Point at the field that drives KV-cache size (hint: it is a product of three config values and the sequence length).
- Explain, in two sentences, how `rope_theta` and `rope_scaling` together determine whether a base is "usable at 32k" or "only usable at 8k."

If any of those three is fuzzy, re-read the chapter section for that field and try the step again — the objective here is reflexive familiarity with `config.json`, not a one-time read-through.

## Hand-off

Keep the scratchpad from step 2. [`exercise-01`](../exercises/exercise-01-decoder-only-forward-pass-walkthrough.md) will have you confirm every one of your one-liners against an actual `named_modules()` listing from a loaded model; the config is the *map*, the module tree is the *territory*.
