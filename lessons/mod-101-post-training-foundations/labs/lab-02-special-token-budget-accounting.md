# lab-02: Special-Token Budget Accounting for a Two-Turn Chat

**Estimated duration:** 60 minutes.
**Companion chapter:** [`02-tokenization-and-chat-templates.md`](../02-tokenization-and-chat-templates.md).

## Objective

Walk through, token by token, the three accounts that make up any conversation sent to a chat-template-trained base — **structural tokens, system-prompt tokens, content tokens** — on a single real model. By the end you will have a scripted procedure that, given any `max_seq_length`, tells you how much room is actually left for assistant content before silent right-truncation starts eating your training examples.

This lab is scripted, single-model, and focused on the *accounting* mechanics. It does not duplicate [`exercise-02`](../exercises/exercise-02-tokenizer-and-chat-template-audit.md), which has you diff *three* tokenizers and hunt for arbitrary footguns — here we build budgeting fluency on one.

## Prerequisites

- Python 3.10+ with `transformers` installed.
- A Hugging Face Hub login with Llama 3 license acceptance, or substitute `Qwen/Qwen2.5-1.5B-Instruct` throughout if you prefer a license-lighter alternative.
- You have read chapter 2, particularly "The special-token budget" and "Three tokenization footguns."

## Steps

### Step 1 — Load the tokenizer and inspect its special-token map

```python
from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"   # or "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL_ID)

print("tokenizer class:", type(tok).__name__)
print("vocab_size     :", tok.vocab_size)
print("bos_token      :", tok.bos_token, tok.bos_token_id)
print("eos_token      :", tok.eos_token, tok.eos_token_id)
print("pad_token      :", tok.pad_token, tok.pad_token_id)
print("special tokens :", tok.special_tokens_map)
```

**Expected output (Llama 3 8B Instruct):** `bos_token` is `<|begin_of_text|>` (128000), `eos_token` is `<|eot_id|>` (128009) — note it is `<|eot_id|>`, not `<|end_of_text|>`, matching the chapter's warning. `pad_token` is likely `None`.

### Step 2 — Render an "empty" conversation to measure the structural overhead

Render a conversation whose content is as close to zero tokens as possible. This isolates the structural overhead the chat template imposes regardless of content.

```python
empty = [
    {"role": "system", "content": ""},
    {"role": "user",   "content": ""},
]

ids_empty = tok.apply_chat_template(empty, tokenize=True, add_generation_prompt=False)
print("ids:", ids_empty)
print("decoded pieces:")
for i in ids_empty:
    print(f"  {i:>6}  {tok.decode([i])!r}")
print("structural token count (no content):", len(ids_empty))
```

**Expected output:** a sequence of 10–15 token IDs on Llama 3, each of which should decode to a special marker (`<|begin_of_text|>`, `<|start_header_id|>`, role strings, `<|end_header_id|>`, `<|eot_id|>`) or a very short whitespace piece. Write down the exact count your tokenizer produced — call it `S_struct`. On Llama 3 Instruct it should be in the chapter's ballpark of "~15 structural tokens for a two-turn conversation."

### Step 3 — Measure the system-prompt account separately

Render the conversation with only a realistic system prompt, keeping the user turn empty:

```python
SYSTEM = (
    "You are a terse Python tutor. Reply with a short code block and no prose. "
    "If the user asks for anything that is not a Python coding question, decline politely."
)

msg_sys_only = [
    {"role": "system", "content": SYSTEM},
    {"role": "user",   "content": ""},
]
ids_sys = tok.apply_chat_template(msg_sys_only, tokenize=True, add_generation_prompt=False)
S_system = len(ids_sys) - len(ids_empty)
print("system-prompt token count:", S_system)
```

**Expected output:** `S_system` is however many tokens the system prompt itself costs — for the example string above, typically ~40–50 tokens on Llama 3. Write this down.

### Step 4 — Measure content for a realistic user turn and assistant response

Now add real content for both turns and measure the content cost:

```python
USER = "Write a function that returns the first n Fibonacci numbers."
ASSISTANT = (
    "```python\n"
    "def fib(n):\n"
    "    a, b = 0, 1\n"
    "    out = []\n"
    "    for _ in range(n):\n"
    "        out.append(a)\n"
    "        a, b = b, a + b\n"
    "    return out\n"
    "```"
)

full = [
    {"role": "system",    "content": SYSTEM},
    {"role": "user",      "content": USER},
    {"role": "assistant", "content": ASSISTANT},
]
ids_full = tok.apply_chat_template(full, tokenize=True, add_generation_prompt=False)
print("full conversation token count:", len(ids_full))

# Content-only: everything not already accounted for in S_struct + S_system.
# This is an approximation — a clean tokenizer should land very close.
S_content = len(ids_full) - len(ids_sys)
print("content token count (user+assistant):", S_content)
```

**Expected output:** `len(ids_full)` is in the 150–200 range. `S_content` captures the user prompt plus the code block (code blocks are relatively token-expensive because of the triple backticks and indentation whitespace).

### Step 5 — Build the budget table

Compile the three accounts into one table, then compute how much room is left under a hypothetical `max_seq_length` of 4096:

```python
MAX_SEQ = 4096
remaining = MAX_SEQ - len(ids_full)

print("Token budget:")
print(f"  structural overhead : {len(ids_empty):>5}")
print(f"  system prompt       : {S_system:>5}")
print(f"  content (this turn) : {S_content:>5}")
print(f"  total used          : {len(ids_full):>5}")
print(f"  remaining of {MAX_SEQ:<5}: {remaining:>5}")
```

**Expected output:** a four-row table with a non-negative `remaining`. If your content is longer than ~3800 tokens, remaining goes negative — which means the trainer's collator will truncate silently.

### Step 6 — Reproduce the "end mid-sentence" symptom in a dry run

Simulate what would happen if `MAX_SEQ` were small enough that the assistant turn got cut mid-response. Do *not* actually train anything — this is a diagnostic drill.

```python
small_max = len(ids_full) - 20   # cut 20 tokens from the end
truncated = ids_full[:small_max]
print("truncated last 5 token IDs and their decoded forms:")
for i in truncated[-5:]:
    print(f"  {i:>6}  {tok.decode([i])!r}")
print()
print("is <|eot_id|> present in the truncated sequence?",
      tok.eos_token_id in truncated)
```

**Expected output:** the last five tokens decode to a fragment of the assistant's code block, and `<|eot_id|>` is **not** present in the truncated sequence. This is exactly the silent failure the chapter warns about: the model, if trained on many examples truncated this way, learns that assistant turns can simply end without an `<|eot_id|>`.

### Step 7 — The `add_generation_prompt` lever, in one check

Finally, render the same `full - assistant` prefix two ways and confirm the exact difference between training-time and inference-time strings:

```python
msgs_prompt = [m for m in full if m["role"] != "assistant"]
train_text = tok.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=False)
infer_text = tok.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=True)
print("--- training-time (no generation prompt) ---")
print(repr(train_text[-120:]))
print("--- inference-time (with generation prompt) ---")
print(repr(infer_text[-120:]))
print("tokens added by add_generation_prompt:",
      len(tok(infer_text, add_special_tokens=False).input_ids)
      - len(tok(train_text, add_special_tokens=False).input_ids))
```

**Expected output:** the inference-time string ends with an `<|start_header_id|>assistant<|end_header_id|>` block (plus a trailing newline on Llama 3), and the token count difference should be a small positive integer (around 4 on Llama 3). This is exactly the "missing in training, present at inference" mismatch the chapter calls out.

## Checkpoint

You are done with this lab when you can state, without rerunning the script:

- The approximate `S_struct` number for your tokenizer, and why it is a per-conversation constant rather than a per-message one.
- The approximate `S_system` number for your actual system prompt, and why that number is often the first thing to shrink when `max_seq_length` is tight.
- The concrete symptom at inference time of training on truncated assistant turns (the model fails to emit `<|eot_id|>` and runs to `max_new_tokens`).
- Why `add_generation_prompt=True` is required at inference but usually `False` at training, per the chapter.

## Hand-off

Keep the budget table from step 5. When you get to [`mod-102`](../../mod-102-sft-dataset-engineering/) and start writing an SFT data collator, this table is the back-of-envelope you will refer to whenever a dataset entry's token count comes in high.
