# exercise-02: Tokenizer and Chat Template Audit

**Estimated effort:** 3 hours
**Companion chapter:** [`02-tokenization-and-chat-templates.md`](../02-tokenization-and-chat-templates.md)

## Objective

Build the instinct to inspect a tokenizer and chat template *before* your first training step. By the end you will have (a) a side-by-side diff of the chat templates of three open base models, (b) a token-by-token disassembly of the same conversation rendered by each, and (c) at least one identified footgun — either in your own rendering code or in a public tokenizer config.

Silent tokenization bugs are the most common cause of "I trained something, but it doesn't work" incidents. Learning to catch them is the highest-leverage habit in this track.

## Prerequisites

- Python 3.10+ with `transformers` and `tokenizers`. Install `sentencepiece` and `tiktoken` too — some models need one or the other.
- HF Hub access with any required license acknowledgements accepted for the models you pick.

Pick three base models spanning the tokenizer families in the chapter. Suggested set (but any comparable trio works):

- One SentencePiece BPE base: `mistralai/Mistral-7B-Instruct-v0.2` or `meta-llama/Llama-2-7b-chat-hf`.
- One tiktoken-family base: `meta-llama/Meta-Llama-3-8B-Instruct` or `meta-llama/Llama-3.2-3B-Instruct`.
- One ChatML-style base: `Qwen/Qwen2.5-7B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct`.

You do not need to download model weights — the tokenizers alone are enough for this exercise. `AutoTokenizer.from_pretrained(model_id)` will pull just the tokenizer files.

## Requirements

### Part A — Load and inspect

For each of your three tokenizers, print and record:

- The `vocab_size`.
- `bos_token`, `eos_token`, `pad_token`, `unk_token`, and their IDs. Note where any are `None`.
- The entire `special_tokens_map`.
- The first ~30 lines of the raw `chat_template` string (found on `tokenizer.chat_template` or in the `tokenizer_config.json`).
- The tokenizer class (`type(tok).__name__`) — is it `LlamaTokenizerFast`, `PreTrainedTokenizerFast`, `Qwen2TokenizerFast`, or something else?

Assemble this into a comparison table in a `notes.md` (or the top of your writeup).

### Part B — Render the same conversation three ways

Choose a canonical two-turn conversation with a system prompt and a code snippet in the assistant response — something whose whitespace matters. For example:

```json
[
  {"role": "system", "content": "You are a terse Python tutor. Reply with a short code block, no prose."},
  {"role": "user",   "content": "Write a function that returns the first n Fibonacci numbers."},
  {"role": "assistant", "content": "```python\ndef fib(n):\n    a, b = 0, 1\n    out = []\n    for _ in range(n):\n        out.append(a)\n        a, b = b, a + b\n    return out\n```"}
]
```

For each tokenizer, produce three artifacts:

1. **The rendered string** — `apply_chat_template(messages, tokenize=False, add_generation_prompt=False)`. Print it as-is (do not strip whitespace). Then print it again with `add_generation_prompt=True` and diff visually.
2. **The token IDs** — `apply_chat_template(messages, tokenize=True, add_generation_prompt=False)`, then `[tok.decode([t]) for t in ids]` so you can see the token boundary between every ID and its text form. Print as a two-column table.
3. **A round-trip check** — decode the full ID list back to text and diff against the rendered string. If anything is not byte-identical, dig in.

### Part C — Hunt three footguns

Write up findings on all three of these questions per tokenizer:

1. **BOS-token count.** Does `apply_chat_template` add a BOS at the front? What happens if you *also* pass that string into `tok(..., add_special_tokens=True)`? Show the token ID sequence in both cases and identify which one is what you actually want to send to a trainer.
2. **`pad_token` policy.** Is `pad_token` set? If not, what would you set it to for SFT? What is the risk if you naïvely set `pad_token = eos_token` and do not mask pad positions in the loss?
3. **EOS ambiguity.** For the Llama 3 model in your trio (or another model with multiple end-of-sequence tokens), enumerate every "end" token: `<|end_of_text|>`, `<|eot_id|>`, `<|eom_id|>` if present. Which one does the model actually emit at the end of an assistant turn? Which one should a serving stack stop generation on? Cite the model card or official prompt-format docs; do not guess.

For each footgun, note one concrete symptom you would expect at fine-tuning or inference time if you got it wrong. Be specific — "the model gets worse" is not a symptom; "the model never emits an EOS and generates until `max_new_tokens`" is.

### Part D — Public tokenizer config audit

Pick one open model *not* in your original three (browse the trending page of the HF Hub for something recent) and read its `tokenizer_config.json` end to end. In one short paragraph:

- Which template family does it belong to (ChatML, Llama-style, custom)?
- Is `add_bos_token` set explicitly, and to what?
- Is there anything in the Jinja template that looks like a source of surprise — a conditional on the system prompt being present, a suffix-only rendering when there is no assistant message, a hard-coded default that overrides your `system` message?

You are not looking for a *bug* per se — you are proving that you can read one of these files critically. If you happen to find a real issue, note it clearly (and consider filing an issue upstream).

## Starter guidance

- The `--verbose` version of `apply_chat_template` doesn't exist; you inspect by printing intermediate results yourself.
- To read the raw Jinja template on a saved tokenizer: `print(tok.chat_template)`. It's a string. If it uses `raise_exception`, the template is protecting against a specific invalid state — worth understanding.
- Every `tok.decode([id])` where the returned string has a leading `▁` is a SentencePiece token; leading `Ġ` is a GPT-2-style byte-level BPE token. The presence or absence of these markers changes how the tokenizer will re-encode the string later.
- For part C question 3, official sources: the Meta Llama 3 model card / prompt-format doc lives on the meta-llama HuggingFace org and in the meta-llama/llama-models GitHub repo. Both list `<|eot_id|>` as the turn terminator.

## Acceptance criteria

You have completed the exercise when:

- Your comparison table for part A is complete for all three tokenizers with no `TODO`s.
- Your rendered-string + token-ID + round-trip artifacts for part B exist for all three tokenizers, and any diff between round-trip decode and the original rendered string is investigated and explained (not ignored).
- All three footgun questions are answered with concrete token IDs, not vague descriptions.
- The public-config audit paragraph names the model, cites its `tokenizer_config.json`, and demonstrates you read it.
- The write-up is short and structured — a colleague could grep it for a specific model's answer in under a minute.

## Stretch goals

- Rewrite one of the chat templates from scratch as a Python function that produces the same byte-identical output as `apply_chat_template`. This is a good way to prove you actually understand the Jinja rules.
- Do the tokenizer inspection against a non-English prompt (Japanese, Arabic, code with lots of underscores). Which tokenizer compresses your test text most efficiently in bytes-per-token? What does that imply about training cost and context budget for that language?
- Load two different tokenizer versions of the same base (e.g., an older and newer revision of a Qwen or Llama tokenizer). Diff `special_tokens_map`. Any silent additions? What would happen at fine-tuning time if you trained against version A and served with version B?
- Reproduce the "double-BOS" footgun in code — actually construct a training example both ways and confirm the token-ID sequences differ. Then reproduce the "pad_token = eos_token unmasked" footgun and describe what a single training step would look like on it.
