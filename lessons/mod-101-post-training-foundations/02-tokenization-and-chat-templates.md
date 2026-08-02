# Tokenization, Chat Templates, and the Special-Token Budget

**Estimated read:** 40 minutes.
**You should be able to, by the end:** explain how BPE, SentencePiece, and tiktoken differ; render a conversation through a Hugging Face chat template and read the resulting token IDs; and name the three most common tokenization footguns that will corrupt an SFT run before the first gradient step.

## Why this matters more than a fine-tuner expects

The single most common reason a "correct-looking" fine-tune produces a broken model is **tokenization mismatch** — the training data was rendered differently than the model saw during pretraining, or than the inference stack renders at serving time. There are no obvious errors, no exceptions, no loss spike. The model just learns something subtly wrong.

Because these bugs are silent, they eat weeks. Learning to spot them requires a working mental model of how a modern tokenizer converts text to token IDs and how a chat template arranges those tokens into a conversation the base model recognizes.

## The three tokenizer families you will meet

There are dozens of tokenizer implementations and three algorithmic families you actually need to know:

### 1. Byte-Pair Encoding (BPE) — the GPT-2 lineage

Introduced for NLP by Sennrich, Haddow, and Birch (2016) and popularized by GPT-2 (Radford et al., 2019). The algorithm starts from a base vocabulary of bytes (or characters) and greedily merges the most frequent adjacent pair, recording each merge in an ordered list. At inference, text is split by the same merge rules and each resulting piece is mapped to a token ID.

Reference implementations: the `tokenizers` crate/library (Rust, with Python bindings) exposes this as `BPE`. Original GPT-2 uses a byte-level variant that guarantees any Unicode string can be encoded losslessly by first encoding it as UTF-8 bytes and then running BPE over the 256-byte alphabet.

### 2. SentencePiece — the T5 / Llama lineage

Kudo & Richardson's SentencePiece (2018) supports two algorithms — BPE and **Unigram Language Model** tokenization — and treats the whitespace-splitting step as part of the model rather than an assumed preprocessing pass. It represents leading spaces with a visible marker (`▁`, U+2581 lower one eighth block) and can round-trip text without a separate detokenizer.

Llama 1 and Llama 2 use SentencePiece with a 32k-vocabulary BPE model. Llama 3 replaced that with a tiktoken-style byte-level BPE at 128k vocab. T5, mT5, and the Gemma family use SentencePiece Unigram.

The practical differences you will feel:

- SentencePiece serializes to a `tokenizer.model` protobuf file. `AutoTokenizer.from_pretrained` handles it transparently, but if you are debugging you may need `sentencepiece` installed.
- Unigram tokenization can produce **different segmentations at inference than at training** if you enable subword regularization; almost no production stack uses this at inference.
- The `▁` prefix is not just cosmetic — training data that lacks the leading-space marker where the model expects one is a real source of mismatch.

### 3. tiktoken — the OpenAI lineage

OpenAI's `tiktoken` library is a fast byte-level BPE implementation with a handful of released encodings:

- `cl100k_base` — GPT-3.5 and the original GPT-4 (~100k vocab).
- `o200k_base` — GPT-4o and o-series (~200k vocab, better multilingual coverage).
- `p50k_base` / `r50k_base` — earlier codex / GPT-3 models.

You will encounter `tiktoken` in two contexts a fine-tuner cares about:

1. **Managed fine-tuning APIs (mod-110):** OpenAI's cost estimator and JSONL uploader use `tiktoken` to count tokens against the model's price sheet. Miscount your token counts and you miscount your bill.
2. **Llama 3 and derivatives use a tiktoken-compatible tokenizer** — the vocab was built with `tiktoken`'s BPE, then wrapped in Hugging Face's `PreTrainedTokenizerFast` for portability. If you are counting tokens for cost or context budgeting against a Llama 3 checkpoint locally, either implementation should give matching results.

## The vocabulary size trend

Vocab sizes have grown as multilingual and code coverage has become table stakes:

| Model | Tokenizer | Vocab size |
|---|---|---|
| GPT-2 | byte-level BPE | 50,257 |
| Llama 2 | SentencePiece BPE | 32,000 |
| Mistral 7B | SentencePiece BPE | 32,000 |
| Llama 3 / 3.1 | tiktoken byte-level BPE | 128,256 |
| Qwen 2.5 | byte-level BPE | 151,936 |
| GPT-4o (`o200k_base`) | tiktoken | ~200,000 |

The consequences of a bigger vocab: each token carries more information (better compression → fewer tokens for the same text → cheaper training and inference), the embedding table and LM head grow linearly with vocab size (which matters for VRAM and initialization), and *cross-tokenizer text comparisons stop being apples-to-apples*. Never compare "tokens per second" or "context length in tokens" across models without noting which tokenizer you are using.

## Chat templates: the invisible protocol

A base model does not natively know about "system prompts" or "user turns." A fine-tuned instruct/chat model has learned to expect a specific arrangement of special tokens that marks who is speaking. **The exact byte sequence matters** — differing by a single space or by using the wrong header token yields a base model that responds like it is still doing raw next-token prediction on random text.

### Three of the templates you will meet

**Llama 3 Instruct** uses an explicit header protocol with four special tokens per turn:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Where does the sun rise?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The sun rises in the east.<|eot_id|>
```

`<|eot_id|>` (end of turn) is what the model has been trained to emit when it wants to stop for that turn. It is a different token from `<|end_of_text|>`. Stopping generation on the wrong one is a very common mistake.

**ChatML** (used by Qwen, some Mistral variants, and the OpenAI-inspired `chatml` template) uses paired role markers:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Where does the sun rise?<|im_end|>
<|im_start|>assistant
The sun rises in the east.<|im_end|>
```

**Alpaca** — from the original Stanford Alpaca project (2023) — uses no special tokens at all, only English scaffolding:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Where does the sun rise?

### Response:
The sun rises in the east.
```

Alpaca-style templates were common in the early open-model era but have largely been supplanted by special-token templates because they are ambiguous (the model can hallucinate `### Instruction:` inside a response) and waste tokens (English words instead of single-token markers).

### How Hugging Face renders them

Modern Hugging Face tokenizers store the chat template as a **Jinja2 template string** in `tokenizer_config.json` under the `chat_template` key. `apply_chat_template` renders that Jinja template against a list of `{"role": ..., "content": ...}` dicts:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Where does the sun rise?"},
]

# For training data: no generation prompt appended
train_text = tok.apply_chat_template(messages, tokenize=False)

# For inference: add the assistant header so the model completes into it
infer_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

The `add_generation_prompt` flag is one of the single most consequential arguments in the entire fine-tuning stack. Forget it at inference and the model will hallucinate a `<|start_header_id|>user<|end_header_id|>` before it answers, because it thinks the conversation has not yet turned to the assistant. Include it in training data and every training example will end mid-conversation with a dangling assistant header the model must learn to complete from — which is a defensible choice, but only when made deliberately.

## The special-token budget

Every conversation you tokenize has three token accounts to keep straight:

1. **Structural tokens** — `<|begin_of_text|>`, role headers, `<|eot_id|>`, blank lines. On Llama 3 a two-turn conversation costs about 15 structural tokens regardless of content.
2. **System prompt tokens** — often the longest single block; can dominate short conversations.
3. **Content tokens** — the actual user and assistant text.

If your effective sequence limit is `max_seq_length` and you are packing multi-turn chats for SFT (mod-103), you need to budget all three against that limit *before* the trainer's collator truncates for you. Silent right-truncation of a long assistant response is a devastating training bug: the model learns to end responses mid-sentence with no `<|eot_id|>`, and you will see it at inference for the rest of the adapter's life.

Concrete audit checklist for any new chat template you fine-tune against:

- Encode a canonical two-turn conversation with `apply_chat_template(..., tokenize=True)` and print the resulting token IDs and their decoded string forms. Look for: duplicated BOS, missing turn terminators, unexpected whitespace.
- Verify the tokenizer's `eos_token_id` matches what the model will emit at end-of-turn. On Llama 3 Instruct this is `<|eot_id|>` (128009), *not* `<|end_of_text|>` (128001). Serving stacks that stop only on the latter will run past turn boundaries.
- Encode a full training example with the response, then use `tok.decode(...)` on the round-trip and diff against the original text. Any diff — including whitespace — is a bug in your pipeline.
- Read the tokenizer's `special_tokens_map.json`. Every entry there is a token that (a) may appear in your training data and (b) needs a policy for whether it is masked in the loss.

## Three tokenization footguns

**1. `pad_token = eos_token`.** Many open models ship with `pad_token = None` on the tokenizer and every SFT tutorial reflexively sets `tok.pad_token = tok.eos_token` to make batching work. Do this without also masking the padding positions from the loss and the model learns to emit EOS everywhere. Always confirm your training collator sets `labels[pad_positions] = -100`.

**2. Two BOS tokens.** `apply_chat_template` prepends BOS itself. If you then wrap the rendered string with a `SFTTrainer` that also adds special tokens (`add_special_tokens=True`), you double-encode. Loss will still go down, but the model will subtly under-perform. Rule of thumb: after `apply_chat_template`, always pass `add_special_tokens=False` in the tokenizer call.

**3. Template drift between training and serving.** You fine-tune on a hand-written wrapper, then serve with `apply_chat_template`. The two produce a different byte sequence. The model still generates something, so the bug hides behind "the model just seems worse than the eval said it was." Ship one function that renders training text and reuse it for serving-time inputs.

## Exercise pointer

`exercises/exercise-02-tokenizer-and-chat-template-audit.md` walks you through diffing the chat templates of three different open base models and hunting for at least one real-world footgun in a public tokenizer config.

## Summary

- BPE (byte-level, GPT-2 / tiktoken / Llama 3) and SentencePiece (BPE or Unigram, Llama 1–2 / T5 / Gemma) are the two families you will encounter. `tiktoken` is OpenAI's fast BPE implementation; the Llama 3 tokenizer is compatible with it.
- Vocab sizes have grown from 32k to 128k–200k as multilingual and code coverage improved. Cross-tokenizer comparisons are not apples-to-apples.
- Chat templates are stored as Jinja strings in `tokenizer_config.json`. `apply_chat_template` is the one API to use — never hand-render.
- The special-token budget has three pieces (structure, system, content). Budget all three or watch your SFT quietly learn to end mid-sentence.
- The three killer footguns: unmasked `pad_token = eos_token`, double-BOS from double special-token addition, and template drift between training and serving. Ship one rendering function used for both.
