# Chat Templates and Multi-Turn Conversation Construction

**Estimated read:** 45 minutes.
**You should be able to, by the end:** produce, for a target base model, the exact `input_ids` and `labels` tensors an SFT trainer should consume — with the prompt tokens masked out of the loss and only assistant tokens contributing — for both single-turn and multi-turn conversations, and know when to pack, when to truncate, and when to split.

## From "message list" to "training tensor"

Mod-101 chapter 2 taught you what a chat template *is* — the Jinja string in `tokenizer_config.json` that turns a list of `{role, content}` dicts into a byte sequence the base model recognizes. This chapter turns that byte sequence into an SFT training example.

The distinction that matters: **inference-time rendering** ends with an assistant header the model completes into. **Training-time rendering** ends with the assistant response *already there* — the model is being taught to produce those specific tokens after that specific prompt. The loss must reflect that: prompt tokens are context, assistant tokens are targets.

## The invariant: loss only on assistant tokens

For SFT, the training loss at token position `t` should be included when the token at position `t` is part of an assistant response, and excluded — usually by setting `labels[t] = -100`, which PyTorch's `CrossEntropyLoss` skips — for every other position.

Why this specific rule:

- **Prompt (system + user) tokens are context, not targets.** If you leave them in the loss, the model learns to predict the *user's next message from its own prior message*, which is neither what you want nor what inference will do.
- **Structural tokens (`<|start_header_id|>assistant<|end_header_id|>`) are borderline.** The common convention on Llama 3, Qwen, and most modern instruct sets is to *include* the assistant-turn opener in the loss (so the model learns to emit its own header) but that decision is made once per template. Read the training recipe of the target base model family before deciding — if you copy their template, copy their masking.
- **Turn terminators (`<|eot_id|>`, `<|im_end|>`) go in the loss.** The model must learn to stop. Excluding these produces a model that will not naturally end assistant turns.
- **Padding tokens are always masked.** If you set `pad_token = eos_token` (mod-101 chapter 2 footgun), you must mask pad positions or the model learns to emit EOS everywhere. This is worth double-checking every time.

The general form:

```python
def build_sft_example(tokenizer, messages) -> dict:
    """
    messages: list of {role, content} dicts ending in an 'assistant' message.
    Returns input_ids, attention_mask, labels ready for a trainer.
    """
    # 1. Render the prompt (everything up to but not including the final assistant response).
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,   # emit the assistant-turn opener
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # 2. Render the whole conversation, then take the response tokens as the tail.
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # response is already present
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # Sanity: full should start with prompt.
    assert full_ids[:len(prompt_ids)] == prompt_ids, "template drifted between renders"

    input_ids = full_ids
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
```

Two properties of this construction that pay off later:

- The assertion catches template drift between "render for prompt" and "render for full" — a real class of bug.
- The response tokens are taken from `full_ids`, not re-tokenized from `messages[-1]["content"]`. Re-tokenizing the response separately produces subtly different token boundaries at the seam (especially with SentencePiece leading spaces), and that seam is where mysterious bugs live.

## Multi-turn conversations

For a `[system, user_1, assistant_1, user_2, assistant_2, …]` conversation, you have a design choice:

### Choice 1: Train on every assistant turn

All assistant tokens across all turns contribute to loss. Prompt tokens (system + all user turns + template scaffolding) are masked.

```
labels:  [ -100 -100 -100 | A1 tokens | -100 -100 | A2 tokens | -100 -100 | A3 tokens ]
                system+u1     assist_1    u2         assist_2      u3         assist_3
```

This is the default for most open recipes (TRL's `SFTTrainer`, Axolotl's default multi-turn config). The model learns each assistant turn conditioned on all prior context.

### Choice 2: Train only on the final assistant turn

All earlier assistant turns are treated as context and masked out; only the last turn contributes to loss.

Trade-off: choice 1 gets ~N examples worth of gradient from one N-turn conversation; choice 2 gets 1. Choice 1 also implicitly teaches the model to condition on its own past outputs (which is what happens at inference); choice 2 forces the model to always start "fresh" per turn.

The literature and most production recipes favor choice 1 unless the assistant turns are of very uneven quality (e.g., early turns are auto-generated placeholders and only the final turn is human-authored).

### Assistant-only masking, in code

The precise multi-turn masking is easier if you render each turn separately and concatenate, tracking assistant spans as you go. TRL's `DataCollatorForCompletionOnlyLM` and `apply_chat_template(..., return_assistant_tokens_mask=True)` both give you an "is this token part of an assistant response" mask directly — prefer those over hand-rolled span tracking when they are available.

## Packing: many examples per sequence

A 2k-token training example on an 8k `max_seq_length` wastes 6k of the sequence budget per row. **Packing** stuffs multiple examples into one sequence to burn less compute on padding.

Two packing styles:

### 1. Naïve concatenation ("pretraining-style packing")

Concatenate examples with a document-boundary token between them, ignore the boundary. Very simple, but examples now "see" each other across boundaries because the causal mask is a single triangle over the whole packed sequence.

Consequence: the model can attend from example B back into example A's tokens. In pretraining this is tolerable; in SFT — where every example is meant to be an independent conversation — it is quality-eroding, especially at short contexts where cross-example attention makes up a large fraction of visible tokens.

### 2. Packed with per-example attention masks

Same physical concatenation, but the attention mask is *per-example*: within each packed sub-sequence, tokens can attend only to prior tokens *inside the same sub-example*. This is what modern SFT stacks default to when they claim to "pack."

Implementation options:

- **FlashAttention 2 / 3 with `cu_seqlens`** (Dao et al., 2022, 2023) — the fastest path. Provide a `cu_seqlens` tensor of cumulative sequence lengths and the kernel builds a block-diagonal attention mask implicitly.
- **Position IDs reset per-example.** The RoPE positions restart at 0 for each packed sub-example, so positional information does not leak either.
- **HF `transformers` — `attn_implementation="flash_attention_2"` plus a packed collator** (Axolotl and TRL both ship one). Confirm your target model supports FA2.

Rules for packing SFT:

- Always use per-example attention masks, not naïve concatenation.
- Do not pack across chat template boundaries in a way that duplicates system prompts — that wastes tokens on repeated boilerplate.
- Log the effective packing efficiency (`sum(len(example)) / (batch_size * max_seq_length)`). A well-packed SFT set typically hits 0.8–0.95; below 0.5, your batch size is over-large or your `max_seq_length` under-tuned.

## Truncation: the silent quality killer

You have set `max_seq_length = 4096`. A training example renders to 4,200 tokens. What happens?

The naïve trainer collator truncates from the right — dropping the tail of the assistant response and the turn terminator with it. The model then sees a training row that ends mid-sentence with no `<|eot_id|>`, learns exactly that, and emits mid-sentence assistant turns forever after.

Correct policies, in decreasing preference:

1. **Drop the row.** If the row does not fit, it should not train. Report the drop rate; if it is high, increase `max_seq_length` or curate shorter examples.
2. **Left-truncate the prompt.** Preserves the full assistant response and its terminator. Only valid if truncating the earliest user turns still leaves the model a comprehensible context.
3. **Sliding-window across turns.** For very long multi-turn conversations, split into overlapping windows where each window contains one target assistant turn plus as many prior turns as fit. Effectively converts one long conversation into several training examples.

Whatever policy you pick, put a hard assertion in your data-loader test that the last token of every training example is a valid turn terminator (`<|eot_id|>`, `<|im_end|>`, or the model's actual EOS). This one assertion catches truncation bugs at build time instead of at eval time.

## Tool calls and multi-message assistant turns

Modern chat templates (Llama 3.1+, Qwen 2.5, Hermes-3-style) support tool-calling turns where the assistant emits a structured tool call, a `tool` role responds with the result, and the assistant continues. Chat-template Jinja for these families renders the tool call inside the assistant turn, wrapped in a template-specific marker (`<|python_tag|>`, `<tool_call>…</tool_call>`, etc.).

Two things to get right in the training example:

1. **Loss over the tool call itself.** The model must learn to emit the tool call; every tool-call token is a target, including the wrapping markers.
2. **Tool responses are context, not target.** Mask them like a user message.

The chat template will render this correctly if you give it messages with the right roles; the mistake is manually assembling the string and mis-attributing tokens to assistant vs. tool. Trust `apply_chat_template` for tool-calling formats — hand-rendering them is a bug generator.

## System-prompt handling

Design decisions to record explicitly:

- **Every row has a system prompt or none of them do.** Mixing rows with and without a system prompt teaches the model that the system prompt is optional. That is a legitimate design choice; make it consciously.
- **System-prompt tokens are always masked.** Never contribute to loss.
- **A single canonical system prompt vs. diverse system prompts.** Training only on `"You are a helpful assistant."` makes the model brittle to different system prompts at inference. Training on diverse system prompts (varying persona, style, tool availability) generalizes better but requires much more curation.

## The training-vs-inference render diff

The single most common multi-turn SFT bug is: training renders the conversation one way, inference renders it another way, model quality mysteriously suffers.

The invariant that fixes this: **exactly one function renders a conversation to a byte string, and it is used at both training and inference time**. If your inference stack renders with `apply_chat_template(..., add_generation_prompt=True)` and your training pipeline emits a hand-rolled string, you have this bug even if the outputs *look* identical, because whitespace and BOS handling almost always drift by a byte or two.

The `assert full_ids[:len(prompt_ids)] == prompt_ids` from the code block above is one guard. A second guard: at build time, render a canonical example through both your training pipeline and your inference-time serving stack; diff the bytes. Any diff is a bug.

## Exercise pointer

`exercises/exercise-03-chat-template-and-multi-turn-formatting.md` walks you through building the trainer-ready tensors for a real multi-turn conversation against a Llama-family and a ChatML base, verifying the assistant-only masking with a print-out, and demonstrating that your rendering matches `apply_chat_template` byte-for-byte.

## Summary

- SFT loss must include only assistant tokens; every prompt / system / user / tool-response / padding token is masked with `-100`.
- The safe recipe: render prompt-with-generation-prompt to get `prompt_ids`, render the full conversation to get `full_ids`, assert the prefix match, label the tail. Never re-tokenize the response separately.
- Multi-turn default: train on every assistant turn (choice 1). Only mask earlier turns if their quality is meaningfully lower than the final turn's.
- Pack with per-example attention masks (FlashAttention `cu_seqlens`), not naïve concatenation. Log packing efficiency.
- Never let the trainer silently right-truncate. Drop or left-truncate; assert every example ends in a turn terminator.
- Tool-calling turns: trust `apply_chat_template`; tool responses are context, tool calls (and their markers) are targets.
- Ship one rendering function; use it at training and inference; diff the bytes.
