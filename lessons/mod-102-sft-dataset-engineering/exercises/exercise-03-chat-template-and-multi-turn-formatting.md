# exercise-03: Chat Template and Multi-Turn Formatting

**Estimated effort:** 3 hours
**Companion chapter:** [`04-chat-templates-and-multi-turn.md`](../04-chat-templates-and-multi-turn.md)

## Objective

Turn a raw multi-turn conversation into a byte-identical training example that an SFT trainer would consume — with the prompt tokens masked to `-100` and only assistant tokens contributing to the loss — for two different base model families. By the end you will have (a) rendered the same conversation through a Llama-family and a ChatML-family tokenizer, (b) produced trainer-ready `input_ids` / `attention_mask` / `labels` tensors with per-token verification, and (c) demonstrated a bug you deliberately introduced and fixed.

The point of this exercise is to prove that "assistant-only loss on a multi-turn chat" is a concrete tensor operation you can perform from scratch, not a phrase you have seen in someone else's `SFTTrainer` config.

## Prerequisites

- Python 3.10+ with `transformers`, `torch`, and enough compute to load two tokenizers (weights not required).
- HF Hub access with any required license acknowledgements accepted.

Pick two base models spanning template families:

- One Llama-family: `meta-llama/Meta-Llama-3-8B-Instruct` (or `Llama-3.2-3B-Instruct`).
- One ChatML: `Qwen/Qwen2.5-7B-Instruct` (or `Qwen/Qwen2.5-1.5B-Instruct`).

You do not need the weights — `AutoTokenizer.from_pretrained` alone is enough for this exercise.

## Requirements

### Part A — Author the multi-turn conversation

Author a conversation of at least four turns that includes:

- A system prompt of at least 50 tokens.
- A user turn with a code snippet in a fenced block.
- An assistant turn with a mixed prose + code response.
- A user follow-up.
- A final assistant turn ending in `<eot>` / `<im_end>` naturally.

Store it as `conversation.json` in the standard `[{"role": ..., "content": ...}, ...]` shape.

### Part B — Render and tokenize, both tokenizers

For each of your two tokenizers, produce three artifacts:

1. **Rendered string with `add_generation_prompt=False`** — the training-time render. Print it as-is (whitespace visible).
2. **Rendered string with `add_generation_prompt=True`** — the inference-time render. Print and diff visually.
3. **Token ID sequence for the full training render** with a two-column table of `(id, tok.decode([id]))` so you can see every token boundary.

Note any surprises (unexpected BOS placement, leading spaces, whitespace before code fences, tool-call markers if applicable). One paragraph per tokenizer.

### Part C — Build the trainer-ready tensors

Implement, per tokenizer, a function `build_sft_example(tokenizer, messages) -> dict` that returns `{"input_ids": ..., "attention_mask": ..., "labels": ...}` following the recipe in chapter 4:

- `input_ids` = tokens of the full rendered conversation.
- `attention_mask` = all ones.
- `labels` = `-100` for every prompt / user / system / template-scaffold token; the token itself for every assistant-response token *including* the turn terminator (`<|eot_id|>` or `<|im_end|>`).

For multi-turn: **every** assistant turn contributes to loss, not just the last one (choice 1 in the chapter).

Then produce a **loss-mask visualization** for each tokenizer: a three-column table with (`position`, `decoded_token`, `label_is_active`) — where `label_is_active` is `"train"` if `labels[i] != -100` and `"mask"` otherwise. This is your ground truth that the masking is right. Read it end to end and check every span.

### Part D — Assert the invariants

Add the following assertions to your code and make them all pass on both tokenizers:

1. The prompt-render (`add_generation_prompt=True`) tokens are a strict prefix of the full-conversation tokens. If they are not, print the two token sequences and find the divergence point.
2. Every training example's final token is one of the tokenizer's known turn terminators (`<|eot_id|>`, `<|im_end|>`, or whatever the tokenizer declares).
3. Every user, system, and tool-response token has `labels[i] == -100`.
4. Every assistant-response token has `labels[i] == input_ids[i]`.
5. The total count of `labels != -100` positions divided by `len(input_ids)` is between 0.15 and 0.85 for your authored conversation. (This is a sanity check that neither everything nor nothing is masked; the exact threshold is soft.)

### Part E — Introduce, catch, and fix a bug

Deliberately introduce one of the following template-drift bugs in your pipeline, then show how one of the assertions above catches it (or, if none catches it, add an assertion that would). Fix the bug and re-run.

Choose one:

- Re-tokenize the response separately (with `add_special_tokens=True`) and concatenate to the prompt tokens instead of taking the tail from the full-conversation render.
- Set `pad_token = eos_token` on a tokenizer that has no pad, and neglect to mask pad positions.
- Rebuild `labels` from the string offset of `messages[-1]["content"]` in the rendered text — this fails on multi-turn where earlier assistant turns should also contribute to loss.
- Render the training text with a hand-rolled f-string wrapper instead of `apply_chat_template`.

Report the bug, the failing assertion, and the fix. Two paragraphs is enough.

## Starter guidance

- Prefer `apply_chat_template(..., return_assistant_tokens_mask=True)` if your `transformers` version supports it — it returns a per-token mask that you can convert directly to `labels`. If your version does not have this, fall back to the prompt-prefix / full-render approach in chapter 4.
- Print rendered strings with `repr()` when debugging so you see literal `\n` and whitespace boundaries.
- For Llama 3, the turn terminator you want to see at the end of every training example is token ID `128009` (`<|eot_id|>`), not `128001` (`<|end_of_text|>`).
- For Qwen 2.5 (ChatML), the turn terminator is `<|im_end|>`.
- TRL's `DataCollatorForCompletionOnlyLM` and Axolotl's masking recipes are reference implementations of the same idea — worth reading after you have written your own once.

## Acceptance criteria

You have completed the exercise when:

- Both tokenizers render your `conversation.json`, and you can walk a reader through the token-ID diff between the two.
- `build_sft_example` is implemented for both tokenizers.
- The loss-mask visualization exists for both, and by reading it end to end you can point at every span you would want to train on and every span you would want to mask.
- All five invariant assertions pass on both tokenizers.
- The introduced-bug section names the bug, shows the assertion that caught it (or the one you added), and shows the fix.

## Stretch goals

- Add packing: build a packed sequence of three of your conversations end-to-end. Emit the correct `cu_seqlens` tensor for FlashAttention 2 and verify that within-example position IDs restart at 0 per sub-example.
- Extend one conversation with a **tool call** (assistant emits a tool call in the format the tokenizer's chat template supports; a `tool`-role message returns a result; assistant continues). Confirm your masking treats the tool call as target and the tool response as context.
- Take a real multi-turn conversation from `OpenAssistant/oasst1` (thread of at least 4 turns, English, ending in an assistant message) and run it through your `build_sft_example`. Report anything the real conversation exposes that your authored one did not.
- Fuzz-test your masker: sample 100 random-length conversations, run through both tokenizers, assert all five invariants. Any failure is a bug hiding in your rendering code — fix it and add a regression test.
