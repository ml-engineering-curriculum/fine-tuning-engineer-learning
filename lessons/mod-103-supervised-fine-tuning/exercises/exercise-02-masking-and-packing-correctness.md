# exercise-02: Masking and Packing Correctness

**Estimated effort:** 3 hours
**Companion chapter:** [`02-masking-packing-and-effective-batch-size.md`](../02-masking-packing-and-effective-batch-size.md)

## Objective

Prove — with a decoded batch print-out, a written assertion, and a paired-run loss comparison — that (a) your SFT trainer computes loss only on assistant tokens, (b) your training-time template matches your inference-time template byte-for-byte, and (c) packing does not leak across example boundaries.

Every mask bug in this chapter's failure catalog is one where the loss curve looked fine and the trained model was broken. This exercise turns those invariants into checked-in test assertions so they cannot silently regress.

## Prerequisites

- The trained checkpoint from `exercise-01-first-sft-run-on-a-1b-model.md`. If you skipped exercise 01, run its Part B up through the sanity-check phase; you do not need a full training run for this exercise.
- Same Python environment as exercise 01: `torch`, `transformers`, `trl`, `datasets`.
- A short (≤ 50 rows) multi-turn conversation dataset — either a slice of `OpenAssistant/oasst1` or three hand-authored conversations of your own.

## Requirements

### Part A — Decode-and-print a collated batch

Write `verify_mask.py` that:

1. Loads the tokenizer from your exercise-01 checkpoint (or the base model, either works).
2. Loads 3 rows from your multi-turn dataset.
3. Runs them through the same collator your training script used (`SFTTrainer` default, `DataCollatorForCompletionOnlyLM`, whatever you chose).
4. For each row, prints one line per token, in the format:

```
[LOSS  ] <|start_header_id|>assistant<|end_header_id|>
[LOSS  ] Hello
[LOSS  ] !
[LOSS  ] <|eot_id|>
[MASKED] <|start_header_id|>user<|end_header_id|>
[MASKED] What
[MASKED] is
[MASKED] 2
[MASKED] +
[MASKED] 2
...
```

Where `[LOSS  ]` means `labels[i] != -100` and `[MASKED]` means `labels[i] == -100`.

Read the output line by line. For each of the 3 rows, confirm by eye:

- Every system-message and user-message token is masked.
- Every assistant-message token contributes to loss.
- The **turn terminator** (`<|eot_id|>`, `<|im_end|>`, or your template's equivalent) at the end of every assistant turn is in the loss. This is the chapter-5 template-leak protection.
- No padding position (if present) contributes to loss.

### Part B — Assertions in a test file

Turn Part A's manual inspection into automated assertions. Write `test_mask.py` (pytest-compatible or plain-python-with-`assert`) with at least these tests:

1. `test_assistant_mask_supported` — construct a two-message conversation, call `tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)`, assert `assistant_masks` is not all-1s and not all-0s and that the number of `1`s roughly matches the assistant response length.
2. `test_terminator_in_loss` — for a canonical example, assert the token at the position of the assistant turn terminator has `labels[i] != -100`.
3. `test_prompt_not_in_loss` — for the same example, assert *at least* 20 consecutive `-100` positions exist in the prompt region.
4. `test_no_pad_in_loss` — for a batch that requires padding, assert every position where `attention_mask == 0` has `labels == -100`.
5. `test_template_byte_identity` — render one canonical conversation two ways: (i) `apply_chat_template(messages)` and (ii) whatever your inference stack does to construct the prompt. Assert the two byte strings are equal. If your inference stack is not code you can call from the test, hard-code its expected bytes and check against those.

These tests must PASS on the code you actually train with. If any fail, fix the training script — do not weaken the test.

### Part C — Packing-leak experiment

The claim from chapter 2: packing with per-example attention masks (via FlashAttention `cu_seqlens` or Liger Kernel) does not leak information across examples. Test it.

1. Take two short training examples A and B (each < `max_length / 2`).
2. Compute the training loss on example A alone (`packing=False`, batch of just A).
3. Concatenate A and B into one packed sequence of length ≤ `max_length` and compute the loss on the sub-slice corresponding to example A, using the same model and the same collator with `packing=True`.
4. Assert the two losses are within a small tolerance (say, 1e-4 for BF16). If they differ meaningfully, the per-example attention mask is not being honored — packing is leaking.

To be certain the test is doing its job, run one more variant: repeat step 3 with an intentionally *wrong* packing setup (e.g., naive concatenation with a single causal mask over the whole sequence). Assert the loss now differs from the un-packed baseline. This confirms your test can detect leakage when it happens.

### Part D — Write up

Write `verify_report.md` (one page) containing:

1. The output of the Part A decode-and-print for one example (paste the 40–100 lines).
2. The list of assertions in `test_mask.py`, marked pass/fail, with the command used to run them.
3. The two numbers from Part C: loss on A alone, loss on A inside a correctly-packed sequence. State the tolerance and whether they match.
4. Any surprise you found. (Templates hide surprises; you will probably find at least one.)

## Starter guidance

- The chapter-2 debug script is the direct starting point for Part A. Do not overthink it.
- For Part B `test_assistant_mask_supported`, if the assertion fails, your tokenizer's chat template does not support `return_assistant_tokens_mask`. You have two options: (i) upgrade to a template that does (Llama 3.x's default template, Qwen 2.5's default template, or the ChatML template from chapter 1's script), or (ii) switch to `DataCollatorForCompletionOnlyLM` and re-do the assertions against that path. Either is fine; document which you chose.
- For Part C, if you cannot easily construct a `packing=False` baseline via TRL (some `SFTTrainer` versions require packing to be on with certain flags), compute the loss manually: forward pass on the same tokens, mask non-assistant to `-100`, compute cross-entropy yourself. It is five lines and it is more transparent.
- If your Part C variant with "intentionally wrong packing" produces a loss identical to the correct packing, congratulations — you have not actually enabled the per-example attention path. Check `attn_implementation` on the model and the packing implementation in your TRL / Axolotl version.

## Acceptance criteria

You have completed the exercise when:

- `verify_mask.py` produces the decode-and-print output for at least 3 examples, and the output is committed as an artifact.
- `test_mask.py` contains all five assertions listed in Part B, they pass on your training code, and `pytest test_mask.py` (or the equivalent runner) exits 0.
- `verify_report.md` includes the packing-leak numbers with an explicit pass/fail against your tolerance.
- The report identifies at least one non-trivial thing you learned or a mask/template detail you were previously fuzzy on.

## Stretch goals

- **Add the tests to CI.** If your project has a CI pipeline, wire the mask tests to run on every push. A silently-regressed mask is the specific failure mode CI catches best.
- **Test on two base models.** Repeat Part B with a Qwen 2.5 base *and* a Llama 3.2 base, using each's native chat template. The tests should pass on both without code changes — if they do not, your test is coupling to template-specific details.
- **Test tool-calling turns.** Extend the test data to include a `tool` role message (see the Llama 3.1+ or Qwen 2.5 chat template for the format). Assert tool responses are masked and tool calls (inside the assistant turn) are in the loss.
- **Instrument the training run.** Add a callback that once per epoch decodes and print-checks the first batch of the epoch. If your template ever silently changes (e.g., after a `tokenizer.save_pretrained` + reload), the callback will catch it in real time.
