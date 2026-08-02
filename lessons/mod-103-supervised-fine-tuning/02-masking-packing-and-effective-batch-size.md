# Masking, Packing, and Effective Batch Size

**Estimated read:** 45 minutes.
**You should be able to, by the end:** point at the exact places a TRL / Axolotl / Unsloth run can silently violate the "loss on assistant tokens only" invariant, choose a packing strategy that fits your dataset and hardware, and compute — from `per_device_train_batch_size`, `gradient_accumulation_steps`, `num_gpus`, and `max_length` — the effective batch size and per-step token throughput you should expect.

## The invariant, restated for the trainer

Mod-102 chapter 4 established the data-side rule: **the SFT loss should be non-zero only on assistant tokens**. Everything else — system prompts, user turns, tool responses, padding, chat-template scaffolding that is not the assistant response itself — is `-100` in `labels`.

This chapter is about the trainer-side of that same invariant. The mask can be right in your dataset row and wrong by the time it reaches the loss, or right in one framework and wrong when you switch frameworks. The failure mode is quiet: the run finishes, the loss curve looks fine, and the model has learned to mimic user prompts.

## The three routes to a masked label tensor in TRL

TRL supports three data shapes and each masks differently. Know which one you are using; do not switch mid-project.

### 1. Messages format with the tokenizer's assistant mask (default, preferred)

If your rows are `{"messages": [...]}` and your tokenizer's chat template supports `return_assistant_tokens_mask=True`, TRL asks the tokenizer directly which tokens belong to assistant turns and masks the rest. This is the correct default in TRL 0.11+ for any Llama 3.x / Qwen 2.x / Gemma / Phi-3.x tokenizer that ships a modern template.

```python
enc = tokenizer.apply_chat_template(
    messages,
    return_dict=True,
    return_assistant_tokens_mask=True,
    tokenize=True,
)
# enc["input_ids"], enc["attention_mask"], enc["assistant_masks"]
# Where assistant_masks[i] == 1 iff token i is inside an assistant turn.
labels = [tok if m == 1 else -100 for tok, m in zip(enc["input_ids"], enc["assistant_masks"])]
```

This is the safest route because the template author (Meta, Alibaba, Google) decided *for you* which tokens count as "assistant." You inherit their decision; you do not re-derive it.

**How this breaks:** the template does not support `return_assistant_tokens_mask` (older or hand-rolled templates), in which case the mask silently becomes all-1s and the model trains on every token including the user's prompt. Verify support at build time:

```python
enc = tokenizer.apply_chat_template(
    [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
    return_dict=True, return_assistant_tokens_mask=True, tokenize=True,
)
assert sum(enc["assistant_masks"]) < len(enc["assistant_masks"]), \
    "assistant_masks is all 1s — template lacks assistant-mask support; either fix the template or switch to DataCollatorForCompletionOnlyLM"
```

### 2. Prompt / completion format with `DataCollatorForCompletionOnlyLM`

If your data is `{"prompt": ..., "completion": ...}` — no chat structure — TRL's `DataCollatorForCompletionOnlyLM` handles masking by string-matching a **response template** in the tokenized sequence and marking everything before it as `-100`.

```python
from trl import DataCollatorForCompletionOnlyLM

response_template = "\n### Response:\n"     # exact string that separates prompt from response
collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
```

Two footguns specific to this collator:

- **Tokenization drift.** The collator tokenizes `response_template` separately and searches for that token sequence inside each example's `input_ids`. If leading whitespace, BOS handling, or SentencePiece boundary conventions differ, the search fails, the collator warns and masks the *entire example* (loss = 0 on the whole row). Read the warnings; a run with many "unable to find response template" warnings is silently training on nothing.
- **Chat templates with multi-turn assistant spans.** `DataCollatorForCompletionOnlyLM` masks everything before the *first* occurrence of the response template. For multi-turn conversations, that trains only on the first assistant turn. Use the messages format above for multi-turn.

`DataCollatorForCompletionOnlyLM` is the right tool for single-turn instruction data with a bespoke separator (e.g., an Alpaca-style `### Instruction: … ### Response: …`). It is the wrong tool for multi-turn or modern chat-templated data.

### 3. Instruction format helpers

TRL and Axolotl both ship helpers (`instruction_template` + `response_template` in the same collator; Axolotl's `type: alpaca`, `type: sharegpt`, etc.) that generalize the case above to a few common non-chat schemas. They are still string-match collators under the hood — same footguns, same recommendation to prefer messages format when the source data supports it.

## Verify the mask, do not trust it

Before your first real training run, write a debug script:

```python
batch = collator([train_ds[0]])
ids = batch["input_ids"][0].tolist()
labs = batch["labels"][0].tolist()

for tid, lid in zip(ids, labs):
    tok = tokenizer.decode([tid])
    if lid == -100:
        print(f"  [MASKED] {tok!r}")
    else:
        print(f"  [LOSS  ] {tok!r}")
```

Read the output with your eyes. Every masked span should correspond to a system / user / tool message or template scaffolding; every loss span should correspond to an assistant response (and the terminator that closes it). If it does not, no downstream diagnosis will help you.

Add this as a `pytest`-style assertion in your training repo so a future dependency bump does not silently break the invariant. Templates evolve; TRL evolves; assertions do not.

## Packing: many short examples per sequence

Almost all SFT examples are shorter than `max_length`. Without packing, every example is right-padded to the longest example in the batch and the padding tokens are compute you paid for and threw away.

Packing concatenates multiple examples into a single sequence of length up to `max_length` and reclaims that compute. Modern SFT stacks pack by default. The question is *how* they pack.

### Naive packing (do not use for SFT)

Concatenate examples with a separator token and treat the packed sequence as one document under the standard causal attention mask. Tokens in example B can attend back into example A because the attention triangle spans the whole sequence.

For pretraining, this cross-example leakage is tolerable — the model already sees noisy contexts. For SFT, where every example is a curated conversation, it is a quality bug: the model learns to condition its response on unrelated prior conversations. The stronger the packing (more examples per sequence), the worse the effect.

TRL's `packing=True` **without** additional flags historically used this style. TRL 0.13+ defaults to per-example attention masks; check your version.

### Packing with per-example attention masks (what you want)

Same physical concatenation, but the attention mask is block-diagonal: within a packed sub-example, tokens attend only to prior tokens *in the same sub-example*. Position IDs also reset per sub-example so RoPE (mod-101 chapter 3) does not leak positional information across boundaries either.

Two implementations you will meet:

- **FlashAttention 2 / 3 with `cu_seqlens`** — the fastest path. The kernel takes a `cu_seqlens` tensor (cumulative sequence lengths) and builds the block-diagonal attention pattern implicitly. TRL, Axolotl, and Unsloth all support this on models that support FA2.
- **`use_liger_kernel=True`** (Liger Kernel, an open-source set of Triton kernels for HF training) — similar API, additional throughput on supported architectures.

Practically:

```python
cfg = SFTConfig(
    packing=True,
    packing_strategy="ffd",           # first-fit-decreasing bin packing across the dataset
    max_length=4096,
)
# Under the hood, TRL emits packed sequences with per-example attention via FA2 when the model supports it.
```

Verify:

- Confirm your model is loaded with `attn_implementation="flash_attention_2"` (or the `sdpa` backend, which is slower but also honors `cu_seqlens` in recent PyTorch versions).
- Log the effective packing ratio: `sum(len(example)) / (num_packed_sequences * max_length)`. A well-packed SFT set typically hits 0.8–0.95. Below 0.5, either your `max_length` is way over the median example length, or your batch is being padded (packing is broken or not on).

### When to turn packing off

- **Very long examples.** If your dataset median is close to `max_length`, packing gains nothing and complicates debugging. Turn it off.
- **You are actively debugging masking.** Padded, un-packed batches are easier to eyeball. Turn packing back on once you have the mask verified.
- **Your model architecture does not support FA2.** Naive packing would introduce cross-example leakage, and non-packed padding is not that expensive at small scales. Turn it off.

## Effective batch size and gradient accumulation

The four knobs that determine the shape of one optimizer step:

- `per_device_train_batch_size` — sequences (or packed sequences) per micro-batch per device.
- `gradient_accumulation_steps` — number of micro-batches the trainer sums gradients over before calling `optimizer.step()`.
- `world_size` — number of data-parallel replicas (GPUs) contributing to each step.
- `max_length` — tokens per sequence.

The **effective batch size in sequences** is:

```
effective_batch_size = per_device_train_batch_size × gradient_accumulation_steps × world_size
```

The **effective batch size in tokens** — the more meaningful quantity for loss scaling — is:

```
effective_tokens_per_step ≈ effective_batch_size × max_length × packing_efficiency
```

Where `packing_efficiency` is 1.0 without packing (with a lot of padding waste) and ~0.85–0.95 with well-configured packing.

### Why this matters

The SFT literature reports learning rates and schedules with respect to token-count. Recent Llama and Mistral fine-tuning recipes routinely use effective batch sizes of ~256k–1M tokens per step. If your GPU can only fit `per_device_train_batch_size=2` at `max_length=4096` (16k tokens per device), you need `gradient_accumulation_steps` around 16 on a single-GPU rig to reach ~256k tokens per step. Halve the accumulation without changing the learning rate and your loss curve will be noisier and warmup will effectively be shorter (chapter 3 spells out why).

### Gradient accumulation, mechanically

Instead of one `forward → backward → step` per micro-batch, the trainer does:

```
for i in range(gradient_accumulation_steps):
    loss = model(**micro_batch_i).loss / gradient_accumulation_steps    # note the divide
    loss.backward()      # gradient accumulates in .grad
optimizer.step()
scheduler.step()
optimizer.zero_grad()
```

Two consequences worth internalizing:

- The `/ gradient_accumulation_steps` divide is what makes the accumulated gradient equal to the gradient you would get with a single big batch of the same total sequences. TRL and `transformers.Trainer` do this correctly by default. Do not roll your own gradient accumulation and forget the divide — a common bug that inflates the effective learning rate by `gradient_accumulation_steps`.
- Because there is only one `optimizer.step()` per accumulated batch, the **scheduler advances once per accumulated step, not once per micro-batch**. `warmup_steps=100` with `gradient_accumulation_steps=16` means warmup lasts `100 * 16 = 1600` micro-batches — plan `logging_steps` accordingly or you will misread the loss during warmup.

### The trainer's `num_train_epochs` and the step count

The trainer reports `Total optimization steps` at startup:

```
total_steps = ceil(len(train_dataset) / effective_batch_size) × num_train_epochs
```

If that number is wildly different from what you expected — either much smaller (packing collapsed your dataset more than you thought) or much larger (you forgot `gradient_accumulation_steps`, or your dataset is bigger than you thought) — stop the run and figure out why before spending compute.

## The "why is throughput awful?" checklist

If your training tokens-per-second is lower than expected, work through these in order:

1. **Is `attn_implementation="flash_attention_2"` (or `sdpa`) actually being used?** Check the model repr or trainer log. On some architectures the default is a slow reference kernel.
2. **Is BF16 (or FP16) on?** `bf16=True` in `SFTConfig`. Full FP32 is 2–3× slower on modern GPUs and offers no fine-tuning quality benefit at this scale.
3. **Is packing on with per-example masks?** `packing=True`, and confirm packing efficiency > 0.7 in the logs. Padding is dead compute.
4. **Are you dataloader-bound?** Watch `nvidia-smi`: GPU utilization periodically dropping to 0% means the dataloader is not keeping up. Increase `dataloader_num_workers`.
5. **Is `torch.compile(model)` an option?** For inference-shaped forward passes it can help; for SFT the wins vary. Test on your target model.
6. **Are you accidentally on `use_cache=True` in the trainer?** KV cache is for inference, not training. TRL disables it automatically for you; verify if you subclassed anything.
7. **Is gradient checkpointing on when you did not need it?** `gradient_checkpointing=True` roughly halves activation memory and roughly halves throughput. Only turn it on when you are memory-bound.

Almost every "training is slow" problem in a 1B–8B SFT setup is one of these seven items.

## The memory budget, in one paragraph

At BF16 on a 7B model with AdamW, the memory footprint is roughly `7B × (2 bytes/param weights + 2 bytes/param gradients + 8 bytes/param optimizer state)` = ~84 GB just for the parameter-and-optimizer state, before activations. This is why a 7B full-parameter SFT does not fit on a single 24 GB GPU. Options: gradient checkpointing to shrink activations, ZeRO / FSDP to shard optimizer state across GPUs (mod-107), or LoRA / QLoRA so the trainable parameters are ~0.1% of the whole (mod-104). At the 1B scale — this module's target — full parameter fits on a single 24 GB GPU without gymnastics.

## Exercise pointer

`exercises/exercise-02-masking-and-packing-correctness.md` has you print the `input_ids` / `labels` tensors of a real collated batch, add the assertions above to a test, and demonstrate that packed sequences do not leak across example boundaries by comparing loss on the same example packed vs. un-packed.

## Summary

- The SFT masking invariant is a trainer-side concern, not just a dataset-side one; verify with a decode-and-print debug script and lock the invariant in with an assertion.
- Three routes to masked labels in TRL: messages format with `return_assistant_tokens_mask` (default, prefer), `DataCollatorForCompletionOnlyLM` with a response template (single-turn only), instruction-format helpers (also single-turn). Do not switch routes mid-project.
- Packing SFT with per-example attention masks and reset position IDs (via FA2 `cu_seqlens` or Liger Kernel) reclaims the compute that padding would waste; naive packing leaks across examples and hurts SFT quality.
- Effective batch size in tokens is what matters for LR scaling: `per_device_train_batch_size × gradient_accumulation_steps × world_size × max_length × packing_efficiency`.
- Gradient accumulation divides the per-micro-batch loss by `gradient_accumulation_steps`; the trainer does this correctly, do not roll your own.
- If throughput is bad, walk the seven-item checklist. Almost every SFT throughput problem is on that list.
