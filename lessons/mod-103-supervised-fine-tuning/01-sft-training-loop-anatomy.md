# The SFT Training Loop, End to End

**Estimated read:** 45 minutes.
**You should be able to, by the end:** walk another engineer through the objects a TRL `SFTTrainer` run instantiates (model, tokenizer, dataset, collator, training args, trainer), explain what each one contributes to a forward/backward step, and know which of them Axolotl or Unsloth take over vs. leave to you.

## What SFT actually is, in one sentence

Supervised fine-tuning is **next-token cross-entropy on a curated set of `(prompt, response)` conversations, with the loss masked so only the response tokens contribute to the gradient**. The training objective is the same one the base model was pretrained with; what changes is (a) the specific distribution of text the model sees, and (b) the fact that most tokens are masked out of the loss.

If you read that sentence and thought "there is no separate 'instruction-following loss' or 'chat loss'" — correct. SFT is a data problem in a `-100` mask wrapper. Everything in this module is either "get the data through the loop" or "know what the loop is telling you."

## The five objects an SFT run instantiates

Any TRL `SFTTrainer` script — and every Axolotl / Unsloth config, once you strip the syntactic sugar — reduces to five objects wired together:

1. A **base model** (a `PreTrainedModel` loaded from a Hugging Face checkpoint or local path).
2. A **tokenizer** (an `AutoTokenizer` with a chat template — see mod-101 chapter 2, mod-102 chapter 4).
3. A **dataset** (a `datasets.Dataset` of message lists, plus an optional eval split).
4. A **data collator** (a callable that turns a batch of examples into `input_ids`, `labels`, `attention_mask` tensors, applying the loss mask).
5. A **trainer** (an `SFTTrainer`, which is a subclass of `transformers.Trainer` that knows about the collator and the chat template).

The trainer holds a `TrainingArguments` (or `SFTConfig`) object that carries the hyperparameters: learning rate, schedule, batch size, gradient accumulation, precision, checkpoint cadence, logging, evaluation. Chapters 2–6 tour those hyperparameters. This chapter is about the wiring.

## A minimum viable SFT script

Here is the smallest useful TRL script for a 1B-class base model:

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

MODEL_ID = "meta-llama/Llama-3.2-1B"          # base, not Instruct — you are creating the instruct-ness
DATASET_ID = "HuggingFaceH4/no_robots"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# The base model's tokenizer_config.json may not ship a chat template.
# For SFT you MUST pick one — pick one and pin it in your config.
if tokenizer.chat_template is None:
    tokenizer.chat_template = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
)

train_ds = load_dataset(DATASET_ID, split="train_sft")
eval_ds  = load_dataset(DATASET_ID, split="test_sft")

cfg = SFTConfig(
    output_dir="./out/first-run",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.0,
    bf16=True,
    logging_steps=10,
    eval_steps=200,
    save_steps=500,
    max_length=2048,
    packing=True,
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,   # `tokenizer=` on older TRL; TRL 0.12+ renamed it
)

trainer.train()
trainer.save_model()
```

That is the whole run. Everything about SFT-as-a-training-loop is a variation on one of the arguments in the `SFTConfig`. The remaining chapters give you enough context to argue for each value.

## What `SFTTrainer` does that `Trainer` does not

TRL's `SFTTrainer` extends `transformers.Trainer` with three responsibilities specific to SFT:

1. **Dataset formatting.** If your dataset rows are `{"messages": [...]}` conversations, `SFTTrainer` calls `tokenizer.apply_chat_template` on them for you. If your rows are `{"prompt": ..., "completion": ...}` or `{"text": ...}` it handles those too. The `dataset_text_field` and `formatting_func` arguments give you an escape hatch when your rows do not match a standard shape.
2. **Loss masking.** By default, when your data is in messages form, `SFTTrainer` uses `apply_chat_template(..., return_assistant_tokens_mask=True)` to build labels that are `-100` everywhere except assistant tokens. For `(prompt, completion)` data, it can accept a `DataCollatorForCompletionOnlyLM` with a `response_template` string that says "everything before this literal string is prompt, everything after is response" — chapter 2 covers this in detail.
3. **Packing.** With `packing=True` the trainer concatenates multiple short examples into a single sequence up to `max_length`, applying per-example attention masks so examples do not leak across each other. Chapter 2 covers when packing helps and when it hurts.

Everything else — the optimizer step, mixed precision, gradient accumulation, checkpointing, evaluation, logging, distributed hooks — comes from `transformers.Trainer` and behaves the same way it does for any other Hugging Face training job. That means the wide `Trainer` documentation and every `TrainingArguments` field is fair game for SFT; TRL is a thin (but load-bearing) SFT-specific shell around it.

## Where Axolotl and Unsloth fit

`SFTTrainer` is the reference implementation. Two other tools cover the same ground with different tradeoffs.

### Axolotl

Axolotl is an opinionated YAML-driven wrapper that codifies known-good SFT recipes for a wide set of open base models. A minimal Axolotl config for the run above:

```yaml
base_model: meta-llama/Llama-3.2-1B
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer

datasets:
  - path: HuggingFaceH4/no_robots
    type: chat_template
    chat_template: chatml

sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true

adapter: null                   # full-parameter SFT for this example
bf16: true
flash_attention: true

num_epochs: 1
micro_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2e-5
lr_scheduler: cosine
warmup_ratio: 0.03
weight_decay: 0.0

output_dir: ./out/first-run-axolotl
save_steps: 500
eval_steps: 200
logging_steps: 10
seed: 42
```

What Axolotl does for you:

- **Per-model template presets.** `chat_template: llama3` / `chatml` / `qwen` / etc. resolve to the correct Jinja and masking rules for that base. You do not hand-set `response_template` strings.
- **Multi-GPU launcher wiring.** Axolotl composes `accelerate`, DeepSpeed, and FSDP configs for you; you write one YAML.
- **Sensible logging and eval defaults.** WandB / TensorBoard integrations are one line each.

What Axolotl does *not* remove:

- The dataset still needs the mod-102 curation, dedup, and decontamination work.
- The training report and reproducibility manifest (chapter 6) are still your responsibility.
- Loss-curve reading (chapter 4) and failure-mode diagnosis (chapter 5) do not go away.

The right way to think about Axolotl: it is a config format for the same TRL-style trainer, worth using when your target model has a preset and worth stepping out of when it does not.

### Unsloth

Unsloth is a set of custom Triton kernels and drop-in `FastLanguageModel` wrappers that make LoRA / QLoRA SFT (mod-104) 1.5–2× faster and cut peak memory substantially for a specific set of supported architectures. It plugs into TRL — you swap `AutoModelForCausalLM.from_pretrained` for `FastLanguageModel.from_pretrained`, and pass the returned model into a standard `SFTTrainer`.

Use Unsloth when:

- You are doing LoRA / QLoRA (its main win is in PEFT paths; the wins for full-parameter SFT are smaller).
- The base model is on Unsloth's supported list (Llama 3.x, Mistral, Qwen 2.x, Phi, Gemma at time of writing — check current docs).
- You are on a single GPU or a small-multi-GPU setup that Unsloth supports.

Do not use Unsloth if:

- Your base model is not on its supported list — the wrapper will fall back or fail loudly.
- You need distributed strategies (FSDP, DeepSpeed ZeRO-3) that Unsloth does not currently support for that architecture.

Unsloth is a performance tool, not a training-methodology tool. Everything in the rest of this module — masking, loss curves, failure modes, reproducibility — still applies unchanged.

## The forward pass, one training step

If you have never watched a single training step happen tensor-by-tensor, it is worth doing once. In the loop above, one step of SFT does the following:

1. The collator produces `input_ids: [B, T]`, `labels: [B, T]` (with `-100` on prompt positions), and `attention_mask: [B, T]` — where B is `per_device_train_batch_size` and T is up to `max_length`.
2. The model runs the forward pass mod-101 chapter 1 walked you through, producing `logits: [B, T, V]` where V is the vocab size.
3. The model shifts labels by one position (predicting token `t` from context `< t`) and computes `CrossEntropyLoss(logits.view(-1, V), labels.view(-1))` with `ignore_index=-100`. The `-100` positions contribute nothing to the sum or the count in the mean.
4. The gradient is backpropagated. In BF16 mixed precision, activations and gradients are BF16; parameters and optimizer state are FP32 in AdamW.
5. If `gradient_accumulation_steps > 1`, the gradient is *accumulated* into the parameter's `.grad` field; the optimizer step happens only every `gradient_accumulation_steps` micro-batches. Chapter 2 covers the math.
6. The optimizer applies the update, the scheduler advances the learning rate one step, the grad is zeroed.
7. Every `logging_steps` micro-batches, the trainer logs the running loss and learning rate. Every `eval_steps`, it runs a full eval pass on the eval dataset. Every `save_steps`, it writes a checkpoint.

There is no reinforcement learning, no reward model, no reference model, no sampling loop, no KL penalty. The gap between an SFT trainer and a preference-optimization trainer (mod-105) is significant; the gap between an SFT trainer and a pretraining trainer is the loss mask.

## Sanity checks to run before the first `train()`

The three things worth verifying before you sink an hour into a training run:

1. **Print one collated batch** and inspect `input_ids`, `labels`, and `attention_mask` shapes, and confirm that `labels` is `-100` on the tokens you expect it to be `-100` on. Chapter 2 exercise runs you through this.
2. **Run one forward pass** by calling `model(**batch).loss` and confirming it produces a finite scalar in a sane range (2–5 for a base model on well-formatted text; something in the tens or with NaNs is a bug).
3. **Run a very short training run** (`max_steps=10`) and confirm loss decreases at all. If it does not, no amount of hyperparameter tuning fixes what is a data / mask / model config bug.

These three checks catch 90% of the "the run finished but the model is broken" bugs before they cost you a full training run's compute.

## Choosing a base model for a first run

For this module — and specifically for exercise 01 — pick a 1B-class base model. Reasons:

- Fits comfortably on a single 24 GB consumer GPU (or a small cloud instance) at BF16 with `max_length=2048` and `per_device_train_batch_size=2`. You will not lose a day to OOM debugging.
- Trains a small SFT dataset in under an hour, so the diagnosis-loop-of-a-run is fast enough to iterate.
- Has enough capacity that the model behavior visibly changes with training. Very small models (under 500M) can be too underparameterized to demonstrate SFT effects reliably.

Concrete options that work well as of this module's authoring:

- `meta-llama/Llama-3.2-1B` — the reference "small base" for the current Llama family; Instruct variant is available for comparison.
- `Qwen/Qwen2.5-1.5B` — ChatML template baked in; multilingual base.
- `microsoft/Phi-3.5-mini-instruct` is 3.8B — a step up if you have the memory.
- `HuggingFaceTB/SmolLM2-1.7B` — trained on well-documented open data; useful when you want to explain everything about the base.

Whichever you pick, use the **base** (non-instruct) checkpoint. The point of SFT is that *you* make it instruct-y. Fine-tuning an already-instruct model is a legitimate real-world exercise (mod-105 does it), but it makes the "before / after" contrast harder to see.

## The dataset contract, one more time

The trainer expects your dataset to be one of the shapes it knows:

- **Messages format:** `{"messages": [{"role": "system"|"user"|"assistant"|"tool", "content": "..."}]}`. This is the canonical SFT shape and the one you should default to. Templates and masks come free.
- **Prompt / completion format:** `{"prompt": "...", "completion": "..."}`. Simpler, but you take on the responsibility of handling multi-turn yourself.
- **Raw text format:** `{"text": "..."}`. Pretraining-style; the trainer computes loss over every token. Do not use this for SFT unless you understand exactly what you are giving up.

The mod-102 output should already be in messages format with the full metadata envelope. Extra keys in each row are harmless — the trainer ignores them; you can still slice loss by them in analysis (chapter 4).

## Exercise pointer

`exercises/exercise-01-first-sft-run-on-a-1b-model.md` walks you through the script above end to end on a real 1B-class base and a small open SFT dataset. The point is not the checkpoint. The point is that you have run the loop and can reason about the timing of every argument.

## Summary

- SFT is next-token cross-entropy with a loss mask; there is no separate SFT objective.
- Any SFT run reduces to five objects — base model, tokenizer, dataset, collator, trainer — plus a `TrainingArguments` / `SFTConfig`. This chapter's script is the minimum viable form.
- `SFTTrainer` extends `transformers.Trainer` with dataset formatting, loss masking, and packing. Everything else (optimizer, mixed precision, checkpointing, evaluation) comes from `Trainer`.
- Axolotl is a YAML wrapper around the same trainer with per-model presets; Unsloth is a kernel-level speedup for supported LoRA / QLoRA paths. Neither removes the reasoning you do in this module.
- Always print a batch, run one forward pass, and run a short `max_steps=10` sanity job before you commit to a full training run.
- Choose a 1B-class base for a first run so the diagnosis loop is fast. Use the **base**, not the Instruct.
