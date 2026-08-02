# Running a Managed Fine-Tune on OpenAI End-to-End

**Estimated read:** 40 minutes.
**You should be able to, by the end:** produce a validated OpenAI JSONL from a `messages`-format dataset, upload it via the Files API, create and monitor a fine-tuning job via the Fine-Tunes API, interpret the job events and reported loss curves, integrate the resulting `ft:...` model ID into your inference code, and estimate the job's training and long-term inference cost from the vendor's published pricing.

OpenAI's Fine-tuning API is the reference implementation for this module. It is the most-used managed API, the one with the deepest tooling, and the one whose seven-step shape (chapter 1) is closest to what every other vendor's shape converges toward. Learn it once and 80% of chapter 3 becomes "here is what Anthropic / Google / Bedrock does differently."

## The whole thing in ten lines of code

Before drilling in, here's the entire pipeline at its simplest — a fine-tune from an already-prepared JSONL file to a model ID you can call:

```python
from openai import OpenAI
client = OpenAI()

train_file = client.files.create(file=open("train.jsonl", "rb"), purpose="fine-tune")
val_file   = client.files.create(file=open("val.jsonl", "rb"),   purpose="fine-tune")

job = client.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=val_file.id,
    model="gpt-4.1-mini-2025-07-14",
    suffix="support-triage-v3",
)

# ... wait for job.status == "succeeded" ...

resp = client.chat.completions.create(
    model=job.fine_tuned_model,           # e.g., "ft:gpt-4.1-mini-...:my-org:support-triage-v3:AbCdEf"
    messages=[{"role": "user", "content": "Refund on order #40219"}],
)
```

The rest of this chapter unpacks each of those calls: what the JSONL must look like, what the job actually does, how to interpret its events, what the model ID means, and what it costs.

## Step 1 — The JSONL, in one page

The **chat model format** for OpenAI is a JSONL file where each line is one training example whose top-level `messages` field is the OpenAI Chat Completions message list you would send to the base model for that same conversation:

```jsonl
{"messages": [{"role": "system", "content": "You are a support triage agent."}, {"role": "user", "content": "My charge failed."}, {"role": "assistant", "content": "I can help. Which card were you using?"}]}
{"messages": [{"role": "system", "content": "You are a support triage agent."}, {"role": "user", "content": "Refund on order #40219"}, {"role": "assistant", "content": "Locating order #40219 now. One moment."}]}
```

Rules that matter and are silently enforced:

- Every example must have at least one `assistant` message. The loss is computed on assistant tokens; an example with no assistant turn trains nothing.
- The final message is not required to be `assistant`, but the assistant tokens up to and including the last assistant turn are what the loss is computed on. Concretely: for multi-turn conversations, all assistant turns contribute to loss unless you explicitly mask them (see below).
- `role` values: `system`, `user`, `assistant`, and `tool` (for tool-calling training). No `function` in current schema — that was the deprecated Functions API's role name.
- Each message needs `content` (string) OR a `tool_calls`/`tool_call_id` structure for tool-calling training. Chapter 4 has the full tool-training JSONL schema; keep the simple chat form in your head for now.
- File is JSONL, not JSON — one row per line, no top-level array, no trailing comma nonsense. UTF-8, no BOM.

### Turn-level loss masking

By default, all assistant turns in a multi-turn example contribute to the loss. To exclude a specific assistant turn from the loss (e.g., a canned greeting you don't want the model to learn to reproduce), add `"weight": 0` to that message:

```json
{"role": "assistant", "content": "Hi! How can I help?", "weight": 0}
```

The `weight` field accepts `0` (excluded) or `1` (included, the default). This is the OpenAI equivalent of the SFT masking discipline you learned in mod-103 chapter 2 — worth using anytime a multi-turn example contains scaffolding turns you do not want the model to memorize.

### The `prompt`/`completion` legacy format

Older docs and older base models expected a `{"prompt": "...", "completion": "..."}` JSONL. That path is deprecated for chat models and only relevant if you are fine-tuning one of the legacy `babbage-002` / `davinci-002` completion base models. New work should always be in the messages format. Chapter 4 covers the automatic conversion from a Hugging Face messages dataset.

## Step 2 — Validation before upload

Every hour spent uploading a broken JSONL is an hour of debugging a job that then rejects it. OpenAI publishes a validation script; the mental model is: **run the same schema check the vendor will run, locally, before you upload.**

The minimum validator you should own:

```python
import json, sys
from collections import Counter

def validate(path):
    seen_roles = Counter()
    errors = []
    n = 0
    with open(path) as f:
        for i, line in enumerate(f, 1):
            n += 1
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append((i, f"invalid JSON: {e}")); continue
            if "messages" not in ex or not isinstance(ex["messages"], list):
                errors.append((i, "missing or non-list 'messages'")); continue
            if not any(m.get("role") == "assistant" for m in ex["messages"]):
                errors.append((i, "no assistant message — training would be a no-op"))
            for m in ex["messages"]:
                seen_roles[m.get("role")] += 1
                if m.get("role") not in {"system", "user", "assistant", "tool"}:
                    errors.append((i, f"invalid role: {m.get('role')}"))
                if "content" not in m and "tool_calls" not in m:
                    errors.append((i, "message has neither 'content' nor 'tool_calls'"))
    print(f"rows: {n}, role counts: {dict(seen_roles)}, errors: {len(errors)}")
    for line, err in errors[:20]:
        print(f"  line {line}: {err}")
    return errors

if __name__ == "__main__":
    errors = validate(sys.argv[1])
    sys.exit(1 if errors else 0)
```

Extend this with:

- Token counting per row (via `tiktoken` for the base model you're targeting). Reject rows over the base model's context window, or over your training-time `max_tokens` budget if you're capping it.
- Duplicate-row detection (hash on canonicalized JSON). Dedup upstream in the mod-102 pipeline, but a sanity check here catches accidental double-writes.
- Per-role token accounting so you know what fraction of your data is assistant tokens (the ones that actually train). If assistant tokens are <10% of total, your `n_epochs=1` is going to look under-trained.
- A random-sample dump — print 5 random rows in a readable form so a human can eyeball them. This catches the "we accidentally trained on JSON-escaped garbage" class of bug.

OpenAI's cookbook has a reference implementation; treat it as one input, not the whole answer.

## Step 3 — Upload

Files up to the current per-request limit (multi-GB in 2026, check the current docs) can be uploaded directly:

```python
train_file = client.files.create(file=open("train.jsonl", "rb"), purpose="fine-tune")
```

The `purpose` field is important — `fine-tune` is the only value that a fine-tuning job will accept. Files uploaded with `purpose="assistants"` or `purpose="user_data"` will be rejected at job creation.

For truly large uploads, OpenAI supports a multipart upload flow (`client.uploads.create` / `.add_part` / `.complete`) that resumes cleanly on interruption. Use it if you are pushing >1 GB training files over an unreliable connection.

Files linger in your account until you explicitly delete them (`client.files.delete(file_id)`) or the retention window (currently 30 days for non-fine-tune files, longer for fine-tune files while any job or model references them) elapses. In a hygienic pipeline, delete the file after the job completes if you have the JSONL in source control.

## Step 4 — Job creation

The full surface of `fine_tuning.jobs.create` — the fields you'll actually reach for:

```python
job = client.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=val_file.id,                     # optional but strongly recommended
    model="gpt-4.1-mini-2025-07-14",                 # base model — versioned!
    suffix="support-triage-v3",                      # goes into the resulting model ID
    seed=42,                                         # reproducibility
    method={                                         # method-scoped hyperparameters
        "type": "supervised",
        "supervised": {
            "hyperparameters": {
                "n_epochs": "auto",                  # or an integer
                "learning_rate_multiplier": "auto",  # or a float, usually 0.01–4.0
                "batch_size": "auto",                # or an integer
            }
        },
    },
    integrations=[                                    # optional
        {"type": "wandb", "wandb": {"project": "my-project", "tags": ["mod-110"]}},
    ],
)
```

Three of these deserve a paragraph each.

### `model`

Always pin the versioned model ID, not the alias. `gpt-4.1-mini` is an alias that moves; `gpt-4.1-mini-2025-07-14` is the pinned version that will not silently swap under you. If OpenAI ships a new snapshot in a month, an unversioned base means your next fine-tune trains a *different* base and your regressions become impossible to attribute. This is the same "pin the base revision" discipline from mod-103 chapter 1, applied to a managed API.

### `method`

Two supported types worth knowing:

- `"supervised"` — the SFT path. This is what "fine-tuning" means unless otherwise specified. Hyperparameters as above.
- `"dpo"` — direct preference optimization, available on chat models. The JSONL format is different — each row has `input.messages` (the shared context), `preferred_output`, and `non_preferred_output`. Method-scoped hyperparameters include `beta` (the DPO regularization strength). Everything you learned in mod-105 chapter 2 about DPO applies; the difference is that OpenAI runs the loop.

<!-- needs-research: OpenAI has also shipped Reinforcement Fine-Tuning (RFT) on o-series models with a `"type": "reinforcement"` method that consumes a grader function. RFT has narrower model coverage, higher cost, and its own JSONL / grader schema — verify current model list and grader API before including in production discussion. -->

For a first fine-tune on a chat model, `method={"type": "supervised", ...}` is the right default. Chapter 3 covers when Bedrock's Continued Pretraining or Vertex's tuning modes are the better fit for a specific vendor's model.

### Hyperparameter `"auto"`

`n_epochs="auto"`, `learning_rate_multiplier="auto"`, and `batch_size="auto"` let OpenAI pick values based on your dataset size. The auto policy targets convergence on typical datasets; it usually picks 3 epochs on smallish sets (<1k rows), 1–2 on larger ones, and adjusts batch size to fit the model's memory budget.

When to override:

- **Small dataset (< 200 rows)** — you often want `n_epochs=4` or higher. `"auto"` will under-train.
- **Large dataset (> 100k rows)** — you often want `n_epochs=1`. `"auto"` usually agrees, but pin it.
- **Loss curve stalled** — bump `learning_rate_multiplier` (0.5–4.0 range). Reference mod-103 chapter 3's LR-diagnosis rubric; the multiplier semantics are the same.
- **Loss curve too spiky** — halve `learning_rate_multiplier`. Same rubric.

The multiplier is *relative* to the internal base LR — a value of 1.0 is the default. Values outside 0.01–4.0 are usually not sensible; if you find yourself wanting 10x, you probably want more data or a different base model.

## Step 5 — Monitoring the job

The job returns immediately with `status="validating_files"` and moves through `queued` → `running` → `succeeded` (or `failed` or `cancelled`). To watch:

```python
import time

while True:
    job = client.fine_tuning.jobs.retrieve(job.id)
    print(f"[{job.status}] created={job.created_at} finished={job.finished_at}")
    if job.status in {"succeeded", "failed", "cancelled"}:
        break
    time.sleep(30)
```

More usefully, stream events:

```python
for event in client.fine_tuning.jobs.list_events(fine_tuning_job_id=job.id, limit=100):
    print(f"[{event.created_at}] {event.level}: {event.message}")
```

Events include: file validation, queue position, epoch start / end, per-step loss snapshots, validation loss snapshots (if you passed a validation file), and terminal status. The validation-loss events are what you actually want — if train loss is dropping but val loss is flat or rising, you are overfitting and should stop the job (`client.fine_tuning.jobs.cancel(job.id)`).

The `result_files` on the succeeded job contain a CSV with per-step training loss, per-step validation loss (if val file provided), and token counts. Download it:

```python
job = client.fine_tuning.jobs.retrieve(job.id)
for f_id in job.result_files:
    content = client.files.content(f_id).text
    with open(f"metrics_{f_id}.csv", "w") as out:
        out.write(content)
```

Plot it, sanity-check it against your intuition from mod-103 chapter 3 (loss curve shapes are model-agnostic), and archive the CSV alongside the eval report.

### The failure modes to expect

Roughly in decreasing frequency:

1. **Validation failed** — JSONL schema error, invalid role, empty assistant. Fix upstream, re-upload, retry.
2. **Content policy violation** — your training data tripped OpenAI's safety filters at ingestion. The event message will not always tell you which rows; you'll need to run the moderation endpoint against your data locally to find them. Mod-106 chapter 4's safety-review discipline is the prevention.
3. **Queued for a long time** — capacity constraints, particularly on newly released base models. Not much you can do beyond wait; the event stream shows queue position.
4. **Failed at training start** — very rare, usually infrastructure. Retry with the same file and hyperparameters; if it fails twice, open a support ticket.
5. **Loss did not decrease** — the job succeeds but the model is no better than base. Usually a data problem (assistant tokens are noise, or the task is not learnable from your dataset). Diagnose with mod-108's eval before you conclude the job was wrong.

## Step 6 — The resulting model ID

Success gives you a model ID with the shape:

```
ft:<base-model>:<org-name>::<suffix>:<hash>
```

Concrete example: `ft:gpt-4.1-mini-2025-07-14:acme::support-triage-v3:AbCdEf12`.

This ID is the artifact. It replaces the base model name everywhere you would call the base model. It is scoped to your org — nobody else can call it — and it persists indefinitely (with a monthly hosting rent while it is available).

`suffix` is your friend for humans. Bake a version and a task into it (`suffix="support-triage-v3"`) so a glance at the model ID tells you what fine-tune this is without cross-referencing a spreadsheet. The trailing hash disambiguates two fine-tunes with the same suffix.

To list all your fine-tuned models: `client.models.list()`. To delete one: `client.models.delete(model_id)` — the hosting rent stops when you delete.

## Step 7 — Inference

The fine-tuned model IS a chat model. You call it the same way you call `gpt-4.1-mini`:

```python
resp = client.chat.completions.create(
    model="ft:gpt-4.1-mini-2025-07-14:acme::support-triage-v3:AbCdEf12",
    messages=[
        {"role": "system", "content": "You are a support triage agent."},
        {"role": "user", "content": "Refund on order #40219"},
    ],
    temperature=0,
)
```

Rules that trip up first-time users:

- **The system prompt is not baked in.** If your training data all included the same system prompt, you still need to send that same system prompt at inference. The model has learned "given this system prompt, produce this shape of assistant response" — omit the system prompt at inference and quality drops.
- **Sampling matters.** The eval numbers you get on the fine-tuned model are meaningful only at the sampling params you set. If your training data resembled deterministic responses, run eval at `temperature=0` first. Then, if production uses `temperature=0.7`, run a "production-mirror" eval lane too (mod-108 chapter 7's discipline).
- **Rate limits are separate.** Fine-tuned models have their own rate-limit buckets, typically lower than the base model. Check the current limits in the OpenAI dashboard; do not assume base model TPM applies.
- **Batch API works.** For high-throughput offline eval or backfill, submit fine-tune-model requests via the Batch API for the standard discount. Chapter 7 uses this for eval cost control.

## Cost — the arithmetic

Managed fine-tuning has three cost lanes. Numbers below are illustrative — always cross-check current [OpenAI pricing](https://openai.com/api/pricing/).

### Lane 1 — Training

Priced per token trained, typically as:

```
training_tokens_billed = total_tokens_in_training_file × n_epochs
training_cost           = training_tokens_billed × price_per_training_token[base_model]
```

Example: 50k rows × 500 tokens/row × 3 epochs = 75M training tokens. At a hypothetical $0.025 per 1M training tokens for `gpt-4.1-mini`, that's ~$1.9. For the same job on `gpt-4.1` at higher price tier, it might be ~$25.

<!-- needs-research: verify current training-token pricing for GPT-4.1 family, GPT-4o family, and o-series before publishing exact per-run numbers. Prices shift; the arithmetic above is the shape, not a commitment. -->

### Lane 2 — Hosting

The fine-tuned model has a monthly rent while it exists. This is a small line item per model, but it multiplies with the number of fine-tunes you retain. If you produce a new fine-tuned model every week and never delete the old ones, in a year you are paying rent on 50 fine-tunes.

The operational discipline: after a new fine-tune is promoted, delete the previous one unless you need it for a specific comparison. Keep a "last-known-good" model retained explicitly.

### Lane 3 — Inference

Priced per token at a **premium over the base model** — typically 2× the base model's input and output token prices, though this ratio varies by model tier. Concrete: if base `gpt-4.1-mini` is $0.15 / 1M input, $0.60 / 1M output, the fine-tuned version might be $0.30 / 1M input, $1.20 / 1M output.

For high-volume production traffic, this premium is the dominant cost — a factor that chapter 5's cost comparison foregrounds against self-hosted alternatives.

### The rough total-cost formula

```
monthly_cost ≈ training_cost / amortization_months
             + hosting_rent
             + inference_cost_per_1M × monthly_tokens
```

Chapter 5 has the full formula and shows how to compare it against the self-hosted stack from mod-109 chapter 7.

## Integrations and reproducibility

Two things worth wiring on your first fine-tune:

**W&B integration.** Pass `integrations=[{"type": "wandb", "wandb": {"project": "my-project", "tags": ["mod-110"]}}]` in the job create call. OpenAI streams training loss, validation loss, and job metadata directly to your W&B run. This makes the loss curve inspectable in your existing tooling and makes the OpenAI job discoverable from your usual experiment tracker.

**Manifest.** Same discipline as mod-103 chapter 1's run manifest, adapted to managed:

```yaml
job_id: ftjob-AbCdEf12345
model_id: ft:gpt-4.1-mini-2025-07-14:acme::support-triage-v3:AbCdEf12
base: gpt-4.1-mini-2025-07-14
suffix: support-triage-v3
train_file:
  path: s3://data/support/train-2026-08-01.jsonl
  sha256: 3c...
  rows: 50247
  tokens: 25123456
val_file:
  path: s3://data/support/val-2026-08-01.jsonl
  sha256: 9b...
  rows: 512
method: supervised
hyperparameters:
  n_epochs: 3
  learning_rate_multiplier: auto (resolved to 0.5)
  batch_size: auto (resolved to 32)
seed: 42
created_at: 2026-08-01T14:12:07Z
finished_at: 2026-08-01T15:47:33Z
training_tokens_billed: 75370368
cost_estimate_usd: 1.88
integrations:
  - wandb: my-project/mod-110/AbCdEf12
```

This is what the release-gate report from mod-108 chapter 6 will reference and what your platform's model registry (mod-111) will index.

## Deletion, portability, and the exit

Two exit paths worth planning for from day one:

1. **Delete the model.** `client.models.delete(model_id)` stops hosting rent and removes the model. You cannot un-delete; the JSONL that trained it and the manifest above are what let you re-run.

2. **Vendor exit.** OpenAI has deprecated model bases before (see: `davinci-002` sunset). When a base model is deprecated, fine-tunes on that base become unusable. The safe posture: (a) keep the training JSONL in your source of truth (source control or object storage), (b) keep the manifest with pinned base version, (c) know which alternate base you'd re-fine-tune to and have run at least one comparison eval against it, so the migration is a re-run of a known recipe, not new engineering.

The whole point of the manifest is that "OpenAI deprecates the base you're on" is a fire drill, not an outage.

## Common bugs and their fixes

Compiled from real fine-tuning-team failure logs. Prevent, don't diagnose.

**Bug: fine-tune has learned to always start responses with "Sure! Here's..."**
Cause: your assistant messages in training all started that way. `weight: 0` was never applied to the boilerplate.
Fix: mask or rewrite the scaffolding. Also, evaluate with the real production system prompt — the model may be responding to *that*, not the training data.

**Bug: fine-tune ignores the system prompt in production.**
Cause: training data had no system prompt (or an empty one) but production sends one.
Fix: retrain with the production system prompt present in every training row, or rewrite the production call to omit the system prompt (rarely the right answer — system prompts control safety in production).

**Bug: loss curve looks great, quality is worse than base.**
Cause: the model learned the format / persona of your data but lost general capability. This is the mod-103 chapter 4 catastrophic-forgetting story, with the same shape and no direct knob to fix — you cannot inject a replay-buffer into a managed job.
Fix: reduce `n_epochs`, mix in general instruction-following data, or lower the `learning_rate_multiplier`. Chapter 4 of mod-103 explains the mechanism; the mitigation surface here is smaller than self-hosted, so prevention matters more.

**Bug: fine-tune costs 5× what you estimated.**
Cause: the token count in the JSONL was estimated by character count / 4, not by tokenizing with `tiktoken`. Long code / non-English / emoji-heavy content tokenizes to more tokens per character than English prose.
Fix: always tokenize with `tiktoken` for the specific base model before estimating cost. Store the token count in the manifest.

**Bug: job succeeded, but calling the model returns "model not found."**
Cause: you called `models.retrieve(base_model_id)` instead of `models.retrieve(job.fine_tuned_model)`, or the API key you're calling with is a different org than the one that trained.
Fix: fine-tuned models are org-scoped. Use the same API key / org that ran the job.

## What this chapter did not cover

Deliberately deferred:

- **The dataset conversion.** How your `datasets.Dataset` becomes the JSONL you upload — chapter 4 owns this.
- **Tool-calling fine-tunes.** The JSONL for tool training has `tool_calls` structure per turn — chapter 4 covers the schema.
- **Multimodal fine-tunes.** GPT-4.1 supports image fine-tuning; format is `{"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {...}}]}`. Chapter 4 covers the schema.
- **Eval integration.** How the fine-tuned model feeds mod-108's release gate via `lm-evaluation-harness --model openai-chat-completions --model_args model=ft:...` — chapter 7 owns this.
- **The other vendors.** Chapter 3 tours Anthropic, Vertex, Mistral, Cohere, Bedrock, and Azure — same seven-step pipeline, different fields and quirks.
- **When not to do this at all.** Chapter 6 makes the closed-frontier vs. open-base decision explicit; nothing in this chapter argues you *should* fine-tune on OpenAI — it teaches you *how*, if you have decided to.

## Summary

- OpenAI's Fine-tuning API is the seven-step pipeline from chapter 1 with concrete calls: `files.create(purpose="fine-tune")`, `fine_tuning.jobs.create(training_file, model, method, ...)`, poll or stream events, receive a `ft:...` model ID, call it as a chat model.
- The JSONL is `messages`-format chat, one example per line. All assistant turns contribute to loss unless masked with `"weight": 0`. Every example needs at least one assistant turn.
- Validate locally with the same schema check the vendor will run. Broken JSONL wastes an upload, an ingest, and 20 minutes of debugging.
- Pin the versioned base model (`gpt-4.1-mini-2025-07-14`, not `gpt-4.1-mini`). Set `seed` for reproducibility. Use `"auto"` hyperparameters as the default and override with the same rubric mod-103 chapter 3 taught for self-hosted SFT.
- Cost has three lanes: training tokens × epochs at the training-token price, monthly hosting rent per retained fine-tune, and per-token inference at a premium over the base model. Chapter 5 puts the total into a full comparison.
- The model ID is your artifact. Write a manifest that pins the base version, JSONL hashes, hyperparameters, cost estimate, and W&B run. Deletion of the model stops hosting rent; deletion of the manifest turns a base deprecation from a fire drill into an outage.
