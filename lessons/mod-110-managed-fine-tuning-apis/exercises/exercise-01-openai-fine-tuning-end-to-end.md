# exercise-01: OpenAI Fine-Tuning End-to-End

**Estimated effort:** 3 hours
**Companion chapters:** [`02-openai-end-to-end.md`](../02-openai-end-to-end.md), [`04-dataset-format-conversion.md`](../04-dataset-format-conversion.md), [`07-eval-handoff-across-backends.md`](../07-eval-handoff-across-backends.md)

## Objective

Take an existing SFT-style dataset from Hugging Face `messages` format all the way through OpenAI's Fine-tuning API — validate, convert, upload, train, receive the model ID, call it, and run at least one lane of the mod-108 release gate against it. The output is a working `ft:...` model, a manifest that pins every reproducibility field, a cost tally, and an eval report — the artifact bundle a real project would ship to a review.

The point of this exercise is mechanics. You will spend $2–$20 depending on model tier and dataset size; keep the dataset small (≤ 2000 rows) so the training run is cheap and the whole exercise completes inside 3 hours end-to-end.

## Prerequisites

- An OpenAI API key with fine-tuning enabled (paid account; verify billing is set up before starting — the job will fail fast otherwise).
- A recent `openai` Python SDK (>=1.50), `tiktoken`, `datasets`, `jsonlines`, and the pinned `lm-evaluation-harness` install from mod-108 exercise 01.
- One SFT-style dataset in messages format. Any of:
  - A subset of your mod-102 exercise output.
  - A small public instruct dataset (e.g., 1000-row slice of `HuggingFaceH4/no_robots`, `databricks/databricks-dolly-15k`, or an equivalent messages-format instruct set with a permissive license — verify licensing before uploading).
  - A synthetic mini-dataset you generated for mod-102 exercise 04.
- A pre-authored task-specific held-out set (mod-108 exercise 02) — 50–200 items is enough for this exercise.
- Budget: expect $2–$20 for training, small for hosting for the exercise duration, and $1–$10 for the eval-through-API pass. Delete the fine-tune when done to stop hosting rent.

## Requirements

### Part A — Environment and pinning

Author `requirements.txt` and `pins.md` recording versions for at minimum:

- `openai`
- `tiktoken`
- `datasets`
- `lm-eval` (commit SHA — same as mod-108 exercise 01 for like-for-like eval)
- Python version, OS, region you'll call from

Include a `.env.template` with `OPENAI_API_KEY` and note that the real `.env` is git-ignored. Never commit a real key.

### Part B — Choose the base model and record why

In `base_choice.md`, pin the exact versioned base model you will fine-tune (e.g., `gpt-4.1-mini-2025-07-14`, not `gpt-4.1-mini`). Explain in one paragraph:

- Which tier you picked (mini / standard / o-series) and why.
- The cost tier per training and inference tokens (link to current pricing page; note the retrieval date).
- The base model's context length and whether your dataset fits inside it.

This is the "record decisions so a reviewer can audit them" discipline; do it every time even for a training exercise.

### Part C — Convert the dataset to OpenAI JSONL

Author `convert.py` implementing the `to_openai` converter from chapter 4. Consume your source dataset in messages format; produce `artifacts/train.jsonl` and `artifacts/val.jsonl` (an 80/10/10 or 90/5/5 split is fine).

Requirements for the converter:

- Every JSONL row is exactly the OpenAI chat schema — `{"messages": [...]}` with no extra top-level fields.
- The system prompt (if any) is the first message in `messages`.
- No assistant turn is empty.
- Roles are one of `system`, `user`, `assistant`, `tool`.
- UTF-8 without BOM; `ensure_ascii=False`.

### Part D — Validate the JSONL

Author `validate.py` implementing the six-check validator from chapter 4. Run against both `train.jsonl` and `val.jsonl`. The validator must exit non-zero on any error and print an error-count summary. Fix any errors before proceeding.

Additionally, add a token-count report using `tiktoken.encoding_for_model(...)` for your chosen base:

- Total tokens per file.
- Per-row: min / p50 / p90 / p99 / max.
- Assistant-tokens-only per-row (the tokens that actually contribute to loss).

Save the report as `artifacts/token_report.md`. This is the input to your cost estimate below.

### Part E — Estimate cost before you spend it

Author `cost_estimate.md` with:

- Training cost estimate: `total_train_tokens × n_epochs × price_per_1M_train_token[base]`.
- Hosting cost for the exercise's expected duration (a few hours; note the rounded-up billing unit if per-hour).
- Inference cost for the eval pass (Part I): `est_input_tokens_across_all_eval_items + est_output_tokens_across_all_eval_items × per-token prices at fine-tune tier`.
- Sum with a 30% headroom.

If the estimate exceeds your budget cap for this exercise, reduce the training set size or drop `n_epochs` before spending.

### Part F — Upload the files

Author `upload.py` that:

- Uploads `train.jsonl` and `val.jsonl` with `purpose="fine-tune"`.
- Prints the returned file IDs.
- Writes them to `artifacts/upload_result.yaml` for the next step.

Verify by calling `client.files.retrieve(file_id)` and confirming `status == "processed"` before proceeding to Part G.

### Part G — Create the fine-tune job

Author `create_job.py` that:

- Calls `client.fine_tuning.jobs.create(...)` with the file IDs from Part F, your pinned base, and hyperparameters.
- Uses `"auto"` for `n_epochs`, `learning_rate_multiplier`, `batch_size` for this exercise unless you have a specific reason to override; document the reason if you do.
- Sets `seed=42`.
- Sets a meaningful `suffix` (e.g., `mod110-ex01-yourname`).
- If you have a W&B account, wires the `wandb` integration.
- Writes the returned job object to `artifacts/job.yaml`.

### Part H — Monitor the job and record events

Author `monitor.py` that streams events (`client.fine_tuning.jobs.list_events`) until the job reaches a terminal status. Log every event to `artifacts/events.jsonl`. On success, download the training-metrics CSV(s) from `result_files` to `artifacts/metrics/`.

Plot `artifacts/metrics/loss.png`:
- Training loss over steps.
- Validation loss over steps (if val file provided).

Include the plot in `artifacts/report.md` (Part J).

### Part I — Call the fine-tuned model and run a mini eval

Author `smoke.py` that calls the resulting `job.fine_tuned_model` on 3 held-out task inputs and prints the completions. This is a sanity check — if outputs are obvious junk, stop and diagnose before spending on the full eval.

Then author `eval.py` that runs the `lm-evaluation-harness` `openai-chat-completions` backend against the fine-tuned model on at least:

- Your task-specific held-out set from mod-108 exercise 02.
- IFEval (5-shot) OR GSM8K (5-shot) — pick one that's meaningful for your data.

Use `--gen_kwargs temperature=0`, `--apply_chat_template`, and the pinned `lm-eval` version. Save results to `results/openai-ft/`.

Also run the same tasks against the **base model** (no fine-tune) with the same settings, saving to `results/base/`. This gives you the delta.

### Part J — Report and artifact bundle

Author `artifacts/report.md` with:

1. **Setup.** Base model + pinned version, dataset source + row count + token count, hyperparameters used, seed, region.
2. **Cost.** Estimated (Part E) vs. actual (from `job.result_files` and inference metering). Difference explained.
3. **Job.** Job ID, model ID, wall-clock duration, event count, any warnings or errors during training.
4. **Loss curves.** The plot from Part H, with a two-paragraph interpretation (did it converge? overfitting? underfitting?).
5. **Eval.** Comparison table of the fine-tuned model vs. base model on the tasks from Part I, with bootstrap CIs. Include per-item disagreements on the task-specific held-out set (the ones where FT and base differ) — helps interpret whether the FT is actually helping.
6. **Manifest.** The full manifest from chapter 2's template.
7. **Cleanup.** Whether you deleted the fine-tuned model (and when) — for exercise hygiene, delete it after eval unless you have a reason to keep it.
8. **What surprised you.** One paragraph.

### Part K — Directory layout

Final tree:

```
outputs/exercise-01/
├── requirements.txt
├── pins.md
├── .env.template
├── base_choice.md
├── convert.py
├── validate.py
├── upload.py
├── create_job.py
├── monitor.py
├── smoke.py
├── eval.py
├── cost_estimate.md
├── artifacts/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── token_report.md
│   ├── upload_result.yaml
│   ├── job.yaml
│   ├── events.jsonl
│   ├── metrics/
│   │   ├── train_metrics.csv
│   │   └── loss.png
│   ├── manifest.yaml
│   └── report.md
└── results/
    ├── base/
    └── openai-ft/
```

## Starter guidance

- **Estimate cost twice: once in tokens, once in dollars.** The token count comes out of `tiktoken`; the dollar figure comes out of the current pricing page. If you find yourself skipping either, you will be surprised by the bill.
- **Do the smoke test (Part I `smoke.py`) before you commit to the full eval.** A five-minute smoke test that reveals the FT is broken saves an hour of eval cost and time.
- **Delete the fine-tune when done.** `client.models.delete(model_id)` after Part J is finished. Hosting rent accrues for as long as the model exists.
- **Do NOT commit `.env` or the training JSONL if it contains anything sensitive.** The training JSONL of most exercise datasets is public; verify.
- **If the job takes over an hour to start (`status: "queued"`), don't panic.** New-base capacity is uneven. The event stream will show queue position; the model still trains, it just waits.
- **If validation fails partway through the pipeline, resume — don't restart.** Re-uploading the same file is fine (files dedupe on hash on some vendor implementations; a duplicate upload is at worst a small extra cost). Re-creating the job on the same file IDs is free.
- **Watch the loss curves.** If training loss is dropping but validation loss is flat / rising after epoch 1, you are overfitting on a tiny dataset — cancel and reduce epochs.
- **Log the exact `lm-eval` command lines** used for Part I in `eval.py` as a docstring or in `artifacts/report.md` — reproducibility.

## Acceptance criteria

You have completed the exercise when:

- `pins.md`, `base_choice.md`, and `cost_estimate.md` are populated with real numbers, not TBD.
- `artifacts/train.jsonl` and `artifacts/val.jsonl` pass the six-check validator with zero errors.
- The fine-tune job reached `status: "succeeded"`; `artifacts/job.yaml` and `artifacts/events.jsonl` exist.
- The loss curve plot exists and is interpretable.
- The fine-tuned model returns non-degenerate completions on the smoke test.
- `results/base/` and `results/openai-ft/` contain full `lm-eval` outputs for the same tasks at the same sampling.
- `artifacts/report.md` includes all eight sections, with a delta table showing FT vs. base on at least the task-specific held-out set.
- `artifacts/manifest.yaml` is complete per chapter 2's template.

## Stretch goals

- **Add DPO on top.** Take a small preference dataset (mod-105 chapter 2 shape), convert to OpenAI's DPO JSONL, and fine-tune-on-the-fine-tune. Compare against SFT-only in the report.
- **Add turn-level loss masking.** Pick a subset of your training data with obvious scaffolding turns (canned greetings, boilerplate). Add `"weight": 0` to those turns and rerun the fine-tune. Compare quality against the unmasked run — does masking measurably help?
- **Do the same fine-tune on `gpt-4.1-mini` and on `gpt-4o-mini` (or a smaller/larger tier pair).** Compare cost, wall-clock, and quality. Which tier is the right pick for your task at your expected volume?
- **Do the same fine-tune on OpenAI direct and on Azure Foundry OpenAI models.** Chapter 3 covers the differences. What was different in practice — API surface, ingest speed, cost, region behavior?
- **Write a mini `openai-direct-vs-hf` eval convergence check.** Take the same model (base, unfine-tuned), evaluate with `--model openai-chat-completions --model_args model=gpt-4.1-mini-2025-07-14` and (separately) download the equivalent open model (if a truly equivalent one exists) and evaluate with `--model hf`. Do the scores agree within CI? Diagnose any gap. Chapter 7 pitfall discussion applies.
