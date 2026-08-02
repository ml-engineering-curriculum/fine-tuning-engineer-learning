# exercise-01: `lm-evaluation-harness` Baseline vs. Tuned

**Estimated effort:** 3 hours
**Companion chapters:** [`01-evaluation-landscape.md`](../01-evaluation-landscape.md), [`02-lm-evaluation-harness.md`](../02-lm-evaluation-harness.md), [`07-eval-pitfalls-and-handoff.md`](../07-eval-pitfalls-and-handoff.md)

## Objective

Install `lm-evaluation-harness` at a pinned commit, run the canonical six-task suite (MMLU, ARC-Challenge, GSM8K, HumanEval, TruthfulQA-mc2, IFEval) against **both** a base model and a fine-tuned checkpoint (yours from mod-103 / mod-104 / mod-105, or a public fine-tune), and produce a per-task delta table with 95% bootstrap confidence intervals and paired p-values.

The point is measurement discipline: getting numbers you would put in a release notes and defend under scrutiny — not maximizing any single score.

## Prerequisites

- Access to at least 1 GPU (A100 40 GB, A100 80 GB, H100, or comparable). A 7B model in BF16 needs ~14 GB of weights alone; the log-likelihood suite comfortably runs on any card that fits the model.
- A base model + a fine-tuned checkpoint pair. Options:
  - Your own from an earlier module (e.g., `meta-llama/Llama-3.1-8B` and your SFT LoRA from mod-104 exercise 01).
  - A public pair (e.g., `meta-llama/Llama-3.1-8B` and `meta-llama/Llama-3.1-8B-Instruct`).
- Python 3.10+, PyTorch 2.4+.
- Enough disk space for one model's shards plus a `results/` directory (~50 GB is safe).
- API access is not required — every task here runs offline.

## Requirements

### Part A — Install and pin

Install `lm-evaluation-harness` at a specific commit SHA and vLLM at a pinned version. Write a `requirements.txt` that pins:

- `lm-eval` at a specific commit
- `transformers` at a specific version
- `torch` at a specific version
- `vllm` at a specific version
- `flash-attn` if you use it
- The tokenizer's dependency (`sentencepiece` / `tiktoken` as required)

Do not float `main`. Record the commit SHA and versions in a `pins.md` for the write-up. Verify the install:

```bash
lm_eval --tasks list | grep -E "^(mmlu|arc_challenge|gsm8k|humaneval|truthfulqa_mc2|ifeval)"
```

All six task names should appear.

### Part B — Compute the base-model baseline

Run the six-task suite against the base model. Use two invocations because the right backend differs per task:

- **Log-likelihood tasks** (MMLU, ARC-Challenge, TruthfulQA-mc2): `hf` backend.
- **Generation tasks** (GSM8K, HumanEval, IFEval): `vllm` backend.

Required flags on every invocation:

- `--seed <fixed>` (pick one and hold it constant across all runs in this exercise).
- `--batch_size auto:4` (or your own choice — document it).
- `--log_samples` (per-item scores; you need these for Part D).
- `--write_out` (dumps first few items' prompts/responses to disk for sanity-check).
- `--output_path results/base/<lane>/`.

For **instruct** base models (e.g., `Llama-3.1-8B-Instruct`), also pass `--apply_chat_template --fewshot_as_multiturn`. For **base** models (no instruct tuning), do not.

Record in a `base_metrics.md`:

- Model id, dtype, backend per task.
- Per-task aggregate score (with `stderr` from the harness output).
- Wall-clock per task.
- Peak GPU memory (`torch.cuda.max_memory_allocated`).
- Exact `--num_fewshot` used per task.
- The `versions` block from the results JSON (per-task version numbers).
- The `git_hash` from the results JSON.

### Part C — Run against the fine-tuned checkpoint

Repeat Part B against your fine-tuned model, using **the same seed, same batch size, same flags, same backends**. Two rules:

1. For a LoRA / QLoRA adapter, pass `--model_args "pretrained=<base>,peft=<adapter-dir>,dtype=bfloat16"`. Verify from `--write_out` that the adapter actually loaded (loss of the base model's identity in the generations is a quick tell; more rigorously, print `type(model).base_model` after load).
2. For a merged / full-parameter fine-tune, pass the merged directory as `pretrained`. Confirm the tokenizer used matches the fine-tune's tokenizer, not the base's.

Save to `results/tuned/<lane>/`. Record the same metadata as in Part B.

### Part D — Compute deltas with confidence intervals

Write a small Python script (call it `compute_deltas.py`) that reads the per-item scores from both `results/base/` and `results/tuned/`, and for each task:

1. Aggregates the per-item score (accuracy, exact-match, pass@1 as appropriate).
2. Computes a **95% bootstrap CI** on the aggregate (1,000+ resamples of items with replacement).
3. Computes a **paired test** between base and tuned on per-item scores:
   - McNemar's exact test for binary metrics (`scipy.stats.contingency.mcnemar` or a direct exact-test computation).
   - Wilcoxon signed-rank for continuous metrics (`scipy.stats.wilcoxon`).
4. Writes a Markdown table with columns: `task | base [CI] | tuned [CI] | Δ | paired p | verdict`.

`verdict` is one of `improved (p<0.05)`, `regressed (p<0.05)`, `no significant change`, or `insufficient signal` (task's stderr band too wide to make either call).

### Part E — Sanity-check chat-template handling

For at least one log-likelihood task (MMLU or ARC-Challenge), run *both*:

- With `--apply_chat_template --fewshot_as_multiturn`.
- Without either flag.

Report both numbers for both models (base and fine-tuned). Discuss in the writeup whether the fine-tuned model's regression (if any) on that task is a real capability change or a template artifact.

Additionally: open five items from `write_out/` for the fine-tuned model and verify the prompts look correct — special tokens present where expected, no obviously-wrong exemplar formatting, no stray tokens leaking into the assistant turn.

### Part F — Write the report

`report.md`, structured as:

1. **Setup.** Model ids, harness commit SHA, backend / version pins, seed, hardware.
2. **Per-task delta table.** From Part D. One row per task. Include CIs and p-values.
3. **Chat-template comparison.** From Part E. The four numbers for the log-likelihood task run both ways, with interpretation.
4. **Reproducibility notes.** What would break if someone reran this a month from now (harness moved, backend upgraded, tokenizer changed, etc.). Which of the eight reproducibility dimensions from ch. 1 you have pinned and which you have not.
5. **Findings.** One paragraph on what the fine-tune actually did — which capabilities improved, which regressed, which stayed the same. This is what would go in the release notes.

### Part G — Save

Directory layout:

```
outputs/exercise-01/
├── requirements.txt
├── pins.md
├── base_metrics.md
├── tuned_metrics.md
├── results/
│   ├── base/loglik/results_*.json
│   ├── base/gen/results_*.json
│   ├── tuned/loglik/results_*.json
│   └── tuned/gen/results_*.json
├── write_out/                     # spot-check samples
├── compute_deltas.py
├── delta_table.md                 # output of compute_deltas.py
└── report.md
```

## Starter guidance

- **Do the log-likelihood suite first.** It runs in <2 hours on a single H100 for a 7B and it does not need `vllm`. Once you have it working, the generation suite is straightforward.
- **Use `--limit N` for the first pass.** `--limit 50` on every task gives you a smoke test that finishes in ~10 minutes and confirms the pipeline works before you commit to the full run. Remove `--limit` for the real numbers.
- **Log the `versions` and `git_hash` fields from the results JSON in your report.** They are how a future reader knows whether your numbers are still comparable to theirs.
- **If your base is an instruct model, `--apply_chat_template --fewshot_as_multiturn` is almost always right.** If you get numbers 5–10 pp below the model card's reported numbers, flip these flags and see if that closes the gap.
- **HumanEval needs code execution.** Set `HF_ALLOW_CODE_EVAL=1` (or the harness's current env var — check for your version). Run inside a container or sandboxed environment; the model can and will emit `os.system` calls.
- **Do not compare against the model card without matching the flags.** Model cards report a specific setup; matching it takes ten minutes of reading and saves you an afternoon of "why don't my numbers match."

## Acceptance criteria

You have completed the exercise when:

- `pins.md` records exact commit SHAs and versions for harness, transformers, torch, vllm, and any other install-time dependency.
- `base_metrics.md` and `tuned_metrics.md` include aggregate score, stderr, wall-clock, peak memory, num_fewshot, and the JSON `versions` block for all six tasks.
- `delta_table.md` shows per-task deltas with 95% bootstrap CIs and paired-test p-values for all six tasks.
- `report.md` includes the setup section, delta table, chat-template comparison, reproducibility notes, and a findings paragraph.
- Rerunning the same commands with the same seed on the same hardware produces byte-identical results JSONs (log-likelihood tasks) or the same aggregate score to 3 decimal places (generation tasks with greedy decoding).

## Stretch goals

- **Repro a model card.** Pick a base model whose card reports MMLU, ARC-Challenge, and GSM8K. Match your numbers to within 1 pp. If you cannot, hunt down which flag or version is different — this is the debugging drill that pays off later.
- **Add HellaSwag and WinoGrande.** Both are log-likelihood tasks. Same recipe, adds ~30 minutes.
- **Instrument with `wandb`.** `--wandb_args project=<yours>` logs the run's results and lineage. Useful when this suite becomes a per-commit gate (chapter 6).
- **Test with `--limit` on a synthetic contamination probe.** Add a fake "eval" of 20 items that appear verbatim in your training set. Confirm that the fine-tuned model scores much higher on those than on unseen items — a sanity check that your eval infrastructure would in fact reveal contamination if it existed.
- **Try FSDP-loaded eval.** If your fine-tune saved as sharded FSDP checkpoints (mod-107), verify that `lm_eval --model hf --model_args "pretrained=<sharded-dir>"` loads correctly, or consolidate first with `torch.distributed.checkpoint`'s offline consolidator.
- **Time the two backends against each other.** Run GSM8K on both `hf` and `vllm` and record the wall-clock. This is the receipt for chapter 2's "use vllm for generation."
