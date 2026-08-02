# Running EleutherAI `lm-evaluation-harness`

**Estimated read:** 45 minutes.
**You should be able to, by the end:** install `lm-evaluation-harness`, pick the right model backend for your artifact and cluster, run a fixed six-task suite (MMLU, ARC-Challenge, GSM8K, HumanEval, TruthfulQA, IFEval) against a fine-tuned model, distinguish `log_likelihood` tasks from `generate_until` tasks and know why the distinction matters for chat templates, and read the results JSON well enough to spot when a number is not what you think it is.

## Why this tool, and what it is

`lm-evaluation-harness` (EleutherAI) is the field-standard runner for academic capability benchmarks. It was originally built to reproduce the eval numbers in the GPT-2 / GPT-3 papers; it has since grown to ~200 tasks and is the tool almost every open-weights model card cites when reporting MMLU, ARC, GSM8K, HellaSwag, HumanEval, TruthfulQA, IFEval, and their friends. When a model card says "MMLU 63.4 (5-shot)," it almost always came from this harness.

Why it matters for a fine-tuning engineer:

1. **Comparability.** If your reported numbers came from the same harness at the same version as the base model's reported numbers, deltas mean something. If they came from a different tool, they don't.
2. **Coverage.** It bundles the wiring for every canonical benchmark. You do not write MMLU prompt templates from scratch.
3. **Backends.** It supports `hf`, `vllm`, `sglang`, and API backends (`openai-completions`, `openai-chat-completions`) — so the same task definitions run against a local safetensors checkpoint, a served vLLM instance, or a hosted model.
4. **Reproducibility.** Task definitions are versioned; the exact prompt, few-shot exemplars, and scoring code are pinned by task version.

The tool is not perfect: `main` moves fast, chat-template handling has changed several times, and getting a number that matches the model card exactly can take an afternoon of digging. This chapter is the map through those foot-guns.

## Install and pin

**Do not `pip install lm-eval` from PyPI and forget about it.** The PyPI release lags `main`, and the harness's `main` is a moving target. Pin a specific commit SHA in your `requirements.txt` and only bump when you have a reason.

```bash
pip install "lm-eval[vllm] @ git+https://github.com/EleutherAI/lm-evaluation-harness@<commit-sha>"
```

Optional extras that matter for common tasks:

- `[vllm]` — the vLLM backend for fast generation-heavy evals (GSM8K, HumanEval, IFEval).
- `[math]` — pulls in `sympy` for MATH-style grading.
- `[ifeval]` — pulls in the rule library IFEval scoring depends on.
- `[wandb]` — for logging results to Weights & Biases.

Verify the install:

```bash
lm_eval --tasks list | head
```

Task names are stable but their internal versions bump. Always record the harness commit SHA *and* the task versions the run used (both appear in the results JSON).

## The four things every invocation specifies

Every `lm_eval` invocation is at minimum:

```bash
lm_eval \
  --model <backend> \
  --model_args <backend-specific args> \
  --tasks <comma-separated task list> \
  --output_path results/<run-name>
```

Plus a handful of near-required flags in practice:

```bash
  --batch_size auto:4          # dynamic batch selection, 4 rounds of tuning
  --num_fewshot <N>            # per-task default is usually correct; override cautiously
  --seed <N>                   # sets torch/np/python seeds
  --write_out                  # dump the full per-item prompts + responses
  --log_samples                # per-item scores in the output JSON
  --apply_chat_template        # for chat/instruct models, use the tokenizer's chat template
  --fewshot_as_multiturn       # pair with --apply_chat_template on instruct models
```

The four choices you must make consciously:

### 1. Backend

| Backend | Best for | Notes |
|---|---|---|
| `hf` | log-likelihood tasks; tight integration with PEFT adapters | slow on generation; loads via `transformers` |
| `vllm` | generation tasks (GSM8K, HumanEval, IFEval) | requires vLLM install; 5–20× faster on generation |
| `sglang` | generation, especially structured / multi-turn | similar shape to vLLM |
| `openai-completions` | scoring a hosted completion model | requires API key; no log-likelihood on chat models |
| `openai-chat-completions` | scoring hosted chat models (OpenAI, and OpenAI-compatible) | generation only — log-likelihood tasks unavailable |
| `local-chat-completions` | scoring a locally-served OpenAI-compatible endpoint | for mod-109 / mod-110 comparisons |

Rule of thumb: **`hf` for log-likelihood tasks (MMLU, ARC, HellaSwag, TruthfulQA-mc), `vllm` for generation tasks (GSM8K, HumanEval, IFEval, TruthfulQA-gen).** Split the run in two if you need both — the setup cost is worth it because generation on `hf` is painfully slow.

### 2. `--model_args`

Backend-specific. For `hf`:

```
--model_args "pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16,attn_implementation=flash_attention_2,parallelize=True"
```

For a LoRA adapter on top of a base:

```
--model_args "pretrained=meta-llama/Llama-3.1-8B,peft=path/to/adapter,dtype=bfloat16"
```

For `vllm`:

```
--model_args "pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16,gpu_memory_utilization=0.9,tensor_parallel_size=1,max_model_len=8192"
```

Two gotchas:

- **`parallelize=True`** on `hf` spreads the model across visible GPUs with naive pipeline parallel. It works, it is slow. Prefer `vllm` with `tensor_parallel_size=N` for real multi-GPU eval.
- **`max_model_len`** on `vllm` must be at least prompt+generation for the longest task. IFEval and MT-Bench-style prompts routinely need 4K–8K.

### 3. `--tasks`

`lm-evaluation-harness` bundles tasks as *task groups* and *individual tasks*. Groups are convenience aggregates:

- `mmlu` — all 57 MMLU subjects (reported as a single average).
- `mmlu_stem`, `mmlu_humanities`, etc. — subject-family subsets.
- `gsm8k_cot` vs. `gsm8k` — chain-of-thought vs. bare-answer scoring.
- `truthfulqa_mc1`, `truthfulqa_mc2`, `truthfulqa_gen` — the three TruthfulQA modes.
- `arc_easy`, `arc_challenge`.
- `ifeval` — IFEval prompt-level and instruction-level accuracy.
- `humaneval` — pass@1 (needs `code_eval` env var to run tests).
- `bbh` — Big-Bench Hard aggregate.
- `agieval_en` — AGIEval English subset.

Always specify a fixed comma-separated list, not a wildcard. Wildcards mean you can add a task inadvertently and change the aggregate report.

### 4. `--num_fewshot`

Each task has a "canonical" few-shot count that matches the paper / leaderboard convention. Override only with a very good reason, because comparing to the base model's reported number requires matching the shot count.

Field conventions (defaults in the harness usually match these):

| Task | Convention |
|---|---|
| MMLU | 5-shot |
| ARC-Challenge | 25-shot |
| ARC-Easy | 25-shot |
| GSM8K | 5-shot (8-shot for chain-of-thought papers) |
| HumanEval | 0-shot |
| TruthfulQA (mc) | 0-shot |
| IFEval | 0-shot |
| HellaSwag | 10-shot |
| WinoGrande | 5-shot |
| BBH | 3-shot (CoT) |

## `log_likelihood` vs `generate_until` — the taxonomy that trips fine-tunes

Every task in the harness is one of two types, and the type determines how a fine-tuned model behaves.

### `log_likelihood` tasks

MMLU, ARC, HellaSwag, TruthfulQA-mc, WinoGrande. The harness builds one prompt per candidate answer, computes the log-probability the model assigns to each answer's tokens, and picks the highest.

Example (MMLU-style):

```
Question: The capital of France is
A. Paris
B. London
C. Berlin
D. Madrid
Answer: A
```

Four forward passes (one per candidate), the harness picks the one with the highest log-prob on `A`/`B`/`C`/`D`.

**Fine-tuning risk:** if your SFT reformatted answers into a different template ("**Answer:** \boxed{Paris}"), the model's log-prob on the raw `A`/`B`/`C`/`D` continuation can crater — not because the model got dumber, but because it now prefers a different output format. The number drops and it looks like a regression.

Two mitigations, both important:

- Run with `--apply_chat_template` for chat/instruct models. This wraps each prompt in the tokenizer's chat template so the log-prob is computed on the assistant-turn continuation.
- Compare against the *base model's* number *also* re-run with the same flags. Do not compare your `--apply_chat_template` fine-tune against a base number computed without it.

### `generate_until` tasks

GSM8K, HumanEval, IFEval, TruthfulQA-gen, MATH. The harness prompts the model, samples a continuation (until stop tokens), and scores the continuation against a rule (exact match, unit-test pass, IFEval verifier, etc.).

Example (GSM8K):

```
Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many
clips in May. How many clips did Natalia sell altogether in April and May?
A:
```

The model generates until `\n\nQ:` or another stop token. The harness extracts the last `#### <number>` and compares to gold.

**Fine-tuning risks:**

- **Wrong stop tokens.** If your chat template introduces an `<|im_end|>` token and the task-defined stop is `\n\nQ:`, the model may run off the end of `max_gen_toks` or produce garbage after the answer. Check the harness task's `until` list against your model's stop tokens.
- **Wrong answer format.** GSM8K scoring extracts `#### <number>`. If your fine-tune taught the model to end with `**Answer: <number>**`, GSM8K scores it wrong. This is a scoring artifact, not a math regression.
- **Sampling parameters.** Default is greedy (`temperature=0`). If you override to `temperature=0.7`, `top_p=0.95`, the numbers become non-deterministic and lower. For a release gate, keep greedy for score reproducibility even if production uses sampling.

## The six-task baseline suite

The suite this module treats as canonical for release-gating a chat/instruct fine-tune:

| Task | Type | Shots | Backend | Wall-clock (7B, 1×H100) |
|---|---|---|---|---|
| `mmlu` | log-lik | 5 | `hf` | 30–90 min |
| `arc_challenge` | log-lik | 25 | `hf` | 5–15 min |
| `gsm8k` | generate | 5 | `vllm` | 20–60 min |
| `humaneval` | generate | 0 | `vllm` | 15–45 min |
| `truthfulqa_mc2` | log-lik | 0 | `hf` | 10–30 min |
| `ifeval` | generate | 0 | `vllm` | 15–40 min |

Two invocations, because you want the right backend per task:

```bash
# Log-likelihood suite (hf backend, no chat template for base models;
# add --apply_chat_template --fewshot_as_multiturn for instruct models).
lm_eval \
  --model hf \
  --model_args "pretrained=$MODEL,dtype=bfloat16,attn_implementation=flash_attention_2" \
  --tasks mmlu,arc_challenge,truthfulqa_mc2 \
  --batch_size auto:4 \
  --seed 1234 \
  --output_path results/$RUN/loglik \
  --log_samples --write_out

# Generation suite (vllm backend, greedy).
lm_eval \
  --model vllm \
  --model_args "pretrained=$MODEL,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=8192" \
  --tasks gsm8k,humaneval,ifeval \
  --batch_size auto \
  --seed 1234 \
  --output_path results/$RUN/gen \
  --log_samples --write_out
```

Notes on the flags:

- `--log_samples` writes the per-item score to the output JSON; without it you only get aggregates and cannot compute the per-item deltas chapter 6 needs.
- `--write_out` writes the actual prompt and model response for a subset of items to disk. Enable it on at least the first run against a new model so you can visually verify the prompt is what you think it is.
- `--seed 1234` — chapter 6 covers why the seed matters even for log-likelihood tasks (it affects few-shot exemplar selection when the task samples them).

## Reading the results JSON

The harness writes one JSON per run. The structure that matters:

```json
{
  "results": {
    "mmlu": {"acc,none": 0.6243, "acc_stderr,none": 0.0038, ...},
    "mmlu_stem": {"acc,none": 0.5411, ...},
    ...
    "gsm8k": {"exact_match,strict-match": 0.6588, "exact_match,flexible-extract": 0.6906, ...}
  },
  "versions": {"mmlu": 2, "gsm8k": 3, ...},
  "config": {
    "model": "hf",
    "model_args": "pretrained=...",
    "batch_size": "auto:4",
    "num_fewshot": null,
    "seed": [1234, 1234, 1234, 1234]
  },
  "git_hash": "abc123",
  "samples": {
    "mmlu_college_biology": [
      {"doc_id": 0, "prompt": "...", "target": "A", "resps": [...], "filtered_resps": [...], "acc": 1.0},
      ...
    ]
  }
}
```

Three fields to check on every run:

1. **`versions`.** The per-task version. If it changed between your base-model run and your fine-tune run, the numbers are not comparable. This happens quietly when you bump the harness commit.
2. **`git_hash`.** The harness commit. Should match your pinned SHA.
3. **`config.model_args`.** Should match your intended model artifact. Careless bash substitution has shipped comparisons of `model-v3` against `model-v3` more than once.

And the two output fields to interpret carefully:

- **`acc,none` vs `acc_norm,none`**. Some tasks report both raw and length-normalized accuracy. HellaSwag reports both; ARC does. The convention on ARC is `acc_norm`; MMLU has only `acc`. Follow the leaderboard convention or you will be comparing apples to oranges.
- **`exact_match,strict-match` vs `exact_match,flexible-extract`** on GSM8K. Strict-match requires the `#### <answer>` format. Flexible-extract pulls the last number in the response. Report both; use strict for the gate.

## Per-task pitfalls

### MMLU

- Report the *aggregate* `mmlu` acc, not just `mmlu_stem`. Leaderboards report the aggregate.
- Chat-template drift is the most common regression cause. If your fine-tune drops 5+ points on MMLU, first re-run with and without `--apply_chat_template` and see which one matches the base.
- The published Llama 3 numbers use `--apply_chat_template --fewshot_as_multiturn` on instruct variants and the base template on base variants. Match this on your comparison.

### ARC-Challenge

- 25-shot is the convention. Some older reports use 0-shot; note the difference in your report.
- Uses `acc_norm` (length-normalized). Reporting `acc` gives a different (usually lower) number.

### GSM8K

- Sample count matters. Default is 1 sample per problem, greedy. Some papers report self-consistency majority vote over `K` samples (e.g., `K=32`); that is a strictly different eval and gives higher numbers. Do not compare across sample budgets.
- Strict-match is stricter than the numbers you often see; flexible-extract is closer to what most model cards report but is slightly lenient.
- Watch for `Q:`/`A:` bleed. Some fine-tuned chat models produce `\nQ: ...` and continue answering the next question. The `until` stop should catch this, but chat-template contamination sometimes prevents it.

### HumanEval

- `pass@1` at temperature 0 with 1 sample is the low-cost default. `pass@1` at `T=0.2` with 20 samples per problem is the "real" pass@1 (mean pass rate across samples); the two differ.
- Requires `HF_ALLOW_CODE_EVAL=1` and a way to actually execute Python — the harness pops a subprocess. Do not run untrusted models under this without sandboxing (the model can and will emit `os.system` calls).
- Uses the raw problem docstring as the prompt. Instruct models sometimes prepend "Sure, here is the function:" which breaks the code-only extractor. Chapter 7 covers the fix.

### TruthfulQA

- Three modes: `mc1` (single correct), `mc2` (multi-correct with weighted scoring), `gen` (open-ended, judged by a fine-tuned GPT model). `mc2` is the field convention. `gen` is much more expensive and requires a judge model.
- The "GPT-Judge" model referenced in the original paper (`curie-tuned` on truthfulness) has been deprecated; several open re-implementations exist in the harness but do not fully match the paper's judge. Note this in your report.

### IFEval

- Rule-based grading — no judge model. Fast and reproducible.
- Reports four numbers: prompt-level strict, prompt-level loose, instruction-level strict, instruction-level loose. Report all four; leaderboards typically headline prompt-level strict.
- Very sensitive to `max_gen_toks`. Some instructions require ~1000 tokens of output; a low cap silently zeros the score.

## Aggregate scores and open leaderboards

Two commonly-cited aggregates:

- **Open LLM Leaderboard v1** (Hugging Face, deprecated) averaged ARC, HellaSwag, MMLU, TruthfulQA-mc2, WinoGrande, GSM8K.
- **Open LLM Leaderboard v2** replaced those with IFEval, BBH, MATH-Lvl-5, GPQA, MuSR, MMLU-Pro on grounds that the v1 tasks had saturated. This is closer to what current publications report.

Neither aggregate is a magic number. A "leaderboard score" is a convenience for cross-model ranking; for a release gate, report the per-task numbers with confidence intervals and let the reader (or the automated gate) aggregate as they see fit.

<!-- needs-research: the Open LLM Leaderboard v2 task list has evolved; verify the current set at HF before citing in a training report. -->

## Chat template and few-shot: the interaction that eats afternoons

For an instruct-tuned model, the harness offers two flags whose interaction is not obvious:

- `--apply_chat_template` — wrap each prompt in the tokenizer's chat template.
- `--fewshot_as_multiturn` — when the task has few-shot exemplars, format each exemplar as its own user/assistant turn instead of jamming them all into one user turn.

The four combinations:

| `--apply_chat_template` | `--fewshot_as_multiturn` | What happens |
|---|---|---|
| off | off | Raw prompt with in-context exemplars in the user turn. Right for base models. |
| on | off | Chat-templated user turn with all exemplars stuffed into it. Sometimes matches published instruct numbers, sometimes doesn't. |
| on | on | Each exemplar is its own user/assistant pair, then the real prompt. Closest to how the model was trained to consume examples. |
| off | on | Ignored — `--fewshot_as_multiturn` requires `--apply_chat_template`. |

If your instruct-model MMLU number is 5–10 points below the base's reported number, this interaction is almost certainly the reason. Match the flag combination the base model's card documents; if it doesn't, run both `on/on` and `off/off` and report both.

## Sanity checks before you trust a number

Every new fine-tuned model, before you commit its number to any report:

1. **Run against the base model with identical flags.** Numbers should be close to the published base numbers. If not, your flags are wrong before your fine-tune is a factor.
2. **Spot-check `--write_out` samples.** Open the dumped prompts and verify they look like the task expects. A wrong `apply_chat_template` interaction is very visible here.
3. **Check `versions`.** Same task versions as the previous run.
4. **Check `config`.** Model, args, seed, batch size all as intended.
5. **Compute delta vs. previous.** If a task moved ±10 points and you did not expect it, find out why before writing anything up. The most common cause is not a model regression — it is a flag change, a template change, or a stop-token change.

Chapter 6 turns those five checks into a script that the release gate runs automatically.

## Exercise pointer

`exercises/exercise-01-lm-eval-harness-baseline-vs-tuned.md` walks the concrete first run: install the harness at a pinned SHA, run the six-task suite against a base model and its fine-tune, compute per-task deltas with 95% CIs, and hand back a table you would put in a release notes.

## Summary

- `lm-evaluation-harness` is the field-standard runner for academic capability benchmarks. Pin a commit SHA; do not float on `main`.
- Every invocation specifies four things: backend (`hf` for log-likelihood, `vllm` for generation), `--model_args` (with PEFT adapter if separate), `--tasks` (fixed comma-separated list), and `--num_fewshot` (matching leaderboard convention).
- Tasks are one of two types: `log_likelihood` (MMLU, ARC, TruthfulQA-mc) — chat-template drift is the risk; `generate_until` (GSM8K, HumanEval, IFEval, TruthfulQA-gen) — stop tokens and answer format are the risks.
- The canonical six-task suite: MMLU (5-shot), ARC-Challenge (25-shot), GSM8K (5-shot), HumanEval (0-shot), TruthfulQA-mc2 (0-shot), IFEval (0-shot). Split across two invocations for `hf` and `vllm` backends.
- Read the results JSON with three fields in mind: `versions` (task versions must match across runs), `git_hash` (harness commit), and `config` (model and args must be what you intended). Enable `--log_samples --write_out` so you can compute per-item deltas later.
- `--apply_chat_template` and `--fewshot_as_multiturn` interact non-obviously; mis-set flags are the single most common cause of "the fine-tune regressed" false alarms on instruct models. Match the base model's convention or report both settings.
