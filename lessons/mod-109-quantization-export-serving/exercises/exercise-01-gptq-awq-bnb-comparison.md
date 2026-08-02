# exercise-01: GPTQ vs. AWQ vs. bitsandbytes 4-bit Comparison

**Estimated effort:** 3 hours
**Companion chapters:** [`01-quantization-landscape.md`](../01-quantization-landscape.md), [`02-gptq-awq-bitsandbytes.md`](../02-gptq-awq-bitsandbytes.md)

## Objective

Take one fine-tuned checkpoint through three 4-bit post-training quantization pipelines — **GPTQ**, **AWQ**, and **bitsandbytes NF4** — using calibration data drawn from your training distribution, run the mod-108 fast-lane suite against each variant plus the BF16 baseline, and produce a defended comparison matrix over quality, VRAM, latency, and disk. The write-up recommends one quantization variant for a specified serving scenario, backed by your own numbers.

The point of this exercise is measurement discipline against a real trade-off: three methods that all claim to give you a 4× VRAM saving with "small" quality delta, and you need to know which is actually true for your model on your evaluation.

## Prerequisites

- A fine-tuned checkpoint you can merge into a plain BF16 model. Options:
  - Your own merged model from mod-104 (`peft.merge_and_unload()` applied to the SFT LoRA).
  - Your own merged model from mod-105 (DPO / KTO / ORPO on top of SFT).
  - A public fine-tune with a permissive license (e.g., `meta-llama/Llama-3.1-8B-Instruct`).
- At least 1 GPU with enough VRAM to hold the BF16 model (~14 GB for a 7B, ~26 GB for a 13B) plus a bit of overhead. One A100 40 GB, one H100 80 GB, or one L40S is comfortable for a 7B.
- Python 3.10+, PyTorch 2.4+, and disk space for four artifact directories (BF16, GPTQ-4bit, AWQ-4bit, bnb-NF4 — roughly 14 + 4 + 4 + 4 = 26 GB for a 7B).
- The `lm-evaluation-harness` install from mod-108 exercise 01 (or a fresh install pinned to the same commit).
- Access to a small representative sample (200–500 rows) of your training data or a domain-matched public dataset for calibration.

## Requirements

### Part A — Pin the environment

Author `requirements.txt` and `pins.md` with commit SHAs / versions for at minimum:

- `transformers`
- `torch`
- `accelerate`
- `bitsandbytes`
- `auto-gptq` (or `gptqmodel`, whichever your ecosystem is on)
- `autoawq`
- `datasets`
- `lm-eval` (with the commit SHA you used in mod-108 exercise 01, or a fresh pinned commit)

Verify each install by importing and printing the version. Include a machine spec block in `pins.md` (GPU model, VRAM, CUDA version, driver version).

### Part B — Prepare the source artifact

Produce a single merged BF16 checkpoint that will feed all three quantization pipelines. If your fine-tune is a LoRA, do the merge here (do not merge inside each pipeline separately — you want one source of truth):

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.bfloat16, revision=BASE_REV)
peft_model = PeftModel.from_pretrained(base, ADAPTER_PATH)
merged = peft_model.merge_and_unload()

merged.save_pretrained("artifacts/bf16")
AutoTokenizer.from_pretrained(BASE_ID).save_pretrained("artifacts/bf16")
```

Verify the merge with a single-prompt greedy generation and compare against the attached-adapter version — they should be effectively identical. If not, halt and diagnose (chapter 3 covers the merged-vs-attached comparison procedure).

Record in `source_metadata.md`: base id, base revision hash, adapter path (if applicable), merge date, disk size of `artifacts/bf16`, and the greedy-generation smoke-test output.

### Part C — Prepare calibration data

Draw 256–512 rows from your training distribution (or a task-matched public dataset — e.g., a C4 slice for a general instruct model, a code slice for a code model, a chat slice for a chat model). Format as raw text (or messages passed through the tokenizer's chat template — match what the model actually sees in production).

Save to `calibration/calibration.jsonl` with one row per line. Include in `calibration/README.md` the source, filter criteria, row count, and mean token length. This is the input to the GPTQ and AWQ quantization runs.

**Warning:** do not include eval-set items in the calibration data. Chapter 5 of mod-108 covers the contamination story; if you leak an MMLU item into calibration, your quantized model's MMLU score is not comparable. Cross-check by hashing the calibration items against your eval sets.

### Part D — Quantize with GPTQ

Author `quantize_gptq.py` that:

1. Loads `artifacts/bf16` in BF16.
2. Loads the calibration data and tokenizes each row to `max_length=2048` (or your model's typical context).
3. Runs the GPTQ pipeline with `bits=4, group_size=128, desc_act=False, damp_percent=0.01`.
4. Saves to `artifacts/gptq-4bit` including the tokenizer.
5. Logs wall-clock and peak GPU memory.

Verify the artifact loads with `AutoGPTQForCausalLM.from_quantized("artifacts/gptq-4bit")` (or `AutoModelForCausalLM.from_pretrained` — recent `transformers` handles the auto-detect). Run the three smoke tests from chapter 2 (load-and-generate, perplexity, 100-item MMLU). Record numbers in `metrics/gptq.md`.

### Part E — Quantize with AWQ

Author `quantize_awq.py` that:

1. Loads `artifacts/bf16` via `AutoAWQForCausalLM.from_pretrained`.
2. Runs `model.quantize(tokenizer, quant_config={"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"})` with the same calibration data.
3. Saves to `artifacts/awq-4bit`.
4. Logs wall-clock and peak GPU memory.

Verify load and run the same three smoke tests. Record in `metrics/awq.md`.

### Part F — Load with bitsandbytes NF4

Author `load_bnb.py` that loads `artifacts/bf16` with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)` and saves the resulting model to `artifacts/bnb-nf4/` via `model.save_pretrained`. Verify reload with the same `BitsAndBytesConfig`. Run the three smoke tests. Record in `metrics/bnb.md`.

Note: `bitsandbytes` does not use calibration data. That is one of the trade-offs your write-up will discuss.

### Part G — Run the fast-lane suite against all four artifacts

Using the pinned `lm-evaluation-harness` from Part A, run the following six tasks against each of `artifacts/bf16`, `artifacts/gptq-4bit`, `artifacts/awq-4bit`, and `artifacts/bnb-nf4`:

- `mmlu` (5-shot)
- `arc_challenge` (25-shot)
- `truthfulqa_mc2`
- `gsm8k` (5-shot, generation, `hf` backend or `vllm` if you have it stood up)
- `ifeval` (generation)
- Your task-specific held-out eval from mod-108 exercise 02, if you have one

Use `--seed <fixed>`, `--batch_size auto:4`, `--log_samples`, `--output_path results/<variant>/`. For instruct models, `--apply_chat_template --fewshot_as_multiturn` on the log-likelihood tasks (see mod-108 exercise 01 for the flag discipline).

Save all four result trees. This is the input to Part H's matrix.

### Part H — Compute the comparison matrix

Author `build_matrix.py` that reads the four result trees and produces `comparison_matrix.md`:

| Task | BF16 | GPTQ Δ | AWQ Δ | bnb NF4 Δ |
|---|---|---|---|---|
| MMLU | X.XX | ±X.XX | ±X.XX | ±X.XX |
| ARC-Challenge | ... | ... | ... | ... |
| ... | | | | |

Where BF16 is the point score, and each Δ column is the quantized score minus the BF16 score (negative = regression). Include 95% bootstrap CIs on each cell for the BF16 column and the Δs.

Also produce `system_matrix.md`:

| Metric | BF16 | GPTQ | AWQ | bnb NF4 |
|---|---|---|---|---|
| Disk size (GB) | | | | |
| Load-time peak VRAM (GB) | | | | |
| Inference-time steady VRAM (GB, single-request greedy generation) | | | | |
| Wall-clock / token (ms, single-request greedy) | | | | |
| Quantization wall-clock (min) | N/A | | | ~0 |
| Quantization peak VRAM (GB) | N/A | | | ~ = load |

### Part I — The write-up

Author `report.md` with the following sections:

1. **Setup.** Model, base revision, adapter (if any), calibration data source and size, hardware, pinned versions.
2. **The comparison matrix.** From Part H.
3. **Quality analysis.** Which tasks each method regressed on and by how much. Is the regression uniform (chapter 1's "shape 1"), task-selective (shape 2), or catastrophic (shape 3)? Any task-selective regression should be diagnosed — is it a calibration mismatch, a numerical-precision-sensitive task, or something else?
4. **System analysis.** Disk / VRAM / latency deltas. Which method is smallest, which is fastest to quantize, which uses the least serve-time VRAM.
5. **Recommendation.** For **this specified serving scenario**, which variant do you ship?
   > "Single-tenant chatbot deployment on one L40S (48 GB VRAM). p95 TTFT < 400 ms is the SLO. Traffic mix is 60% short queries (mean 200 tokens in / 200 out), 30% medium (500 in / 500 out), 10% long-context RAG (3000 in / 500 out). Release-gate tolerance: ≤ 2 pp regression on academic tasks, ≤ 3 pp on the task-specific held-out set."

   Argue with reference to your numbers. There is not one right answer; the point is that the answer is defended.
6. **Reproducibility notes.** What would need to be different to rerun the exercise a month from now (tooling updated, base revision moved, calibration data refreshed). Which of the eight reproducibility dimensions from mod-108 chapter 1 you have pinned and which you have not.
7. **What surprised you.** One paragraph. This is where the real learning surfaces.

### Part J — Directory layout

Final tree:

```
outputs/exercise-01/
├── requirements.txt
├── pins.md
├── artifacts/
│   ├── bf16/
│   ├── gptq-4bit/
│   ├── awq-4bit/
│   └── bnb-nf4/
├── calibration/
│   ├── calibration.jsonl
│   └── README.md
├── source_metadata.md
├── quantize_gptq.py
├── quantize_awq.py
├── load_bnb.py
├── metrics/
│   ├── gptq.md
│   ├── awq.md
│   └── bnb.md
├── results/
│   ├── bf16/
│   ├── gptq-4bit/
│   ├── awq-4bit/
│   └── bnb-nf4/
├── build_matrix.py
├── comparison_matrix.md
├── system_matrix.md
└── report.md
```

## Starter guidance

- **Merge once.** The single most common mistake in this exercise is merging the LoRA separately in each pipeline. Merge in Part B, then every pipeline consumes the same BF16 checkpoint. If you re-merge, the pipelines' inputs differ subtly (numeric order of operations) and the comparison is contaminated.
- **Calibrate on task-relevant data.** A generic C4 calibration will make an instruct model quantize acceptably, but a task-specific calibration will make it quantize *better* on the tasks that matter. If you have training data, use it. Chapter 2 has the guidance.
- **Do the 100-item MMLU smoke test before the full suite.** Full-suite MMLU takes 30–90 minutes; the 100-item version takes 3–5. If it drops 15+ pp, do not run the full suite — go back and fix the pipeline. Saves hours.
- **Watch for `desc_act` incompatibility.** GPTQ with `desc_act=True` produces slightly better quality but historically some inference kernels (older ExLlama, some Marlin paths) do not consume it. This exercise uses `desc_act=False` for safety; note it in your report.
- **Do not enable both `lm-eval`'s `--apply_chat_template` and none-of-the-above.** All four variants must be evaluated with the same template settings. If BF16 was evaluated with the template applied, GPTQ must be too. A template mismatch will look like a quantization regression.
- **`bitsandbytes` quantization is on-the-fly at load.** There is no separate quantization step; `save_pretrained` after a 4-bit load persists the config, and reload reproduces the quantization. If your `metrics/bnb.md` says "quantization wall-clock: 30 min," you are timing the model *load*, not the quantization — clarify.
- **VRAM measurement.** Use `torch.cuda.reset_peak_memory_stats()` before each measured operation and `torch.cuda.max_memory_allocated()` after. Do it in a fresh process to avoid pollution from earlier operations.

## Acceptance criteria

You have completed the exercise when:

- `pins.md` records exact versions for every tool used.
- `artifacts/bf16`, `artifacts/gptq-4bit`, `artifacts/awq-4bit`, `artifacts/bnb-nf4` all exist and each loads cleanly with the appropriate loader.
- `calibration/calibration.jsonl` has 256+ rows, source and filter criteria documented, and no overlap with eval sets.
- Three smoke tests (load-and-generate, perplexity, 100-item benchmark) run cleanly against each of the three quantized variants.
- `results/` has full `lm-eval` output for all four variants across at least five of the six specified tasks (the task-specific held-out is optional if you do not have one).
- `comparison_matrix.md` shows quality deltas with CIs; `system_matrix.md` shows disk, VRAM, latency, quantization cost.
- `report.md` includes all seven sections, with the recommendation defended by concrete numbers.

## Stretch goals

- **Add a fifth variant: SmoothQuant W8A8.** Follow chapter 2's SmoothQuant path (via `optimum` / `compressed-tensors` / a similar tool). Compare on the same suite. This is the "throughput-first" reference against the "memory-first" 4-bit variants.
- **Add a sixth variant: FP8.** If you are on an H100, quantize to FP8 via Nvidia's `Model Optimizer` / `ammo` or vLLM's built-in FP8 quantization. Same suite. Expect near-BF16 quality with much higher throughput.
- **Vary the calibration set.** Rerun GPTQ and AWQ with three different calibration sources: C4 (generic), your training data (matched), and adversarial (data very different from training — e.g., code prompts for a chat model). Report how the eval numbers change per calibration choice. This is the empirical version of chapter 2's calibration guidance.
- **Ablate `group_size`.** Rerun GPTQ with `group_size=32`, `64`, `128`. Report file size and quality trade-off.
- **Deploy each variant behind vLLM.** Chapter 4 covers the flags. Measure single-request latency and small-batch throughput on each. This bridges into exercise 02.
- **Include contamination scan.** Run mod-108 exercise 04's contamination probes against the calibration set; disclose in the report.
- **Cost projection.** Extend the recommendation with a rough dollars-per-million-tokens number under the specified traffic mix (chapter 7's model). Sanity-check against a managed-API alternative for the same tier of model.
