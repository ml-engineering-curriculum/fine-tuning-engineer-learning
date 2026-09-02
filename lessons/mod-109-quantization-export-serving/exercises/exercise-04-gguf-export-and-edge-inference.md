# exercise-04: GGUF Export and Edge Inference

**Estimated effort:** 2 hours
**Companion chapters:** [`06-gguf-and-edge-inference.md`](../06-gguf-and-edge-inference.md), [`01-quantization-landscape.md`](../01-quantization-landscape.md), [`07-serving-runbook-and-cost.md`](../07-serving-runbook-and-cost.md)

## Objective

Convert the merged BF16 fine-tuned checkpoint from exercise 01 to **GGUF** at **two quantization levels** (a workhorse and an alternative), prove equivalence against the source weights with three graded checks (load-and-generate parity, perplexity delta, mod-108 fast-lane release-gate subset), benchmark tokens/sec on the CPU or edge hardware you have access to, and produce an **edge deployment brief** that a mobile or on-prem product team could execute cold.

The point of this exercise is that the GGUF path is a different serving world: not vLLM, not TGI, no datacenter GPU. The conversion is easy; making the conversion actually *equivalent enough to ship* is where the discipline lives. You should finish with a concrete recommendation on which k-quant to ship for your model and target hardware, backed by your own numbers.

## Prerequisites

- The merged BF16 checkpoint from exercise 01 (`artifacts/bf16`). If you skipped exercise 01, merge your LoRA into the base first with `peft.merge_and_unload()` and save to a plain `transformers` directory. GGUF conversion does not consume `.peft` adapters directly.
- A `llama.cpp` build. Clone `https://github.com/ggml-org/llama.cpp`, install its `requirements/requirements-convert_hf_to_gguf.txt`, and build with `cmake -B build && cmake --build build --config Release`. Note the commit hash for `pins.md`.
- Python 3.10+, PyTorch (for the source-side parity check), and disk space for one F16 GGUF (~14 GB for a 7B) plus two quantized variants (~4–6 GB each). Total: ~25 GB.
- **Target hardware for the benchmark.** At least one of:
  - A datacenter or workstation **CPU** pod / laptop CPU (any modern x86-64 or ARM64).
  - An **M-series Mac** (Metal backend).
  - A **consumer GPU** (CUDA / Vulkan / ROCm — build `llama.cpp` with the matching backend flag).
  - An **edge device** (Raspberry Pi 5, Jetson Orin Nano, an Android or iOS device via a demo app, or anything else you have).
- Access to the mod-108 fast-lane setup (`lm-evaluation-harness` with the pinned commit from exercise 01 Part A) if you want to run Check 3.
- A small held-out text corpus (200–500 lines, ~10–100k tokens total) for the perplexity check. Use your task-specific held-out set from mod-108 exercise 02 if you have one; otherwise a slice of your training data with the eval items filtered out.

## Requirements

### Part A — Pin the environment

Author `requirements.txt` and `pins.md` capturing:

- `llama.cpp` commit SHA at build time (`git -C llama.cpp rev-parse HEAD`).
- The `cmake` build flags used (CPU-only? `-DGGML_CUDA=ON`? `-DGGML_METAL=ON`? `-DGGML_VULKAN=ON`?).
- Python versions for `transformers`, `torch`, and the `convert_hf_to_gguf.py` requirements.
- The source checkpoint's provenance: base id, base revision, adapter path (if merged), merge commit / date, `sha256sum` of `model.safetensors` (or the sharded files).
- Target-hardware spec block for whichever machine(s) you benchmark on: CPU model + core count + RAM, GPU model + VRAM (if applicable), OS.

### Part B — Convert to F16 GGUF (the reference)

Run stage 1 of the two-stage conversion pipeline from chapter 6:

```bash
python llama.cpp/convert_hf_to_gguf.py \
  artifacts/bf16 \
  --outfile gguf/support-v3-f16.gguf \
  --outtype f16
```

Verify:

- No warnings about a missing tokenizer, missing chat template, or unsupported architecture. If any appear, fix before continuing — a silent tokenizer skip breaks inference at load time.
- The output file exists and is ~2× the parameter count in bytes (a 7B is ~14 GB).
- `llama.cpp/build/bin/llama-cli -m gguf/support-v3-f16.gguf -p "The capital of France is" -n 20 --temp 0` produces a coherent Paris completion.

Record the conversion wall-clock, output file size, and the smoke-generation output in `conversion/f16.md`.

### Part C — Quantize to two k-quant levels

Choose one workhorse and one alternative from chapter 6's ladder. The recommended starting picks by model size:

- **7B–13B general:** workhorse `Q4_K_M`, alternative `Q5_K_M` (bump if quality-critical) *or* `Q8_0` (near-BF16 reference for comparison).
- **<3B:** workhorse `Q5_K_M`, alternative `Q6_K` (smaller models degrade harder at 4-bit).
- **Code / math–heavy fine-tune:** workhorse `Q5_K_M`, alternative `Q6_K` — numerical-precision tasks regress more at Q4.

Justify your two choices in `conversion/README.md` with reference to the target hardware, quality tolerance, and any prior evidence from exercise 01.

Run stage 2 for each:

```bash
./llama.cpp/build/bin/llama-quantize \
  gguf/support-v3-f16.gguf \
  gguf/support-v3-q4_k_m.gguf \
  q4_k_m

./llama.cpp/build/bin/llama-quantize \
  gguf/support-v3-f16.gguf \
  gguf/support-v3-q5_k_m.gguf \
  q5_k_m
```

Record for each variant in `conversion/<variant>.md`: quantize wall-clock, output file size, and one smoke generation on the "The capital of France is" prompt (should still produce Paris).

### Part D — Equivalence Check 1: load-and-generate parity (cheap)

For a fixed set of 10 prompts and `max_new_tokens=32`, `temperature=0.0` (greedy), compare outputs across:

- Source `transformers` BF16 (via a small script that loads `artifacts/bf16` and generates).
- `llama-cli` on `gguf/support-v3-f16.gguf`.
- `llama-cli` on `gguf/support-v3-q4_k_m.gguf` (or your workhorse).
- `llama-cli` on `gguf/support-v3-q5_k_m.gguf` (or your alternative).

Author `equivalence/check1_greedy.py` (or a shell script that invokes `llama-cli` and diffs). Save a per-prompt table to `equivalence/check1_greedy.md`:

| Prompt | BF16 output | F16 GGUF | Q4_K_M | Q5_K_M | First divergence (token idx) |
|---|---|---|---|---|---|
| ... | | | | | |

Expectations at greedy:

- BF16 and F16 GGUF should agree for the first ~all tokens (they are the same weights in a different container). Divergence in the first several tokens means the conversion is broken — usually a tokenizer or chat-template issue.
- Q4_K_M and Q5_K_M should agree with F16 for the first several tokens and drift only slightly after. Complete divergence from token 1 means the quantization is broken.

If any variant fails, halt and diagnose before continuing.

### Part E — Equivalence Check 2: perplexity delta (numeric)

Compute perplexity on your held-out text corpus with `llama-perplexity` for each of the F16 reference and both quantized variants:

```bash
./llama.cpp/build/bin/llama-perplexity \
  -m gguf/support-v3-f16.gguf \
  -f data/held_out.txt \
  --chunks 32

./llama.cpp/build/bin/llama-perplexity \
  -m gguf/support-v3-q4_k_m.gguf \
  -f data/held_out.txt \
  --chunks 32

./llama.cpp/build/bin/llama-perplexity \
  -m gguf/support-v3-q5_k_m.gguf \
  -f data/held_out.txt \
  --chunks 32
```

Save the raw output to `equivalence/check2_perplexity_raw/` and a summary table to `equivalence/check2_perplexity.md`:

| Variant | Perplexity | Δ vs. F16 (absolute) | Δ vs. F16 (%) |
|---|---|---|---|
| F16 GGUF | | 0 | 0% |
| Q4_K_M | | | |
| Q5_K_M | | | |

Expectations:

- F16 vs. source BF16 model perplexity delta should be near zero (a fraction of a percent at most). If not, either the conversion introduced numerical drift the module does not typically see, or you are computing perplexity on different tokenized text. Investigate.
- Q4_K_M should be within ~1–5% of F16 for a well-behaved 7B+ model.
- Q5_K_M should be closer to F16 than Q4_K_M is.

More than ~10% above F16 for any k-quant is a problem — bump precision, use `imatrix` calibration (stretch goal), or investigate the fine-tune's sensitivity.

### Part F — Equivalence Check 3: release-gate fast lanes (definitive)

Start `llama-server` for each GGUF variant and point `lm-evaluation-harness`'s `local-chat-completions` backend at it. Use the mod-108 fast-lane subset:

```bash
./llama.cpp/build/bin/llama-server \
  -m gguf/support-v3-q4_k_m.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 4096 \
  --chat-template <TEMPLATE_NAME>

# In another shell:
lm_eval --model local-chat-completions \
  --tasks mmlu,arc_challenge,gsm8k,ifeval \
  --model_args base_url=http://localhost:8080/v1,model=support-v3 \
  --seed 42 \
  --output_path results/q4_k_m/
```

Repeat for the F16 reference and the alternative quant. Also include the source BF16 (via `transformers` or the vLLM deployment from exercise 03) as the "true baseline" the GGUF outputs are proving equivalence to.

Save `results/{bf16,f16,q4_k_m,q5_k_m}/` and a summary table to `equivalence/check3_gate.md`:

| Task | BF16 | F16 GGUF | Q4_K_M Δ | Q5_K_M Δ | Inside chapter-1 tolerance? |
|---|---|---|---|---|---|
| MMLU | | | | | |
| ARC-Challenge | | | | | |
| GSM8K | | | | | |
| IFEval | | | | | |

Chapter 1's tolerance table (GGUF Q4_K_M: ≤ 1–2 pp academic, ≤ 2–4 pp task-specific) is the pass/fail bar.

### Part G — Benchmark tokens/sec on target hardware

For each target device you have access to, measure sustained tokens/sec at:

- Batch=1 greedy generation of 256 tokens after a 128-token prompt (typical chat-turn shape).
- Batch=`-np 4` (four concurrent decode streams) if the runtime supports it.
- Prompt-processing throughput: how fast does prefill run on a 2048-token prompt?

The `llama.cpp` bench tool is the reference:

```bash
./llama.cpp/build/bin/llama-bench \
  -m gguf/support-v3-q4_k_m.gguf \
  -p 128,2048 \
  -n 256 \
  -t <cpu_threads> \
  -ngl <gpu_layers_or_0>
```

Save raw output to `bench/<device>/` and a summary table to `bench/summary.md`:

| Device | Backend | Variant | Prompt processing (tok/s @ 2048) | Generation (tok/s @ 256, batch=1) | Peak resident RAM / VRAM (GB) |
|---|---|---|---|---|---|
| e.g., 1× M2 Pro 32 GB | Metal | Q4_K_M | | | |
| e.g., 1× x86-64 8-core | CPU | Q4_K_M | | | |
| e.g., RTX 3090 24 GB | CUDA (-ngl -1) | Q4_K_M | | | |

Track battery / thermal impact on any mobile device benchmark (chapter 6's "cost is battery / thermal budget" observation).

### Part H — The edge deployment brief

Author `edge_deployment_brief.md`, structured so a mobile or on-prem product team could hand it to an engineer and expect a working install. Sections:

1. **Artifact.** File paths, `sha256sum`, quantization level, source-model provenance (base + revision + fine-tune commit).
2. **Runtime.** `llama.cpp` commit + build flags, or the downstream runtime (Ollama, LM Studio, MLC LLM) the target uses. Include install commands.
3. **Equivalence evidence.** One paragraph summarizing Checks 1–3: greedy parity, perplexity delta, release-gate deltas. Cite the numbers.
4. **Performance envelope.** From Part G. Tokens/sec on each tested device, RAM / VRAM footprint, prompt-processing latency for the p95 prompt length.
5. **Runtime configuration.** The `llama-server` (or Ollama `Modelfile`, or LM Studio config) command / manifest with every flag pinned. Chat template, context size, parallel-slot count. Rationale for each.
6. **Known limitations.** Which quantization level was chosen and why the other was rejected. Which target devices were tested and which were extrapolated. Any tasks that regressed enough to require a warning (e.g., "GSM8K dropped 4 pp at Q4_K_M; do not use for math-heavy customer use cases").
7. **Rollback and monitoring.** How does the customer downgrade if a shipped update misbehaves? What signals should be watched?
8. **Reproducibility notes.** The pin set from Part A, plus the calibration corpus for the imatrix (if used), plus the eval commit.

### Part I — The write-up

Author `report.md` — the narrative that connects Parts D–H:

1. **Setup.** Source, target hardware(s), why the two chosen quant levels.
2. **Conversion story.** Any surprises during Part B / C. Warnings, dtype casts, tokenizer packing.
3. **Equivalence.** All three checks summarized. Which variant passed, which did not, which is marginal.
4. **Performance.** Cross-device benchmark. Where does Q4_K_M beat Q5_K_M enough to matter, and where is Q5_K_M's quality worth the memory / speed cost?
5. **Recommendation.** For **this specified serving scenario**, which variant do you ship?
   > "A B2C desktop application that runs the fine-tune locally on the customer's laptop for privacy reasons. Target hardware: M-series Mac (M1 or newer) with ≥16 GB unified memory, and x86-64 Windows laptop with ≥16 GB RAM. p95 TTFT < 1.5 s. Users tolerate 15+ tokens/sec generation. Quality tolerance: ≤ 3 pp regression on general instruct tasks; a small regression on math is acceptable if quality on the primary task (customer-support Q&A) is preserved."
6. **What surprised you.** One paragraph.
7. **Reproducibility notes.** Same shape as prior exercises.

### Part J — Directory layout

```
outputs/exercise-04/
├── requirements.txt
├── pins.md
├── artifacts/                       # symlink to exercise-01 artifacts/bf16
├── gguf/
│   ├── support-v3-f16.gguf
│   ├── support-v3-q4_k_m.gguf
│   └── support-v3-q5_k_m.gguf
├── conversion/
│   ├── README.md
│   ├── f16.md
│   ├── q4_k_m.md
│   └── q5_k_m.md
├── data/
│   └── held_out.txt
├── equivalence/
│   ├── check1_greedy.py
│   ├── check1_greedy.md
│   ├── check2_perplexity.md
│   ├── check2_perplexity_raw/
│   └── check3_gate.md
├── results/
│   ├── bf16/
│   ├── f16/
│   ├── q4_k_m/
│   └── q5_k_m/
├── bench/
│   ├── summary.md
│   └── <device>/
├── edge_deployment_brief.md
└── report.md
```

## Starter guidance

- **Merge before you convert.** The GGUF conversion is of a single merged model. If your fine-tune is a LoRA, `peft.merge_and_unload()` first into a BF16 model and convert that. Trying to convert base + separate LoRA is the alternative path (chapter 6 covers `convert_lora_to_gguf.py`) and has more failure modes; save it for the stretch goal.
- **Verify the chat template is baked into the GGUF.** `llama-cli --chat-template <name>` will override whatever the GGUF specifies. Leave the flag unset for the smoke tests and confirm the GGUF's embedded template matches your fine-tune. If the template is wrong or missing, refuse to ship — chapter 6 calls this out as a common failure.
- **Perplexity on the same tokenized corpus.** `llama-perplexity` uses the GGUF's tokenizer; the source-side perplexity must use the same tokenizer. If you tokenize with an off-the-shelf `AutoTokenizer` from a different model, the perplexity numbers are not comparable. Load the same tokenizer for both sides.
- **`llama-server` chat template.** The `local-chat-completions` backend of `lm-evaluation-harness` posts to `/v1/chat/completions`, which applies the server-side chat template. Verify with a one-request smoke first (the same "hidden phrase" test from chapter 4) — if the wrong template is applied, your gate numbers will look like a quantization regression when it is really a template bug.
- **Warm up before benchmarking.** `llama.cpp`'s first few tokens on a cold process are slower (memory-mapping, first cache warm-up). `llama-bench` handles this; a hand-rolled benchmark should skip the first 32 tokens.
- **CPU thread count matters.** `-t <n>` matches the number of physical cores by default in recent `llama.cpp`, but on containers with restricted CPU quota, over-specifying threads hurts throughput. Match `-t` to actual available cores.
- **GPU offload is a knob, not a switch.** `-ngl -1` offloads all layers to GPU (fastest, most VRAM). `-ngl 0` is CPU-only. Intermediate values are useful when the model is bigger than the GPU can hold. Report tokens/sec at both extremes for the GPU device.
- **Stretch goals aside, ship Q4_K_M by default.** Chapter 6's "canonical edge default" is a starting point, not a floor. Move up (Q5_K_M / Q6_K) only when Check 3 shows the numbers demand it — and document the reason in the brief.

## Acceptance criteria

You have completed the exercise when:

- `pins.md` records `llama.cpp` commit, build flags, source-model provenance, and target-hardware specs.
- Two GGUF variants exist under `gguf/` and both load-and-generate a coherent response on the smoke prompt.
- `equivalence/check1_greedy.md` shows greedy parity between BF16 and F16 GGUF, and characterizes drift for the two quantized variants.
- `equivalence/check2_perplexity.md` reports perplexity for F16 + both variants with a Δ column.
- `equivalence/check3_gate.md` has the fast-lane numbers for all four points (BF16 source, F16 GGUF, both quantized variants) with pass/fail annotations against chapter 1's tolerance table.
- `bench/summary.md` reports tokens/sec on at least one real device (do not fabricate numbers if you do not have hardware; document the deferral explicitly).
- `edge_deployment_brief.md` includes all eight sections and is executable — an engineer could follow it to install and configure the runtime.
- `report.md` includes all seven sections; the recommendation is defended by concrete numbers from your own equivalence checks and benchmarks.

## Stretch goals

- **Add an `imatrix`-calibrated `IQ` variant.** Generate an importance matrix from a calibration corpus (chapter 6's `llama-imatrix` step) and quantize to `IQ4_XS` or `IQ3_M`. Add it to Checks 1–3 and the benchmark table. Report whether the imatrix bought back quality relative to the plain k-quant at the same bit width.
- **Ship a `Q8_0` reference.** Add `Q8_0` as a third variant — the "near-BF16 in GGUF form" reference. Confirm it is essentially indistinguishable from F16 on the release gate. Useful as an internal reference when a customer reports a bug: "does it reproduce on Q8_0?" isolates whether the issue is quantization or something else.
- **LoRA-on-GGUF path.** Instead of merging, convert the base to GGUF and the adapter to a GGUF LoRA (`convert_lora_to_gguf.py`), load with `llama-cli --lora`. Compare Check 1 outputs to the merged path. Report support quality and any caveats — chapter 6 flagged this as evolving.
- **Ollama-native workflow.** Package the chosen GGUF as an Ollama model via `Modelfile`, publish to a local Ollama library, and repeat the smoke tests through `ollama run`. Add the `Modelfile` to the deployment brief. This is often what customer-facing installers actually consume.
- **Mobile deployment.** Build the `llama.cpp` Android or iOS demo (or use MLC LLM's mobile runtime), load the Q4_K_M artifact, and record tokens/sec + battery / thermal behavior for a five-minute chat session. Add a mobile row to `bench/summary.md`. Discuss what would need to change for a production mobile ship.
- **Chat-template edge cases.** Compare `/v1/chat/completions` behavior between `llama-server` and vLLM / TGI on a workload with tool calls or multi-turn system prompts. Report any divergences — chat-template implementations differ subtly across runtimes and this is a common source of "the model behaves differently in Ollama than in production" bug reports.
- **Cost projection.** For the CPU pod scenario, compute dollars-per-million-tokens on a Graviton or `c7i` instance running `llama-server`. Compare to the datacenter-GPU cost from exercise 03's write-up. Chapter 7's cost model is the reference; the interesting axis here is when a "CPU-only serving" story is cost-competitive.
- **Contamination scan on the calibration corpus.** If you used an imatrix calibration, run mod-108 exercise 04's contamination probes against the calibration data vs. the eval sets used in Check 3. Disclose in the brief. This closes the loop on the contamination discipline from exercise 01.
