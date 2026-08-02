# Resources for mod-104-peft-lora-qlora

Curated primary references. Prefer these — the paper of record, the official docs, the current tool README — over blog-post summaries when you go deep.

## Chapter 1 — LoRA math and implementation

### Papers

- **Hu et al., 2021** — "LoRA: Low-Rank Adaptation of Large Language Models." The paper of record. Read sections 3 (method), 4 (relation to existing PEFT), and 7 (empirical rank ablations). https://arxiv.org/abs/2106.09685
- **Aghajanyan et al., 2020** — "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." The prior work LoRA cites for its intrinsic-rank motivation. https://arxiv.org/abs/2012.13255
- **Kalajdzievski, 2023** — "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA" (rsLoRA). Motivates `alpha / sqrt(r)` scaling. https://arxiv.org/abs/2312.03732
- **Liu et al., 2024** — "DoRA: Weight-Decomposed Low-Rank Adaptation." A modern LoRA variant that decomposes the update into magnitude + direction; supported by `peft`. https://arxiv.org/abs/2402.09353

### Official tooling

- Hugging Face `peft` — main docs and API reference. https://huggingface.co/docs/peft
- `peft` LoRA conceptual guide — https://huggingface.co/docs/peft/conceptual_guides/lora
- `peft` GitHub, examples directory — https://github.com/huggingface/peft/tree/main/examples

## Chapter 2 — QLoRA and 4-bit base weights

### Papers

- **Dettmers et al., 2023** — "QLoRA: Efficient Finetuning of Quantized LLMs." The paper of record for NF4, double quantization, paged optimizers, and the Guanaco recipe. https://arxiv.org/abs/2305.14314
- **Dettmers et al., 2022** — "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." Background on the 8-bit quantization path in `bitsandbytes`. https://arxiv.org/abs/2208.07339
- **Dettmers and Zettlemoyer, 2023** — "The case for 4-bit precision: k-bit Inference Scaling Laws." Empirical justification for 4-bit being on the Pareto frontier. https://arxiv.org/abs/2212.09720

### Official tooling

- `bitsandbytes` — main repository and docs. https://github.com/bitsandbytes-foundation/bitsandbytes
- `bitsandbytes` 4-bit integration guide (in `transformers` docs) — https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes
- `peft` k-bit training utilities (`prepare_model_for_kbit_training`) — https://huggingface.co/docs/peft/main/en/package_reference/peft_types

## Chapter 3 — Merging and multi-adapter serving

### Papers

- **Kwon et al., 2023** — "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM). The PagedAttention paper. Background for how a modern inference stack can support multi-adapter LoRA at high throughput. https://arxiv.org/abs/2309.06180
- **Sheng et al., 2023** — "S-LoRA: Serving Thousands of Concurrent LoRA Adapters." The reference paper for large-scale multi-adapter serving; techniques that vLLM / TGI / LoRAX drew from. https://arxiv.org/abs/2311.03285
- **Yadav et al., 2023** — "TIES-Merging: Resolving Interference When Merging Models." Background for `peft`'s TIES combination in `add_weighted_adapter`. https://arxiv.org/abs/2306.01708

### Official tooling and docs

- vLLM LoRA support — https://docs.vllm.ai/en/latest/models/lora.html
- Hugging Face TGI (Text Generation Inference) — https://huggingface.co/docs/text-generation-inference
- Predibase LoRAX — https://github.com/predibase/lorax
- `peft` multi-adapter guide — https://huggingface.co/docs/peft/main/en/developer_guides/mixed_models
- `PeftModel.merge_and_unload` reference — https://huggingface.co/docs/peft/main/en/package_reference/peft_model

## Chapter 4 — PEFT methods in context

### Papers

- **Liu et al., 2022** — "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning" (IA³). https://arxiv.org/abs/2205.05638
- **Li and Liang, 2021** — "Prefix-Tuning: Optimizing Continuous Prompts for Generation." https://arxiv.org/abs/2101.00190
- **Lester et al., 2021** — "The Power of Scale for Parameter-Efficient Prompt Tuning." https://arxiv.org/abs/2104.08691
- **Liu et al., 2022** — "P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks." https://arxiv.org/abs/2110.07602
- **Houlsby et al., 2019** — "Parameter-Efficient Transfer Learning for NLP." The original Houlsby adapter paper. Historical but foundational. https://arxiv.org/abs/1902.00751
- **He et al., 2022** — "Towards a Unified View of Parameter-Efficient Transfer Learning." A useful survey that puts LoRA / prefix / prompt / adapter methods in one framework. https://arxiv.org/abs/2110.04366
- **Han et al., 2024** — "Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey." A current survey of the space. https://arxiv.org/abs/2403.14608

### Official tooling

- `peft` conceptual guide (per-method overview) — https://huggingface.co/docs/peft/main/en/conceptual_guides/adapter
- `peft` supported methods list — https://huggingface.co/docs/peft/main/en/index

## Chapter 5 — Adapter failure modes

### Papers

- The Hu et al. LoRA paper's ablation section (again — cited above). Sections 7.1 and 7.2 have the empirical rank / target-module analyses.
- The Dettmers et al. QLoRA paper's Appendix on hyperparameter sensitivity. Cited above.
- **Roy and Vetterli, 2007** — "The Effective Rank: A Measure of Effective Dimensionality." Background for the stable-rank metric used in the health check. https://ieeexplore.ieee.org/document/7098875
- **Kalajdzievski, 2023** — rsLoRA (cited above); directly relevant to the alpha-mis-scaling failure mode.

### Tooling for diagnostics

- `torch.linalg` documentation — https://pytorch.org/docs/stable/linalg.html
- `weights & biases` custom charts (for logging per-layer SVD over training) — https://docs.wandb.ai/guides/track/log/plots

## Recipes and reference runs

- **Guanaco recipe** (QLoRA paper's canonical config; a good starting point for any 7B–13B QLoRA run) — released with the QLoRA paper's code. https://github.com/artidoro/qlora
- **Zephyr-7B recipes** (HuggingFaceH4) — including QLoRA-based SFT recipes. https://github.com/huggingface/alignment-handbook
- **Axolotl** — YAML-driven LoRA / QLoRA / prefix-tuning recipes for a wide model set. https://github.com/axolotl-ai-cloud/axolotl
- **Unsloth** — kernel-level LoRA / QLoRA acceleration for supported bases. https://github.com/unslothai/unsloth
- **LLaMA-Factory** — an alternative all-in-one PEFT training framework covering LoRA, QLoRA, prefix, prompt, IA³. https://github.com/hiyouga/LLaMA-Factory

## Suggested reading order for this module

1. Skim the LoRA paper's introduction and method section (§3) to anchor the low-rank picture.
2. Read chapter 1, then implement `LoRALinear` from scratch (exercise 01, Part A) before touching `peft`.
3. Skim the QLoRA paper's introduction and §3 (their technique details) before reading chapter 2.
4. Read chapter 2 and run exercise 02 alongside the QLoRA paper's Guanaco recipe.
5. Read chapter 3 alongside the vLLM LoRA docs and the S-LoRA paper (for the ambitious version of multi-adapter serving).
6. Read chapter 4 before doing exercise 04 — the decision matrix is more useful after you have felt LoRA and QLoRA in your hands.
7. Read chapter 5 last; the failure modes are more memorable when you have already run at least two adapters.

<!-- needs-research: on the next authoring cycle, add version pins for the tooling links (peft, bitsandbytes, transformers, trl, vllm) so future readers land on the version this module was authored against. Also verify vLLM / TGI LoRA URLs still route to their current LoRA docs, since those docs sites reorganize frequently. -->
