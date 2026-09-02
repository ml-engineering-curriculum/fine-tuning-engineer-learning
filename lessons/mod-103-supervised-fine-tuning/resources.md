# Resources for mod-103-supervised-fine-tuning (Supervised Fine-Tuning End-to-End)

Curated primary references. Prefer these sources — the paper of record, the official framework documentation, the current tool repo — over blog-post summaries when you go deep. Where a chapter cites a claim, the citation lands here.

## Chapter 1 — The SFT training loop, end to end

### Papers and technical reports

- **Ouyang et al., 2022** — "Training language models to follow instructions with human feedback." The InstructGPT paper; the reference framing of SFT as the first stage of instruction-tuning before preference optimization. https://arxiv.org/abs/2203.02155
- **Touvron et al., 2023b** — "Llama 2: Open Foundation and Fine-Tuned Chat Models." Section 3 documents the Llama 2 SFT recipe used as an anchor throughout this module. https://arxiv.org/abs/2307.09288
- **Tunstall et al., 2023** — "Zephyr: Direct Distillation of LM Alignment." SFT-then-DPO recipe on Mistral 7B; a reference for a compact open recipe. https://arxiv.org/abs/2310.16944
- **Lambert et al., 2024** — "Tülu 3: Pushing Frontiers in Open Language Model Post-Training." End-to-end open post-training pipeline with an SFT stage; a reference for a full modern recipe. https://arxiv.org/abs/2411.15124
- **Zhou et al., 2023** — "LIMA: Less Is More for Alignment." The reference for the quality-over-quantity SFT effect that shapes epoch-count and dataset-size decisions later in the module. https://arxiv.org/abs/2305.11206

### Official framework documentation

- Hugging Face **TRL** documentation (home of `SFTTrainer` / `SFTConfig`) — https://huggingface.co/docs/trl
- TRL **`SFTTrainer`** guide specifically — https://huggingface.co/docs/trl/en/sft_trainer
- Hugging Face **`transformers.Trainer`** reference — https://huggingface.co/docs/transformers/main/en/main_classes/trainer
- Hugging Face **`TrainingArguments`** reference — https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
- Hugging Face **`datasets`** documentation — https://huggingface.co/docs/datasets
- Hugging Face **chat templating** documentation — https://huggingface.co/docs/transformers/main/en/chat_templating
- Hugging Face **`accelerate`** documentation (the launcher TRL delegates multi-device concerns to) — https://huggingface.co/docs/accelerate

### Alternative trainer stacks referenced in the chapter

- **Axolotl** — YAML-driven SFT / DPO / continued-pretraining wrapper. https://github.com/axolotl-ai-cloud/axolotl and docs at https://axolotl-ai-cloud.github.io/axolotl/
- **Unsloth** — custom Triton kernels for LoRA / QLoRA acceleration. https://github.com/unslothai/unsloth and docs at https://docs.unsloth.ai
- **`torchtune`** — PyTorch-native fine-tuning library (an alternative to TRL when you want fewer HF abstractions). https://pytorch.org/torchtune/stable/

### Base models suggested for a first SFT run

- `meta-llama/Llama-3.2-1B` — https://huggingface.co/meta-llama/Llama-3.2-1B
- `Qwen/Qwen2.5-1.5B` — https://huggingface.co/Qwen/Qwen2.5-1.5B
- `HuggingFaceTB/SmolLM2-1.7B` — https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B
- `microsoft/Phi-3.5-mini-instruct` — https://huggingface.co/microsoft/Phi-3.5-mini-instruct

## Chapter 2 — Masking, packing, and effective batch size

### Papers

- **Dao et al., 2022** — "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." The packed-sequence / `cu_seqlens` primitive the chapter's per-example-mask packing relies on. https://arxiv.org/abs/2205.14135
- **Dao, 2023** — "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." https://arxiv.org/abs/2307.08691
- **Shah et al., 2024** — "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." Hopper-generation update. https://arxiv.org/abs/2407.08608

### Framework references for masking and packing

- TRL **`DataCollatorForCompletionOnlyLM`** — https://huggingface.co/docs/trl/en/sft_trainer#train-on-completions-only
- `tokenizer.apply_chat_template` **`return_assistant_tokens_mask`** — https://huggingface.co/docs/transformers/main/en/chat_templating#advanced-adding-and-editing-chat-templates
- Axolotl sample-packing documentation — https://axolotl-ai-cloud.github.io/axolotl/docs/config-reference.html
- **`flash-attention`** repo (implementation of `flash_attn_varlen_func` used for variable-length packed sequences) — https://github.com/Dao-AILab/flash-attention
- **Liger Kernel** (LinkedIn) — open-source Triton kernels for HF training including packed-sequence paths. https://github.com/linkedin/Liger-Kernel
- PyTorch **scaled_dot_product_attention** (SDPA) reference — https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

### Gradient accumulation and mixed-precision references

- Hugging Face **performance and scalability guide** (training) — https://huggingface.co/docs/transformers/main/en/performance
- PyTorch **automatic mixed precision** reference — https://pytorch.org/docs/stable/amp.html
- **Micikevicius et al., 2018** — "Mixed Precision Training." The reference for FP16 / BF16 training dynamics. https://arxiv.org/abs/1710.03740

## Chapter 3 — Learning-rate schedule, warmup, weight decay, and checkpoint cadence

### Papers

- **Loshchilov & Hutter, 2019** — "Decoupled Weight Decay Regularization." The AdamW paper; grounds the "AdamW with decoupled weight decay" default. https://arxiv.org/abs/1711.05101
- **Loshchilov & Hutter, 2017** — "SGDR: Stochastic Gradient Descent with Warm Restarts." Introduces the cosine annealing schedule used as default in TRL. https://arxiv.org/abs/1608.03983
- **Kingma & Ba, 2015** — "Adam: A Method for Stochastic Optimization." The original Adam paper. https://arxiv.org/abs/1412.6980
- **Dettmers et al., 2022** — "8-bit Optimizers via Block-wise Quantization." The paper behind `bitsandbytes` 8-bit AdamW. https://arxiv.org/abs/2110.02861
- **Chen et al., 2023** — "Symbolic Discovery of Optimization Algorithms." The Lion optimizer. https://arxiv.org/abs/2302.06675
- **Hu et al., 2021** — "LoRA: Low-Rank Adaptation of Large Language Models." Grounds the "LoRA takes an order-of-magnitude larger LR than full-parameter SFT" heuristic. https://arxiv.org/abs/2106.09685
- **Dettmers et al., 2023** — "QLoRA: Efficient Finetuning of Quantized LLMs." Reference for the `adamw_bnb_8bit` recipe used in QLoRA. https://arxiv.org/abs/2305.14314

### Framework references

- `transformers.optimization` schedule utilities (cosine, linear, constant_with_warmup, etc.) — https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules
- `TrainingArguments` `lr_scheduler_type`, `warmup_ratio`, `warmup_steps`, `weight_decay` — https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
- `TrainingArguments` `save_strategy`, `save_steps`, `save_total_limit`, `load_best_model_at_end`, `metric_for_best_model` — same reference.
- **`bitsandbytes`** (8-bit optimizers, quantization) — https://github.com/bitsandbytes-foundation/bitsandbytes
- **`lion-pytorch`** — https://github.com/lucidrains/lion-pytorch

### Recipes to compare against

- OpenHermes-2.5 (Teknium) — synthetic instruction mix; often paired with modest LR SFT. https://huggingface.co/datasets/teknium/OpenHermes-2.5
- Zephyr recipe (SFT step): the config used in the Zephyr paper — https://github.com/huggingface/alignment-handbook

## Chapter 4 — Reading loss curves

### Papers and references on loss diagnostics

- **Nakkiran et al., 2019** — "Deep Double Descent: Where Bigger Models and More Data Hurt." A general reference for why "eval loss rising while train loss falls" has a well-studied structure. https://arxiv.org/abs/1912.02292
- **Kaplan et al., 2020** — "Scaling Laws for Neural Language Models." Grounds the intuition that loss curves have predictable shapes at a given compute/data scale. https://arxiv.org/abs/2001.08361
- **Hoffmann et al., 2022** — "Training Compute-Optimal Large Language Models" (Chinchilla). Not directly about SFT loss curves, but the reference for the "under-trained" shape you compare an SFT run against. https://arxiv.org/abs/2203.15556

### Tooling for logging and plotting

- **Weights & Biases** documentation — https://docs.wandb.ai
- **TensorBoard** (via `SummaryWriter`) — https://www.tensorflow.org/tensorboard and PyTorch wrapper https://pytorch.org/docs/stable/tensorboard.html
- **MLflow** tracking — https://mlflow.org/docs/latest/tracking.html
- **Aim** — https://github.com/aimhubio/aim
- `TrainerCallback` reference (for the "log per-example loss" pattern described in the chapter) — https://huggingface.co/docs/transformers/main/en/main_classes/callback

## Chapter 5 — Overfitting, catastrophic forgetting, and template leak

### Papers on catastrophic forgetting

- **McCloskey & Cohen, 1989** — "Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem." *Psychology of Learning and Motivation*, Vol. 24. The original reference for the phenomenon. https://doi.org/10.1016/S0079-7421(08)60536-8
- **French, 1999** — "Catastrophic forgetting in connectionist networks." *Trends in Cognitive Sciences*, 3(4), 128–135. The canonical review. https://doi.org/10.1016/S1364-6613(99)01294-2
- **Kirkpatrick et al., 2017** — "Overcoming catastrophic forgetting in neural networks." Elastic weight consolidation; a mitigation reference from continual learning. https://arxiv.org/abs/1612.00796
- **Biderman et al., 2024** — "LoRA Learns Less and Forgets Less." Empirical evidence that LoRA reduces forgetting relative to full-parameter fine-tuning — the source of chapter 5's LoRA-as-mitigation claim. https://arxiv.org/abs/2405.09673
- **Luo et al., 2023** — "An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning." An LLM-specific empirical study. https://arxiv.org/abs/2308.08747

### Benchmarks used to detect forgetting

- **Hendrycks et al., 2021** — MMLU. https://arxiv.org/abs/2009.03300 — dataset: https://huggingface.co/datasets/cais/mmlu
- **Clark et al., 2018** — ARC. https://arxiv.org/abs/1803.05457 — dataset: https://huggingface.co/datasets/allenai/ai2_arc
- **Zellers et al., 2019** — HellaSwag. https://arxiv.org/abs/1905.07830 — dataset: https://huggingface.co/datasets/Rowan/hellaswag
- **Cobbe et al., 2021** — GSM8K. https://arxiv.org/abs/2110.14168 — dataset: https://huggingface.co/datasets/openai/gsm8k
- **Chen et al., 2021** — HumanEval. https://arxiv.org/abs/2107.03374 — dataset: https://huggingface.co/datasets/openai/openai_humaneval
- **Zhou et al., 2023** — IFEval (instruction-following eval). https://arxiv.org/abs/2311.07911 — dataset: https://huggingface.co/datasets/google/IFEval
- **Zheng et al., 2023** — MT-Bench and Chatbot Arena. https://arxiv.org/abs/2306.05685 — prompts: https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge
- **Röttger et al., 2023** — XSTest (measures over-refusal, a specific safety-tuning forgetting mode). https://arxiv.org/abs/2308.01263

### Eval frameworks

- **EleutherAI `lm-evaluation-harness`** — the standard framework for MMLU, ARC, HellaSwag, GSM8K, IFEval evaluation. https://github.com/EleutherAI/lm-evaluation-harness
- **`bigcode-evaluation-harness`** — HumanEval and code-eval variants. https://github.com/bigcode-project/bigcode-evaluation-harness
- **`lighteval`** (Hugging Face) — an alternative modern eval harness. https://github.com/huggingface/lighteval

### Chat templates (where template-leak bugs actually originate)

- Llama 3 prompt format reference — https://github.com/meta-llama/llama-models/blob/main/models/llama3/README.md
- Qwen 2 / 2.5 chat template documentation — https://qwen.readthedocs.io
- Hugging Face **chat templating** documentation (the source of truth for `return_assistant_tokens_mask` and template diffing) — https://huggingface.co/docs/transformers/main/en/chat_templating

## Chapter 6 — Reproducibility and the training report

### Papers and frameworks on ML reproducibility

- **Pineau et al., 2021** — "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)." *Journal of Machine Learning Research* 22(164):1−20. https://www.jmlr.org/papers/v22/20-303.html
- **Mitchell et al., 2019** — "Model Cards for Model Reporting." The reference structure for the model card that ships alongside the training report. https://arxiv.org/abs/1810.03993
- **Gebru et al., 2018 / 2021** — "Datasheets for Datasets." Companion structure for documenting the training set. https://arxiv.org/abs/1803.09010
- **Reproducible ML Checklist** (NeurIPS) — https://neurips.cc/public/guides/PaperChecklist
- **The Turing Way — Guide for Reproducible Research** — https://the-turing-way.netlify.app/reproducible-research/reproducible-research.html

### Tooling references for reproducibility

- **PyTorch reproducibility notes** (`torch.use_deterministic_algorithms`, `torch.backends.cudnn.benchmark`, `PYTHONHASHSEED`) — https://pytorch.org/docs/stable/notes/randomness.html
- **`transformers.set_seed`** — https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.set_seed
- Hugging Face **Hub revision pinning** (`revision=` kwarg on `from_pretrained`) — https://huggingface.co/docs/huggingface_hub/main/en/guides/download#download-from-a-specific-revision
- **`uv`** (Astral) — modern Python package resolver + lockfile format used for the `requirements.lock` referenced in the chapter. https://docs.astral.sh/uv/
- **`pip freeze`** and `pip-compile` (pip-tools) — https://pip.pypa.io/en/stable/cli/pip_freeze/ and https://github.com/jazzband/pip-tools
- **DVC** (Data Version Control) — one option for versioning the dataset files the manifest references. https://dvc.org
- **`safetensors`** format used for checkpoint hashing — https://github.com/huggingface/safetensors

### Training-report and run-registry references

- **Hugging Face model cards** — https://huggingface.co/docs/hub/model-cards
- **Hugging Face dataset cards** — https://huggingface.co/docs/hub/datasets-cards
- **Google Data Cards Playbook** — https://sites.research.google/datacardsplaybook/
- WandB **Reports** (an example of a rendered training report format) — https://docs.wandb.ai/guides/reports/
- **`sigstore` / `cosign`** for signing model artifacts (stretch-goal reference in exercise 05) — https://docs.sigstore.dev

## Suggested reading order for this module

1. Read the TRL `SFTTrainer` guide and skim `transformers.Trainer` reference before chapter 1. You do not need to memorize either; you need to know what to grep when you get stuck.
2. Read the Llama 2 paper section 3 (SFT) and skim the Tülu 3 tech report SFT section before chapter 3 — those two grounded the anchor LR ranges and the epoch-count recommendations.
3. Skim FlashAttention (Dao 2022) and the TRL packing docs before chapter 2. The paper explains *why* per-example masks matter; the docs explain *how* to turn them on.
4. Read AdamW (Loshchilov & Hutter, 2019) and the SGDR / cosine schedule paper before chapter 3. Both are short and both anchor defaults you will use for the rest of the module.
5. Read the Biderman et al. "LoRA Learns Less and Forgets Less" paper before chapter 5 — it grounds the LoRA-as-forgetting-mitigation claim you will act on in exercise 04.
6. Read Model Cards for Model Reporting (Mitchell et al., 2019) and one full open model card of your choice (Llama 3 or Tülu 3) before chapter 6 and exercise 05. The report you produce in exercise 05 is a compressed cousin of the model-card structure they describe.

<!-- needs-research: on the next research cycle, verify current URLs for the Llama 3.x prompt-format README and the Qwen 2.x chat-template docs; both repos have moved before and may move again. Also add version pins for TRL, transformers, and Axolotl matching the versions this module was authored against. -->
