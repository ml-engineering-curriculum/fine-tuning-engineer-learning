# Prerequisites for the Fine-Tuning Engineer track

This curriculum starts at **level 30 (deep specialist — peer to Senior ML Engineer)**. It does not re-teach the build-altitude Machine Learning Engineer foundation. Before working through it, you should already be comfortable with the items below. The [`ml-engineer-learning`](https://github.com/ml-engineering-curriculum/ml-engineer-learning) track (level 20) is the canonical place to acquire them, and [`ai-infra-junior-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning) (level 10) below that.

## Required

- **Intermediate Python** — functions, classes, modules, virtual environments, packaging (`uv` / `poetry`), structured logging, unit testing, and the ability to read a moderately complex open-source library.
- **PyTorch fluency** — write a training loop from scratch (forward / loss / backward / step), use `nn.Module`, `DataLoader`, optimizers, learning-rate schedules, mixed precision (`autocast` / `GradScaler`), and gradient accumulation. PyTorch is non-negotiable for this track.
- **Hugging Face `transformers` working familiarity** — load a model and tokenizer, run `generate()`, understand `AutoModelForCausalLM` vs. `AutoModelForSequenceClassification`, and know what a `pad_token` / `eos_token` is.
- **Deep learning fundamentals** — backprop, common optimizers (AdamW), regularisation, attention at the level "I can explain query / key / value and the multi-head attention shape." If you cannot, do `mod-105` of `ml-engineer-learning` first.
- **Experiment tracking** — at least one of MLflow / Weights & Biases used in anger on a small project.
- **Model packaging basics** — FastAPI / Docker enough to package a small model. `mod-108` and `mod-109` of `ml-engineer-learning` cover this.
- **Linux command line and Git** — assumed at the level of `ai-infra-junior-engineer-learning`.
- **Basic linear algebra and statistics** — vectors, matrices, probability, distributions, gradient intuition. Cited papers in this track assume this comfort.

## Recommended

- **One end-to-end ML project shipped** (Kaggle, course capstone, hobby model) at the level of `ml-engineer-learning` Projects 101–103. Without one, mod-103 and the projects in this track will feel abrupt.
- **One prior fine-tuning attempt at any depth** — even a Colab QLoRA notebook is enough. You will move much faster if you have already felt the wall-clock and VRAM constraints once.
- **GPU access plan** — either local consumer GPU (≥ 16 GB VRAM, ideally 24 GB), Colab Pro / Pro+, Modal / RunPod / Lambda / Vast.ai credits, or a cloud account. mod-104 and onwards expect at least a 16-GB consumer GPU or equivalent rented; mod-107 expects multi-GPU access on at least one occasion (rented is fine).
- **Hugging Face account** — datasets, models, and inference endpoints are easier to exercise with an authenticated account.
- **Basic LLM application exposure** — having called an OpenAI / Anthropic / open-source LLM API once and built a small toy with prompts helps frame the build-vs-buy discussion in `mod-110` and `mod-111`.

## Out of scope as prerequisites

You do **not** need to know any of the following before starting; they are either taught in this curriculum or owned by a peer/upstream track:

- TRL, PEFT, Accelerate, DeepSpeed, FSDP, bitsandbytes, Axolotl, Unsloth, vLLM, TGI, GGUF, lm-evaluation-harness — taught here.
- LoRA, QLoRA, DPO, PPO-RLHF, ORPO, KTO, GRPO, reward modeling — taught here.
- Multi-GPU / multi-node launchers and ZeRO partitioning — taught at the operator altitude in `mod-107` (not at the platform-engineering altitude).
- Cluster engineering / multi-tenant GPU schedulers / NCCL fabric tuning — owned by [`training-pipeline-engineer-learning`](https://github.com/ml-engineering-curriculum/training-pipeline-engineer-learning). Linked out, not re-taught.
- LLM application engineering depth (prompting, agents, tool use) — owned by [`llm-application-developer-learning`](https://github.com/ml-engineering-curriculum/llm-application-developer-learning).
- RAG depth — owned by [`rag-engineer-learning`](https://github.com/ml-engineering-curriculum/rag-engineer-learning).
- Eval platform engineering depth — owned by [`model-evaluation-engineer-learning`](https://github.com/ml-engineering-curriculum/model-evaluation-engineer-learning) / [`ai-eval-engineer-learning`](https://github.com/ml-engineering-curriculum/ai-eval-engineer-learning).
- ML/AI security depth — owned by [`ai-infra-security-learning`](https://github.com/ai-infra-curriculum/ai-infra-security-learning).
- Governance / compliance / risk depth — owned by [`ai-governance-analyst-learning`](https://github.com/ml-engineering-curriculum/ai-governance-analyst-learning) and [`ai-risk-engineer-learning`](https://github.com/ml-engineering-curriculum/ai-risk-engineer-learning).

## Hardware

- Modules 101–103 can be done on a 16-GB consumer GPU or a free Colab / Kaggle notebook (1B-class base model).
- Modules 104 (QLoRA on 7B–13B) and 108 expect a 24-GB GPU or equivalent rented time (an RTX 3090 / 4090, A10G, L4, A100 40GB, or rented hourly equivalent works).
- Module 105 (preference optimization, especially PPO-RLHF) is the most VRAM-hungry — plan for a 40-GB-class GPU or paired-GPU setup for a few sessions. DPO is materially cheaper than PPO and is fine on 24 GB.
- Module 107 (distributed fine-tuning) expects at least one multi-GPU session — rent it.
- Module 109 (serving) needs the same VRAM tier as 104; vLLM and TGI run happily on a single 24-GB consumer GPU for the lab.
- Module 110 (managed APIs) needs API credits (≤ \$50 with care) but no local GPU.

<!-- needs-research: confirm the hardware budget against the live posting evidence on the next autonomous research cycle; if postings consistently demand only A100 / H100 multi-node experience, raise the floor on the recommended-hardware list. -->
