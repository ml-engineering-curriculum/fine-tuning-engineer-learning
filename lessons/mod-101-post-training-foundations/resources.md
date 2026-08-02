# Resources for mod-101-post-training-foundations

Curated primary references for the four objectives of this module. Prefer these sources over blog-post summaries when you go deep; every one is either the paper of record, the official implementation, or the current maintained documentation.

## Chapter 1 — Decoder-only forward pass and PEFT targets

### Architecture papers

- **Vaswani et al., 2017** — "Attention Is All You Need." The original transformer paper. Encoder-decoder, but every decoder-only model in production descends from its decoder stack. https://arxiv.org/abs/1706.03762
- **Radford et al., 2018 / 2019** — the GPT and GPT-2 papers. Establishes the decoder-only recipe. https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf and https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- **Touvron et al., 2023a** — "LLaMA: Open and Efficient Foundation Language Models." Describes the Llama 1 architecture choices (RMSNorm, SwiGLU, RoPE) that essentially every modern open base has kept. https://arxiv.org/abs/2302.13971
- **Touvron et al., 2023b** — "Llama 2: Open Foundation and Fine-Tuned Chat Models." Llama 2 architecture plus grouped-query attention at 34B / 70B. https://arxiv.org/abs/2307.09288
- **Shazeer, 2020** — "GLU Variants Improve Transformer." The SwiGLU FFN used by Llama, Mistral, Qwen, Gemma. https://arxiv.org/abs/2002.05202
- **Zhang & Sennrich, 2019** — "Root Mean Square Layer Normalization." The RMSNorm used in place of LayerNorm in modern decoder-only stacks. https://arxiv.org/abs/1910.07467
- **Ainslie et al., 2023** — "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." Grouped-query attention. https://arxiv.org/abs/2305.13245

### PEFT method papers

- **Hu et al., 2021** — "LoRA: Low-Rank Adaptation of Large Language Models." The original LoRA. https://arxiv.org/abs/2106.09685
- **Dettmers et al., 2023** — "QLoRA: Efficient Finetuning of Quantized LLMs." 4-bit NF4 quantization + LoRA. https://arxiv.org/abs/2305.14314
- **Liu et al., 2022** — "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning." Introduces IA3. https://arxiv.org/abs/2205.05638
- **Li & Liang, 2021** — "Prefix-Tuning: Optimizing Continuous Prompts for Generation." https://arxiv.org/abs/2101.00190
- **Lester et al., 2021** — "The Power of Scale for Parameter-Efficient Prompt Tuning." Prompt tuning. https://arxiv.org/abs/2104.08691
- **Houlsby et al., 2019** — "Parameter-Efficient Transfer Learning for NLP." The original adapter modules. https://arxiv.org/abs/1902.00751
- **Ben Zaken et al., 2022** — "BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models." https://arxiv.org/abs/2106.10199

### Official docs and code you will actually use

- Hugging Face `transformers` — https://huggingface.co/docs/transformers
- Hugging Face `peft` — https://huggingface.co/docs/peft
- The `LlamaForCausalLM` source in `transformers` is the single best reference for a modern decoder-only block: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

## Chapter 2 — Tokenization and chat templates

### Papers

- **Sennrich, Haddow & Birch, 2016** — "Neural Machine Translation of Rare Words with Subword Units." The BPE paper. https://arxiv.org/abs/1508.07909
- **Kudo & Richardson, 2018** — "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing." https://arxiv.org/abs/1808.06226
- **Kudo, 2018** — "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates." The Unigram algorithm used by SentencePiece Unigram. https://arxiv.org/abs/1804.10959

### Official tokenizer implementations and specs

- Hugging Face `tokenizers` (Rust + Python) — https://huggingface.co/docs/tokenizers
- OpenAI `tiktoken` — https://github.com/openai/tiktoken
- Google `sentencepiece` — https://github.com/google/sentencepiece

### Chat templates

- Hugging Face chat-template documentation — https://huggingface.co/docs/transformers/main/en/chat_templating
- The Llama 3 Instruct prompt format, from the official Llama repo — https://github.com/meta-llama/llama-models/blob/main/models/llama3/README.md
- Qwen 2 / 2.5 chat template (ChatML variant) — https://qwen.readthedocs.io/en/latest/getting_started/concepts.html
- Original Stanford Alpaca template (historical reference) — https://github.com/tatsu-lab/stanford_alpaca

## Chapter 3 — Positional encodings

- **Su et al., 2021 / 2023** — "RoFormer: Enhanced Transformer with Rotary Position Embedding." The RoPE paper. https://arxiv.org/abs/2104.09864
- **Press, Smith & Lewis, 2022** — "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." The ALiBi paper. https://arxiv.org/abs/2108.12409
- **Chen et al., 2023** — "Extending Context Window of Large Language Models via Positional Interpolation." Introduces PI. https://arxiv.org/abs/2306.15595
- **Peng et al., 2023** — "YaRN: Efficient Context Window Extension of Large Language Models." https://arxiv.org/abs/2309.00071
- Meta's Llama 3.1 blog post explains the shipped `rope_scaling` configuration and long-context evaluation methodology — https://ai.meta.com/blog/meta-llama-3-1/
- Hugging Face `transformers` RoPE-scaling documentation — https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaConfig.rope_scaling

## Chapter 4 — Post-training pipeline

### Foundational pipeline papers

- **Ouyang et al., 2022** — "Training language models to follow instructions with human feedback." InstructGPT — the paper that operationalized the SFT → RM → RLHF three-stage recipe. https://arxiv.org/abs/2203.02155
- **Christiano et al., 2017** — "Deep Reinforcement Learning from Human Preferences." The RLHF origin paper. https://arxiv.org/abs/1706.03741
- **Rafailov et al., 2023** — "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." DPO. https://arxiv.org/abs/2305.18290
- **Hong et al., 2024** — "ORPO: Monolithic Preference Optimization without Reference Model." https://arxiv.org/abs/2403.07691
- **Ethayarajh et al., 2024** — "KTO: Model Alignment as Prospect Theoretic Optimization." https://arxiv.org/abs/2402.01306
- **Meng et al., 2024** — "SimPO: Simple Preference Optimization with a Reference-Free Reward." https://arxiv.org/abs/2405.14734
- **Bai et al., 2022** — "Constitutional AI: Harmlessness from AI Feedback." Anthropic's RLAIF recipe. https://arxiv.org/abs/2212.08073
- **Gururangan et al., 2020** — "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks." Origin of the DAPT terminology used in the pipeline map. https://arxiv.org/abs/2004.10964

### Quantization papers (reference for chapter 4's serving box; depth in mod-109)

- **Frantar et al., 2022** — "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." https://arxiv.org/abs/2210.17323
- **Lin et al., 2023** — "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." https://arxiv.org/abs/2306.00978
- **Dettmers et al., 2022** — "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." The bitsandbytes 8-bit quantization paper. https://arxiv.org/abs/2208.07339

### Official tooling docs

- TRL (Transformers Reinforcement Learning) — https://huggingface.co/docs/trl
- Axolotl — https://github.com/axolotl-ai-cloud/axolotl
- Unsloth — https://github.com/unslothai/unsloth
- vLLM — https://docs.vllm.ai
- Text Generation Inference (TGI) — https://huggingface.co/docs/text-generation-inference
- llama.cpp / GGUF — https://github.com/ggerganov/llama.cpp
- lm-evaluation-harness — https://github.com/EleutherAI/lm-evaluation-harness

## Suggested reading order for this module

1. Skim the Vaswani, Radford, and Touvron (Llama 2) papers to anchor the architecture vocabulary.
2. Read the LoRA paper carefully; skim QLoRA.
3. Read chapters 1–4 of this module in order alongside those papers.
4. Read the RoPE and ALiBi papers when you get to chapter 3; skim the PI / YaRN papers if you plan to do long-context work.
5. Read InstructGPT and DPO before you start mod-103; the rest of the preference-optimization papers are covered in mod-105.

<!-- needs-research: on the next autonomous cycle, confirm the current URLs still resolve and add version pins for the tooling documentation links (transformers, peft, trl, vllm) so the reader lands on the version this module was authored against. -->
