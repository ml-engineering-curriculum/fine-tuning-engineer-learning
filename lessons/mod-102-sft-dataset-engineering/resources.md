# Resources for mod-102-sft-dataset-engineering

Curated primary references. Prefer these sources — the paper of record, the official dataset card, the current tool documentation — over blog-post summaries when you go deep.

## Chapter 1 — Curation and quality filtering

### Papers and dataset reports

- **Zhou et al., 2023** — "LIMA: Less Is More for Alignment." The reference for the quality-over-quantity effect in SFT data. https://arxiv.org/abs/2305.11206
- **Taori et al., 2023** — "Alpaca: A Strong, Replicable Instruction-Following Model." Reference distillation run from `text-davinci-003`. https://crfm.stanford.edu/2023/03/13/alpaca.html
- **Köpf et al., 2023** — "OpenAssistant Conversations — Democratizing Large Language Model Alignment." The OASST1 paper; a full end-to-end description of a crowd-sourced curation pipeline. https://arxiv.org/abs/2304.07327
- **Rajani et al., 2023** — the No Robots dataset card. https://huggingface.co/datasets/HuggingFaceH4/no_robots
- **Chen et al., 2023** — "AlpaGasus: Training a Better Alpaca with Fewer Data." Judge-model quality filtering on Alpaca. https://arxiv.org/abs/2307.08701
- **Li et al., 2024** — "From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning." Introduces the Instruction-Following Difficulty (IFD) metric. https://arxiv.org/abs/2308.12032

### Official tooling and datasets

- Hugging Face `datasets` documentation — https://huggingface.co/docs/datasets
- OpenAssistant / OASST1 dataset — https://huggingface.co/datasets/OpenAssistant/oasst1
- No Robots dataset — https://huggingface.co/datasets/HuggingFaceH4/no_robots
- LIMA dataset — https://huggingface.co/datasets/GAIR/lima
- Detoxify (toxicity classifier) — https://github.com/unitaryai/detoxify
- Microsoft Presidio (PII detection / redaction) — https://microsoft.github.io/presidio/
- fastText `lid.176` language identifier — https://fasttext.cc/docs/en/language-identification.html

## Chapter 2 — Deduplication

### Papers

- **Lee et al., 2022** — "Deduplicating Training Data Makes Language Models Better." The canonical case for dedup in language-model corpora. https://arxiv.org/abs/2107.06499
- **Broder, 1997** — "On the resemblance and containment of documents." The MinHash paper of record. https://www.cs.princeton.edu/courses/archive/spring13/cos598C/broder97resemblance.pdf
- **Charikar, 2002** — "Similarity Estimation Techniques from Rounding Algorithms." The SimHash paper. https://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/CharikarEstim.pdf
- **Mining of Massive Datasets, ch. 3 (Leskovec, Rajaraman, Ullman)** — the standard textbook treatment of shingling + MinHash + LSH. http://www.mmds.org

### Tooling

- `datasketch` (MinHash + LSH in Python) — https://github.com/ekzhu/datasketch
- `text-dedup` (bigcode-project — production-grade dedup pipeline used on The Stack) — https://github.com/ChenghaoMou/text-dedup
- FAISS (dense-vector nearest-neighbor for semantic dedup) — https://github.com/facebookresearch/faiss
- `sentence-transformers` — https://www.sbert.net

## Chapter 3 — Eval-set decontamination

### Papers with published decontamination methodology

- **Brown et al., 2020** — "Language Models are Few-Shot Learners" (GPT-3). Appendix C is the reference decontamination recipe. https://arxiv.org/abs/2005.14165
- **Touvron et al., 2023b** — the Llama 2 paper, decontamination section. https://arxiv.org/abs/2307.09288
- **Meta Llama 3 model card** — decontamination methodology as reported in the model release. https://github.com/meta-llama/llama-models/blob/main/models/llama3/MODEL_CARD.md

### The four eval suites

- **Hendrycks et al., 2021** — "Measuring Massive Multitask Language Understanding" (MMLU). https://arxiv.org/abs/2009.03300 — dataset: https://huggingface.co/datasets/cais/mmlu
- **Chen et al., 2021** — "Evaluating Large Language Models Trained on Code" (HumanEval). https://arxiv.org/abs/2107.03374 — dataset: https://huggingface.co/datasets/openai_humaneval
- **Cobbe et al., 2021** — "Training Verifiers to Solve Math Word Problems" (GSM8K). https://arxiv.org/abs/2110.14168 — dataset: https://huggingface.co/datasets/gsm8k
- **Zheng et al., 2023** — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (MT-Bench). https://arxiv.org/abs/2306.05685 — prompts: https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge

### Contamination-tracking tools

- `bigcode-project/bigcode-evaluation-harness` — includes decontamination utilities for code evals. https://github.com/bigcode-project/bigcode-evaluation-harness
- EleutherAI `lm-evaluation-harness` — the standard eval framework; documents contamination-check helpers. https://github.com/EleutherAI/lm-evaluation-harness

## Chapter 4 — Chat templates and multi-turn

- Hugging Face chat-template documentation — https://huggingface.co/docs/transformers/main/en/chat_templating
- TRL `SFTTrainer` and `DataCollatorForCompletionOnlyLM` — https://huggingface.co/docs/trl
- Axolotl (opinionated SFT config framework with well-tested multi-turn masking) — https://github.com/axolotl-ai-cloud/axolotl
- **Dao et al., 2022** — "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (packed-sequence attention with `cu_seqlens`). https://arxiv.org/abs/2205.14135
- **Dao, 2023** — "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." https://arxiv.org/abs/2307.08691
- The Llama 3 Instruct prompt format — https://github.com/meta-llama/llama-models/blob/main/models/llama3/README.md
- Qwen 2 / 2.5 chat template (ChatML variant) — https://qwen.readthedocs.io

## Chapter 5 — Synthetic data (Self-Instruct / Evol-Instruct)

### Papers

- **Wang et al., 2023** — "Self-Instruct: Aligning Language Models with Self-Generated Instructions." https://arxiv.org/abs/2212.10560
- **Xu et al., 2023** — "WizardLM: Empowering Large Language Models to Follow Complex Instructions." (Evol-Instruct). https://arxiv.org/abs/2304.12244
- **Luo et al., 2023** — "WizardCoder: Empowering Code Large Language Models with Evol-Instruct." Evol-Instruct applied to code. https://arxiv.org/abs/2306.08568
- **Mukherjee et al., 2023** — "Orca: Progressive Learning from Complex Explanation Traces of GPT-4." Rationale-distillation approach. https://arxiv.org/abs/2306.02707
- **Ding et al., 2023** — "Enhancing Chat Language Models by Scaling High-quality Instructional Conversations" (UltraChat). https://arxiv.org/abs/2305.14233
- **Sun et al., 2023** — "Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision" (SELF-ALIGN). https://arxiv.org/abs/2305.03047
- **Yuan et al., 2024** — "Self-Rewarding Language Models." https://arxiv.org/abs/2401.10020
- **Wei et al., 2024 / NVIDIA** — the Nemotron-4 340B technical report; a synthetic-data pipeline released explicitly to be a permissive teacher. https://arxiv.org/abs/2406.11704

### Reference datasets

- Alpaca — https://huggingface.co/datasets/tatsu-lab/alpaca
- WizardLM Evol-Instruct — https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k
- OpenHermes (Teknium's curation of synthetic instruction data) — https://huggingface.co/datasets/teknium/OpenHermes-2.5
- SlimOrca — https://huggingface.co/datasets/Open-Orca/SlimOrca

## Chapter 6 — Dataset mixing

- **Ivison et al., 2023** — "Camels in a Changing Climate: Enhancing LM Adaptation with Tülu 2." https://arxiv.org/abs/2311.10702
- **Lambert et al., 2024** — Tülu 3 technical report. https://arxiv.org/abs/2411.15124 — datasets: https://huggingface.co/collections/allenai/tulu-3
- **Xue et al., 2021** — "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer." Popularized temperature-adjusted sampling across sources of very different sizes. https://arxiv.org/abs/2010.11934
- **Meta Llama 3.1 model card / paper** — https://arxiv.org/abs/2407.21783
- **Röttger et al., 2023** — "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models." Measures over-refusal. https://arxiv.org/abs/2308.01263

## Chapter 7 — Licensing, provenance, and the model card

### Governance / documentation frameworks

- **Mitchell et al., 2019** — "Model Cards for Model Reporting." https://arxiv.org/abs/1810.03993
- **Gebru et al., 2018 / 2021** — "Datasheets for Datasets." https://arxiv.org/abs/1803.09010
- **Google Data Cards Playbook** — https://sites.research.google/datacardsplaybook/
- **Hugging Face model card documentation** — https://huggingface.co/docs/hub/model-cards
- **Hugging Face dataset card documentation** — https://huggingface.co/docs/hub/datasets-cards
- **BigScience BLOOM model card** — https://huggingface.co/bigscience/bloom — the reference for governance-heavy multi-org disclosure.
- **Le Scao et al., 2022** — the BLOOM technical report. https://arxiv.org/abs/2211.05100

### License references

- **SPDX License List** — https://spdx.org/licenses/ (the standard identifiers for OSS licenses).
- **Creative Commons license chooser** — https://creativecommons.org/choose/ (BY, BY-SA, BY-NC families).
- **Meta Llama 3 Community License** — https://github.com/meta-llama/llama-models/blob/main/models/llama3/LICENSE
- **Qwen license** — https://github.com/QwenLM/Qwen2.5/blob/main/LICENSE
- **NVIDIA Open Model License / Nemotron terms** — released with the Nemotron-4 340B family; see the model card on the HF Hub. https://huggingface.co/nvidia/Nemotron-4-340B-Instruct

<!-- needs-research: on the next research cycle, verify the current OpenAI Terms of Use and Anthropic Commercial Terms sections that govern outputs used to train competing models, and add the precise URLs + effective dates. These terms shift more often than an authored curriculum can track, and citing them here without a live check risks going stale. -->

### Reference open-model cards worth reading end to end

- Llama 3 / 3.1 model cards (on `meta-llama/*` on the HF Hub).
- Tülu 3 model card (on `allenai/*`).
- BLOOM model card (on `bigscience/bloom`).
- Nemotron-4 340B Instruct model card (on `nvidia/*`).

## Suggested reading order for this module

1. Skim LIMA (Zhou et al.) and the OASST1 paper to anchor the "curation matters" and "documented curation" pictures.
2. Read chapters 1 and 2 alongside `text-dedup`'s README and the MinHash/LSH chapter of *Mining of Massive Datasets*.
3. Read Brown et al. appendix C (GPT-3 decontamination) before chapter 3, then chapter 3, then run exercise 02.
4. Skim the Self-Instruct paper before chapter 5; skim WizardLM (Evol-Instruct) if you plan to use it.
5. Read the Tülu 3 data card end to end before chapter 6 and exercise 05 — it is the reference for what a documented open mix looks like.
6. Read Model Cards for Model Reporting (Mitchell et al.) and one full model card of your choice before writing your own in exercise 05.

<!-- needs-research: on the next authoring cycle, add version pins for the tooling links (datasets, transformers, trl, axolotl) so future readers land on the version this module was authored against. -->
