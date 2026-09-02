# Resources for mod-108-evaluating-fine-tuned-models

Curated primary references. Prefer these — the paper of record, the official docs, the current tool README — over blog-post summaries when you go deep. Versions and leaderboard task lists move; verify the current state before quoting a specific figure.

## Chapter 1 — The evaluation landscape

### Framing / meta-reviews

- **Chang et al., 2023** — "A Survey on Evaluation of Large Language Models." A broad taxonomy of eval approaches; useful as the mental-model paper before drilling into any specific benchmark. https://arxiv.org/abs/2307.03109
- **Liang et al., 2022** — "Holistic Evaluation of Language Models (HELM)." Stanford CRFM's framework for multi-dimensional, multi-scenario LLM eval. https://arxiv.org/abs/2211.09110
- **Srivastava et al., 2022** — "Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models (BIG-Bench)." The 200+ task benchmark that predates and complements the harness-standard six. https://arxiv.org/abs/2206.04615

### Official frameworks

- HELM main site — https://crfm.stanford.edu/helm/
- BIG-bench repository — https://github.com/google/BIG-bench

## Chapter 2 — Running EleutherAI `lm-evaluation-harness`

### Benchmark papers (the "six-task suite")

- **Hendrycks et al., 2021** — "Measuring Massive Multitask Language Understanding" (MMLU). The paper of record for the 57-subject multiple-choice benchmark. https://arxiv.org/abs/2009.03300
- **Clark et al., 2018** — "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge." ARC-Easy and ARC-Challenge. https://arxiv.org/abs/1803.05457
- **Cobbe et al., 2021** — "Training Verifiers to Solve Math Word Problems" (GSM8K). https://arxiv.org/abs/2110.14168
- **Chen et al., 2021** — "Evaluating Large Language Models Trained on Code" (HumanEval, pass@k). https://arxiv.org/abs/2107.03374
- **Lin et al., 2022** — "TruthfulQA: Measuring How Models Mimic Human Falsehoods." The mc1 / mc2 / gen modes and the "GPT-Judge" concept originate here. https://arxiv.org/abs/2109.07958
- **Zhou et al., 2023** — "Instruction-Following Evaluation for Large Language Models" (IFEval). Rule-based verifiable instruction following. https://arxiv.org/abs/2311.07911
- **Zellers et al., 2019** — "HellaSwag: Can a Machine Really Finish Your Sentence?" Common-sense NLI, often paired with ARC. https://arxiv.org/abs/1905.07830
- **Sakaguchi et al., 2019** — "WinoGrande: An Adversarial Winograd Schema Challenge at Scale." Commonsense reasoning; the other frequent log-likelihood companion to MMLU. https://arxiv.org/abs/1907.10641

### Newer / v2-leaderboard tasks

- **Suzgun et al., 2022** — "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them" (BBH). https://arxiv.org/abs/2210.09261
- **Rein et al., 2023** — "GPQA: A Graduate-Level Google-Proof Q&A Benchmark." https://arxiv.org/abs/2311.12022
- **Sprague et al., 2024** — "MuSR: Testing the Limits of Chain-of-Thought with Multistep Soft Reasoning." https://arxiv.org/abs/2310.16049
- **Wang et al., 2024** — "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark." https://arxiv.org/abs/2406.01574
- **Hendrycks et al., 2021** — "Measuring Mathematical Problem Solving with the MATH Dataset" (MATH; MATH-Lvl-5 is the hardest subset used on the v2 leaderboard). https://arxiv.org/abs/2103.03874

### Official tooling

- EleutherAI `lm-evaluation-harness` repository (README, task catalog, backend docs) — https://github.com/EleutherAI/lm-evaluation-harness
- `lm-evaluation-harness` task README index — https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
- vLLM main docs (the `vllm` backend) — https://docs.vllm.ai/
- SGLang main docs — https://github.com/sgl-project/sglang
- BigCode Evaluation Harness (HumanEval, MBPP, MultiPL-E, and the code-eval task family) — https://github.com/bigcode-project/bigcode-evaluation-harness

### Leaderboards (verify current task lists before citing)

- Hugging Face Open LLM Leaderboard (v2 tasks: IFEval, BBH, MATH-Lvl-5, GPQA, MuSR, MMLU-Pro) — https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- Open LLM Leaderboard v2 blog post — https://huggingface.co/spaces/open-llm-leaderboard/blog

<!-- needs-research: verify the current v2 task list against the leaderboard page before quoting any specific set in a training report; the mix has evolved. -->

## Chapter 3 — Task-specific held-out evals

### Papers and framing

- **Ribeiro et al., 2020** — "Beyond Accuracy: Behavioral Testing of NLP models with CheckList." The reference for capability-oriented, behavioral testing that translates directly to task-specific eval design. https://arxiv.org/abs/2005.04118
- **Bowman & Dahl, 2021** — "What Will it Take to Fix Benchmarking in Natural Language Understanding?" On why held-out benchmarks decay and how to build ones that don't. https://arxiv.org/abs/2104.02145
- **Bragg et al., 2021** — "FLEX: Unifying Evaluation for Few-Shot NLP." The stratification and confidence-interval accounting that this chapter's sizing rules-of-thumb draw on. https://arxiv.org/abs/2107.07170
- **Muennighoff et al., 2022** — "MTEB: Massive Text Embedding Benchmark." Referenced as an example of a well-designed domain-specific benchmark suite; useful when cold-starting a retrieval-adjacent eval. https://arxiv.org/abs/2210.07316

### Domain-specific evals (adaptable to your held-out set)

- **Jin et al., 2019** — "PubMedQA: A Dataset for Biomedical Research Question Answering." https://arxiv.org/abs/1909.06146
- **Guha et al., 2023** — "LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models." https://arxiv.org/abs/2308.11462
- **Cassano et al., 2022** — "MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation." Multi-language HumanEval; a template for adapting an academic benchmark to your language mix. https://arxiv.org/abs/2208.08227

### Labeling / dataset tooling

- Argilla — https://argilla.io/
- Label Studio — https://labelstud.io/
- doccano — https://github.com/doccano/doccano
- Prodigy (paid) — https://prodi.gy/

### Statistical sizing background

- scikit-learn "Cross-validation" and metrics guides — https://scikit-learn.org/stable/modules/model_evaluation.html
- SciPy statistics reference (`bootstrap`, `mcnemar`, `wilcoxon`) — https://docs.scipy.org/doc/scipy/reference/stats.html

## Chapter 4 — LLM-as-judge protocols

### Foundational papers

- **Zheng et al., 2023** — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." The paper of record for LLM-as-judge, MT-Bench single-answer and pairwise protocols, and the four-bias discussion. https://arxiv.org/abs/2306.05685
- **Li et al., 2023** — "AlpacaEval: An Automatic Evaluator of Instruction-following Models." The predecessor / companion to MT-Bench and the origin of the pairwise-win-rate-vs-reference pattern. https://github.com/tatsu-lab/alpaca_eval
- **Dubois et al., 2024** — "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators." The published statistical correction for verbosity bias that chapter 4 recommends adopting for release-gate pairwise numbers. https://arxiv.org/abs/2404.04475

### Bias / reliability studies

- **Chiang & Lee, 2023** — "Can Large Language Models Be an Alternative to Human Evaluations?" Empirical study of judge-vs-human agreement across tasks; source for several bias caveats. https://arxiv.org/abs/2305.01937
- **Wang et al., 2023** — "Large Language Models are not Fair Evaluators." Documents position bias and prescribes the randomize-and-average control. https://arxiv.org/abs/2305.17926
- **Panickssery et al., 2024** — "LLM Evaluators Recognize and Favor Their Own Generations." The self-preference paper. https://arxiv.org/abs/2404.13076
- **Zeng et al., 2024** — "Evaluating Large Language Models at Evaluating Instruction Following." Evidence that many judges systematically fail on specific rubric dimensions. https://arxiv.org/abs/2310.07641

### Judge-model options

- **Kim et al., 2024** — "Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models." Open judge model tuned for rubric grading. https://arxiv.org/abs/2405.01535
- **Wang et al., 2023** — "PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization." Another open judge model. https://arxiv.org/abs/2306.05087
- **Zhu et al., 2023** — "JudgeLM: Fine-tuned Large Language Models are Scalable Judges." https://arxiv.org/abs/2310.17631

### Official tooling and recipes

- MT-Bench code (in FastChat) — https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge
- AlpacaEval — https://github.com/tatsu-lab/alpaca_eval
- Chatbot Arena / LMSYS Arena leaderboard — https://lmarena.ai/
- OpenAI Structured Outputs (JSON mode, function-schema judging) — https://platform.openai.com/docs/guides/structured-outputs
- Anthropic Messages API + tool-use for structured judging — https://docs.anthropic.com/en/docs/build-with-claude/tool-use

### Agreement metric references

- Cohen, 1960 — "A Coefficient of Agreement for Nominal Scales" (Cohen's κ). https://journals.sagepub.com/doi/10.1177/001316446002000104
- Krippendorff's alpha reference — https://en.wikipedia.org/wiki/Krippendorff%27s_alpha (for pointers to the primary papers)
- `scipy.stats.spearmanr` — https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
- `sklearn.metrics.cohen_kappa_score` — https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html

## Chapter 5 — Contamination detection

### Core papers

- **Brown et al., 2020** — "Language Models are Few-Shot Learners" (GPT-3). Appendix C — the first widely-cited discussion of benchmark contamination in a frontier-model paper. https://arxiv.org/abs/2005.14165
- **Sainz et al., 2023** — "NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for Each Benchmark." The survey that catalogs known contamination across academic evals. https://arxiv.org/abs/2310.18018
- **Oren et al., 2023** — "Proving Test Set Contamination in Black-Box Language Models." The membership-inference probe with formal statistical guarantees. https://arxiv.org/abs/2310.17623
- **Golchin & Surdeanu, 2023** — "Time Travel in LLMs: Tracing Data Contamination in Large Language Models." The guided-prompting probe and the canonical-continuation setup. https://arxiv.org/abs/2308.08493
- **Golchin & Surdeanu, 2024** — "Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models." A follow-up multiple-choice probe. https://arxiv.org/abs/2311.06233
- **Deng et al., 2024** — "Investigating Data Contamination in Modern Benchmarks for Large Language Models." Empirical measurements across many models and evals. https://arxiv.org/abs/2311.09783
- **Zhou et al., 2023** — "Don't Make Your LLM an Evaluation Benchmark Cheater." On overfitting-to-benchmark dynamics that adjacent contamination detection needs to guard against. https://arxiv.org/abs/2311.01964

### Sentence encoders (for the embedding-overlap lens)

- **Reimers & Gurevych, 2019** — "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." https://arxiv.org/abs/1908.10084
- Sentence-Transformers library — https://sbert.net/
- BGE embeddings (`BAAI/bge-large-en-v1.5`) — https://huggingface.co/BAAI/bge-large-en-v1.5
- E5 embeddings (`intfloat/e5-large-v2`) — https://huggingface.co/intfloat/e5-large-v2
- FAISS (approximate nearest-neighbor for large training corpora) — https://github.com/facebookresearch/faiss

### Model-card contamination-disclosure exemplars

- Llama 3 model card (contamination section) — https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md
- Llama 2 paper (§4.6, "Data Contamination") — https://arxiv.org/abs/2307.09288
- Mistral / Mixtral model cards — https://huggingface.co/mistralai
- Qwen 2.5 report — https://arxiv.org/abs/2412.15115
- Gemma 2 technical report — https://arxiv.org/abs/2408.00118

<!-- needs-research: on next authoring cycle, verify which of these model cards still contain a dedicated data-contamination section (the vendors' disclosure practices vary release-to-release). -->

## Chapter 6 — The release-gate eval suite and regression report

### Statistical treatment

- **Dror et al., 2018** — "The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing." Practical reference for choosing paired tests and CIs on NLP metrics. https://aclanthology.org/P18-1128/
- **Koehn, 2004** — "Statistical Significance Tests for Machine Translation Evaluation." Origin of the bootstrap-CI convention for score aggregates in NLP. https://aclanthology.org/W04-3250/
- **McNemar, 1947** — the original McNemar test paper. https://link.springer.com/article/10.1007/BF02295996
- SciPy `bootstrap` — https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
- SciPy `mcnemar` — https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.mcnemar.html
- SciPy `wilcoxon` — https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html

### Cost / batch-API references (verify current pricing before quoting)

- OpenAI Batch API — https://platform.openai.com/docs/guides/batch
- Anthropic Message Batches — https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
- Google Vertex AI batch prediction — https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#batch

### Experiment tracking and results storage

- Weights & Biases — https://docs.wandb.ai/
- MLflow — https://mlflow.org/docs/latest/index.html
- Aim — https://aimstack.io/
- Neptune.ai — https://docs.neptune.ai/

## Chapter 7 — Common pitfalls and handoff

### Chat template and tokenization

- Hugging Face chat-templating docs — https://huggingface.co/docs/transformers/chat_templating
- Hugging Face tokenizer `apply_chat_template` reference — https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.apply_chat_template
- `lm-evaluation-harness` chat-template / `fewshot_as_multiturn` guide — https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md

### Sampling and reproducibility

- vLLM sampling parameters reference — https://docs.vllm.ai/en/latest/dev/sampling_params.html
- vLLM deterministic-mode discussion (per-release, so check current README) — https://github.com/vllm-project/vllm
- Transformers `GenerationConfig` — https://huggingface.co/docs/transformers/main_classes/text_generation

### Safety-eval references (handoff to mod-106)

- **Mazeika et al., 2024** — "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal." https://arxiv.org/abs/2402.04249
- HarmBench repository — https://github.com/centerforaisafety/HarmBench
- **Röttger et al., 2023** — "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models." The reference for over-refusal (helpfulness under safety tuning). https://arxiv.org/abs/2308.01263
- **Zou et al., 2023** — "Universal and Transferable Adversarial Attacks on Aligned Language Models" (GCG). https://arxiv.org/abs/2307.15043

### Quantization handoff (to mod-109)

- **Frantar et al., 2022** — "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." https://arxiv.org/abs/2210.17323
- **Lin et al., 2023** — "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." https://arxiv.org/abs/2306.00978
- **Dettmers et al., 2022** — "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." https://arxiv.org/abs/2208.07339

### Managed-API handoff (to mod-110)

- OpenAI fine-tuning docs — https://platform.openai.com/docs/guides/fine-tuning
- OpenAI evals framework — https://github.com/openai/evals
- Anthropic evaluation guidance — https://docs.anthropic.com/en/docs/test-and-evaluate/eval-tool
- Google Vertex AI Gen AI Evaluation Service — https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview

## Reference model cards (for eval-reporting conventions)

Read the eval sections end-to-end when in doubt about how to present a number.

- Llama 3 (8B / 70B / 405B) model card — https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md
- Llama 3 technical report — https://arxiv.org/abs/2407.21783
- Mistral / Mixtral papers — https://arxiv.org/abs/2310.06825 (Mistral 7B) and https://arxiv.org/abs/2401.04088 (Mixtral of Experts)
- Qwen 2.5 technical report — https://arxiv.org/abs/2412.15115
- Gemma 2 technical report — https://arxiv.org/abs/2408.00118
- Phi-3 technical report — https://arxiv.org/abs/2404.14219
- DeepSeek-V3 technical report — https://arxiv.org/abs/2412.19437

## Suggested reading order for this module

1. Skim the MT-Bench paper (Zheng et al., 2023) §3–4 and Table 1. It sets the vocabulary the rest of the module uses (single-answer, pairwise, the four biases).
2. Read chapter 1 of this module, then read chapter 2 alongside the `lm-evaluation-harness` README and the MMLU / GSM8K / HumanEval papers' abstracts. Run exercise 01 before finishing chapter 2 if you can — the tool's ergonomics are the fastest route to intuition.
3. Read chapter 3, then read the CheckList paper (Ribeiro et al., 2020) sections 3–4. Both frame the eval as a *behavioral test suite* rather than an accuracy number.
4. Read chapter 4 alongside the MT-Bench paper and the length-controlled AlpacaEval paper (Dubois et al., 2024). Exercise 03 is the payoff.
5. Read chapter 5 alongside the Oren et al. (2023) contamination paper's introduction. Read the Llama 3 model card's contamination section as a reference for the disclosure format.
6. Read chapter 6 last of the core chapters — everything before it produces the numbers; chapter 6 is what turns them into a decision. Exercise 05 wires it end to end.
7. Return to chapter 7 as a reference the first time you hit a "the eval regressed but the model looks fine" mystery. Its checklist is the fastest triage tool in the module.

<!-- needs-research: on the next authoring cycle, add version pins for the `lm-evaluation-harness`, `alpaca_eval`, `bigcode-evaluation-harness`, `sentence-transformers`, `vllm`, and `sglang` links so future readers land on the version this module was authored against. Judge-model version tags (`gpt-4o-2024-08-06`, `claude-sonnet-4-5`, etc.) also drift and should be re-verified. -->
