# Resources for mod-106-safety-alignment-tuning (Safety and Alignment Tuning)

Curated primary references. Prefer these — the paper of record, the official framework documentation, the current tool repository — over blog summaries when you go deep. Every claim the chapters cite lands here.

## Chapter 1 — The Safety Alignment Problem

### Papers

- **Weidinger et al., 2021** — "Ethical and social risks of harm from Language Models." DeepMind. The reference survey chapters cite for a six-category harm taxonomy. https://arxiv.org/abs/2112.04359
- **Askell et al., 2021** — "A General Language Assistant as a Laboratory for Alignment." Introduces the helpful / harmless / honest (HHH) triad the module keeps returning to. https://arxiv.org/abs/2112.00861
- **Bai et al., 2022a** — "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." Anthropic's HH-RLHF paper; source of both the corpus and the operational Anthropic harmlessness taxonomy chapter 1 summarizes. https://arxiv.org/abs/2204.05862
- **Röttger et al., 2023** — "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviors in Large Language Models." The first systematic over-refusal benchmark; cited whenever the chapters name "over-refusal / false refusal." https://arxiv.org/abs/2308.01263
- **Cui et al., 2024** — "OR-Bench: An Over-Refusal Benchmark for Large Language Models." The 80k over-refusal test-prompt corpus; the "large-enough to split" over-refusal set the chapters recommend. https://arxiv.org/abs/2405.20947
- **Wallace et al., 2024** — "The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions." The reference for the safe-completion response class and for training the model to weight system-prompt policy above user-turn overrides. https://arxiv.org/abs/2404.13208

### Policy and taxonomy references

- **MLCommons AILuminate** benchmark and 12-hazard taxonomy — https://mlcommons.org/benchmarks/ai-luminate/
- **Anthropic Usage Policies** — https://www.anthropic.com/legal/aup
- **OpenAI Usage Policies** — https://openai.com/policies/usage-policies/
- **Meta Llama Acceptable Use Policy** — https://www.llama.com/llama3/use-policy/
- **Google Generative AI Prohibited Use Policy** — https://policies.google.com/terms/generative-ai/use-policy

## Chapter 2 — Refusal and Harm-Reduction Datasets

### Papers behind the source corpora

- The Bai et al. 2022a HH-RLHF paper (cited above). The `harmless-base` split is the module's primary open safety-preference source.
- **Ji et al., 2023** — "BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset." Behind PKU-SafeRLHF and BeaverTails. https://arxiv.org/abs/2307.04657
- **Wang et al., 2023** — "Do-Not-Answer: A Dataset for Evaluating Safeguards in LLMs." 939 prompts across 12 harm subtypes. https://arxiv.org/abs/2308.13387
- **Bhardwaj and Poria, 2023** — "Red-Teaming Large Language Models Using Chain of Utterances for Safety-Alignment" (HarmfulQA). https://arxiv.org/abs/2308.09662
- **Zhang et al., 2024** — "SafetyBench: Evaluating the Safety of Large Language Models with Multiple Choice Questions." https://arxiv.org/abs/2309.07045
- **Ghosh et al., 2024** — "AEGIS: Online Adaptive AI Content Safety Moderation with Ensemble of LLM Experts." The paper behind NVIDIA's Aegis content-safety dataset and classifier. https://arxiv.org/abs/2404.05993
- **Zou et al., 2023** — "Universal and Transferable Adversarial Attacks on Aligned Language Models." Source of AdvBench and the GCG attack. https://arxiv.org/abs/2307.15043
- **Mazeika et al., 2024** — "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal." The 400-behavior, seven-category attack corpus with its own judge model. https://arxiv.org/abs/2402.04249
- **Chao et al., 2024** — "JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models." 100 harmful + 100 benign behaviors with reference responses and a leaderboard. https://arxiv.org/abs/2404.01318
- **Inan et al., 2023** — "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations." Source of the Llama Guard classifier and its taxonomy. https://arxiv.org/abs/2312.06674
- **Han et al., 2024** — "WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs." Behind WildGuardMix and the WildGuard classifier. https://arxiv.org/abs/2406.18495
- **Lambert et al., 2024** — "Tülu 3: Pushing Frontiers in Open Language Model Post-Training." Documents Tülu 3's safety mix and blend ratios chapter 2 references. https://arxiv.org/abs/2411.15124

### Datasets referenced in the chapter (Hugging Face Hub)

- `Anthropic/hh-rlhf` — https://huggingface.co/datasets/Anthropic/hh-rlhf
- `PKU-Alignment/PKU-SafeRLHF` — https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF
- `PKU-Alignment/BeaverTails` — https://huggingface.co/datasets/PKU-Alignment/BeaverTails
- `LibrAI/do-not-answer` — https://huggingface.co/datasets/LibrAI/do-not-answer
- `allenai/wildguardmix` — https://huggingface.co/datasets/allenai/wildguardmix
- `walledai/AdvBench` — https://huggingface.co/datasets/walledai/AdvBench
- `walledai/HarmBench` — https://huggingface.co/datasets/walledai/HarmBench
- `JailbreakBench/JBB-Behaviors` — https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors
- `declare-lab/HarmfulQA` — https://huggingface.co/datasets/declare-lab/HarmfulQA
- `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` — https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0

### Model / dataset cards referenced

- **Meta Llama 3 model card** — https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md
- **Meta Llama Guard 3 model card** — https://huggingface.co/meta-llama/Llama-Guard-3-8B
- **Anthropic Claude system cards** (Claude 3, 3.5, and later) — https://www.anthropic.com/news
- **Ai2 Tülu 3 project page** — https://allenai.org/tulu

## Chapter 3 — Constitutional AI and RLAIF

### Papers

- **Bai et al., 2022b** — "Constitutional AI: Harmlessness from AI Feedback." Anthropic. The two-stage SL-CAI + RL-CAI recipe the chapter dissects, and the source of the published example constitution. https://arxiv.org/abs/2212.08073
- **Anthropic, 2023** — "Collective Constitutional AI: Aligning a Language Model with Public Input." The crowdsourced-constitution companion write-up. https://www.anthropic.com/news/collective-constitutional-ai-aligning-a-language-model-with-public-input
- **Lee et al., 2023** — "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback." The paper that formally names RLAIF and measures it against RLHF on summarization and helpful-dialogue tasks. https://arxiv.org/abs/2309.00267
- **Sun et al., 2023** — "SELF-ALIGN: Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision" (Dromedary). The 16-principle + 195-seed bootstrapping variant. https://arxiv.org/abs/2305.03047
- **Wang et al., 2022** — "Self-Instruct: Aligning Language Models with Self-Generated Instructions." The instruction-generation-side companion pattern. https://arxiv.org/abs/2212.10560
- **Madaan et al., 2023** — "Self-Refine: Iterative Refinement with Self-Feedback." Inference-time critique-and-revise. https://arxiv.org/abs/2303.17651
- **Shinn et al., 2023** — "Reflexion: Language Agents with Verbal Reinforcement Learning." The related inference-time reflection-loop pattern. https://arxiv.org/abs/2303.11366
- **Yu et al., 2024** — "RLAIF-V: Aligning MLLMs through Open-Source AI Feedback for Super GPT-4V Trustworthiness." The multimodal CAI variant. https://arxiv.org/abs/2405.17220
- **Kim et al., 2024** — "Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models." The reference open judge model the chapters cite as an alternative to closed frontier judges. https://arxiv.org/abs/2405.01535
- **Jiang et al., 2023** — "LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion" (source of PairRM). https://arxiv.org/abs/2306.02561

### Framework references

- **TRL `DPOTrainer` / `KTOTrainer`** — https://huggingface.co/docs/trl/en/dpo_trainer and https://huggingface.co/docs/trl/en/kto_trainer (the trainers CAI-generated pairs feed into)
- **Hugging Face chat templating** — https://huggingface.co/docs/transformers/main/en/chat_templating
- **vLLM** — https://github.com/vllm-project/vllm (batched generation infrastructure for the SL-CAI / RL-CAI loops)
- **Text Generation Inference (TGI)** — https://github.com/huggingface/text-generation-inference
- **Prometheus-2** — https://huggingface.co/prometheus-eval/prometheus-7b-v2.0
- **PairRM** — https://huggingface.co/llm-blender/PairRM

## Chapter 4 — Red-Team Data to Training-Ready Signal

### Papers

- **Ganguli et al., 2022** — "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned." The reference paper on internal red-team intake shape and cadence. https://arxiv.org/abs/2209.07858
- **Perez et al., 2022** — "Red Teaming Language Models with Language Models." The foundational automated-red-teaming paper. https://arxiv.org/abs/2202.03286
- **Wei et al., 2023** — "Jailbroken: How Does LLM Safety Training Fail?" The reference paper for the jailbreak-family taxonomy chapter 4 organizes attack categories around. https://arxiv.org/abs/2307.02483
- **Chao et al., 2023** — "Jailbreaking Black Box Large Language Models in Twenty Queries" (PAIR). The reference automated-black-box attack. https://arxiv.org/abs/2310.08419
- **Mehrotra et al., 2023** — "Tree of Attacks: Jailbreaking Black-Box LLMs Automatically" (TAP). Tree-search extension of PAIR. https://arxiv.org/abs/2312.02119
- The Zou et al. GCG paper (cited above) — universal, transferable adversarial suffixes.
- **Liu et al., 2024** — "AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models." Readable jailbreak generation via genetic algorithm. https://arxiv.org/abs/2310.04451
- **Jiang et al., 2024a** — "WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models" (WildJailbreak). https://arxiv.org/abs/2406.18510
- **Jiang et al., 2024b** — "ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs." https://arxiv.org/abs/2402.11753
- **Russinovich et al., 2024** — "Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack." https://arxiv.org/abs/2404.01833
- The Wallace et al. 2024 Instruction Hierarchy paper (cited above) — the reference training pattern for instruction-hierarchy defenses.

### Attack-corpus and tooling references

- **HarmBench evaluation code and attack scripts** — https://github.com/centerforaisafety/HarmBench
- **JailbreakBench reference implementation** — https://github.com/JailbreakBench/jailbreakbench
- **PAIR reference implementation** — https://github.com/patrickrchao/JailbreakingLLMs
- **GCG reference implementation** — https://github.com/llm-attacks/llm-attacks
- **AutoDAN reference implementation** — https://github.com/SheltonLiu-N/AutoDAN
- **garak** (NVIDIA / Leon Derczynski) — general LLM vulnerability scanner covering many of the above attack families. https://github.com/NVIDIA/garak
- **PyRIT** (Microsoft) — Python Risk Identification Tool for generative AI; automates red-team probing across multiple attack strategies. https://github.com/Azure/PyRIT

### Standards and operational references

- **OWASP Top 10 for Large Language Model Applications** — https://genai.owasp.org/llm-top-10/ (referenced for LLM01: prompt injection)
- **MITRE ATLAS** — adversarial threat landscape for AI systems. https://atlas.mitre.org/
- **Anthropic HackerOne bug bounty program** — https://hackerone.com/anthropic
- **OpenAI bug bounty program** — https://bugcrowd.com/openai
- **Google Vulnerability Reward Program (AI focus)** — https://bughunters.google.com/about/rules/google-friends/6625378258649088/abuse-vulnerability-reward-program-rules

## Chapter 5 — Safety Evals: Refusal, Over-Refusal, Jailbreak Resistance, Harm Adherence

### Papers

- The Mazeika et al. HarmBench paper (cited above).
- The Chao et al. JailbreakBench paper (cited above).
- The Wang et al. Do-Not-Answer paper (cited above).
- The Röttger et al. XSTest paper (cited above).
- The Cui et al. OR-Bench paper (cited above).
- The Inan et al. Llama Guard paper (cited above).
- The Han et al. WildGuard paper (cited above).
- The Chao et al. PAIR paper (cited above).
- The Mehrotra et al. TAP paper (cited above).
- The Zou et al. GCG paper (cited above).
- The Liu et al. AutoDAN paper (cited above).
- The Russinovich et al. Crescendo paper (cited above).
- The Jiang et al. ArtPrompt paper (cited above).
- **Zheng et al., 2023** — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." The reference paper for LLM-as-judge methodology and MT-Bench. https://arxiv.org/abs/2306.05685
- **Gehman et al., 2020** — "RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models." https://arxiv.org/abs/2009.11462
- **Hartvigsen et al., 2022** — "ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection." https://arxiv.org/abs/2203.09509
- **Ouyang et al., 2022** — "Training language models to follow instructions with human feedback." The InstructGPT paper; source of the "alignment tax" framing chapter 5 uses for the capability-regression report. https://arxiv.org/abs/2203.02155
- **Greenblatt et al., 2024** — "Alignment Faking in Large Language Models." The Anthropic paper on model behavior that differs between believed-eval and believed-deployment contexts, cited under "what the dashboard cannot measure." https://arxiv.org/abs/2412.14093

### Judge / classifier models

- `meta-llama/Llama-Guard-3-8B` (and 1B / 11B-Vision siblings) — https://huggingface.co/meta-llama/Llama-Guard-3-8B
- `cais/HarmBench-Llama-2-13b-cls` and `cais/HarmBench-Mistral-7b-val-cls` — https://huggingface.co/cais/HarmBench-Llama-2-13b-cls
- `allenai/wildguard` — https://huggingface.co/allenai/wildguard
- **Google ShieldGemma** (2B / 9B / 27B) — https://huggingface.co/google/shieldgemma-9b and paper: https://arxiv.org/abs/2407.21772
- **NVIDIA NeMo Guardrails** — https://github.com/NVIDIA/NeMo-Guardrails
- **OpenAI Moderation API** — https://platform.openai.com/docs/guides/moderation
- **Google Perspective API** — https://perspectiveapi.com/
- **Azure AI Content Safety** — https://learn.microsoft.com/en-us/azure/ai-services/content-safety/

### Eval-set references (Hugging Face Hub)

- `walledai/HarmBench` — https://huggingface.co/datasets/walledai/HarmBench
- `JailbreakBench/JBB-Behaviors` — https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors
- `LibrAI/do-not-answer` — https://huggingface.co/datasets/LibrAI/do-not-answer
- `natolambert/xstest-v2-copy` — https://huggingface.co/datasets/natolambert/xstest-v2-copy (community mirror of Röttger et al.'s XSTest v2)
- `bench-llm/or-bench` — https://huggingface.co/datasets/bench-llm/or-bench
- `allenai/wildguardmix` — https://huggingface.co/datasets/allenai/wildguardmix
- `allenai/wildjailbreak` — https://huggingface.co/datasets/allenai/wildjailbreak
- `HuggingFaceH4/mt_bench_prompts` — https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts (for the general-capability report line)

### Reporting and statistical tooling

- **Wilson-score confidence interval** — see `statsmodels.stats.proportion.proportion_confint(..., method='wilson')` at https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html
- **LM Evaluation Harness** — https://github.com/EleutherAI/lm-evaluation-harness (background eval framework; several safety benchmarks are wired in)
- **Ai2 `safety-eval`** — https://github.com/allenai/safety-eval (Ai2's open safety-eval harness bundling many of the benchmarks above)

## Chapter 6 — Handoff to Security and Governance

### Papers and standards

- **Mitchell et al., 2019** — "Model Cards for Model Reporting." The reference paper for the model-card artifact. https://arxiv.org/abs/1810.03993
- The Wallace et al. 2024 Instruction Hierarchy paper (cited above) — training pattern for system-prompt / tool-output override defenses.
- The Zou et al. GCG paper (cited above) — persistent adversarial-suffix attacks, cited under "escalate rather than train against."
- The Ganguli et al. 2022 red-team paper (cited above) — operational cadence for red-team ops.
- The Greenblatt et al. 2024 alignment-faking paper (cited above).
- The OWASP Top 10 for LLM Applications (cited above) — LLM01 prompt injection is the enterprise vulnerability chapter 6 flags.

### Regulatory frameworks and voluntary commitments

- **NIST AI Risk Management Framework (AI RMF 1.0, NIST AI 100-1, 2023)** — https://www.nist.gov/itl/ai-risk-management-framework
- **NIST Generative AI Profile (NIST AI 600-1)** — https://airc.nist.gov/AI_RMF_Knowledge_Base/AI_RMF/Foundational_Information/2-sec-characteristics
- **EU AI Act** (Regulation (EU) 2024/1689) — https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- **US Executive Order 14110** — "Safe, Secure, and Trustworthy Development and Use of AI" (October 30, 2023) — https://www.federalregister.gov/documents/2023/11/01/2023-24283/safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence <!-- needs-research: EO 14110 was revoked by EO 14148 on 2025-01-20; confirm the current federal AI-executive-order landscape before quoting compliance requirements to a governance reviewer. -->
- **UK AI Safety Institute (now UK AI Security Institute) evaluations** — https://www.aisi.gov.uk/
- **ISO/IEC 42001:2023** — AI management systems standard. https://www.iso.org/standard/81230.html
- **MLCommons AILuminate** benchmark (cited above) — the industry-published safety benchmark referenced under "third-party evaluations."

### Voluntary lab commitments (RSP-style documents)

- **Anthropic Responsible Scaling Policy** — https://www.anthropic.com/rsp-updates
- **OpenAI Preparedness Framework** — https://openai.com/preparedness/
- **Google DeepMind Frontier Safety Framework** — https://deepmind.google/discover/blog/updating-the-frontier-safety-framework/

### Runtime guardrails and moderation

- All Llama Guard, WildGuard, ShieldGemma, NeMo Guardrails, OpenAI Moderation, Perspective, and Azure AI Content Safety references from chapter 5 (cited above) apply.
- **Meta Prompt Guard** — https://huggingface.co/meta-llama/Prompt-Guard-86M (small input-side prompt-injection classifier)
- **Rebuff** — https://github.com/protectai/rebuff (open-source prompt-injection detector)
- **Anthropic constitutional classifiers** — https://www.anthropic.com/research/constitutional-classifiers

### Content provenance (adjacent, chapter 6 names for pointer only)

- **C2PA (Coalition for Content Provenance and Authenticity)** — https://c2pa.org/
- **SynthID** (Google) — https://deepmind.google/technologies/synthid/

## Reference open pipelines and tooling

- **Hugging Face Alignment Handbook** — SFT + DPO recipes that the safety mix from exercise 01 blends into. https://github.com/huggingface/alignment-handbook
- **Ai2 OpenInstruct / Tülu 3** — full open post-training pipeline (SFT, DPO, PPO, safety) and public safety training sets. https://github.com/allenai/open-instruct
- **Axolotl** — YAML-driven trainer that consumes the exercise-01 and exercise-03 dataset shapes directly. https://github.com/axolotl-ai-cloud/axolotl
- **LLaMA-Factory** — all-in-one PEFT + preference-tuning framework, another consumer of the safety-tuning data shapes. https://github.com/hiyouga/LLaMA-Factory
- **HarmBench evaluation code** — the canonical reference for reproducing the HarmBench numbers reported across the field. https://github.com/centerforaisafety/HarmBench
- **JailbreakBench evaluation code** — https://github.com/JailbreakBench/jailbreakbench
- **Ai2 `safety-eval`** — bundle of safety benchmarks the chapter-5 dashboard reuses. https://github.com/allenai/safety-eval
- **`garak`** — general vulnerability scanner covering many attack strategies from chapter 4. https://github.com/NVIDIA/garak
- **`PyRIT`** — Microsoft's Python Risk Identification Tool. https://github.com/Azure/PyRIT

## Suggested reading order for this module

1. Read Askell et al. 2021 first — the HHH triad is under every subsequent chapter and it is short.
2. Skim the Weidinger et al. survey — it is the taxonomy vocabulary; you need to know the categories to read the corpora.
3. Read the Bai et al. 2022a HH-RLHF paper (or at least §3–§4) before chapter 2. Sets the shape of both the dataset and the response-class choices.
4. Read the Röttger et al. XSTest paper before chapter 2's over-refusal section. Short; the specific benign-adversarial construction pattern is repeated everywhere downstream.
5. Read the Bai et al. 2022b Constitutional AI paper before chapter 3. §2–§3 (SL-CAI) and §4 (RL-CAI); the appendix lists the published example constitution.
6. Read the Ganguli et al. 2022 red-team paper before chapter 4. The intake and iteration cadence in the chapter comes directly from this write-up.
7. Read the Wei et al. 2023 "Jailbroken" paper alongside chapter 4's attack-taxonomy section. Its failure-mode analysis is the reason the taxonomy has the shape it does.
8. Read the Mazeika et al. HarmBench paper and the Chao et al. JailbreakBench paper before exercise 04. Their evaluation protocols are what the exercise reproduces.
9. Read the Zheng et al. MT-Bench / LLM-as-judge paper before wiring any LLM judge in exercise 04. Position bias, length bias, and sycophancy in judges are covered in §5–§6.
10. Read the Wallace et al. 2024 Instruction Hierarchy paper before chapter 6. It is the reference for the training-vs-runtime handoff on system-prompt policy.
11. Skim the NIST AI RMF and the EU AI Act preamble before chapter 6's regulatory-frameworks section. You need only enough to know what a governance reviewer will ask; the depth belongs to the governance track.

<!-- needs-research: on the next authoring cycle, add version pins for the frameworks referenced here (transformers, TRL, vLLM, Llama-Guard-3, WildGuard, HarmBench evaluation code) matching the versions this module was authored against. Also verify the URLs still resolve — the alignment-handbook, open-instruct, and HarmBench repos have restructured before, and some of the government AI-policy URLs shift with each US administration. -->
