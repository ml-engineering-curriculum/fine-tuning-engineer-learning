# Resources for mod-105-preference-optimization (Preference Optimization: RLHF, DPO, ORPO, KTO, and the Modern Family)

Curated primary references. Prefer these sources — the paper of record, the official framework documentation, the current tool repo — over blog-post summaries when you go deep. Where a chapter cites a claim, the citation lands here.

## Chapter 1 — From SFT to Preferences

### Papers

- **Bradley and Terry, 1952** — "Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons." *Biometrika* 39(3/4), 324–345. The 70-year-old paper that supplies the preference model every method in this module uses. https://doi.org/10.2307/2334029
- **Christiano et al., 2017** — "Deep Reinforcement Learning from Human Preferences." The paper that introduced the reward-model-plus-RL-against-it pattern that InstructGPT would later specialize to language. https://arxiv.org/abs/1706.03741
- **Ouyang et al., 2022** — "Training language models to follow instructions with human feedback." The InstructGPT paper. The reference framing for the full SFT → RM → PPO-RLHF pipeline this module dissects. https://arxiv.org/abs/2203.02155
- **Schulman et al., 2017** — "Proximal Policy Optimization Algorithms." The PPO paper — the RL algorithm every chapter-4 pipeline runs. https://arxiv.org/abs/1707.06347
- **Rafailov et al., 2023** — "Direct Preference Optimization: Your Language Model Is Secretly a Reward Model." The DPO paper. Read sections 3 (derivation) and 4 (analysis); the appendix on why both log-probs can drop is worth returning to when you hit chapter 5's failure mode. https://arxiv.org/abs/2305.18290
- **Bai et al., 2022** — "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." Anthropic's HH paper. The reference for a full RLHF pipeline with a helpful/harmless split, and the source of the HH-RLHF dataset. https://arxiv.org/abs/2204.05862
- **Touvron et al., 2023** — "Llama 2: Open Foundation and Fine-Tuned Chat Models." Section 3.2 documents Llama-2's SFT-then-iterative-RM-then-PPO pipeline, which the chapter cites as a canonical PPO-RLHF example. https://arxiv.org/abs/2307.09288

## Chapter 2 — Preference Pair Datasets

### Papers behind the source corpora

- **Cui et al., 2023** — "UltraFeedback: Boosting Language Models with High-quality Feedback." The paper behind the UltraFeedback dataset that most Zephyr/alignment-handbook DPO tutorials consume. https://arxiv.org/abs/2310.01377
- **Ethayarajh et al., 2022** — "Understanding Dataset Difficulty with V-Usable Information" (background) and the SHP corpus paper of record on the dataset card. The SHP dataset itself lives on the Hub. https://arxiv.org/abs/2110.08420
- **Ji et al., 2023** — "BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset." The paper behind PKU-SafeRLHF and related safety-focused preference sets. https://arxiv.org/abs/2307.04657
- **Zhu et al., 2023** — "Starling-7B: Improving LLM Helpfulness & Harmlessness with RLAIF" — the paper behind the Nectar preference dataset. https://arxiv.org/abs/2310.19678 and dataset: https://huggingface.co/datasets/berkeley-nest/Nectar
- **Köpf et al., 2023** — "OpenAssistant Conversations — Democratizing Large Language Model Alignment." The OpenAssistant conversational + preference corpus. https://arxiv.org/abs/2304.07327
- **Tunstall et al., 2023** — "Zephyr: Direct Distillation of LM Alignment." The recipe that pinned UltraFeedback-Binarized as the DPO reference dataset. https://arxiv.org/abs/2310.16944
- **Lambert et al., 2024** — "Tülu 3: Pushing Frontiers in Open Language Model Post-Training." Documents `allenai/tulu-3-pref-*` preference sets with per-domain coverage. https://arxiv.org/abs/2411.15124

### Datasets referenced in the chapter (Hugging Face Hub)

- `HuggingFaceH4/ultrafeedback_binarized` — https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
- `Anthropic/hh-rlhf` — https://huggingface.co/datasets/Anthropic/hh-rlhf
- `stanfordnlp/SHP` — https://huggingface.co/datasets/stanfordnlp/SHP
- `PKU-Alignment/PKU-SafeRLHF` — https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF
- `argilla/dpo-mix-7k` — https://huggingface.co/datasets/argilla/dpo-mix-7k
- `Intel/orca_dpo_pairs` — https://huggingface.co/datasets/Intel/orca_dpo_pairs
- `berkeley-nest/Nectar` — https://huggingface.co/datasets/berkeley-nest/Nectar
- `allenai/tulu-3-pref-personas-instruction-following` (and siblings) — https://huggingface.co/datasets?search=allenai%2Ftulu-3-pref

### Framework references

- Hugging Face **`datasets`** documentation — https://huggingface.co/docs/datasets
- Hugging Face **chat templating** documentation — https://huggingface.co/docs/transformers/main/en/chat_templating
- TRL **dataset format guide** (for preference-shaped datasets consumed by `DPOTrainer`, `KTOTrainer`, `ORPOTrainer`) — https://huggingface.co/docs/trl/en/dataset_formats

## Chapter 3 — Reward Modeling

### Papers

- The Ouyang et al. InstructGPT paper (cited above). Section 3.5 covers the RM training recipe.
- The Bai et al. HH-RLHF paper (cited above). Section 4 covers the RM.
- **Coste et al., 2023** — "Reward Model Ensembles Help Mitigate Overoptimization." The reference for the "ensemble RMs to fight reward hacking" mitigation. https://arxiv.org/abs/2310.02743
- **Gao, Schulman, and Hilton, 2023** — "Scaling Laws for Reward Model Overoptimization." The paper that measured RM-vs-policy-size scaling of the over-optimization curve. Foundational for chapter 3 and chapter 7. https://arxiv.org/abs/2210.10760
- **Singhal et al., 2023** — "A Long Way to Go: Investigating Length Correlations in RLHF." The paper of record for length bias in RMs; the source for the length-conditional accuracy diagnostic in exercise 02. https://arxiv.org/abs/2310.03716
- **Sharma et al., 2023** — "Towards Understanding Sycophancy in Language Models." The paper behind the sycophancy failure mode; cited as one of the RM audit prompts in exercise 02. https://arxiv.org/abs/2310.13548
- **Lambert et al., 2024** — "RewardBench: Evaluating Reward Models for Language Modeling." The public reward-model benchmark referenced in exercise 02's stretch goal. https://arxiv.org/abs/2403.13787

### Framework references

- TRL **`RewardTrainer`** guide — https://huggingface.co/docs/trl/en/reward_trainer
- Hugging Face **`AutoModelForSequenceClassification`** reference — https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForSequenceClassification
- `peft` **`TaskType.SEQ_CLS`** and `modules_to_save` documentation — https://huggingface.co/docs/peft/main/en/package_reference/peft_types and https://huggingface.co/docs/peft/main/en/package_reference/lora
- **RewardBench** repository — https://github.com/allenai/reward-bench

### Public RMs used as sanity references

- `sfairXC/FsfairX-LLaMA3-RM-v0.1` — https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1
- `RLHFlow/ArmoRM-Llama3-8B-v0.1` — https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1
- `OpenAssistant/reward-model-deberta-v3-large-v2` — https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2

## Chapter 4 — Classical PPO-RLHF with TRL

### Papers

- The Schulman et al. PPO paper (cited above).
- **Schulman et al., 2015** — "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (GAE). The advantage estimator PPO uses. https://arxiv.org/abs/1506.02438
- **Ziegler et al., 2019** — "Fine-Tuning Language Models from Human Preferences." The first paper to apply PPO to LM fine-tuning against a learned reward; the origin of the adaptive KL controller. https://arxiv.org/abs/1909.08593
- **Huang et al., 2024** — "The N+ Implementation Details of RLHF with PPO." The single most valuable reference for reproducing InstructGPT-style PPO-RLHF in practice; TRL's modern `PPOTrainer` follows this document closely. https://arxiv.org/abs/2403.17031
- **Ouyang et al., 2022** (InstructGPT) — reread §3.5 and §3.6 for the RL stage details.

### Framework references

- TRL **`PPOTrainer`** guide (modern rewrite) — https://huggingface.co/docs/trl/en/ppo_trainer
- TRL **`PPOConfig`** reference — same URL as above.
- TRL repository (see `examples/scripts/ppo/`) — https://github.com/huggingface/trl
- Hugging Face **`accelerate`** — https://huggingface.co/docs/accelerate (multi-device backing for PPO runs)

## Chapter 5 — Direct Preference Optimization

### Papers

- The Rafailov et al. DPO paper (cited above). Read the derivation in §4 and the appendix on gradient behavior.
- **Pal et al., 2024** — "Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive." Documents the "both log-probs drop" DPO failure mode and proposes fixes (DPO+SFT auxiliary term). https://arxiv.org/abs/2402.13228
- **Rafailov et al., 2024** — "From r to Q*: Your Language Model is Secretly a Q-Function." The paper that reframes DPO as token-level Q-learning; the reference the chapter alludes to when discussing DPO's implicit-reward view. https://arxiv.org/abs/2404.12358
- **Park et al., 2024** — "Disentangling Length from Quality in Direct Preference Optimization" (R-DPO). Introduces the length-regularized DPO variant cited in chapter 7. https://arxiv.org/abs/2403.19159

### Framework references

- TRL **`DPOTrainer`** guide — https://huggingface.co/docs/trl/en/dpo_trainer
- TRL **`DPOConfig`** reference (includes `loss_type` variants: sigmoid, ipo, simpo, hinge, etc.) — same URL.
- `peft` **disabled-adapter reference model trick** (used by `DPOTrainer` when `ref_model=None`) — https://huggingface.co/docs/peft/main/en/package_reference/peft_model#peft.PeftModel.disable_adapter
- **Hugging Face Alignment Handbook** (Zephyr / DPO recipes) — https://github.com/huggingface/alignment-handbook

## Chapter 6 — ORPO, KTO, IPO, SimPO, GRPO, CPO

### Papers

- **Azar et al., 2023** — "A General Theoretical Paradigm to Understand Learning from Human Preferences" (IPO). https://arxiv.org/abs/2310.12036
- **Ethayarajh et al., 2024** — "KTO: Model Alignment as Prospect Theoretic Optimization." https://arxiv.org/abs/2402.01306
- **Hong et al., 2024** — "ORPO: Monolithic Preference Optimization without Reference Model." https://arxiv.org/abs/2403.07691
- **Meng et al., 2024** — "SimPO: Simple Preference Optimization with a Reference-Free Reward." https://arxiv.org/abs/2405.14734
- **Shao et al., 2024** — "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." The GRPO paper. Read §3.2 for the objective and §4 for the training details. https://arxiv.org/abs/2402.03300
- **DeepSeek-AI et al., 2025** — "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." The paper that made GRPO famous outside the DeepSeekMath community. https://arxiv.org/abs/2501.12948
- **Xu et al., 2024** — "Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation" (CPO). https://arxiv.org/abs/2401.08417
- **Kahneman and Tversky, 1979** — "Prospect Theory: An Analysis of Decision under Risk." *Econometrica* 47(2), 263–291. The 1979 paper whose loss-vs-gain asymmetry KTO adopts. https://doi.org/10.2307/1914185

### Framework references

- TRL **`ORPOTrainer`** guide — https://huggingface.co/docs/trl/en/orpo_trainer
- TRL **`KTOTrainer`** guide — https://huggingface.co/docs/trl/en/kto_trainer
- TRL **`CPOTrainer`** guide — https://huggingface.co/docs/trl/en/cpo_trainer
- TRL **`GRPOTrainer`** guide — https://huggingface.co/docs/trl/en/grpo_trainer
- TRL **SimPO** (via `DPOTrainer` with `loss_type="simpo"`) — see the DPO trainer guide's `loss_type` section, https://huggingface.co/docs/trl/en/dpo_trainer#loss-functions
- SimPO reference implementation (Meng et al.) — https://github.com/princeton-nlp/SimPO
- ORPO reference implementation (Hong et al.) — https://github.com/xfactlab/orpo

<!-- needs-research: GRPO's `GRPOTrainer` argument surface has evolved in the TRL 0.14–0.20 series; before quoting `num_generations` / `beta` defaults verbatim in a shipped training script, verify against the exact TRL version you pinned. -->

## Chapter 7 — Reward Hacking, Length Bias, KL Collapse, Reference Drift

### Papers

- The Gao, Schulman, and Hilton over-optimization paper (cited above).
- The Singhal et al. length-correlations paper (cited above).
- **Skalse et al., 2022** — "Defining and Characterizing Reward Hacking." The formal reference for what "reward hacking" means as a mathematical property, not just an anecdote. https://arxiv.org/abs/2209.13085
- **Goodhart, 1975** — "Problems of Monetary Management: The U.K. Experience." The origin of Goodhart's law. Not directly on ML but the reference every reward-hacking paper cites in its introduction. Reprinted in *Papers in Monetary Economics*, Reserve Bank of Australia, 1975.
- **Dubois et al., 2024** — "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators." The paper that formalized length-controlled pairwise evals; the reference for "much of RLHF's apparent gain is length gain in disguise." https://arxiv.org/abs/2404.04475
- The Park et al. R-DPO paper (cited above).
- **Ziegler et al., 2019** (cited above) — adaptive KL controller.
- The Huang et al. "N+ Implementation Details" paper (cited above) — extensive KL-related diagnostics.
- The Coste et al. RM-ensembles paper (cited above).
- **Anthropic, 2022** — "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022b). Referenced for the RLAIF setup and constitutional-critique data as a mitigation for sycophancy. https://arxiv.org/abs/2212.08073

### Framework and diagnostic references

- TRL's per-trainer logged metrics (see each trainer's guide: `DPOTrainer`, `PPOTrainer`, `KTOTrainer`, etc.). Links above.
- **Weights & Biases** — https://docs.wandb.ai (the log-heavy dashboarding referenced in chapter 7's instrumentation section)
- **TensorBoard** — https://pytorch.org/docs/stable/tensorboard.html
- **LM Evaluation Harness** — https://github.com/EleutherAI/lm-evaluation-harness (for the independent held-out eval that catches reward hacking; discussed in chapter 7)
- **MT-Bench / Arena-Hard-Auto** (Zheng et al., 2023) — https://arxiv.org/abs/2306.05685 and prompts at https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge — the pairwise-judged eval baseline used to detect reward hacking

## Reference recipes and open pipelines

- **Hugging Face Alignment Handbook** — Zephyr's SFT + DPO recipe; the canonical reproduction target for a DPO run on UltraFeedback. https://github.com/huggingface/alignment-handbook
- **Ai2 OpenInstruct / Tülu 3** — full open post-training pipeline including SFT, DPO, PPO, and evaluation. https://github.com/allenai/open-instruct
- **DeepSeek-R1 open recipes / community reproductions** — several open reproductions of DeepSeek-R1's GRPO stage; the `open-r1` project (Hugging Face) is a good starting point. https://github.com/huggingface/open-r1
- **Axolotl** — YAML-driven trainer covering SFT / DPO / KTO / ORPO / SimPO. https://github.com/axolotl-ai-cloud/axolotl
- **LLaMA-Factory** — all-in-one PEFT + preference-tuning framework covering DPO / KTO / ORPO / SimPO / PPO. https://github.com/hiyouga/LLaMA-Factory
- **OpenRLHF** — a distributed RLHF/RLVR framework designed for scale. https://github.com/OpenRLHF/OpenRLHF
- **`trlx`** — an earlier community RLHF library predating TRL's rewrite; still cited in some 2023–2024 papers. https://github.com/CarperAI/trlx

## Suggested reading order for this module

1. Skim the InstructGPT paper (Ouyang et al., 2022) end-to-end before reading chapter 1. It supplies the shape every subsequent chapter refines.
2. Read Bradley and Terry, 1952 (short) — it is genuinely the model behind DPO / IPO / ORPO / SimPO / KTO. Knowing the two-page argument sharpens every downstream loss.
3. Read the Rafailov et al. DPO paper (or at least §3 and §4) before chapter 5. The derivation is more legible once you can see the equations directly.
4. Skim the Huang et al. "N+ Implementation Details" paper before chapter 4 and exercise 03. It is the single best debugging reference for PPO-RLHF.
5. Read the Ethayarajh et al. KTO paper before chapter 6's KTO section. The Kahneman-Tversky background makes the asymmetric loss make sense.
6. Read Gao, Schulman, and Hilton, 2023 before chapter 7. It is short, precise, and the empirical bedrock under most of chapter 7's failure-mode framing.
7. Read the Singhal et al. length-correlations paper alongside chapter 7's length-bias section — many of the chapter's diagnostic tests come directly from that paper.
8. Optional but recommended: skim DeepSeek-R1 (2025) before exercise 05's GRPO track. It is the current paper of reference for what GRPO actually delivers at scale.

<!-- needs-research: on the next authoring cycle, add version pins for TRL, transformers, peft, and bitsandbytes matching the versions this module was authored against. Also verify the alignment-handbook and open-instruct URLs still point to the current default branches (both repos have restructured before). -->
