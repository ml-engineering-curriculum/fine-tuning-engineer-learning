# The Post-Training Pipeline, End to End

**Estimated read:** 30 minutes.
**You should be able to, by the end:** name every stage between "someone released a base model" and "we serve an adapter in production," describe the artifact that flows out of each stage, and know which module of this track owns each stage.

## The map

```
        (upstream, not the fine-tuner's job)
                       │
                       ▼
   ┌─────────────────────────────────────────┐
   │           Pretraining                    │  base model checkpoint  ← inputs to this track
   └─────────────────────────────────────────┘
                       │
                       ▼
   ┌─────────────────────────────────────────┐
   │  (optional) Continued Pretraining /      │  domain-adapted base
   │   Domain-Adaptive Pretraining (DAPT)     │
   └─────────────────────────────────────────┘
                       │
                       ▼
   ┌─────────────────────────────────────────┐
   │        Supervised Fine-Tuning (SFT)      │  SFT checkpoint / adapter
   │   full-parameter OR LoRA/QLoRA/PEFT      │
   └─────────────────────────────────────────┘
                       │
                       ▼
   ┌─────────────────────────────────────────┐
   │      Preference / Reward Optimization    │  preference-aligned checkpoint
   │   RLHF (PPO) · DPO · ORPO · KTO · GRPO   │
   └─────────────────────────────────────────┘
                       │
                       ▼
   ┌─────────────────────────────────────────┐
   │           Safety / Alignment tuning      │  safety-tuned checkpoint
   │  refusal training, Constitutional / RLAIF│
   └─────────────────────────────────────────┘
                       │
                       ▼
   ┌─────────────────────────────────────────┐
   │           Evaluation gates               │  release-candidate decision
   │ academic bench · held-out task · safety  │
   └─────────────────────────────────────────┘
                       │
                       ▼
   ┌─────────────────────────────────────────┐
   │          Quantization & export           │  serving artifact
   │ GPTQ · AWQ · bitsandbytes · GGUF · merge │
   └─────────────────────────────────────────┘
                       │
                       ▼
   ┌─────────────────────────────────────────┐
   │              Serving                     │  live endpoint + runbook
   │        vLLM · TGI · llama.cpp            │
   └─────────────────────────────────────────┘
```

Every arrow in that diagram is a **stage contract**: the artifact leaving one stage is the input to the next. If a stage's artifact is malformed (bad chat template, missing config metadata, wrong tokenizer version), every subsequent stage inherits the problem. Half the incident post-mortems in fine-tuning shops trace to a stage-contract violation at an earlier step.

## Stage by stage

### Pretraining (upstream)

A base-model provider (Meta, Mistral, Alibaba, Google, DeepSeek, Microsoft, small labs) runs next-token prediction over trillions of tokens on hundreds to thousands of accelerators for weeks. The output is a **base model** — a checkpoint that speaks fluent text but does not yet follow instructions, refuse harmful requests, or hold a conversation. The base is your input.

- **You do not own this stage.** Its cost is 5–7 figures per model and its methodology is at the frontier of large-scale distributed systems research.
- **What you inherit from it:** the tokenizer (chapter 2), the positional encoding scheme (chapter 3), the architecture and parameter count, the pretraining data mix (only partially disclosed for open models), and the known capabilities and biases of the base.
- **What you need to check before adopting it:** the license, the vocabulary and language coverage, any known safety issues in the base, and the config for `rope_theta` and `max_position_embeddings`.

### Continued pretraining / domain-adaptive pretraining (optional)

For narrow domains — biomedical, legal, low-resource languages, proprietary code — a shop sometimes runs an additional next-token-prediction pass on domain data before any instruction tuning. This is **continued pretraining** or **DAPT** (Gururangan et al., 2020).

- Consumes: base checkpoint + domain corpus of raw text (not instruction-formatted).
- Produces: a domain-adapted base.
- Risks: catastrophic forgetting of general capabilities, and (if the data is not decontaminated against eval sets) benchmark inflation.
- Owned in this track by mod-102 and mod-103.

Most real-world post-training projects skip this stage and go straight from pretrained base to SFT — the ROI on DAPT is unclear except in genuinely low-resource domains.

### Supervised Fine-Tuning (SFT)

The first stage where the fine-tuner is unambiguously in charge. Input: a base (possibly DAPT'd), a curated **instruction / conversation dataset**, and a chat template. Output: a checkpoint (or LoRA adapter) that follows instructions and completes conversations.

- **Objective:** standard next-token cross-entropy, but usually with **loss masking** so that only the assistant's tokens contribute to the gradient. The user's messages and structural tokens are supervised but ignored — you do not want the model learning to mimic user prompts.
- **Tooling in this track:** TRL's `SFTTrainer`, Axolotl, Unsloth (mod-103). PEFT integration via the `peft` library for LoRA/QLoRA (mod-104).
- **Dataset engineering:** owned by mod-102 (curation, dedup, decontamination, chat-template design, synthetic data).
- **Common failure modes:** template drift (chapter 2), overfitting on small datasets, catastrophic forgetting of general capabilities, template leak (the model emits `<|eot_id|>` when it shouldn't or the reverse). All covered in mod-103.

An SFT-only model is often production-usable. The next stage is only necessary when you need behavior that cannot be demonstrated in single "correct" responses — helpfulness, style, refusal quality — where you have preference judgments rather than gold labels.

### Preference / reward optimization

You have SFT-adapted the model to answer, but "answer well" is not a next-token label — it is a judgment about which of two candidate answers is better. Preference optimization stages consume **preference pairs** (`prompt`, `chosen`, `rejected`) and shape the model to produce more chosen-like and less rejected-like outputs.

The main methods you will meet, in rough chronological order:

- **RLHF via PPO** (Christiano et al., 2017; Ouyang et al., 2022 / InstructGPT). Train a reward model on human preferences, then optimize the base with PPO against that reward with a KL penalty back to the SFT reference. Powerful, expensive, brittle. TRL's `PPOTrainer` implements this.
- **DPO** — Direct Preference Optimization (Rafailov et al., 2023). Reframes the RLHF objective as a supervised loss on preference pairs, skipping the reward model. Dominant open-model default because it is cheaper and simpler than PPO with competitive quality.
- **ORPO** (Hong et al., 2024) — folds preference optimization into SFT itself, one stage instead of two.
- **KTO** — Kahneman-Tversky Optimization (Ethayarajh et al., 2024) — works with unpaired "this response was good / bad" signal rather than pairs.
- **SimPO** (Meng et al., 2024) — DPO variant with a length-normalized reference-free objective.
- **GRPO** — Group Relative Policy Optimization, used by DeepSeek in their R1 recipe. Optimizes against relative rewards within a sampled group; important in reasoning-focused post-training.

Owned by mod-105.

- **Consumes:** SFT checkpoint + preference dataset (`prompt`, `chosen`, `rejected`) + (for PPO) a reward model + (for PPO / DPO) a frozen reference model.
- **Produces:** a preference-optimized checkpoint or adapter.
- **Common failure modes:** reward hacking (PPO), length bias (all methods, DPO especially), KL collapse where the policy drifts too far from the reference, and preference dataset contamination or annotator disagreement. Covered in mod-105.

### Safety / alignment tuning

Not always a separate stage — some safety data is folded into SFT or DPO — but a good fine-tuner treats it as a distinct concern with its own metrics. Consumes refusal data, red-team data, and (optionally) Constitutional AI / RLAIF self-critique data (Bai et al., 2022). Produces a safety-tuned checkpoint that meets refusal-rate, over-refusal, jailbreak-resistance, and policy-adherence targets.

Owned by mod-106. Hands off to the ML-security, governance, and risk tracks for depth beyond the fine-tuner's remit.

### Evaluation gates

Threaded through every stage above, not a single point-in-time step. You run:

- **Academic benchmarks** via `lm-evaluation-harness` — MMLU, ARC, GSM8K, HumanEval, TruthfulQA, IFEval.
- **Task-specific held-out evals** tied to the actual production use case.
- **LLM-as-judge** evaluations at MT-Bench granularity for open-ended quality.
- **Safety evaluations** — refusal rate, over-refusal, jailbreak batteries.
- **Contamination checks** to ensure eval sets are not leaked into training.
- **Regression checks** against the base model so you can prove you did not silently break something to gain elsewhere.

The output is a **release-candidate decision**: adapter A vs adapter B, is A worth shipping over B? Owned by mod-108.

### Quantization and export

The trained model is BF16 or FP16. The serving stack usually wants something smaller. This stage takes a trained checkpoint and produces a **serving artifact** — a smaller weight file plus any metadata required to load it.

Common paths:

- **Merge the LoRA adapter into the base** and serve the merged checkpoint. Simple, but loses the ability to hot-swap adapters.
- **Serve the LoRA adapter alongside the base** with a runtime that supports it — vLLM's multi-LoRA feature, TGI's LoRA support. Enables per-tenant adapters.
- **Post-training quantization** — GPTQ (Frantar et al., 2022), AWQ (Lin et al., 2023), bitsandbytes 8-bit / 4-bit, or GGUF for llama.cpp. Trades a small quality delta for a large VRAM saving.

Owned by mod-109.

### Serving

Deploy the artifact to a runtime — vLLM, TGI, llama.cpp — and give the application layer an endpoint. The runbook here covers latency SLOs, throughput sizing, KV-cache memory budgets, multi-tenant routing (if applicable), and the observability hooks to catch quality regressions in production.

Owned by mod-109 for the serving mechanics and mod-111 for the systems-design view (fine-tune vs. RAG vs. prompt engineering, adapter registry, rollback plan, model card, governance handoff).

### Managed fine-tuning APIs (a parallel branch)

For teams that do not want to run their own training, OpenAI, Anthropic, Google, Mistral, Cohere, AWS Bedrock, and Azure AI Foundry offer **managed fine-tuning APIs**: upload a JSONL of examples, receive a fine-tuned model ID. Mod-110 covers the fair-comparison harness, cost / data-residency / IP tradeoffs, and the managed-vs-self-hosted decision.

Managed APIs collapse the stages above into a single stage from the caller's perspective, but the same concerns still apply — chat template consistency, eval discipline, safety measurement — just with less control over the internals.

## What the fine-tuner controls at each stage

| Stage | Fine-tuner owns | Track module | Artifact out |
|---|---|---|---|
| Pretraining | — (upstream) | inherited | base checkpoint + tokenizer + config |
| DAPT | data curation, run config | mod-103 (optional) | domain-adapted base |
| SFT | dataset, template, PEFT config, hyperparams | mod-102, 103, 104 | SFT checkpoint / adapter |
| Preference optimization | pairs, reward model, method choice | mod-105 | preference-optimized checkpoint |
| Safety tuning | refusal data, red-team ingestion | mod-106 | safety-tuned checkpoint |
| Evaluation | eval suite composition, gates | mod-108 | release decision |
| Quantization | method choice, quality-loss tolerance | mod-109 | serving artifact |
| Serving | runtime choice, runbook | mod-109, mod-111 | live endpoint |

## The through-line: reproducibility and provenance

Every arrow in the pipeline map should be reproducible from a manifest — data hashes, seeds, checkpoint hashes, config diffs. Every stage should emit enough metadata that a downstream stage or governance reviewer (mod-111) can answer "what went into this?" The industry name for the collected metadata is the **model card**, and the industry norm is that a professional fine-tuner ships one with every release.

- The dataset used at each stage is content-hashed and its licenses are catalogued.
- The base checkpoint is pinned by revision hash, not just by name.
- The tokenizer version is pinned alongside the model.
- The eval suite version and results are recorded per stage.
- The safety evaluation results are recorded and referenced in the card.

Ship without these and the release is not defensible to security, governance, or risk partners — which is the discipline the capstone in mod-111 makes you rehearse.

## Exercise pointer

`exercises/exercise-04-post-training-pipeline-map.md` has you produce a real, one-page pipeline diagram for a hypothetical customer-support chatbot project — including the data sources, methods, evaluation gates, and go/no-go criteria at each stage — and defend the choices against a review checklist.

## Summary

- Pretraining is upstream. You inherit a base + tokenizer + config; you do not own this stage.
- SFT is the first fine-tuner-owned stage. It shapes format, style, and task-following on instruction/conversation data.
- Preference optimization (DPO / PPO / ORPO / KTO / GRPO) shapes behavior against pairwise or scalar judgments where "correct" is a matter of preference.
- Safety tuning is a distinct concern with its own dataset and metrics, even when technically implemented as SFT or DPO data.
- Evaluation runs through every stage and gates release.
- Quantization and export produce the serving artifact; serving deploys it under a runbook.
- Every arrow is a stage contract. Every artifact needs enough metadata to reconstruct the previous stage — the model card is where that metadata lives.
