# mod-105-preference-optimization: Preference Optimization — RLHF, DPO, ORPO, KTO, and the Modern Family

**Estimated effort:** 18 hours
**Position in the track:** fifth module. Consumes the SFT checkpoint from mod-103 (or the merged LoRA output from mod-104) and produces a preference-tuned model that a downstream module (safety alignment in mod-106, distributed training in mod-107, evaluation in mod-108) can pick up. Feeds directly into mod-106, whose safety-alignment recipes are largely DPO / KTO variants over a curated safety-preference set.

Mod-103 taught you how to imitate a curated distribution with cross-entropy. Mod-104 taught you how to do it cheaply. This module teaches you how to move a model past imitation and toward *preferred* behavior — the shape of "helpful and harmless" that InstructGPT introduced in 2022 and that every current production chat model still owes its shape to. The through-line is that preference optimization is *SFT plus a comparison signal*: the training loop looks almost identical, the loss changes, and a handful of load-bearing hyperparameters (β, KL, length normalization, reference model) become the entire game.

## Learning objectives

- Construct preference-pair datasets (chosen / rejected) with quality and length controls
- Train a reward model and use it inside a TRL `PPOTrainer` pipeline (classical RLHF)
- Run DPO end-to-end and compare to PPO-RLHF on the same SFT base
- Apply ORPO, KTO, IPO, SimPO, and GRPO variants and explain when each fits
- Detect and mitigate reward hacking, length bias, KL collapse, and reference-model drift

## Lecture chapters

1. [`01-from-sft-to-preferences.md`](01-from-sft-to-preferences.md) — From SFT to Preferences. Why SFT alone is not enough, the Bradley-Terry model for pairwise preferences, the KL-constrained RL objective InstructGPT introduced, and the family map from PPO-RLHF through DPO to the reference-free "modern family."
2. [`02-preference-pair-datasets.md`](02-preference-pair-datasets.md) — Preference Pair Dataset Construction. Where preference pairs come from (human, LLM-judge, existing corpora like HH-RLHF / UltraFeedback / SHP), the quality and length controls that decide whether the run will learn what you meant, and the on-disk format TRL trainers expect.
3. [`03-reward-modeling.md`](03-reward-modeling.md) — Reward Modeling. Turning preference pairs into a scalar-reward model via the Bradley-Terry loss, sequence-classifier heads atop a base, TRL `RewardTrainer` mechanics, RM eval (accuracy, calibration), and the failure modes that make an RM look good on validation but fail in RL.
4. [`04-ppo-rlhf.md`](04-ppo-rlhf.md) — Classical PPO-RLHF with TRL. The four models in the loop (policy, value, reward, reference), rollouts and advantages, the KL-penalized reward, TRL's `PPOTrainer` scaffolding, why PPO-RLHF is fussy in practice, and how to read its diagnostics.
5. [`05-dpo.md`](05-dpo.md) — Direct Preference Optimization. Rafailov et al.'s derivation of DPO from RLHF's KL-constrained objective, the closed-form policy that skips the reward model, `DPOTrainer` in practice, the β knob and its trade-offs, and DPO's implicit-reward view.
6. [`06-modern-preference-family.md`](06-modern-preference-family.md) — ORPO, KTO, IPO, SimPO, GRPO. What each method adds or removes relative to DPO — reference-free (ORPO / SimPO), unpaired-label (KTO), regularized (IPO), group-relative for reasoning (GRPO) — and a decision matrix over data shape × compute budget × task.
7. [`07-failure-modes.md`](07-failure-modes.md) — Reward Hacking, Length Bias, KL Collapse, Reference Drift. The four preference-optimization-specific failure modes; their signatures on reward curves, KL curves, and eval; and the first mitigation to try for each.

## Exercises

Each exercise is a hands-on companion to one or more chapters. They live in [`exercises/`](exercises/).

- [`exercise-01-preference-pair-dataset-construction.md`](exercises/exercise-01-preference-pair-dataset-construction.md) — assemble a preference dataset from an existing open corpus, apply quality and length controls, and produce a reproducible dataset card.
- [`exercise-02-reward-model-training.md`](exercises/exercise-02-reward-model-training.md) — train a small reward model with TRL `RewardTrainer` on the dataset from exercise 01 and evaluate its pairwise accuracy and length bias.
- [`exercise-03-ppo-rlhf-run.md`](exercises/exercise-03-ppo-rlhf-run.md) — run a classical PPO-RLHF loop that combines the SFT base, the reward model from exercise 02, and TRL's `PPOTrainer`, and diagnose the run from its KL / reward / value-loss curves.
- [`exercise-04-dpo-run-and-comparison-to-ppo.md`](exercises/exercise-04-dpo-run-and-comparison-to-ppo.md) — run DPO on the same SFT base and preference set, produce a side-by-side comparison to the PPO run from exercise 03, and defend a choice for a specified deployment scenario.
- [`exercise-05-orpo-kto-simpo-grpo-survey-experiment.md`](exercises/exercise-05-orpo-kto-simpo-grpo-survey-experiment.md) — pick three of ORPO / KTO / SimPO / GRPO, run each on the same base, and produce a comparison matrix + a written recommendation.

## Labs and quizzes

Long-form labs land in [`labs/`](labs/) and the knowledge-check quiz in [`quizzes/`](quizzes/) on subsequent authoring cycles.

## Resources

See [`resources.md`](resources.md) for primary papers (InstructGPT, DPO, ORPO, KTO, IPO, SimPO, GRPO, and the RLHF / reward-hacking literature), official documentation (TRL, `trl` trainers, `peft`), and reference recipes cited across the chapters.

## What comes next

- **mod-106 — Safety Alignment Tuning** applies most of this module's machinery to a safety-preference set: refusal training, red-team-driven DPO, and safe-generation KTO.
- **mod-107 — Distributed Fine-Tuning** picks up the multi-GPU story for PPO and DPO, both of which have their own scaling footguns beyond what SFT hit.
- **mod-108 — Evaluating Fine-Tuned Models** covers the pairwise-preference evals (AlpacaEval, MT-Bench, LMSYS Arena Hard) that this module's checkpoints ultimately get judged by.
- **mod-111 — Post-Training Systems Design** situates preference optimization as one component of a full post-training platform.
