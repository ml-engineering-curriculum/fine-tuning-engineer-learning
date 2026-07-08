# mod-105-preference-optimization: Preference Optimization: RLHF, DPO, ORPO, KTO, and the Modern Family

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 18 hours

## Learning objectives

- Construct preference-pair datasets (chosen / rejected) with quality and length controls
- Train a reward model and use it inside a TRL PPOTrainer pipeline (classical RLHF)
- Run DPO end-to-end and compare to PPO-RLHF on the same SFT base
- Apply ORPO, KTO, IPO, SimPO, and GRPO variants and explain when each fits
- Detect and mitigate reward hacking, length bias, KL collapse, and reference-model drift

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
