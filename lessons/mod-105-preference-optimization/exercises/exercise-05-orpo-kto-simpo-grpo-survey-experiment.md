# exercise-05: ORPO / KTO / SimPO / GRPO Survey Experiment

**Estimated effort:** 5 hours (varies with how many methods you take on; GRPO is by far the slowest)
**Companion chapters:** [`06-modern-preference-family.md`](../06-modern-preference-family.md), [`07-failure-modes.md`](../07-failure-modes.md)

## Objective

Pick three of ORPO / KTO / SimPO / GRPO, run each on the same SFT base, produce a comparison matrix, and write a defended recommendation for a specified scenario. Exercise 04 taught you the "same base, one axis varied" muscle for the two-way DPO-vs-PPO comparison. This exercise scales that muscle to a four-way family survey and forces you to reason across methods that swap different pieces (reference model, paired-vs-unpaired data, on-policy sampling).

The point is *not* to identify a universal winner — chapter 6 argues explicitly there isn't one, and the current empirical picture (§ "Empirical picture across 2024–2025 papers" in chapter 6) is that each method wins in its own niche. The point is to feel the differences on the same starting conditions so that the next time a colleague asks "should we try SimPO on our chat data?" you have a run-informed answer, not a paper-of-the-week answer.

## Prerequisites

- The SFT model from mod-103 exercise 01 (or the same public SFT you used in exercises 03–04).
- The preference dataset from exercise 01 (`outputs/preference-set-v1/`). If you attempt KTO, you also need the **KTO-format variant** from exercise 01's stretch goal (`outputs/preference-set-v1-kto/`) or the ~10-minute conversion described in Part A below.
- The DPO run artifacts from exercise 04 (`outputs/dpo-run/`) — DPO is your baseline for every method here.
- `trl` ≥ 0.14 for `GRPOTrainer`. `trl` ≥ 0.9 is enough for `ORPOTrainer`, `KTOTrainer`, and `DPOTrainer` with `loss_type="simpo"`. Pin the version and record it in your report. (Chapter 6 flags GRPO's API is still evolving — take the pin seriously.)
- `peft`, `bitsandbytes`, `accelerate`, `datasets`, `transformers`.
- A GPU with ≥ 24 GB VRAM for QLoRA + LoRA runs on 1B–7B bases across the DPO-family (ORPO, KTO, SimPO). GRPO needs meaningfully more — plan on ≥ 40 GB for a 1B GRPO run because of the K-sample rollout memory (chapter 6 explains why).
- If you take on GRPO, you also need a **reward source** — either the reward model from exercise 02, or a rule-based verifier (see Part B/GRPO below).

## Requirements

### Part A — Pick your three methods

Pick three of the four:

- **ORPO** — reference-free, single-stage SFT+preference. `ORPOTrainer` / `ORPOConfig`. Chapter 6, "ORPO" section.
- **KTO** — reference-based, *unpaired* label per example. `KTOTrainer` / `KTOConfig`. Chapter 6, "KTO" section. Requires the unpaired data variant.
- **SimPO** — reference-free, length-normalized margin. `DPOTrainer` with `loss_type="simpo"` in modern TRL, or the paper's reference implementation. Chapter 6, "SimPO" section.
- **GRPO** — on-policy, group-relative advantages, no value model. `GRPOTrainer` / `GRPOConfig`. Chapter 6, "GRPO" section. This is the odd one out — data shape and reward source both differ.

Justify your picks in a short `picks.md` file (~150 words). Reference chapter 6's decision matrix and connect each pick to something concrete about your data or setup. "I picked ORPO because I'm memory-constrained; SimPO because my exercise-02 RM showed length bias; and GRPO because I want to feel on-policy training." Not "I picked the three that sounded most interesting."

**Constraints on the pick**: to make comparisons meaningful, at least two of your three picks must operate on the same paired preference data. GRPO is fine as the odd-one-out (different data shape is the whole point of surveying it).

### Part B — Run each method with matched settings

For each of the three, run on `outputs/preference-set-v1/train/` (or the KTO variant for KTO; or a prompts-only variant for GRPO). Use the same SFT base as your DPO baseline. Match hyperparameters wherever the methods agree:

- Same LR (adjust per method's paper recommendation — see below).
- Same LoRA config (`r`, `alpha`, `target_modules`) if using LoRA.
- Same effective batch size (`per_device_train_batch_size * gradient_accumulation_steps`).
- Same `num_train_epochs` and `max_length`.
- Same seed for prompt sampling.

Where methods differ meaningfully in their headline hyperparameter, use each paper's recommended defaults and cite the source in your notes:

- **ORPO** — `beta=0.1` in TRL's `ORPOConfig` (the odds-ratio weight `λ` in chapter 6's notation). Hong et al., 2024 uses 0.1 as their headline.
- **KTO** — `beta=0.1`, `desirable_weight ≈ 1.0`, `undesirable_weight ≈ (n_desirable / n_undesirable)` to balance the gradient (chapter 6 notes this). Ethayarajh et al., 2024's recommendation.
- **SimPO** — `beta=2.0` and `simpo_gamma=1.0` — Meng et al., 2024's defaults. Note these are *very* different from DPO's `beta=0.1`; the meaning of β differs in the length-normalized formulation.
- **GRPO** — `num_generations=4` (K=4 samples per prompt is the floor; K=8 is more standard but doubles rollout cost). `beta=0.04` for the KL penalty against the reference. `max_completion_length` sized to your task (256–1024).

Each method's config also needs its data format. Confirm in a scratch script that your dataset loads and iterates cleanly for each trainer *before* you kick off a long training run:

```python
# DPO-family (ORPO, SimPO): {"prompt", "chosen", "rejected"}
# KTO:                       {"prompt", "completion", "label": bool}
# GRPO:                      {"prompt"} + a reward_funcs callable
```

For KTO, if you didn't do exercise 01's stretch, convert on the fly:

```python
def to_kto(row):
    return [
        {"prompt": row["prompt"], "completion": row["chosen"], "label": True},
        {"prompt": row["prompt"], "completion": row["rejected"], "label": False},
    ]
kto_ds = paired_ds.flatten_indices().map(to_kto, ...).flatten()  # sketch
```

For GRPO, define your reward source. Two acceptable options:

1. **Reuse the exercise-02 RM.** Wrap it in a `reward_funcs` callable that takes `(prompts, completions)` and returns a list of scalars. This mirrors PPO-RLHF's classical setup, minus the value model.
2. **A verifiable-reward setup.** If you have a suitable prompts source (GSM8K math prompts, HumanEval-style code prompts, or a synthetic "produce valid JSON" set), define a rule-based reward instead — this is GRPO's design home. Chapter 6 argues this is where GRPO actually shines.

Either is acceptable; document which you chose.

### Part C — Log the same diagnostics for every run

Whatever the method, log at every step to a per-method JSONL:

- Loss (`train/loss`, `eval/loss`).
- Method-specific "the model is learning" signal:
  - ORPO / SimPO / KTO: `rewards/chosen`, `rewards/rejected`, `rewards/margins`, `rewards/accuracies` (whichever TRL surfaces for that trainer).
  - GRPO: mean reward per group, reward std per group (zero std = flat group = no learning signal — flag it), fraction of groups with all-zero reward.
- KL against reference (for reference-based methods) or a proxy drift signal (for reference-free ORPO / SimPO — compare log-probs to the frozen SFT on a held-out canned prompt every 200 steps).
- Response length trajectory — SimPO especially: watch for over-correction (chapter 6, "SimPO — when to be careful").
- Peak VRAM and per-step wall-clock.

Dump 5 canned prompts (the same set used in exercises 03–04) every 200 steps into `outputs/<method>-run/samples.jsonl`. Same prompts every method → trajectory-comparable across methods.

### Part D — The comparison matrix

`outputs/family-survey/comparison.md`. This is the deliverable that separates a "ran three trainers" submission from a survey that would inform a colleague's decision.

Fill in a matrix with **one column per method you ran (plus a DPO baseline column pulled from exercise 04)**:

| Dimension | DPO (ex. 04) | Method 1 | Method 2 | Method 3 |
|---|---|---|---|---|
| Trainer + TRL version | | | | |
| Loaded models (policy / ref / RM / value) | | | | |
| Trainable params | | | | |
| Peak VRAM (GB) | | | | |
| Wall-clock (total; per-step median) | | | | |
| Data shape used | paired | | | |
| Reference model used? | Yes | | | |
| On-policy? | No | | | |
| Headline hyperparameter(s) + value | β=0.1 | | | |
| Final training-loss trajectory (fell / oscillated / plateaued) | | | | |
| Held-out pairwise win-rate vs SFT (same judge, same prompts) | | | | |
| Held-out pairwise win-rate vs the DPO baseline | — | | | |
| Median response length | | | | |
| Length delta vs SFT (median tokens) | | | | |
| RM mean reward on held-out (exercise-02 RM) | | | | |
| Which chapter-7 failure mode (if any) did the run hit | | | | |
| One-sentence qualitative summary of samples | | | | |

Use the same 200-prompt subset of `eval_benchmark` you used in exercise 04 for all pairwise-vs-SFT and pairwise-vs-DPO evals. Same judge across all methods. Same seed for generation temperature.

Below the matrix, write **three short sections**:

1. **What surprised you.** Concrete: "SimPO's median length was 40% shorter than DPO's despite equal training data" is a real observation. "It was interesting how they varied" is not.
2. **Where the method's paper claim held up vs your run.** For each method, name one thing the paper promises and one thing your run showed. Match or mismatch, both are learning.
3. **Cost per unit quality.** Rough. Divide (peak VRAM × wall-clock) by (win-rate-vs-SFT). Higher = worse cost/quality. Rank the methods. Speculate.

### Part E — Defend a recommendation for a scenario

Pick **one** of the following scenarios and write a ~300-word recommendation defending which of the methods you surveyed you would ship for it, citing evidence from your runs:

- **A.** A small product team has 10k unpaired thumbs-up/down signals per week from their deployed chatbot. Weekly training refresh. Single 40 GB GPU. Which method would you use?
- **B.** A team maintains a code-generation model and can execute generated code as a correctness check. They want to push reasoning quality without spending on human annotators. Which method would you use?
- **C.** A team has a well-curated 20k-pair preference set with balanced chosen/rejected lengths. They want the simplest possible training pipeline — no reference model, no two-stage runs — because their infra team is small and every extra artifact is a burden. Which method would you use?
- **D.** A team just shipped a DPO-trained model that eval reports as "wordy" — median generation length is 30% higher than the SFT baseline and human raters comment on filler. They want a targeted retraining step to fix this without a full pipeline rebuild. Which method (or variant of DPO) would you use?

Every choice must cite at least two rows from your comparison matrix. Recommendations that could have been written from chapter 6 alone (without your runs) are failing recommendations — you must ground them in what your survey actually showed.

### Part F — The one-page survey report

`outputs/family-survey/SURVEY_REPORT.md`, ~1 page (~500–800 words). This is the artifact a colleague reads instead of asking you to walk them through everything.

Sections:

- **What I ran.** Methods, base model, data, seed, TRL version.
- **The comparison matrix.** (Link to `comparison.md` above.)
- **What worked, what didn't.** Two-to-three sentences per method.
- **My recommendation for scenario X.** (Copy from Part E, tightened.)
- **What I would run next.** One paragraph — the natural follow-up experiment. This is what makes the survey generative rather than terminal.

## Starter guidance

- **Do the DPO baseline column first.** You already have it from exercise 04; copy the numbers into the matrix on day one. This anchors every other run visually — you can see if a method is 2× worse or 10% better immediately.
- **KTO wants scale.** Chapter 6 flags this — under a few thousand examples, KTO is unstable. If your `eval_benchmark`-adjacent training pool is small, expect noisier KTO curves.
- **SimPO's `β` is not DPO's `β`.** They mean different things in the two objectives. Do NOT reuse `β = 0.1` from DPO; use `β = 2.0` and `simpo_gamma = 1.0` as the paper defaults, and note this in your report.
- **GRPO's slow.** K-sample rollouts every step multiply generation cost by K. If your GPU makes generation slow, GRPO will make it painfully slow. Start with `num_generations=4` and a small `max_completion_length`; scale up only if the run looks stable.
- **Zero-variance groups kill GRPO's learning.** If your reward source returns the same value for all K samples on many prompts (e.g., all 4 samples got 0 on a hard math prompt), the advantage is 0 and no gradient flows for that prompt. Log the fraction of zero-variance groups; if it's above ~30% your reward source or your prompt difficulty is misaligned.
- **Reference-free methods can drift.** Chapter 6 flags this for ORPO and SimPO both. Sample from the trained policy on canned prompts every 200 steps and read them. Divergence from the SFT is *the point* — but if the divergence is toward nonsense, stop and reduce the effective LR.
- **The comparison is honest only if the seeds match.** Pin the seed for shuffling, for dropout (if any), for prompt sampling in generation. Chapter 6's empirical claims come from carefully-matched settings; yours should too.
- **This is a lot of runs.** Reserve wall-clock. On a single 24 GB GPU with QLoRA, expect: DPO baseline (already done) + 2 hours ORPO + 2 hours SimPO + 3–6 hours GRPO. Plan the evening.

## Acceptance criteria

You have completed the exercise when:

- `outputs/orpo-run/`, `outputs/kto-run/`, `outputs/simpo-run/`, or `outputs/grpo-run/` — three of the four — exist, each with per-step diagnostics JSONL, samples JSONL, config, and a brief `diagnostics.md` describing the run's trajectory and any failure mode it hit.
- `outputs/family-survey/comparison.md` exists with the matrix filled in for the DPO baseline plus three new methods. Empty cells or "TBD" are fails.
- `outputs/family-survey/SURVEY_REPORT.md` exists and covers every section in Part F.
- Held-out pairwise win-rate is reported for each new method vs SFT *and* vs the DPO baseline, on the same 200-prompt subset with the same judge.
- The scenario recommendation cites at least two matrix rows.
- `picks.md` explains why you chose the three methods you did.

## Stretch goals

- **Run the fourth method.** All four instead of three. You will feel the odd shape of GRPO relative to the DPO family directly.
- **IPO and CPO too.** Add the two chapter-6 also-rans (`loss_type="ipo"` and `CPOTrainer`). Both are drop-in relative to DPO; total added wall-clock is small.
- **Length-controlled comparison.** Rerun SimPO and DPO on a length-balanced training subset (`|chosen| ≈ |rejected|`) and compare. Isolate how much of SimPO's length correction is data vs objective.
- **KTO on paired data converted to unpaired.** Convert your paired set to KTO format and run KTO on it. Compare to DPO on the same underlying pairs. The paper claims KTO matches DPO on paired data; test the claim on your data.
- **GRPO with two reward sources.** Run GRPO twice — once with the exercise-02 RM as the reward, once with a rule-based verifier (JSON validity, or a math task from GSM8K). Compare stability and final performance. The chapter-6 claim is GRPO shines with verifiable rewards; test it.
- **Cross-method sample ranking.** Take 20 canned prompts; generate one response per method plus SFT; blind-shuffle; hand-rank them. Compare your ranking to the win-rate table. Where do they disagree, and why?
- **Ship the winner.** Merge the LoRA adapter (mod-104 chapter 3) for whichever method won your recommendation, load in vLLM, and serve. Run 20 real prompts. This is what a "survey → decision → ship" cycle looks like end to end.
