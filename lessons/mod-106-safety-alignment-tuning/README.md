# mod-106-safety-alignment-tuning: Safety and Alignment Tuning

**Estimated effort:** 13 hours
**Position in the track:** sixth module. Consumes the preference-tuned checkpoint from mod-105 (DPO or PPO output) and produces a safety-aligned model that mod-108 (evaluation) and mod-109 (quantization / serving) can pick up. Safety-alignment tuning is largely applied preference optimization — the trainers are the ones from mod-103 (`SFTTrainer`) and mod-105 (`DPOTrainer` / `KTOTrainer`); the novelty is in the data curation, the self-critique augmentation, and the eval dashboard that decides whether the run shipped or regressed.

Mod-105 taught you how to move a model past imitation using paired comparisons. This module applies that machinery to a specific kind of comparison — **safer vs less safe** — and adds the operational discipline that turns "we ran DPO on some safety pairs" into "we shipped a checkpoint whose safety numbers a governance reviewer can defend." The through-line is that safety-alignment tuning is 90% dataset engineering and evaluation, 10% training loop, and 100% dependent on a written policy that says what safe means.

## Learning objectives

- Curate refusal / harm-reduction datasets and integrate them into the SFT + preference-optimization mix
- Implement Constitutional AI / RLAIF self-critique loops at a small scale
- Ingest red-team data from peer / risk teams and convert it into training-ready preference pairs
- Measure refusal rate, over-refusal rate, jailbreak-resistance, and harm-policy adherence
- Recognise where to hand off to the security and governance tracks for depth

## Lecture chapters

1. [`01-the-safety-alignment-problem.md`](01-the-safety-alignment-problem.md) — The Safety Alignment Problem. What training-time safety adds on top of SFT and preference optimization, the harm taxonomies (Weidinger, Anthropic, MLCommons AILuminate) you will keep hitting, the helpful / harmless / honest triad, the under-refusal vs over-refusal tension, and the four levers this module pulls.
2. [`02-refusal-and-harm-reduction-datasets.md`](02-refusal-and-harm-reduction-datasets.md) — Refusal and Harm-Reduction Datasets. Where safety data lives (HH-RLHF, PKU-SafeRLHF / BeaverTails, Do-Not-Answer, attack corpora), how to split it into SFT-safe and preference-safe halves, response-style choices (hard refusal vs soft refusal vs safe completion), and the category-coverage table that decides whether your training will actually work.
3. [`03-constitutional-ai-and-rlaif.md`](03-constitutional-ai-and-rlaif.md) — Constitutional AI and RLAIF Self-Critique Loops. Bai et al.'s two-stage recipe (SL-CAI + RL-CAI), how to write a working constitution, how to implement critique-and-revise against an open judge, and when RLAIF earns its complexity relative to plain preference data.
4. [`04-red-team-data-to-training-signal.md`](04-red-team-data-to-training-signal.md) — Red-Team Data to Training-Ready Signal. What raw red-team submissions look like when they land in your inbox, the intake pipeline (schema, reproduction, dedup, categorization, triage), the attack taxonomy that determines training response, the three conversion patterns from failure trace to preference pair, and the model-in-the-loop iteration cadence.
5. [`05-safety-evals-refusal-and-jailbreak.md`](05-safety-evals-refusal-and-jailbreak.md) — Safety Evals: Refusal, Over-Refusal, Jailbreak Resistance, Harm Adherence. The four axes, the standard public benchmarks (HarmBench, JailbreakBench, XSTest, OR-Bench), the judge-model choice (Llama Guard, WildGuard, HarmBench-cls), the report card that ships alongside the checkpoint, and eval hygiene / contamination discipline.
6. [`06-handoff-to-security-and-governance.md`](06-handoff-to-security-and-governance.md) — Handoff to Security and Governance. Where fine-tuning stops and runtime guardrails, operational controls, and policy / legal / regulatory layers pick up. The artifacts you owe the outer layers (report card, dataset card, model card, red-team diff report). What to escalate rather than train against.

## Exercises

Each exercise is a hands-on companion to one or more chapters. They live in [`exercises/`](exercises/).

- [`exercise-01-refusal-and-harm-reduction-dataset.md`](exercises/exercise-01-refusal-and-harm-reduction-dataset.md) — build a safety-focused SFT + DPO mix from PKU-SafeRLHF / BeaverTails / HH-harmless, apply category coverage and length controls, and ship a dataset card with a coverage table.
- [`exercise-02-constitutional-ai-self-critique-loop.md`](exercises/exercise-02-constitutional-ai-self-critique-loop.md) — author a ~10-principle constitution, run the SL-CAI critique-and-revise loop on ~200 red-team prompts using an open generator + critic, produce a `(prompt, revised)` SFT set, and hand-audit the revisions.
- [`exercise-03-red-team-data-to-preference-pairs.md`](exercises/exercise-03-red-team-data-to-preference-pairs.md) — ingest a public attack corpus (HarmBench / JailbreakBench / WildGuardMix), dedupe and categorize, apply the pattern-A/B/C conversion, produce a training-ready DPO dataset, and audit against benchmark contamination.
- [`exercise-04-over-refusal-and-jailbreak-eval.md`](exercises/exercise-04-over-refusal-and-jailbreak-eval.md) — stand up the four-axis safety-eval harness on HarmBench, XSTest, JailbreakBench + PAIR, and Llama-Guard-3 as the harm classifier; compute per-category rates + MT-Bench + length deltas; produce the report card from chapter 5.

## Labs and quizzes

Long-form labs land in [`labs/`](labs/) and the knowledge-check quiz in [`quizzes/`](quizzes/) on subsequent authoring cycles.

## Resources

See [`resources.md`](resources.md) for primary papers (Anthropic HH-RLHF, Constitutional AI, RLAIF, XSTest, OR-Bench, HarmBench, JailbreakBench, Llama Guard, WildGuard, GCG, Instruction Hierarchy), the reference open corpora, and the tooling (TRL, Llama Guard, WildGuard, garak, HarmBench evaluation code) cited across the chapters.

## What comes next

- **mod-107 — Distributed Fine-Tuning** covers the multi-GPU / multi-node story for the DPO / SFT stages this module reuses. Safety-tuning-specific data pipelines usually stay small enough to run single-node; the eval harness scales the same way distributed training does.
- **mod-108 — Evaluating Fine-Tuned Models** covers the general-capability eval side that this module's report card treats as a single number (MT-Bench / MMLU). If your safety-tuned model regresses capability, mod-108 is where you dig in.
- **mod-109 — Quantization, Export, Serving** and **mod-110 — Managed Fine-Tuning APIs** cover the deployment side. The runtime guardrails from chapter 6 (Llama Guard, WildGuard, moderation APIs) are where your safety artifacts get consumed by the serving layer.
- **mod-111 — Post-Training Systems Design** situates safety-alignment tuning as one part of a full post-training platform, alongside the governance and security tracks that this module explicitly hands off to.
