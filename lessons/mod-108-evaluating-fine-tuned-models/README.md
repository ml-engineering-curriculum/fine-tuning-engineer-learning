# mod-108-evaluating-fine-tuned-models: Evaluating Fine-Tuned Models

**Estimated effort:** 15 hours
**Position in the track:** eighth module. Consumes the checkpoints produced by mod-103 (full-parameter SFT), mod-104 (LoRA / QLoRA), mod-105 (DPO / KTO / ORPO), mod-106 (safety alignment), and mod-107 (distributed fine-tuning). Feeds the release-gate signals that mod-109 (quantization / serving), mod-110 (managed APIs), and mod-111 (systems design) all depend on when deciding whether an artifact is ready to ship. Also closes the loop back to mod-102's dataset-engineering discipline: when eval contamination is discovered here, the fix is almost always upstream.

Fine-tuning without evaluation is a vibes exercise. This module is about turning "the loss went down" and "the samples look good" into numbers you can defend in a release review: a fixed benchmark suite where you can point to an unchanged config file, a task-specific held-out set that mirrors your production traffic, a judged eval where the judge itself has been audited for bias, a contamination scan that says your numbers are not inflated, and a regression report that says how the new adapter compares to the last one across all of these.

The through-line: **an eval is a contract, not a script.** Once you publish a number, downstream people (product, safety, execs, other engineers) make decisions on it. This module is about building evals whose numbers are worth making decisions on — reproducible, uncontaminated, calibrated, and reported with enough context that a reader six months from now can reconstruct exactly what "MMLU 62.4" or "MT-Bench 7.8" meant.

## Learning objectives

- Run the EleutherAI `lm-evaluation-harness` against a fine-tuned model for academic benchmarks (MMLU, ARC, GSM8K, HumanEval, TruthfulQA, IFEval)
- Build task-specific held-out evals tied to the production use case
- Design an LLM-as-judge protocol with MT-Bench-style rubrics, pairwise comparison, and position-bias controls
- Detect benchmark contamination and prevent it via decontamination at dataset-engineering time
- Run a release-gate eval suite and write a regression report against the previous adapter

## Lecture chapters

1. [`01-evaluation-landscape.md`](01-evaluation-landscape.md) — The Evaluation Landscape for Fine-Tuned Models. What "eval" actually means at this altitude, the four-lane taxonomy (academic benchmarks / task-specific held-out / judged / safety), how eval budgets differ from training budgets, and the difference between the eval a research paper reports and the eval a release gate needs.
2. [`02-lm-evaluation-harness.md`](02-lm-evaluation-harness.md) — Running EleutherAI `lm-evaluation-harness`. Install, model backends (`hf`, `vllm`, `openai-chat-completions`), the `--tasks` catalog (MMLU, ARC-Challenge, GSM8K, HumanEval, TruthfulQA, IFEval), `log_likelihood` vs `generate_until` task types, few-shot vs zero-shot, chat-template pitfalls, per-task memory / time budget, and how to read the results JSON.
3. [`03-task-specific-held-out-evals.md`](03-task-specific-held-out-evals.md) — Building Task-Specific Held-Out Evals. Why academic benchmarks alone do not tell you whether your model is fit for the production use case, the dataset schema (input / gold / metadata / stratum), sizing rules of thumb, how to build a golden set from production traffic without leaking labels back into training, and the connection to mod-102's SFT dataset discipline.
4. [`04-llm-as-judge.md`](04-llm-as-judge.md) — LLM-as-Judge Protocols. MT-Bench-style single-answer grading and pairwise comparison, rubric design that a judge can follow, the four known biases (position, verbosity, self-preference, style) and the concrete controls for each, judge-model choice and cost, inter-judge agreement, and calibration against a small human-labeled anchor set.
5. [`05-contamination-detection.md`](05-contamination-detection.md) — Detecting Benchmark Contamination at Eval Time. Why decontamination in mod-102 chapter 3 was not enough on its own, model-signal contamination detection (test-set membership probes, canonical-answer echoing, guided-completion tests), fixing contamination at the dataset-engineering layer, and disclosing contamination on the model card.
6. [`06-release-gate-eval-suite.md`](06-release-gate-eval-suite.md) — The Release-Gate Eval Suite and Regression Report. Turning the evals from chapters 2–5 into a single reproducible suite with pass/fail gates, statistical significance (bootstrap CIs, McNemar for pairwise, agreement metrics for judged), a regression-report template that compares adapter N vs. adapter N−1, and the flake / cost management that keeps the gate runnable on every release.
7. [`07-eval-pitfalls-and-handoff.md`](07-eval-pitfalls-and-handoff.md) — Common Pitfalls and Handoff to Serving / Systems. Chat-template mismatch, sampling-parameter drift, tokenizer changes, stop-token bugs, prompt-cache confounds, the "quantized weights change the eval" trap that hands into mod-109, and the eval artifacts mod-111's post-training platform needs.

## Exercises

Each exercise is a hands-on companion to one or more chapters. They live in [`exercises/`](exercises/).

- [`exercise-01-lm-eval-harness-baseline-vs-tuned.md`](exercises/exercise-01-lm-eval-harness-baseline-vs-tuned.md) — install `lm-evaluation-harness`, run a fixed six-task suite against a base model and its fine-tuned checkpoint, and produce a per-task delta table with 95% confidence intervals.
- [`exercise-02-task-specific-held-out-eval-design.md`](exercises/exercise-02-task-specific-held-out-eval-design.md) — design and build a 300–500-item task-specific eval for a named production use case (choose from the four provided or bring your own), including schema, stratification, sizing justification, and the metric that decides pass/fail.
- [`exercise-03-llm-as-judge-mt-bench-style.md`](exercises/exercise-03-llm-as-judge-mt-bench-style.md) — implement single-answer rubric grading and pairwise comparison against your held-out set, measure and correct position bias, and report inter-judge agreement against a small human-labeled anchor.
- [`exercise-04-contamination-detection-and-mitigation.md`](exercises/exercise-04-contamination-detection-and-mitigation.md) — run n-gram, embedding, and model-signal contamination scans against a real SFT set and eval suite, and write the disclosure section for the model card.
- [`exercise-05-release-gate-eval-report.md`](exercises/exercise-05-release-gate-eval-report.md) — package chapters 2–5 into a single release-gate suite, run it against two consecutive adapter versions, and produce the regression report a release captain would sign off on.

## Labs and quizzes

Long-form labs land in [`labs/`](labs/) and the knowledge-check quiz in [`quizzes/`](quizzes/) on subsequent authoring cycles.

## Resources

See [`resources.md`](resources.md) for primary papers (Hendrycks et al. — MMLU; Cobbe et al. — GSM8K; Chen et al. — HumanEval; Clark et al. — ARC; Lin et al. — TruthfulQA; Zhou et al. — IFEval; Zheng et al. — MT-Bench / LLM-as-judge; Brown et al. — GPT-3 contamination appendix; Sainz et al. — data contamination survey; Oren et al. — proving test-set contamination; Chiang & Lee — LLM-as-judge biases), official tooling docs (EleutherAI `lm-evaluation-harness`, BIG-bench, HELM, `bigcode-evaluation-harness`, `alpaca_eval`, MT-Bench code), and reference release-gate templates in the major open model cards (Llama 3, Mistral, Qwen, Gemma).

## What comes next

- **mod-109 — Quantization, Export, and Serving** re-runs your release-gate suite against quantized artifacts (INT8, INT4, GPTQ, AWQ). The eval infrastructure this module builds is what tells you a quantized model still meets the bar.
- **mod-110 — Managed Fine-Tuning APIs** faces a stricter eval problem because you cannot introspect the trained artifact; the held-out evals and judged evals from chapters 3–4 become the primary quality signal.
- **mod-111 — Post-Training Systems Design** stores the eval artifacts this module produces (JSONs, judge transcripts, contamination reports, regression reports) in the experiment tracker and gates deployments on them. Chapter 6's release-gate spec is what mod-111 wires into CI.
