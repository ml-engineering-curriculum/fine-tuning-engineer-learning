# The Evaluation Landscape for Fine-Tuned Models

**Estimated read:** 30 minutes.
**You should be able to, by the end:** enumerate the four lanes of evaluation a fine-tuning engineer routinely runs (academic benchmarks, task-specific held-out sets, judged evals, safety), state what question each lane answers and what it does *not*, describe why the training loss and a handful of vibes samples are not enough, and distinguish an eval a research paper reports from an eval a release gate consumes.

## Why "the loss went down" is not enough

You have a checkpoint from mod-103, mod-104, mod-105, or mod-106. Training loss dropped. A validation-loss curve on a held-out slice of the SFT set went down too. You sampled twenty prompts and the answers look better than the base model. Ship it?

No. Every one of those signals is compatible with a model that is worse in production.

- **Training loss** measures how well the model predicts the next token in the training distribution. Loss can drop because the model memorized templates, tightened its style, or learned to imitate a specific teacher — none of which necessarily means it does the downstream task better.
- **Held-out SFT loss** measures generalization *within the same distribution as training*. A model that overfits an entire dataset genre (all your SFT rows are customer-support tickets) will happily score well on a held-out slice of that genre while regressing on general knowledge, safety, and instruction following.
- **Sampled outputs** are the smallest possible eval. Twenty prompts is not a statistical sample of anything, and the prompts you personally choose are biased toward the kinds of failures you already know about. This is the most common source of "shipped a regression" incidents in fine-tuning.

Evaluation exists to answer the questions loss and vibes cannot:

1. Did the fine-tune improve the model on the tasks users actually care about?
2. Did it regress on capabilities the base model had?
3. Did it regress on safety (refusals, jailbreaks, harmful-content generation)?
4. Are the numbers reproducible — will the next engineer get the same result running the same suite?
5. Are the numbers uncontaminated — did the model score well because it learned, or because it saw the answers?

The rest of this module is about how to build the evaluation harness that answers those five questions well enough that the release captain will sign the deploy.

## The four lanes

Real fine-tuning teams run evals in four distinct lanes. Each lane answers a different question, uses different infrastructure, has different failure modes, and appears in the release-gate report as its own section.

### Lane 1 — Academic / capability benchmarks

Fixed, published, reproducible benchmarks that measure general model capabilities. The canonical set for a chat/instruct model is a subset of:

| Benchmark | Measures | Format | Chapter |
|---|---|---|---|
| **MMLU** (Hendrycks et al., 2021) | broad academic knowledge across 57 subjects | 4-choice multiple choice, ~14k test items | ch. 2 |
| **ARC-Challenge** (Clark et al., 2018) | grade-school science reasoning | 4-choice multiple choice, 1,172 test items | ch. 2 |
| **GSM8K** (Cobbe et al., 2021) | grade-school math word problems | generation, exact-match answer, 1,319 test items | ch. 2 |
| **HumanEval** (Chen et al., 2021) | Python code synthesis | generation, unit-test pass@k, 164 items | ch. 2 |
| **TruthfulQA** (Lin et al., 2022) | resistance to human falsehoods | multiple choice (mc1/mc2) + generation | ch. 2 |
| **IFEval** (Zhou et al., 2023) | verifiable instruction following | generation, rule-based grading, 541 prompts | ch. 2 |

What this lane is good for: comparability across models, resistance to overfitting-to-eval (because the benchmark is public and stable), and a language your reader already knows ("MMLU 62.4" reads the same way to everyone in the field).

What this lane is bad for: telling you whether the model does *your* task well. A model that gains 4 points on MMLU may or may not answer your product's customer questions better; the two are only weakly correlated for narrow use cases.

Run this lane every release. The infrastructure is `lm-evaluation-harness` (chapter 2).

### Lane 2 — Task-specific held-out evals

A dataset you build yourself, mirroring the shape of the traffic your fine-tuned model will actually see. If your model is going to summarize support tickets, this lane is 300–500 support tickets with gold summaries and a metric. If your model is going to answer customer questions about billing, this lane is billing questions with gold answers and a metric.

What this lane is good for: telling you whether the model actually does the job. This is the only lane whose numbers move in lockstep with product quality.

What this lane is bad for: comparability (no one else has your dataset), transferability (a good number here does not tell anyone else anything), and staleness (the eval has to be refreshed as the product changes).

Every fine-tuning project needs at least one of these. Building it well is chapter 3. Not having one is the single most common cause of "we fine-tuned, we shipped, users complained" post-mortems.

### Lane 3 — LLM-as-judge / preference evals

Open-ended generation quality — is answer A better than answer B, does the answer meet a rubric, does it follow the persona — is not scoreable with exact match. The current field-standard workaround is to prompt a strong LLM (GPT-4-class, or a well-scored open model) to grade or compare, in the pattern of MT-Bench (Zheng et al., 2023) and AlpacaEval (Li et al., 2023).

What this lane is good for: measuring quality on generation tasks (writing, roleplay, chat, reasoning) that no exact-match metric can capture.

What this lane is bad for: it is a noisy, biased proxy for a human. Position bias, verbosity bias, self-preference, style bias, and judge-model instability all corrupt the signal if you do not control for them. Chapter 4 is entirely about running this lane correctly.

Run this lane every release for chat / instruct models. Skip it for pure classification or extraction models.

### Lane 4 — Safety / red-team evals

Refusal on harmful requests, resistance to jailbreaks, harmful-content generation, PII leakage. Mod-106 was about *tuning* for these; this module measures whether the tune held. Standard packages include HarmBench (Mazeika et al., 2024), the safety subsets bundled with `lm-evaluation-harness`, and internal red-team sets.

What this lane is good for: catching safety regressions that user testing will never surface until it is a headline.

What this lane is bad for: it is easy to game (a model that refuses everything scores perfectly on refusal), so it must always be run alongside a helpfulness eval (lanes 1–3).

This module treats safety eval as a first-class lane but does not re-derive the taxonomy — mod-106 owns that. Chapter 6 wires the safety numbers into the release gate.

## What a research paper reports vs. what a release gate needs

The eval numbers you see in a model card or arXiv paper answer the question "is this model interesting?" The eval numbers a release gate needs answer the question "should we deploy this?" They are not the same question, and the machinery differs.

| Dimension | Research paper | Release gate |
|---|---|---|
| Task coverage | Whatever tells the paper's story | A fixed suite that catches regression |
| Freshness of eval | Latest published version | Pinned version, byte-identical across runs |
| Reproducibility bar | "Run our code" | Same JSON output every time |
| Statistical treatment | Point estimates, sometimes CIs | CIs *and* per-item diffs vs. previous adapter |
| Contamination check | One-line disclosure or none | Explicit scan, artifacts saved |
| Runtime budget | Whatever the reviewer expects | Fixed per-release budget (dollars and wall-clock) |
| Format of report | Markdown / LaTeX table | Machine-readable JSON *plus* human-readable diff |
| Ownership on failure | Nobody | The release captain, immediately |

Chapter 6 is where the release-gate machinery gets built. Everything before that chapter is about producing the numbers; chapter 6 is about producing them *the same way every time*, which is a strictly harder problem.

## The eval budget

Training a fine-tune is expensive. Evaluating it well is also expensive, and the cost is easy to underestimate because each eval individually looks cheap.

Approximate wall-clock and cost for a well-tuned suite on a single 7B model (single H100, `vllm` backend, unless noted):

| Eval | Wall-clock | Compute cost | Dollars (rough) |
|---|---|---|---|
| MMLU (5-shot, log-likelihood) | 30–90 min | 1 GPU | $2–8 |
| ARC-Challenge (25-shot) | 5–15 min | 1 GPU | <$1 |
| GSM8K (5-shot, generation) | 20–60 min | 1 GPU | $1–5 |
| HumanEval (pass@1, 20 samples) | 15–45 min | 1 GPU | $1–4 |
| TruthfulQA (mc2) | 10–30 min | 1 GPU | <$2 |
| IFEval (generation + rule grading) | 15–40 min | 1 GPU | $1–3 |
| Task-specific held-out (300 items, generation) | 5–20 min | 1 GPU | <$2 |
| Judged eval (300 items × 1 judge call each) | 20–60 min | API | $5–30 (GPT-4-class) |
| Safety eval (HarmBench-style, 400 items) | 15–40 min | 1 GPU + judge API | $10–40 |
| **Total per release** | **~3–6 hours** | **1 GPU + API** | **~$25–100** |

These numbers are illustrative and shift with model size, backend, sampling budget, and judge choice — but the shape holds. **Judged evals dominate cost; benchmark evals dominate wall-clock.** Chapter 6 discusses how to run the fast lanes on every commit and reserve the expensive lanes for release candidates.

<!-- needs-research: the exact wall-clocks depend heavily on backend and hardware; treat as order-of-magnitude anchors, not commitments. -->

## The eval altitude

This module is written at the fine-tuning-engineer altitude — you own choosing the suite, running it, and writing the report. The related altitudes:

**In scope (this module):**

- Choose the tasks in each lane.
- Run `lm-evaluation-harness` and the judged / task-specific harnesses.
- Design task-specific evals and rubrics.
- Detect and disclose contamination.
- Package the suite into a repeatable gate and write the release regression report.

**Out of scope (owned elsewhere):**

- Building the evaluation platform — a scheduler that runs evals across many models nightly, stores results forever, and diffs them automatically — is a training-pipeline-engineer / platform-engineer job. Mod-111 discusses the interface.
- Owning production A/B tests and shadow-traffic evaluation. That is a product-engineering / ML-platform job. This module ends at the release gate; production quality measurement continues past it.
- Deep human-labeling workflows (label-studio pipelines, rater training, disagreement resolution at scale). This module builds anchor sets small enough for one engineer to label; scaling that up is a labeling-ops discipline.
- Novel benchmark research. Building the next MMLU is a research contribution, not a fine-tuning-engineer task.

When you find yourself proposing "we should build our own benchmark platform," escalate to mod-111 and the platform team.

## What "reproducible eval" actually means

Reproducibility is the single most under-invested-in property of fine-tuning evals, and it is what separates evals a release gate can trust from evals a research paper can get away with. Concretely, a reproducible eval must fix:

1. **Model artifact.** The exact adapter / merged checkpoint hash (SHA of the safetensors files, not just "model v3"). Chapter 6 covers artifact identity.
2. **Tokenizer.** Same version, same special tokens, same chat template. Chapter 7 documents the failure modes when this drifts.
3. **Prompt template.** Byte-identical between runs. `lm-evaluation-harness` pins this per task; your task-specific evals must pin it too.
4. **Sampling parameters.** Temperature, top-p, top-k, max_new_tokens, stop tokens. Different values yield different scores on any generation task.
5. **Random seed.** Where sampling is used, the seed must be fixed. For log-likelihood tasks (MMLU, ARC) there is no sampling, so seeding is irrelevant — but for GSM8K, HumanEval, IFEval, and every judged eval, the seed matters.
6. **Task version.** `lm-evaluation-harness` tasks are versioned; a task v1 → v2 bump usually means the prompt or scoring changed, and results are no longer comparable.
7. **Harness version.** Pin `lm-evaluation-harness` to a commit SHA. Do not float on `main`; results shift across releases.
8. **Backend.** `hf` transformers vs. `vllm` vs. `sglang` can give slightly different numbers on generation tasks because of sampling implementation details.

If you cannot say all eight of these things about a reported number, someone will not be able to reproduce it. Chapter 6 provides the release-gate manifest that pins all eight.

## The one-page mental model

The full picture, before we dive in:

```
                              +-----------------------+
                              |  Base model (mod-101) |
                              +-----------+-----------+
                                          |
                       +------------------+------------------+
                       |                                     |
                       v                                     v
           +-----------+-----------+           +-------------+-----------+
           | SFT / LoRA / DPO /    |           | Reference numbers on   |
           | safety tuning         |           | the four eval lanes    |
           | (mods 103-107)        |           | for the BASE           |
           +-----------+-----------+           +-------------+-----------+
                       |                                     |
                       v                                     |
           +-----------+-----------+                         |
           | Fine-tuned checkpoint |                         |
           +-----------+-----------+                         |
                       |                                     |
                       v                                     |
   ,---------------------------------------------------------+
   | Release-gate eval suite (chapter 6)                     |
   |                                                         |
   |  lane 1: lm-evaluation-harness  (chapter 2)             |
   |  lane 2: task-specific held-out (chapter 3)             |
   |  lane 3: LLM-as-judge           (chapter 4)             |
   |  lane 4: safety / red-team      (mod-106 + this module) |
   |                                                         |
   |  gates + contamination + regression report              |
   |  (chapters 5 and 6)                                     |
   `---------------------------------------------------------'
                       |
                       v
           +-----------+-----------+
           |  Ship / hold decision |
           +-----------------------+
```

Every subsequent chapter fills in one of the boxes.

## Summary

- Loss down + samples look good is not enough. Evals answer the questions loss cannot: did the fine-tune help the tasks users care about, did it regress capabilities the base had, did it regress safety, and are the numbers reproducible and uncontaminated.
- Fine-tuning evals live in four lanes: academic benchmarks (lane 1, `lm-evaluation-harness`), task-specific held-out (lane 2, you build it), judged / preference (lane 3, MT-Bench-style), and safety (lane 4, from mod-106). Each answers a different question and has different failure modes.
- A research-paper number and a release-gate number are not the same artifact. The gate needs reproducibility, statistical treatment, contamination disclosure, and per-item diffs against the previous adapter that the paper does not.
- The eval budget is real. A well-tuned suite runs in 3–6 hours and $25–100 per release for a 7B model; judged evals dominate dollars, benchmarks dominate wall-clock. Chapter 6 discusses running the fast lanes often and the expensive lanes at release-candidate boundaries.
- Reproducibility means pinning eight things: model, tokenizer, prompt template, sampling params, seed, task version, harness version, backend. If any drift, the numbers are not comparable across runs.
- This module ends at the release gate. Post-deployment quality measurement, A/B testing, and rater-scale labeling live in adjacent roles; mod-111 owns the platform interface.
