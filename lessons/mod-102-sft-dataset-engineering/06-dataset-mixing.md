# Mixing Datasets Across Capabilities

**Estimated read:** 40 minutes.
**You should be able to, by the end:** design a per-source mixing plan across capability categories, choose between raw proportional sampling and temperature-adjusted sampling for sources of very different sizes, document the "safety data catastrophic forgetting" tradeoff, and run a small ablation that tells you whether your ratios are load-bearing or arbitrary.

## Why mixing is the real design decision

At this point in the module you have a curated pool of examples with provenance metadata. The next question is not "how do I train" but "in what proportion do I show the model each kind of example." This is the *mix*, and it is the single design decision that most shapes the resulting model's behavior — more than learning rate, more than epochs, more than PEFT rank.

Public examples that make the effect concrete:

- **Llama 3.1 model card / paper** describes the pretraining and post-training data mix explicitly, including capability-specific ratios that were tuned via ablation. The point is not the exact numbers (they are trade-secret-flavored and change between versions) but the shape of the disclosure: capability categories, per-category weights, ablation-driven adjustments.
- **Tülu 2 / 3** (Ivison et al., 2023; Lambert et al., 2024) publishes the exact source list and mixing ratios for its SFT stage. Reading its data card end to end is the reference for what a documented open mix looks like.
- **Nemotron-4 340B Instruct** and **SmolLM2** disclose their post-training data mixes in more detail than most; both are worth reading as a template.

If you cannot answer "why is code 15% of the mix and not 30%?" for your own model, you have skipped this chapter.

## Capability categories

Almost every published post-training mix organizes its sources into a small set of capability categories. Common ones:

- **General chat / instruction following** — everyday conversational assistance. Sources: OpenAssistant, No Robots, LIMA, various ShareGPT-derived cleanups.
- **Reasoning** — chain-of-thought math and logic. Sources: GSM8K's train split (used carefully — see decontamination), OpenMath, OpenR1-Math, NuminaMath, MetaMathQA, Camel-AI math.
- **Code** — programming instruction data. Sources: CodeAlpaca, OSS-Instruct (Magicoder), OpenCodeInterpreter, WizardCoder-family, Evol-Instruct-Code.
- **Safety / refusals** — worked examples of refusing harmful requests, and — crucially — worked examples of *not* refusing benign requests. Sources: Anthropic HH-RLHF (SFT parts), PKU-SafeRLHF, WildChat (with safety filters), custom red-team-authored refusals.
- **Tool use / function calling** — assistant emits structured tool calls. Sources: Glaive Function Calling, ToolBench, Hermes-Function-Calling-Dataset.
- **Long-context / summarization** — trains context-length skills. Sources: NarrativeQA-style, book summarization datasets, LongAlign.
- **Persona / style** — teach a specific voice, tone, or brand-of-voice. Usually hand-authored or small-scale distilled.
- **Multilingual** — if the target model supports multiple languages.

You do not need all of these. You need to name the categories your model must serve, source data for each, and set a proportion.

## Setting proportions: three approaches

### 1. Proportional-to-source-size

The default in most naïve pipelines: shuffle every row from every source together, sample uniformly, train.

Consequence: your mix looks like the concatenation of your sources. If OpenAssistant is 90% of your rows and your safety pile is 2%, the model sees 90% general chat and 2% safety. This is almost never what you want; large sources drown out small ones regardless of their per-example value.

### 2. Explicit per-source weights

Assign each source a target proportion of the mix. Sample rows from that source at whatever rate hits the target. If the source has fewer examples than the target proportion demands, it will be upsampled (rows repeat within an epoch); if it has more, it will be downsampled (rows are seen with probability < 1).

This is what most published mixes do. The Tülu 3 data card is a clean example.

### 3. Temperature-adjusted proportional sampling

A middle ground. Given source sizes `n_i`, sample with probability proportional to `n_i^α` for some temperature `α ∈ [0, 1]`. At `α = 1` you get proportional-to-size (default); at `α = 0` you get uniform across sources regardless of size; intermediate `α` softens the effect of source-size disparities.

This was popularized by the **mT5** paper (Xue et al., 2021) for multilingual pretraining and shows up across corpora with heavy per-source imbalance. It is a good starting point when you know you want to boost small sources but do not yet have per-source measurements to justify explicit weights.

## Concrete example: a first-cut mix

For a general-purpose instruction-following model aiming at English chat + code + light reasoning + safety, a reasonable first cut:

| Category | Target proportion | Approximate rows in a 100k mix |
|---|---:|---:|
| General chat / instruction following | 40% | 40,000 |
| Reasoning (math + logic) | 20% | 20,000 |
| Code | 20% | 20,000 |
| Safety refusals (harmful) + non-refusals (benign) | 10% | 10,000 |
| Tool use | 5% | 5,000 |
| Persona / style | 5% | 5,000 |

These numbers are illustrative — they exist to give the exercise a concrete starting point. **You must be able to defend your actual numbers with either a citation to a comparable published mix or your own ablation results.** "It looked reasonable" is not a defense.

## The safety-data catastrophic forgetting tradeoff

Every fine-tuner discovers this the hard way once. There are two failure modes at opposite ends of the safety-data spectrum:

**Too little safety data.** The model happily complies with clearly harmful requests it should refuse. Every red-team pass finds new categories. You add more safety examples and things get worse:

**Too much safety data — over-refusal.** The model refuses ordinary requests that pattern-match to refusal training data. "Write a poem about a chemical reaction" gets refused because "chemical" appears in refusal exemplars. Real-world reports of this include the "Gemini refusing to write Python code that could 'harm the file system'" class of incidents.

The mitigations that consistently help:

1. **Curate a benign-request pile alongside the refusal pile.** For every category you train refusals in ("medical advice," "legal advice," "chemistry"), curate examples of *benign requests* in that same category that the model must not refuse. This teaches the boundary explicitly.
2. **Keep refusals a small fraction of the mix.** Refusal exemplars are template-heavy and low-diversity; too many overwhelm the general-chat distribution.
3. **Measure over-refusal in your evals.** Datasets like **XSTest** (Röttger et al., 2023) exist specifically to measure over-refusal. Track it every eval.

The Llama 3, Claude 3, and GPT-4-class model reports all discuss this tradeoff explicitly. It is not an obscure gotcha; it is one of the two or three central tensions of SFT.

## Curriculum vs. shuffled training

A closely-related mixing decision: within an epoch, do you show hard and easy examples together, or in a curriculum (easy → hard)?

- **Shuffled training** is the default and the safest. Every batch is a stratified sample of the mix.
- **Curriculum training** — showing simpler tasks first, then harder — occasionally shows small wins on reasoning-heavy fine-tunes, but the effect is fragile and setup-dependent. If you experiment with curriculum, start with a strict A/B against shuffled at the same total token count.
- **Task-clustered training** — training an epoch of code, then an epoch of reasoning, then general chat — is a known way to induce catastrophic forgetting of the earlier categories. Do not do this.

The default assumption should be shuffled unless you have a specific hypothesis to test.

## Actually running an ablation

The only real way to know whether your mixing ratios matter is to change one and re-train. A minimal ablation protocol:

1. Fix everything except the ratio you are testing.
2. Train the baseline mix and the modified mix at identical total token counts.
3. Score both against the eval suite (mod-108). Include a small held-out "in-the-wild" prompt set alongside the standard benchmarks, so you catch shifts that MMLU / MT-Bench numbers miss.
4. Log the deltas per capability.
5. Only change the mix if the ablation supports the change with a margin bigger than your eval's noise floor.

Even a single ablation ("code 20% vs. code 30%") gives you far more information than a hundred spreadsheet tweaks. Budget at least one ablation into every SFT project.

## The mix card

Every SFT mix should ship with a **mix card** — a document describing:

- Each source, its category, its license (SPDX or explicit terms).
- The size of each source before and after filters.
- The target proportion and the actual proportion (they diverge when a source runs out of rows or a filter cuts hard).
- Any upsampling / downsampling factors.
- The decontamination result per source per eval.
- The ablations that justify (or fail to justify) the ratios.

This is the input to chapter 7's model card. It is also the artifact that lets a future you — or a future colleague — reproduce or modify the mix without re-deriving every decision.

## Exercise pointer

`exercises/exercise-05-dataset-mix-card-and-licensing-trace.md` has you design and document a 100k-example mix across at least four capability categories, produce the mix card, and trace the license of every included source.

## Summary

- The mix is the biggest design decision in SFT; more than any hyperparameter.
- Organize sources into a small set of capability categories (general chat, reasoning, code, safety, tool use, persona) and set proportions per category.
- Choose between proportional-to-size (naïve), temperature-adjusted (mT5-style), or explicit per-source weights. Explicit weights are the industry default for post-training.
- The safety-data catastrophic forgetting tradeoff has two ends: under-refusal and over-refusal. Curate benign requests in each refusal category; keep refusals a small fraction of the mix; measure over-refusal (e.g., XSTest).
- Default to shuffled training; avoid task-clustered epochs.
- Run at least one ablation per project; the marginal information is huge compared to spreadsheet reasoning.
- Ship a mix card describing every source, category, license, filter drop, and proportion. Chapter 7's model card depends on it.
