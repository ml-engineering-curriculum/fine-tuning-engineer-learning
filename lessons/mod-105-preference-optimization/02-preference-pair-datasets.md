# Preference Pair Dataset Construction

**Estimated read:** 50 minutes.
**You should be able to, by the end:** know where usable preference pairs come from (human, LLM-judge, existing open corpora), design quality and length controls that keep your set from teaching the model the wrong thing, and produce a preference dataset in the shape TRL's DPO / KTO / ORPO / PPO trainers expect.

## What a preference pair is, exactly

The unit of preference data is:

```
(prompt x, chosen y_w, rejected y_l)
```

An annotator — human, LLM, or a proxy — was shown `x`, saw at least two candidate responses, and marked one as preferred. `y_w` is the winner; `y_l` is the loser. Optional metadata: annotator confidence, dimension of preference (helpfulness, harmlessness, correctness, style), source model for each response.

Two things this format does **not** carry that beginners often assume it does:

- **A magnitude of preference.** "chosen" vs "rejected" is binary. There is no "chosen by 3 points." Bradley-Terry (chapter 1) reads this as a single-bit outcome of a stochastic comparison.
- **A ground-truth reward.** There is no scalar reward attached. The reward function `r*` (chapter 1) is *implicit* in the collection of pairs — reward modeling extracts an estimate of it (chapter 3).

Downstream: DPO / KTO / ORPO / SimPO consume this format directly and train against it end-to-end; PPO consumes a *reward model* trained on this format. Same data, two paths.

## Where preference pairs come from

Four sources, listed by increasing scale and decreasing per-item quality:

1. **Human labelers.** A pool of annotators sees `(x, y_A, y_B)` and picks one. Gold standard for the classical RLHF pipeline; expensive and slow. InstructGPT, Anthropic's HH-RLHF, and OpenAssistant were built this way. Typical throughput: a labeler produces on the order of tens to a few hundred labels per hour, depending on prompt complexity.
2. **LLM-as-judge.** A stronger model (GPT-4, Claude, a specialist judge model) is prompted to compare two responses and pick one. Vastly cheaper and faster than human labeling; the quality ceiling is the judge's own preferences. UltraFeedback, Nectar, and the "distillation" recipes (Zephyr) rely on this.
3. **Rule-based / verifiable-reward preferences.** For domains where correctness is checkable (math, code, structured output), you can generate `y_A` and `y_B` and label based on whether each passed a test. This is the RLVR ("Reinforcement Learning with Verifiable Rewards") pattern behind the o-series and DeepSeek-R1 recipes. Preferences are effectively free per item — the "annotator" is a test runner.
4. **Existing preference corpora.** Datasets other people already built. HH-RLHF, UltraFeedback, SHP, PKU-SafeRLHF, Chatbot Arena preference data, etc. Cheapest to consume; you inherit whatever the source's biases and quality controls were.

Most first preference-tuning runs at 2024–2026 scale use option 4 — you take an existing corpus, apply filters and length controls, and get a workable set without paying for fresh annotations. Options 1–3 are for teams building their own house preference distribution, usually because their domain is not well-represented in open corpora.

The rest of this chapter assumes you are working from an existing corpus or a hybrid of the four. The mechanics generalize.

## Reference corpora worth knowing

You will see these names constantly in the DPO / ORPO / SimPO / KTO literature. Skim the top-level structure of at least two before you commit to one:

### HH-RLHF (Anthropic)

Bai et al., 2022. Anthropic's helpful-and-harmless corpus. Multi-turn conversations; each conversation ends with two candidate final assistant turns, one labeled chosen. Split into "helpful-only" and "harmless-only" and "helpful-plus-harmless" subsets. The historical RLHF dataset — most 2022–2023 open reproductions of RLHF used it.

- **Strengths:** Large (~170k prompts across splits), covers a wide surface of chat behavior including refusals.
- **Weaknesses:** The base model that generated the responses is now years old; response quality feels dated. Some annotator disagreements at the margins. Length bias is well-documented in this set (Singhal et al., 2023, cited in chapter 7).

### UltraFeedback (Cui et al., 2023)

64k prompts with 4 responses each, scored by GPT-4 on multiple dimensions (helpfulness, honesty, truthfulness, instruction-following). Preference pairs derive from the scores — the highest-scoring response is chosen, and rejected is drawn from a lower-scoring one. The dataset most used to reproduce Zephyr-7B and the "alignment handbook" recipes.

- **Strengths:** GPT-4 judgments are strong; the multi-dimensional score lets you build task-conditional preference subsets.
- **Weaknesses:** All judgments are one model's (GPT-4). Length correlates positively with chosen-ness — a known bias you have to control for (see later section).

### UltraFeedback-Binarized (HuggingFaceH4)

The `HuggingFaceH4/ultrafeedback_binarized` dataset packages UltraFeedback into a straight `(prompt, chosen, rejected)` shape ready for DPO. This is what most "reproduce Zephyr" tutorials consume.

### SHP (Stanford Human Preferences)

Ethayarajh et al., 2022 / 2023. ~385k preference pairs drawn from Reddit discussions with `>` upvote signal (comment A had more upvotes than comment B on the same post). Naturalistic preferences from a large, heterogeneous population.

- **Strengths:** Very large; genuinely human preferences at scale; no LLM-in-the-loop bias.
- **Weaknesses:** Reddit-community-specific style; the "preference" is upvotes, which is a much noisier signal than a dedicated annotator judgment.

### PKU-SafeRLHF

Ji et al., 2023 (BAIR / Peking). Safety-focused preference pairs — chosen = safer, rejected = less safe — over a wide set of red-team-style prompts.

- **Strengths:** Purpose-built for safety training. Complements a helpfulness set well.
- **Weaknesses:** Safety judgments are policy-specific; you may not agree with every safe/unsafe label.

### Chatbot Arena preference data

LMSYS's Chatbot Arena has released large sets of side-by-side conversation pairs with model-vs-model win/loss labels from real users. `lmsys/chatbot_arena_conversations` is one Hub artifact; there are others.

- **Strengths:** Real users, real prompts, up-to-date. Excellent evaluation signal.
- **Weaknesses:** Coverage skew toward the users of the Arena; annotator preferences are less controlled than a labeled pool.

### Others worth naming

- **Nectar** (Zhu et al., 2023) — GPT-4-scored multi-way preferences. UltraFeedback's cousin.
- **Distilabel-generated sets** — Argilla's `distilabel` tool generates preference data via LLM-judge pipelines; produced many recent DPO datasets.
- **DPO-mix-7k** and related "curated mix" datasets — small, high-quality, hand-curated blends useful for quick DPO smoke tests.

## The two format conventions

Two on-disk shapes dominate the ecosystem. TRL handles both; some conversion is usually required.

### Standard format: `prompt`, `chosen`, `rejected`

```json
{
  "prompt": "Explain the Fermi paradox to a 10-year-old.",
  "chosen": "Imagine the universe is huge — much bigger than we can even picture...",
  "rejected": "The Fermi paradox refers to the apparent contradiction between..."
}
```

`prompt` is the user turn (or the full conversation up to the assistant's response). `chosen` and `rejected` are the two candidate assistant responses. TRL trainers can apply the chat template themselves if you also pass a tokenizer with one set.

### Conversational format: `chosen` and `rejected` as full message lists

```json
{
  "chosen": [
    {"role": "user", "content": "Explain the Fermi paradox to a 10-year-old."},
    {"role": "assistant", "content": "Imagine the universe is huge..."}
  ],
  "rejected": [
    {"role": "user", "content": "Explain the Fermi paradox to a 10-year-old."},
    {"role": "assistant", "content": "The Fermi paradox refers to..."}
  ]
}
```

This shape is used when the "prompt" is a multi-turn conversation and you want the trainer to apply the tokenizer's chat template to each message. Prefer this shape when your data is multi-turn.

TRL's `DPOTrainer` since around version 0.9 accepts both; check your version's docs (linked in `resources.md`).

### KTO format: labeled singletons

KTO (chapter 6) takes *unpaired* data:

```json
{"prompt": "...", "completion": "...", "label": true}   // chosen
{"prompt": "...", "completion": "...", "label": false}  // rejected
```

You can convert a standard pair set to KTO format by splitting each pair into two labeled rows. Chapter 6 covers when this is worth doing.

## Quality controls — what actually makes a set usable

A preference dataset can look enormous on paper and still teach a model the wrong things. The quality controls that matter, in rough order of impact:

### 1. The chosen response must actually be better

This sounds obvious. In LLM-judged sets, it often is not. Common failure patterns:

- **Both responses are wrong**, and the judge picked the "more plausibly-worded" one. The model learns to sound authoritative while being incorrect.
- **The chosen response is more verbose but not more correct**. Judges (especially LLMs) systematically favor longer answers. The model learns to pad.
- **The judge is prompt-primed** toward one style (e.g., markdown-formatted, bulleted). The model learns that style regardless of whether it fits the prompt.

**Mitigation:** hand-inspect at least 50–100 pairs from any dataset before you commit to it. Read the prompt, both responses, and the label. Ask "would I have picked chosen over rejected here?" If your agreement rate with the label is below ~85%, you have a data problem, not a training problem.

For LLM-judged data, run a second-judge cross-check on a sample: sample 500 pairs, ask a *different* judge model to re-label them, measure agreement. Under ~80% agreement with the original judge means the labels are noisier than the training signal will tolerate.

### 2. Chosen and rejected must be from the same task

The pair must respond to the same prompt with a comparable *shape* of answer. If chosen is a full-code implementation and rejected is a one-sentence refusal, you are training preference for "answer over refuse" — not for the substantive difference you probably intended.

**Mitigation:** filter pairs where the two responses differ wildly in *kind*. A cheap classifier ("was this response a refusal / a code block / a bulleted list?") applied to both sides, followed by drop-rows-where-they-mismatch, catches most of these.

### 3. Length must not be the dominant signal

This is important enough it has its own section below.

### 4. Prompts must cover the task distribution you care about

An UltraFeedback-only DPO run makes a model that responds well to *UltraFeedback-shaped prompts* — general chat, one-turn, mildly technical. If your production surface is code assistance or multi-turn agents, that distribution shift will show up in evals.

**Mitigation:** produce a prompt histogram (task type, domain, length) before you train and confirm it matches what production sees. Supplement with domain-specific pairs if the base corpus under-represents your surface.

### 5. Multi-turn / single-turn parity

If you train on only single-turn preferences but serve a multi-turn chat, you will get regressions on long conversations. Include multi-turn examples in the training set at roughly the same rate as production.

### 6. Deduplication

Duplicate `x` across the set — especially with different `y_w` / `y_l` — creates conflicting gradients. Same prompt, chosen A once and rejected A elsewhere confuses the loss. Dedup by prompt hash before training; keep the highest-confidence pair per prompt if you have multiple.

### 7. No train/eval leakage

The eval preference set (used for RM eval or DPO eval loss) must have no prompt overlap with training. Hash the prompts and check. This is the same discipline as SFT decontamination (mod-102 chapter on decontamination); the failure mode is identical.

## Length bias — the specific failure worth naming

Longer responses have a systematically higher chance of being marked chosen, both by humans and by LLM judges. The consequence: a preference-tuned model learns to produce longer responses, whether or not longer is genuinely better.

The empirical evidence is thorough. Singhal et al., 2023 ("A Long Way to Go: Investigating Length Correlations in RLHF") measured length correlations across HH-RLHF and other sets and showed that a reward model trained on the raw pairs can be beaten by a *trivial* length-based scorer on a nontrivial fraction of the eval set. Their conclusion: much of RLHF's apparent "helpfulness" gain is length gain in disguise.

Consequences for your dataset:

- **Chosen / rejected length distribution.** Histogram both. If chosen is systematically 1.5–3× longer than rejected, your set is biased.
- **Length-conditional accuracy.** If you evaluate a candidate RM on your set, split by length delta and compute pairwise accuracy separately. If RM accuracy on "chosen longer" pairs is 90% and on "chosen shorter" pairs is 55%, your RM has learned "longer = better" — not what you meant to train.
- **The trained model's output length.** A model that produces 400-token responses after DPO where it used to produce 200 was likely trained on a length-biased set. This is not automatic evidence of a bug — some tasks want longer answers — but it is a signal to check.

Three mitigations, from lightest to heaviest:

1. **Filter pairs where the length delta is extreme.** Drop pairs where `len(y_w) > 2 · len(y_l)` or `len(y_w) < 0.5 · len(y_l)`. This throws away real signal (some longer responses are genuinely better), but the pairs that remain give a cleaner training signal.
2. **Length-balance the training set.** Sample the set so that "chosen longer than rejected" pairs and "chosen shorter than rejected" pairs are roughly equal. The model cannot learn "longer = better" if the training signal does not carry it.
3. **Use a length-normalized method.** SimPO (chapter 6) normalizes the log-probability by sequence length in the reward-margin term. This does not fix the underlying dataset bias but at least does not compound it in the objective.

Do at least one of these. All three is standard practice for a "we care about response length" production run.

## The dataset-card checklist

Every preference set you train against should be shipped with a dataset card (analogous to mod-102's data mix card) that answers, at minimum:

- **Provenance.** Where did the prompts come from? Where did each side of the pair come from? What model(s) generated the responses? What annotator (human pool / LLM judge / rule) produced the labels?
- **Sizes.** Total rows, unique prompts, per-source counts, held-out eval size.
- **Length statistics.** Median and 90th-percentile tokens for `prompt`, `chosen`, and `rejected`. The chosen-vs-rejected length delta distribution.
- **Label statistics.** For LLM-judged sets, the judge model's ID and a link to its prompt. For human sets, inter-annotator agreement (Cohen's κ or equivalent) on a re-labeled sample.
- **Filter chain applied.** Every filter you ran, in order, with the row count before and after.
- **License and use restrictions.** Every source dataset carries its own; the derived set is only as permissive as its most-restrictive input.
- **Decontamination check.** Which eval benchmarks did you decontaminate against, and with what n-gram / prompt-hash overlap threshold?

Exercise 01 has you produce exactly this artifact. Every downstream chapter assumes you have one — an RM eval, a DPO run, or a PPO run without it is not reproducible in the sense mod-103 defined.

## Common recipes to copy from

Reference constructions worth reading (all on the Hub):

- **`HuggingFaceH4/ultrafeedback_binarized`** — the standard DPO training set used in the Zephyr-7B and alignment-handbook recipes. Directly compatible with `DPOTrainer`.
- **`Anthropic/hh-rlhf`** — the reference dataset for reproducing the InstructGPT-style RLHF pipeline. Both helpful and harmless splits.
- **`Intel/orca_dpo_pairs`** — a 12k-pair DPO set built on top of Orca-2 with GPT-4 judgments. Good for smoke tests.
- **`argilla/dpo-mix-7k`** — a curated 7k blend commonly used for fast DPO smoke tests; small enough to run in an hour on a single GPU.
- **`nvidia/HelpSteer2`** — Nvidia's preference dataset with dimension-specific ratings; strong for reward modeling.
- **`allenai/tulu-3-pref-personas-instruction-following-*`** — Ai2's Tulu-3 preference data, with strong per-domain coverage.

Skim two before you write your own. The Hub dataset viewer + a hand-tour of 50 examples usually tells you within an hour whether the set is a fit.

## Splitting for training and evaluation

Every set gets split into three:

1. **Train.** What the DPO / RM / PPO training loop iterates over.
2. **Eval loss.** A held-out slice you compute the training-time loss / RM accuracy on at every eval step. Same-distribution as train.
3. **Preference-benchmark eval.** A separate, out-of-distribution set for the "is this better" question — e.g., a held-out slice of Chatbot Arena, or a MT-Bench-style pairwise eval.

The first two live inside `datasets.DatasetDict` (`train` and `test` splits). The third is separately managed. Mixing them is the single most common way people accidentally overfit their preference-training set and never notice.

Sizes: for a DPO run on 7B, 5k–50k preference pairs in the train split is the working range. 500–2k in eval-loss. The preference-benchmark eval is whatever your production evaluation stack uses (mod-108 covers this in depth).

## Tokenization gotchas specific to preference data

Two gotchas that do not exist in SFT:

1. **The chat template must be the same for chosen and rejected.** If your tokenizer applies `<|assistant|>\n<response>\n<|end|>` to chosen but somehow leaves the end token off rejected, the log-probability comparison in DPO is measuring the presence of the token, not the response's quality. TRL's `DPOTrainer` handles this correctly if you pass the tokenizer with a chat template; hand-rolled loops must verify.
2. **Prompt length must be equal across chosen and rejected within a pair.** The prompt is the same in both branches of the pair, so `prompt_input_ids` are shared. Some data pipelines have a bug where the tokenizer emits slightly different token sequences for the same prompt because of leading-whitespace handling; assert `len(chosen_prompt_ids) == len(rejected_prompt_ids)` before you trust the batch.

Both of these are catchable by writing 20 pairs' worth of tokenized output to a JSONL file and eyeballing it. Do that before you train.

## Exercise pointer

`exercises/exercise-01-preference-pair-dataset-construction.md` has you take an open corpus (UltraFeedback, HH-RLHF, or another of your choice), apply the length and quality filters above, produce a dataset card, and hand off a `datasets.DatasetDict` ready for `RewardTrainer` and `DPOTrainer` in the following exercises.

## Summary

- A preference pair is `(prompt, chosen, rejected)`. It is binary; there is no magnitude. `chosen` is what an annotator (human / LLM / rule) preferred.
- Four sources, decreasing per-item quality: humans, LLM judges, verifiable rules, and existing open corpora. Most first runs use existing corpora (HH-RLHF, UltraFeedback, SHP, PKU-SafeRLHF, Chatbot Arena preference data).
- Two disk formats matter: the "prompt / chosen / rejected" flat shape and the "conversational message-list" shape. KTO uses a different, unpaired format.
- Quality controls: chosen must actually be better, chosen and rejected must be same-task, length bias must not dominate, prompt distribution must match production, dedup and no train/eval leakage.
- Length bias is severe and specific to this data shape. Filter, balance, or use a length-normalized method (SimPO); pick at least one.
- Ship a dataset card that documents provenance, sizes, length distributions, annotator info, filters applied, license, and decontamination. Every downstream trainer assumes one exists.
- Split into train / eval-loss / preference-benchmark. Do not conflate the eval-loss and benchmark splits — that is how overfit runs escape detection.
