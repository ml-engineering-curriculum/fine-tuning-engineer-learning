# Decontamination Against the Big-Four Evals

**Estimated read:** 40 minutes.
**You should be able to, by the end:** define what "contamination" means for MMLU / HumanEval / GSM8K / MT-Bench, run an n-gram-overlap scan of an SFT set against each, and know which decontamination recipe the major open model cards (Llama 2, Llama 3, Mistral) actually use.

## What contamination is (and is not)

Contamination is the presence, in your training data, of examples that overlap with the evaluation benchmark you plan to score against. If your training set includes MMLU questions and you then evaluate on MMLU, your model has effectively been trained on part of the test — the eval number no longer measures what you think it measures.

Two related-but-distinct concepts to keep straight:

- **Test-set contamination** (this chapter): training data contains items that also appear in the eval's test split. Directly inflates eval scores.
- **Test-set memorization**: the model reproduces eval items verbatim from a contaminated training set. A downstream *symptom* of contamination, not the same thing.

Contamination is not a single yes/no. Overlap is a spectrum from "byte-identical" through "identical prompt with different formatting" to "same underlying question paraphrased." All three inflate scores; the decontamination pipeline is what draws the line for your model card.

## Why SFT is a contamination minefield

Pretraining corpora at trillion-token scale are contaminated by construction — MMLU questions are on the web, GSM8K problems appear in Common Crawl. That is a known artifact and every published base model has a decontamination section acknowledging it.

SFT is different. SFT contamination is largely *avoidable* because the data is much smaller and the sources are named. It also happens *more* easily because:

1. **Instruction datasets scrape the same forums and QA sites** the eval builders drew from. Alpaca and its descendants used a Self-Instruct pipeline seeded by 175 tasks; that pipeline can absolutely regenerate examples that overlap MMLU or GSM8K by topic.
2. **Synthetic pipelines call teacher models** that have themselves seen the evals. A teacher asked to "generate 100 grade-school math word problems" will happily produce GSM8K-like items — sometimes verbatim.
3. **ShareGPT-style dumps** capture users who literally paste eval questions to see how the assistant answers. Those pastes end up in training data if you do not filter.

The published open model cards catch this. Llama 2 (Touvron et al., 2023b, appendix) and Llama 3 (Meta, 2024) both include decontamination analyses against the eval suites they report. If you plan to publish comparable numbers, you owe the reader the same analysis.

## The four evals

You need to know what each of these actually looks like before you can decontaminate against them.

### MMLU — Massive Multitask Language Understanding

**Hendrycks et al. (2021).** ~15,908 multiple-choice questions across 57 subjects (STEM, humanities, social sciences, professional). Format is a stem, four choices A/B/C/D, and a gold letter. The `test` split (14,042 items) is what everyone scores against; `dev` (5 items per subject) is few-shot exemplars.

Contamination target: the stem and the choices together, since a paraphrased stem with the same choice ordering is a near-perfect leak.

### HumanEval

**Chen et al. (2021).** 164 Python programming problems. Each item has a function signature, a docstring describing the task, and a set of hidden unit tests. Solutions are graded pass@k on the tests.

Contamination target: the docstring plus the function signature. A model that has seen the docstring text can produce the canonical solution without any programming ability. Watch also for canonical solutions themselves appearing in code datasets — this is common because the HumanEval problems are simple and appear on tutorial sites.

### GSM8K

**Cobbe et al. (2021).** 8,500 grade-school-level math word problems (7,473 train, 1,319 test). Each item has a natural-language problem and a solution that walks through arithmetic reasoning ending in `####  answer`.

Contamination target: the problem text. Because problems are ~2–4 sentences, exact-match dedup catches obvious cases but semantic paraphrases slip through — hence n-gram overlap at 8- to 13-grams is the recommended threshold in most decontam recipes.

### MT-Bench

**Zheng et al. (2023).** 80 two-turn conversation prompts across 8 categories (writing, roleplay, extraction, reasoning, math, coding, STEM knowledge, humanities). Judged by GPT-4 on a 1–10 scale, not by exact match. Small but expensive to eval.

Contamination target: the first-turn prompt text. Because MT-Bench prompts are hand-written and distinctive ("Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences…"), even light paraphrases are easy to catch. Because the eval is judged, contamination shows up as suspiciously high scores rather than perfect scores.

## The n-gram overlap recipe

The canonical decontamination method — used in the GPT-3 paper (**Brown et al., 2020**, appendix C), the Llama 2 model card, and most published open recipes — is n-gram overlap detection.

The algorithm:

1. Choose an `n` (commonly 8 to 13 words; longer `n` = fewer false positives, more false negatives).
2. For each eval item, extract every word `n`-gram from the prompt text.
3. Build a hash-set (or Bloom filter, at scale) of all eval `n`-grams.
4. For each training row, extract every word `n`-gram of the same size.
5. Score the training row: number of overlapping `n`-grams, or fraction of the training row's `n`-grams that appear in the eval set.
6. Flag / remove rows above a threshold.

A minimal reference implementation:

```python
def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = text.lower().split()
    return {tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)}

# Build the eval n-gram set once
eval_ngrams: set[tuple[str, ...]] = set()
for item in eval_items:
    eval_ngrams |= word_ngrams(item["prompt"], n=13)

# Scan the training set
def contamination_score(row) -> float:
    grams = word_ngrams(row["prompt"] + " " + row["response"], n=13)
    if not grams:
        return 0.0
    return len(grams & eval_ngrams) / len(grams)
```

Parameter defaults that the field has converged on:

- **`n = 8`** for a conservative-catch, higher-false-positive scan (from Brown et al., 2020 for GPT-3).
- **`n = 13`** for a stricter, near-verbatim scan (used in several Llama-family reports).
- **Threshold**: often "any overlap at all" for `n=13`, since matching a 13-gram by chance is vanishingly rare on non-boilerplate text. For `n=8`, thresholds of ≥1% of the row's n-grams overlapping are common.

Do the scan against **both prompt and response** of training rows: contamination sometimes appears only in the response (e.g., a training row where the assistant restates the MMLU question in its answer).

## Where n-gram overlap fails

Three failure modes to know:

1. **Paraphrasing.** "Which of the following best describes photosynthesis?" and "What does the process of photosynthesis primarily involve?" share no long n-gram but are essentially the same eval item. An embedding-similarity pass on top of n-gram is prudent when your eval items are short (MT-Bench, GSM8K) or when you know your synthetic pipeline paraphrases heavily.
2. **Code.** HumanEval problems have short function signatures and docstrings. Char-level overlap or AST-based comparison catches more than word n-gram overlap. Tools like `bigcode-project/bigcode-evaluation-harness` include specific HumanEval-decontam checks.
3. **Multiple-choice format drift.** MMLU items reformatted from "A. x / B. y / C. z / D. w" to a JSON schema will not overlap on n-grams. If your synthetic pipeline can emit multiple-choice questions, decontaminate the stem *without* the choice formatting.

The pragmatic answer: run n-gram at two sizes (8 and 13), run embedding similarity for short evals (MT-Bench, GSM8K), and hand-check the top-100 highest-scoring flagged rows.

## The decontamination report

Every decontamination pass should end in a report with:

- Rows scanned; rows flagged per eval; rows removed per eval.
- Per-eval false-positive rate on a hand-sampled subset (aim for ≥50 sampled flags per eval).
- The n and threshold used per eval.
- Ten example flagged rows per eval, with the matching eval item, so a reader can judge for themselves.
- The final `decontam` field written back onto each surviving training row (see the schema in chapter 1).

This report attaches to the model card. When someone challenges an eval number six months from now, this is what you point at.

## When contamination is discovered late

You published a model, someone downstream ran a contamination scan you did not, and they found leaks. The playbook:

1. Reproduce their finding. Ideally with their code.
2. Determine whether the leak is in *your* training data or was inherited from the base model's pretraining. The former is on you; the latter is a known industry issue but should still be disclosed.
3. Update the model card with the finding, the extent of overlap, and whether reported evals are re-run with contaminated rows removed.
4. Decide whether to re-train — for meaningful contamination, yes; for a handful of border-case n-gram hits, usually no, but disclose.

The failure mode you want to avoid is contamination becoming an accusation you cannot answer because you cannot reproduce your own dataset. This is why chapter 1's per-row metadata schema and chapter 7's provenance discipline are load-bearing.

## Exercise pointer

`exercises/exercise-02-eval-set-decontamination.md` has you build the full pipeline against a real SFT set and produce the report.

## Summary

- Contamination is training data that overlaps eval items; it directly inflates eval scores and is a released-model liability.
- SFT contamination is largely avoidable because sources are named and sets are small — but it happens easily via scraped ShareGPT dumps, teacher-model leaks, and Self-Instruct paraphrases of eval-adjacent topics.
- The four evals to run against: MMLU (~14k multiple-choice), HumanEval (164 Python), GSM8K (~1.3k math), MT-Bench (80 conversation prompts). Know their format before you decontaminate.
- The published recipe is word n-gram overlap (Brown et al., 2020, appendix C). Common defaults: `n=8` conservative, `n=13` strict; scan both prompt and response.
- N-gram overlap misses paraphrases (add embedding similarity for short evals) and code (add AST or char-level for HumanEval).
- Ship a decontamination report with per-eval flag counts, hand-sampled false-positive rates, and the final `decontam` field written back to every training row.
