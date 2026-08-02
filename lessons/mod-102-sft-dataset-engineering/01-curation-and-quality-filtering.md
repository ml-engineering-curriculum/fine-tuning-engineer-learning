# Curation and Quality Filtering for SFT

**Estimated read:** 45 minutes.
**You should be able to, by the end:** define — in a written rubric — what makes an instruction–response pair worth training on for your model, name the small set of filters that most production SFT mixes actually apply, and specify the per-example metadata schema that lets you defend every decision on the way to a model card.

## Why curation is more than "run a filter script"

In pretraining, the working assumption is "more tokens is more better, subject to a light quality bar." SFT flips this: you are teaching a specific behavior — following instructions, answering in a specific format, refusing specific requests — from a comparatively tiny dataset. Every bad example is a *worked example of bad behavior*, and the model is more than capable of copying it.

Two well-cited results anchor how strong this "quality over quantity" effect can be:

- **LIMA** (Zhou et al., 2023) fine-tuned LLaMA-65B on 1,000 carefully curated prompt–response pairs and reached, on their human evaluation, response quality comparable to or preferred over much larger instruction sets. The takeaway is not "1k is the magic number" — it is that curation quality dominates raw count once you clear a very small floor.
- **Alpaca** (Taori et al., 2023), the reference "cheap distillation" run, is 52k `text-davinci-003`-generated instruction pairs, unfiltered beyond a light dedup. Follow-up work repeatedly showed that even modest quality filtering on that set improves downstream evals.

You will not build a dataset the size of pretraining. You will build one you can defend row by row.

## What "quality" means for an SFT example

An SFT example is a `(prompt, response)` pair — or, for multi-turn, a full conversation — that you want the model to produce more of. The quality dimensions that repeatedly show up in dataset papers and open recipes are:

1. **Instruction clarity.** The prompt is well-formed, unambiguous, and specifies the response shape (format, length, tone) when it matters. Ambiguous prompts teach the model that any of several outputs are equally correct, which increases entropy where you did not want it.
2. **Response correctness.** The response actually answers the prompt. For factual claims: the claims are true. For code: the code runs and does what it says. For reasoning: the chain of thought is coherent and the final answer follows from it.
3. **Response completeness.** The response does not truncate mid-thought or leave the user hanging. Truncation is disproportionately dangerous — it teaches the model to stop early, and it silently trains "the assistant abandons long tasks."
4. **Format faithfulness.** If the prompt asks for JSON, the response is valid JSON. If it asks for a code block, the response is *only* a code block, not "sure, here's your code: ```…```" wrapped in prose the user did not ask for.
5. **Absence of teacher artifacts.** For distilled data (mod-102 chapter 5): no "As an AI language model, I…" openers, no meta-commentary about the teacher's own limitations, no leaked system prompts. These artifacts are exactly what the model will learn to emit.
6. **Safety alignment with your policy.** Every refusal in your training set is a worked example of what to refuse. Refusing benign requests is as harmful to the model as failing to refuse harmful ones.
7. **License and provenance cleanliness.** You can name the source, the license, and the terms under which you may redistribute a model trained on this row. Chapter 7 makes this a hard-blocking requirement.

The right rubric is **explicit and written down before you start filtering**. Reviewers can then apply the same rule; automated filters can be tested against a hand-labeled sample; and — critically — you can revisit the rubric when your first eval comes back and adjust rather than guess.

## The per-example metadata schema

The single highest-leverage habit in dataset engineering is refusing to ever store an SFT row as a bare `{"prompt": ..., "response": ...}` object. Every row carries a metadata envelope that follows it through cleaning, mixing, and model-card generation.

A minimal schema — you can add fields, but you should not remove any of these:

```jsonc
{
  "id": "oa-2023-h2-000123",                 // globally unique, stable across runs
  "source": "OpenAssistant/oasst1",          // dataset of origin
  "source_id": "abc123",                     // native ID inside that source
  "license": "Apache-2.0",                   // SPDX identifier — see chapter 7
  "source_url": "https://huggingface.co/datasets/OpenAssistant/oasst1",
  "category": "general_chat",                // for mix ratios in chapter 6
  "language": "en",                          // ISO 639-1 or "multi"
  "generator": null,                         // null if human, model name if synthetic
  "generator_prompt_hash": null,             // for synthetic: hash of the generator prompt
  "quality_score": 0.87,                     // whatever scalar quality signal you use
  "filters_passed": ["length", "language", "toxicity", "response_quality"],
  "decontam": {"mmlu": "clean", "humaneval": "clean", "gsm8k": "clean", "mtbench": "clean"},
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

This looks like paperwork. It is paperwork. It is also what makes the difference between "we trained on some data" and a model card whose data section is auditable. When mod-108 comes back with an eval regression, this schema is what lets you slice loss by source, license, generator, or category and localize the cause in an afternoon instead of a week.

## The filter stack

Almost every production SFT curation pipeline is a chain of small, independently-testable filters. Order matters: cheap filters run first so expensive filters see fewer rows. A representative stack:

### 1. Structural filters (cheap, near-perfect precision)

- **Schema validity.** The row deserializes, has the required fields, roles are one of `{system, user, assistant, tool}`, and alternate correctly (no two assistant messages in a row unless your template supports it).
- **Non-empty.** No empty user or assistant messages. Empty assistant text is a common corruption from scraped forum data.
- **Length bounds.** Drop rows shorter than a small floor (usually a signal of noise — one-word prompts, "ok" responses) and longer than your effective `max_seq_length` after templating, or plan to truncate deliberately (chapter 4).

### 2. Language filters

Use a language identifier (fastText `lid.176`, `langdetect`, or a small multilingual classifier) on user and assistant separately. Decisions to make explicitly:

- Do you keep only your target languages, or all languages? Multilingual mixes are legitimate but change your data-mix decisions in chapter 6.
- Are user and assistant allowed to be in different languages (code-switching, translation prompts)? Say so explicitly in the rubric.

### 3. PII and sensitive-content filters

Real user-submitted data (OpenAssistant, ShareGPT-style dumps) contains phone numbers, emails, home addresses, sometimes credit-card numbers. Presidio, `pii-leakage`, or a purpose-built regex + NER classifier will find the common cases. Two levers:

- **Drop the row entirely**, or **redact the PII in place** and keep the row. Both are defensible; pick one policy and record which per row (`pii_action: "redacted" | "dropped" | "clean"`).
- Distinguish PII-in-the-prompt (users often ask "here is my resume, review it") from PII-in-the-response (the model should generally not emit real PII in trained-on responses).

### 4. Toxicity / harmful-content filters

You almost certainly do not want to train the model to reproduce the most toxic responses in your source data. Small classifiers (Detoxify, Perspective API, or a modern content-moderation model) give a per-message toxicity score. Two important nuances:

- These classifiers have well-documented false positives on medical, legal, and dialect content. Log threshold decisions so you can revisit.
- Do not silently drop *refusals to toxic prompts*. Those are exactly the safety training data you probably want to keep (mod-106 will use them). Separate "user message is toxic" from "assistant response is harmful."

### 5. Response-quality filters

This is where dataset engineering earns its keep. Cheap heuristics catch a lot:

- **Repetition.** Ratio of unique n-grams to total n-grams in the response. Very low ratio → the model looped.
- **Prefix / template artifacts.** Response starts with "Sure! Here is…", "As an AI language model…", "I am ChatGPT…", `<start>` tokens, or leaked system-prompt fragments. Any of these is a worked example of behavior you do not want.
- **Refusal-when-not-warranted.** Response is a refusal ("I cannot help with that") on a manifestly benign prompt. Keeping these teaches the model to over-refuse.
- **Format compliance.** If the prompt asked for JSON, does `json.loads` succeed? For code prompts, does the response contain a fenced code block? These are per-domain checks that pay for themselves.

Expensive but powerful filters use a **judge model** — a stronger LLM scoring each response 1–5 on helpfulness / correctness / format. This is the technique behind **AlpaGasus** (Chen et al., 2023), which showed that filtering Alpaca down to the highest-scoring ~9k of the 52k examples produced a better model. The `Instruction-Following-Difficulty` metric (Li et al., 2024) is a lower-cost proxy that uses the base model itself as scorer. If you use a judge model, decide whether the judge is a hard filter or a soft signal (retained in metadata for later mix decisions), and always sample-check the judge's calls against human labels.

### 6. Deduplication and decontamination

Covered in chapters 2 and 3. They are filters in the same pipeline, not separate stages. Every row that passes should have `decontam` fields filled in.

## Two representative open curation recipes

Two open SFT datasets are worth reading end to end for their published curation methodology:

- **OpenAssistant / OASST1** (Köpf et al., 2023) — a crowd-sourced multi-turn conversation tree. The paper documents contributor guidelines, per-message quality ratings, and how conversation trees were filtered to training-ready threads. Reading the paper alongside the dataset viewer is a masterclass in what "provenance" looks like in practice.
- **No Robots** (Rajani et al., 2023, HuggingFaceH4) — 10k hand-authored prompt–response pairs, no synthetic data. Documented category breakdown and explicit "no LLM output" invariant. The dataset card is the reference for how tightly a small human-authored SFT set can be scoped.
- **LIMA** (Zhou et al., 2023) — 1k examples, hand-curated, spanning multiple domains. The paper's appendix on sourcing is the reference for "how thin can a well-curated SFT set be and still work."

You should not assume these are the *right* data for your model. You should read them to internalize the level of documentation your own set should reach.

## Building the audit report

At the end of a curation pass, generate a report that a colleague can read in five minutes:

- Row counts in and out per filter, with drop rates.
- Per-category and per-source counts after all filters.
- Language distribution.
- Length distribution (histogram of user-message and assistant-message tokens, using the tokenizer you will train with — mod-101 chapter 2 explains why cross-tokenizer comparisons lie).
- Sampled examples of what each filter dropped, so a reviewer can spot false positives. **Never trust a filter you have not eyeballed the drops from.**

This report is the raw input to the model-card data section (chapter 7). Ship it as part of the dataset artifact, not as a one-off diagnostic.

## Exercise pointer

`exercises/exercise-01-sft-data-quality-audit.md` has you build a real curation report against one of the open datasets above, with the metadata schema above, and hand back the before/after diff of your filter stack.

## Summary

- SFT is disproportionately data-quality-bound; bad examples train bad behavior faster than good examples train good behavior.
- Write down a quality rubric before you filter; a filter is only as good as the rubric it enforces.
- Never store rows without a metadata envelope (id, source, license, category, language, generator, quality signals, decontam status). This is what makes a model card auditable later.
- Filter in cheap-to-expensive order: structure → language → PII → toxicity → response quality → dedup → decontamination.
- Cite two or three open datasets whose curation methodology you can name (OASST1, No Robots, LIMA are good starters). Match their level of documentation for your own set.
- Ship a curation report with the dataset, not the pipeline logs.
