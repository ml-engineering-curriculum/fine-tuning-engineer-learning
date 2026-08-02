# exercise-05: Dataset Mix Card and Licensing Trace

**Estimated effort:** 2 hours
**Companion chapters:** [`06-dataset-mixing.md`](../06-dataset-mixing.md), [`07-licensing-provenance-model-card.md`](../07-licensing-provenance-model-card.md)

## Objective

Design a 100k-example SFT mix across at least four capability categories, produce the mix card that would ship with the resulting model, and write the licensing trace from per-source license to the model card. By the end you will have (a) the mix specification, (b) the mix card, (c) the licensing trace with a red / yellow / green verdict per source, and (d) the data section of a draft model card.

You do not need to actually assemble the 100k mix. This is a design and documentation exercise; the artifact is the paperwork.

## Prerequisites

- The metadata schema from exercise 01 and the decontamination results from exercise 02 as reference points (even if you did not run them yourself).
- Familiarity with dataset cards on the Hugging Face Hub. If you have not read one all the way through, start with `HuggingFaceH4/no_robots` or `allenai/tulu-3-sft-mixture`.

## Requirements

### Part A — Define the target

Write a one-paragraph target model description in `mix_target.md`:

- What is the model for (chat assistant, coding assistant, domain specialist, etc.)?
- What languages must it support?
- What is your redistribution intent (internal only, open-weights research release, commercial product)?

These three answers determine everything downstream — especially the license bar.

### Part B — Choose sources for at least four categories

Design a 100k mix across at least four of the capability categories from chapter 6. Suggested categories: general chat, reasoning (math/logic), code, safety refusals + non-refusals, tool use, persona.

For each category, pick 1–3 real public sources you would actually use. For each source, look up (do not guess):

- Its Hugging Face Hub repo (or GitHub, or other canonical home).
- Its declared license (the SPDX identifier if available).
- Its size.
- Whether it is human-authored, synthetic, or mixed.
- If synthetic: the teacher model and the license of that teacher's outputs.

Do not include a source whose license is `license: other` or unknown unless your redistribution intent from part A is "internal only."

Produce `sources.csv` (or `sources.md`) with one row per source containing at minimum:

```
source, url, spdx, size, generator, generator_license, category, notes
```

### Part C — Set proportions and produce the mix card

Decide per-category proportions that sum to 100%. Justify each in one sentence:

- What role does this category play in the target model's behavior?
- Why this proportion and not one bucket higher or lower?

Compute per-source target row counts. For sources smaller than their target: note the upsampling factor. For sources larger than their target: note the downsampling.

Produce `mix_card.md` with:

1. Target model description (from part A).
2. Category table (category, proportion, sources, aggregate rows).
3. Per-source table (source, license, category, filtered size, target rows, up/downsample factor).
4. Rationale paragraph per category.
5. A "known biases" section listing at least three biases the mix inherits from its sources (e.g., "OpenAssistant tilts toward English STEM contributors; the reasoning pile is disproportionately Western-notation math").
6. A "what we did not include and why" section listing at least two sources you considered and rejected, with reason.

### Part D — Licensing trace

Produce `licensing_trace.md` containing:

1. A **red / yellow / green table**, one row per source:
   - **Green** — permissive OSS or explicitly training-and-redistribution-permitted; no obligations beyond attribution.
   - **Yellow** — usable but with material obligations (share-alike, attribution required in the model card, notification required, community-license MAU thresholds, etc.). Note the specific obligation.
   - **Red** — cannot be used given the redistribution intent from part A. If you kept any red-flagged sources, either change the intent or drop the source.
2. A resolution paragraph for every yellow row: what obligation does the mix inherit, and how does the model card discharge it?
3. A named model-card license conclusion: given the mix, what license (or terms) will the resulting model be distributed under?

If the mix contains outputs from a proprietary teacher (OpenAI, Anthropic, Google) and your redistribution intent is anything other than internal, this is where you either resolve the vendor-terms question or drop that source. <!-- needs-research: on the next research cycle, add pointers to current OpenAI and Anthropic Terms of Use sections governing outputs used for training competing models, with URLs and effective dates -->

### Part E — Draft the model card data section

Write `model_card_data_section.md` — the data section of the model card the trained model would ship with. Follow the structure from chapter 7:

- Sources
- Data collection and preprocessing
- Synthetic data disclosure
- Data mix and rationale
- Personal data and safety data
- Reproducibility
- Known limitations

Keep it tight. Two to three pages is the right length. A reader — a downstream user, a security reviewer, a partner-team engineer — should be able to answer "should I use this model for X?" from this document alone.

## Starter guidance

- Do not aim for a novel mix. Steal shamelessly from Tülu 3's published proportions or the Llama 3.1 model card's disclosed capability breakdown, then adjust for your target model description. "Similar to Tülu 3 but with 5% more code and no long-context" is a defensible starting point; "we made up these numbers" is not.
- The SPDX identifier registry (spdx.org/licenses) is the reference for standard license names. Use those identifiers rather than free-form license strings.
- For each source, open its Hugging Face dataset card and read the license section literally. If the card conflicts with the license file in the repo, note the conflict — this is a real thing that happens.
- For synthetic sources built on proprietary teachers, treat vendor Terms of Use as separate from the dataset's stated license. A dataset labeled "MIT" but generated with GPT-4 still inherits OpenAI's terms on its content.
- The Data Cards Playbook (Google, 2022) and Model Cards for Model Reporting (Mitchell et al., 2019) are the reference documents for structure — both linked in `resources.md`.

## Acceptance criteria

You have completed the exercise when:

- `mix_target.md`, `sources.csv`, `mix_card.md`, `licensing_trace.md`, and `model_card_data_section.md` all exist.
- Every source in the mix has a filled-in SPDX identifier or a documented explanation of why one is not applicable.
- The category proportions in the mix card sum to 100% and are justified per category.
- The licensing trace has no `red` rows in the shipped mix; every `yellow` row's obligation is discharged in the model card draft.
- The model card data section can be read in five minutes and answers the "should I use this model" question directly.

## Stretch goals

- Reproduce Tülu 3's published mix (from its dataset card and paper) as your baseline, then design a variant that changes exactly one proportion — and describe the ablation you would run to validate the change (from chapter 6's protocol). Include what eval numbers would justify the change.
- Sketch a **model card diff** between two variants of the same base — one trained on your designed mix and one on an alternative — so downstream users can compare defensibly.
- Extend the licensing trace to include **downstream** obligations you would put in your model card's user-facing license (e.g., "any model fine-tuned from this checkpoint must include the following attribution list"). This is where model-family community licenses (Llama 3) meet share-alike obligations from Creative Commons sources.
- Turn the whole exercise into a reusable template — one directory with each artifact scaffolded so a teammate can produce a comparable mix card in a day. Ship the template internally.
