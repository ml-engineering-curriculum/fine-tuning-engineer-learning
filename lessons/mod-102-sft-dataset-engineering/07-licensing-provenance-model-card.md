# Licensing, Provenance, and the Model Card

**Estimated read:** 40 minutes.
**You should be able to, by the end:** read a dataset license as an engineer (not a lawyer), recognize the common license buckets you will encounter on the Hugging Face Hub, trace per-example provenance from source to model card, and write the data section of an SFT model card that a downstream user can actually rely on.

## Why this matters (and why it is your job)

An SFT model without documented data provenance is a legal and reputational liability. A downstream user cannot know whether they may commercialize it; a security reviewer cannot know whether a leaked prompt came from your training set; a regulator or a customer's procurement team cannot check compliance with their own data-use policies.

You are not the legal team. You should not pretend to be. But you *are* the person with the ground truth about what went into the model, and the model card is your contract with everyone who uses it. Chapter 1's per-row metadata schema, chapter 5's synthetic provenance record, and chapter 6's mix card are the source-of-truth documents from which the model card is derived. This chapter closes the loop.

## Reading a dataset license as an engineer

You are not looking to reach a legal conclusion. You are looking to answer three questions:

1. **May I use this dataset to train a model?** Some licenses are training-permitted by default (Apache, MIT, CC-BY, most permissive Hugging Face-hosted datasets). Some restrict to research-only. Some require negotiated agreements.
2. **May I redistribute a model trained on this dataset?** Distinct from (1). Research-only datasets frequently permit training but forbid commercial redistribution of derived models.
3. **What obligations follow the model?** Attribution requirements, share-alike (copyleft), notification, use-case restrictions (no military, no surveillance, no automated decision-making about people).

You should be able to fill in a short row per source:

| Field | Example |
|---|---|
| `source` | `HuggingFaceH4/no_robots` |
| `spdx` | `CC-BY-NC-4.0` |
| `training_permitted` | yes |
| `commercial_redistribution` | no (NC clause) |
| `attribution_required` | yes |
| `share_alike` | no |
| `use_restrictions` | none beyond NC |
| `notes` | pull URL, dataset-card checksum |

Fill this row in *before* the source enters the mix. Do not defer the licensing question to the model-card phase; it will bite you.

## The license buckets you will meet

The exhaustive list of licenses on the HF Hub is large; the practically-relevant clusters are small.

### Permissive open-source

- **MIT**, **BSD-2/3-Clause**, **Apache-2.0** — training and commercial redistribution permitted; retain attribution.
- Common on curated open datasets (SlimPajama, OpenR1-Math, Nemotron-generated data).

### Creative Commons

- **CC-BY** — attribution required, commercial use permitted.
- **CC-BY-SA** — share-alike (any derivative must inherit the CC-BY-SA license). Contagious; verify compatibility with downstream uses before mixing.
- **CC-BY-NC** — non-commercial. Do **not** include if you plan to distribute a commercial model.
- **CC0** — public domain dedication; no obligations beyond good manners.

### Model-family community licenses

- **Llama 3 / 3.1 / 3.2 Community License** — permissive with named-derivative and MAU-threshold clauses. Data generated from a Llama model inherits terms; verify each release.
- **Gemma Terms of Use**, **Qwen License**, **Mistral Research License** vs. **Apache-licensed Mistral 7B** — read each; they differ, and they change across model versions.

### Research-only / academic

- **CC-BY-NC-SA**, **BAIR license**, various university terms.
- OpenAssistant is Apache-2.0, but many of the "instruction dataset" scrapes derived from ShareGPT are of unclear provenance and should be treated as research-only unless you have reason to believe otherwise.

### Vendor terms of use (not an OSS license)

- **OpenAI Terms of Use**, **Anthropic Commercial Terms**, **Google Vertex terms**. These are *not* licenses in the OSS sense; they are contractual terms between you and the vendor. Outputs generated under them carry the vendor's restrictions; the vendor's terms often explicitly restrict use to train competing models. <!-- needs-research: on the next research cycle, confirm the current OpenAI and Anthropic Terms of Use sections that govern training-on-outputs and re-cite with URL and effective date; these terms shift and any citation goes stale quickly -->

### "License: other" / no license

- The presence of a dataset on the Hub is not a license grant. Datasets marked `license: other` or with no license are effectively **all rights reserved** for legal purposes; you cannot include them in a mix intended for redistribution without permission.

## Provenance: from row to model card

Chapter 1's row schema, chapter 5's synthetic provenance record, and chapter 6's mix card should compose into a single audit chain. Concretely:

1. **Per-row provenance** — every row's `source`, `source_id`, `source_url`, `license`, `generator`, `generator_prompt_hash`, `filters_passed`, `decontam`. Row-level.
2. **Per-source manifest** — one row per source: name, license, size before filters, size after filters, per-eval decontam summary. Source-level.
3. **Mix card** — chapter 6's document listing categories, per-source weights, ablation results. Mix-level.
4. **Model card data section** — the human-facing distillation of the above. Model-level.

If any level cannot be reconstructed from the level below it, you have broken the chain. This is why the row-level metadata is worth the paperwork.

## The model card data section

**Mitchell et al. (2019) — "Model Cards for Model Reporting"** is the original reference. Hugging Face's model card documentation and the Data Cards Playbook (Google, 2022) are the practical templates most teams follow today.

An SFT model card's *data* section — the part this module makes you responsible for — should include:

### Sources

A table of every dataset included, with counts, license, and whether the data was used verbatim or filtered / re-generated.

### Data collection and preprocessing

- What filters ran, in what order. The chapter 1 filter stack, made concrete.
- What was deduplicated (chapter 2): normalization function, thresholds, cross-source vs. per-source.
- What was decontaminated against (chapter 3): eval names, n-gram sizes, thresholds, flag counts, and whether removed or annotated.

### Synthetic data disclosure

If any synthetic data went in:

- Teacher model(s) used, with version pins.
- Approximate volume of synthetic vs. human-authored rows.
- Licenses of the synthetic outputs, per teacher.
- Any Self-Instruct / Evol-Instruct pipeline documented well enough that another team could reproduce (or the exact commit hash of your pipeline).

### Data mix and rationale

- Capability categories and their proportions (from the mix card).
- Rationale for the ratios, ideally with ablation numbers.
- Known biases inherited from sources (e.g., "OpenAssistant tilts toward English STEM users; expected implication is stronger STEM performance and weaker non-Western persona").

### Personal data and safety data

- What PII handling was done (dropped rows vs. redacted-in-place).
- What kinds of refusal training rows were included, and what over-refusal risk mitigations were tested (chapter 6).

### Reproducibility

- Where the exact dataset artifact lives (internal S3 path, HF Hub private dataset, git commit hash of the pipeline).
- Whether the artifact will be released publicly.

### Known limitations

- Any eval where decontamination flagged non-zero rows and you chose to keep them (and why).
- Any source you would have liked to include but excluded for licensing or quality reasons.
- Any capability category the mix under-serves.

You are writing a document a future engineer — likely not you — will read to answer "can I use this model for X?" Aim for that reader.

## Two model cards worth reading

Actually read at least two published model cards end to end before you write yours. Recommendations:

- **Llama 3 / 3.1 model cards** (on `meta-llama/*` on the Hub) — the reference for large-scale disclosure at the vendor level.
- **Tülu 3** (AI2, `allenai/tulu-*`) — the reference for open, fully-reproducible SFT mix documentation.
- **BLOOM** (Le Scao et al., 2022) — the reference for governance-heavy multi-org model card structure.

You will notice they say different things and are structured differently; that is fine. There is no universal template, only a set of questions the reader must be able to answer.

## Governance handoff

At the end of an SFT run, the artifact bundle you hand over should include:

- The trained checkpoint (mod-103's output).
- The exact dataset artifact (rows + per-row metadata).
- The mix card.
- The decontamination report.
- The curation report.
- The model card draft.
- A "known issues" list of things you would fix on the next revision.

If any of these is missing, the handoff is incomplete. The model may work, but it is undocumented, and the next engineer to touch it — including you, six months later — will pay the tax.

## Exercise pointer

`exercises/exercise-05-dataset-mix-card-and-licensing-trace.md` has you produce the full mix card and licensing trace for a designed 100k mix, drawing the whole chain from per-source license to model-card draft.

## Summary

- You are not the legal team, but you are the source of ground truth about what went into the model. That makes documentation your responsibility.
- Fill in a per-source license row *before* the source enters the mix. Answer three engineer-level questions: may I train, may I redistribute, what obligations follow.
- Know the common license buckets: permissive OSS, Creative Commons (BY, BY-SA, BY-NC), model-family community licenses, research-only, vendor Terms of Use.
- "License: other" or missing license = all rights reserved for legal purposes; do not include without permission.
- Provenance chain: row → source manifest → mix card → model card. Each level must be reconstructible from the level below.
- Model card data section: sources, preprocessing, synthetic disclosure, mix rationale, PII / safety handling, reproducibility, known limitations. Read Llama 3, Tülu 3, and BLOOM cards for structure.
- Handoff bundle at end of run: checkpoint, dataset artifact, mix card, decontam report, curation report, model card, known issues.
