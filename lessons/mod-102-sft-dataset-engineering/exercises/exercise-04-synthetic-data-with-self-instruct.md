# exercise-04: Synthetic Data with Self-Instruct

**Estimated effort:** 4 hours
**Companion chapter:** [`05-synthetic-data-self-instruct-evol-instruct.md`](../05-synthetic-data-self-instruct-evol-instruct.md)

## Objective

Implement a small Self-Instruct pipeline (with one or two Evol-Instruct-style depth operators bolted on), generate a small synthetic SFT set, and produce the full provenance record for every generated row. By the end you will have (a) a written teacher-license decision made *before* generation, (b) a runnable pipeline of ~200 generated rows, (c) three quality filters wired in, and (d) provenance metadata that would survive a code review.

This exercise is small on purpose. Two hundred generations is enough to see every failure mode of the pipeline; two hundred thousand is not necessary to prove the point and burns money.

## Prerequisites

- Python 3.10+ with `openai` or `anthropic` or a local inference stack (vLLM, llama-cpp, Ollama). You are the one choosing the teacher — see part A.
- `datasets`, `sentence-transformers`, and any regex / dedup libraries you prefer.
- A small hand-authored seed set (you will write it in part B).

## Requirements

### Part A — Resolve the teacher-license question first

Before writing a line of pipeline code, write `teacher_license.md` (one page) that answers:

1. Which teacher model will you use?
2. Under what license or terms of use are its outputs distributed?
3. Would a model fine-tuned on those outputs be redistributable? Commercially?
4. What obligation follows the data (attribution, sharing, use restrictions)?

If your chosen teacher's terms are ambiguous, say so explicitly. If they are clearly training-prohibited for your intended downstream use, pick a different teacher. Suggested open teachers with permissive-for-training terms include `nvidia/Nemotron-4-340B-Instruct`, Llama 3.1 405B under its community license (read the license carefully), and Qwen 2.5 72B under the Qwen license. <!-- needs-research: on the next research cycle, confirm the current NVIDIA Open Model License terms for Nemotron-4 outputs and the Llama 3.x community license clauses on synthetic-data derivatives, and cite with URLs and effective dates -->

Do not skip this step. The provenance record you build in part D includes the license string; that string must be defensible.

### Part B — Author the seed pool

Write 20 hand-authored seed instruction–response pairs, spanning at least three categories (e.g., reasoning, code, general chat). Store as `seeds.jsonl`. Requirements:

- Each has a clear task type and a well-formed response.
- No two seeds cover the same task at the same difficulty; you are trying to prime *diversity*, not fill space.
- No seed is copied from an existing public dataset — you should be able to hand a reviewer the file and say "I wrote every one of these."

### Part C — Implement the Self-Instruct loop

Build a pipeline that:

1. **Samples** 3 seed instructions per round (plus, once the pool has any, 3 previously-generated instructions).
2. **Prompts** the teacher: "Given the following example instructions, come up with a new, distinct instruction of similar quality but on a different topic." Record the exact generation prompt (its hash goes into provenance).
3. **Generates** a response for the new instruction with the teacher.
4. **Filters** the resulting pair — see part E.
5. **Optionally applies one Evol-Instruct depth operator** — pick either `add_constraint` or `deepen`, wire it in as an every-Nth-round mutation, and record `evol_history` in provenance.
6. **Appends** surviving pairs to the pool.
7. Terminates when the pool has 200 surviving pairs.

Log every generation with its raw teacher output before filtering; you will use this to compute drop rates.

### Part D — Provenance record per row

Every surviving row is written to `synthetic.jsonl` with:

```jsonc
{
  "id": "syn-000123",
  "source": "self-instruct-local",
  "license": "<from your teacher_license.md>",
  "category": "reasoning" | "code" | "general_chat" | "…",
  "generator": "<teacher model id + version>",
  "generator_prompt_hash": "<sha256 of the exact generation prompt>",
  "generator_temperature": 0.7,
  "generator_seed_ids": ["seed-04", "seed-11", "seed-17"],
  "evol_history": [{"operator": "add_constraint", "round": 3, "prior_id": "syn-000042"}],
  "filters_passed": ["dedup", "length", "refusal_ok"],
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
}
```

Any row missing any of these fields is not a completed row.

### Part E — Wire in three quality filters

At minimum:

1. **ROUGE-L (or MinHash Jaccard) dedup** against the existing pool at a documented threshold. Rejected rows: log the near-neighbor from the pool alongside the rejected row.
2. **Refusal detection** on the response. If the teacher refused, route to a separate `refusals.jsonl` (do not silently drop; some may be useful safety training data later, mod-106).
3. **Format / cheap-correctness check** on at least one category. For code prompts: does the response contain a valid Python code block whose `ast.parse` succeeds? For math prompts: does the response end with a `\boxed{...}` or a "The answer is ..." pattern with a numeric answer? Choose based on your seed categories.

Report per-filter drop rates: how many raw generations became how many surviving rows.

### Part F — Produce a mini quality report

Write `synth_report.md` (two pages max) containing:

1. Total raw generations, total survivors, per-filter drop rate.
2. Category distribution of survivors, and the drift from your seed category distribution.
3. Sampled examples of dropped rows per filter (5 each).
4. Ten sampled surviving rows read by hand and rated 1–5 for quality against the chapter-1 rubric. What is the average rating? What percentage of the pool is rated ≤2? (If it is >20%, your filters are not doing their job.)
5. One paragraph describing what you would change on the next iteration.

## Starter guidance

- Cap teacher calls with a `MAX_CALLS` constant early; runaway loops are a real risk. 200 surviving rows should not cost more than a few dollars against a hosted teacher, and pennies against a local one.
- Cache teacher responses to disk keyed on the exact prompt. Re-runs while debugging should not re-hit the teacher.
- For the ROUGE-L / MinHash dedup, `datasketch.MinHash` is fine at this scale; `rouge_score` gives you ROUGE-L directly if you prefer.
- The Self-Instruct paper's appendix contains the exact generation prompts they used. You do not have to reuse them, but reading them is instructive — they are shorter than most modern practitioners assume.
- Do not enable Evol-Instruct on every round out of the gate. It amplifies teacher errors; add it once your baseline pipeline is producing decent Self-Instruct output.

## Acceptance criteria

You have completed the exercise when:

- `teacher_license.md` exists and answers all four questions.
- `seeds.jsonl` contains 20 hand-authored pairs.
- `pipeline.py` (or notebook) is reproducible and reaches 200 surviving rows.
- `synthetic.jsonl` has 200 rows, each with complete provenance metadata.
- `synth_report.md` includes per-filter drop rates, category drift, sampled drops, and hand-rated quality on 10 surviving rows.
- The report's per-filter drop rates are consistent with the raw-generation logs and can be independently recomputed.
- No row in `synthetic.jsonl` has a `license` field of `unknown` or empty.

## Stretch goals

- Add a **judge-model quality filter**: score each surviving row 1–5 with a smaller, independent LLM (not your teacher). Compare the judge's score against your hand-rated 10-example sample. Is the judge biased high or low? If you re-run the pipeline with judge-score-threshold ≥ 4 as a hard filter, does per-row quality improve — and by how much do you shrink the pool?
- Add a **decontamination pass** against MMLU / GSM8K on the generated instructions (reusing the code from exercise 02). Report any flagged rows and remove them. This is the "synthetic pipelines especially need decontam" point from chapter 3, made concrete.
- Add a second Evol-Instruct operator (`concretize` or `add_constraint`) and run one round of both. Diff category distributions after two rounds. Which operator drifts the distribution most?
- Publish `synthetic.jsonl` + `teacher_license.md` + `synth_report.md` to a private HF Hub dataset with a dataset card. This is what a real internal synthetic-data handoff looks like.
