# exercise-02: Constitutional AI Self-Critique Loop

**Estimated effort:** 3 hours (~1 hour authoring + ~1 hour generation wall-clock + ~1 hour hand-audit)
**Companion chapters:** [`03-constitutional-ai-and-rlaif.md`](../03-constitutional-ai-and-rlaif.md), background from [`02-refusal-and-harm-reduction-datasets.md`](../02-refusal-and-harm-reduction-datasets.md)

## Objective

Author a small **constitution** (~10 principles), run the **SL-CAI** critique-and-revise loop from chapter 3 on ~200 red-team prompts using open generator and critic models, produce a `(prompt, revised_response)` SFT dataset ready to blend into the safety mix from exercise 01, and hand-audit the results to see where the pipeline works and where it breaks. This exercise stops at SL-CAI; the RL-CAI half (preference pairs from AI-judged samples) is picked up as a stretch goal here and again in exercise 03's red-team conversion.

The point of the exercise is not to ship a model — the volume is deliberately small — but to feel every component of a CAI pipeline (constitution → critique → revision → filter → SFT-shape output) end-to-end so the trade-offs from chapter 3 (judge misalignment, no-op revisions, monoculture, cost overhang) become concrete rather than theoretical.

## Prerequisites

- The safety mix from exercise 01 (`outputs/safety-mix-v1/`) — you will blend the CAI-revised rows into it.
- `transformers`, `datasets`, `accelerate`. `vllm` (or `text-generation-inference`) strongly recommended for batched generation; the loop is 3–4× more generations than a normal SFT pass, so throughput matters.
- A **generator** model (the one being critiqued). Options: `meta-llama/Llama-3.2-3B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, or the SFT checkpoint from mod-103 exercise 01. Small (2–3B) is fine and keeps wall-clock manageable.
- A **critic / reviser** model. Two viable options:
  - **Self-critique** (chapter 3's original variant): same model as the generator. Cheapest; matches the Bai et al. 2022 setup.
  - **Distilled critique**: a stronger open model (e.g. `Qwen/Qwen2.5-14B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`) as critic. Produces sharper revisions at ~2× the cost. Pick one; document why.
- A GPU with ≥ 24 GB VRAM (16 GB works if you 4-bit-load both models). Multi-GPU is nicer for the batched generation but not required.
- A prompt corpus of ~200 red-team-shaped attack prompts. Pull from `walledai/AdvBench`, `HarmBench/HarmBench` (standard behaviors), or the JailbreakBench harmful split. Do not use XSTest or OR-Bench here — those are for exercise 04's eval, not training.

## Requirements

### Part A — Write the constitution

Author a working constitution and commit it as `outputs/cai-run/constitution.md`. Requirements:

- **10–20 principles.** Chapter 3's empirical guidance: too many principles hurt because the critic gets less discerning. Start narrow.
- **Coverage across the five categories** from chapter 3: harmlessness (1–2 general + 3–5 category-specific), helpfulness (at least 2), honesty (at least 1), voice / style (at least 1), and any domain-specific principle your policy surface needs. A CAI run with only harmlessness principles degenerates to over-refusal; the helpfulness and honesty principles are load-bearing.
- **Anthropic-style phrasing.** Each principle worded to be usable in *both* the critique prompt and (if you do the RL-CAI stretch) the judge prompt. The template from chapter 3:
  > Please choose the response that <property clause>.
  > For example, if the responses discuss <case>, please choose the one that <specific guidance>.
- **A rationale line per principle.** One sentence saying *why* the principle is in the constitution — which failure mode it targets, which category it covers, where it came from (published constitution, your AUP, a red-team finding). Chapter 3 flags "constitutional monoculture" as a failure mode; the rationale is your future self's audit trail for adding / removing principles.

Read Anthropic's published CAI constitution and the Collective Constitutional AI companion paper before writing yours — do not copy them verbatim (they were built for Claude's specific policy surface), but treat them as the reference for what a working constitution looks like.

### Part B — Assemble the prompt corpus

`outputs/cai-run/prompts.jsonl`, ~200 rows with the schema:

```json
{
  "id": "cai-0001",
  "prompt": "<attack prompt text, single-turn>",
  "source": "advbench" | "harmbench" | "jailbreakbench" | "custom",
  "harm_category": "<category from your exercise-01 taxonomy>",
  "expected_response_class": "hard_refusal" | "soft_refusal" | "safe_completion"
}
```

Coverage requirements:

- **≥ 6 harm categories** represented, at least 10 prompts each. If you use HarmBench-standard, its 7 semantic categories map directly.
- **≥ 30% of prompts** should be ones where `expected_response_class` is `soft_refusal` or `safe_completion` — not just hard-refusal cases. Otherwise the constitution will only get to exercise its "refuse" muscle.
- **Decontaminate against exercise-04's eval sets.** Hash every prompt against HarmBench-standard (if you did not draw from it), JailbreakBench behaviors, XSTest, and OR-Bench. Drop overlaps. Record the counts in `prompts.jsonl`'s metadata.

### Part C — Implement the critique-and-revise loop

Following the code sketch in chapter 3, build `run_cai.py` that iterates over your prompt corpus and, for each row, produces:

```json
{
  "id": "cai-0001",
  "prompt": "<same as input>",
  "principle_ids": ["p03", "p07"],           // which principles were sampled per revision round
  "initial_response": "<generator output on prompt>",
  "rounds": [
    {"round": 1, "principle_id": "p03", "critique": "...", "revised_response": "..."},
    {"round": 2, "principle_id": "p07", "critique": "...", "revised_response": "..."}
  ],
  "final_response": "<the last revised_response>",
  "no_op": false,                            // true if all revisions changed nothing
  "meta": {
    "generator_model": "...",
    "critic_model": "...",
    "temperatures": {"initial": 0.7, "critique": 0.3, "revise": 0.3},
    "wall_clock_seconds": 12.4
  }
}
```

Implementation notes:

- **Use the actual chat template.** Both the generator and the critic need `tokenizer.apply_chat_template` (mod-101 chapter on chat templates), not bare-string concatenation. A CAI pipeline that skips the chat template will produce garbage revisions on any modern instruct model.
- **Sample one principle per round**, not all at once. Chapter 3 explains why: injecting all principles at once makes the judge less discerning. Rotate through the constitution.
- **Two rounds is a reasonable default.** Bai et al. cycle 3–4 rounds; each round costs one critique + one revision. Fixed round count is fine for the exercise; a real pipeline can stop early when the revision matches the initial response.
- **Sampling.** `temperature ≈ 0.7` for the initial response (you want the model's real distribution); `temperature ∈ [0.2, 0.4]` for the critique and revision so the outputs are more consistent.
- **Batch aggressively.** Naively, one row is 1 + 2 × (1 + 1) = 5 generations. On 200 rows, that is 1000 generations. Even with vLLM at a few hundred tok/s, expect ~30–60 min of wall-clock; sequential `.generate()` calls will run several hours. Batch the initial responses first, then the critiques, then the revisions.
- **Detect no-op revisions.** After each round, cheap-check whether the revised response is materially different from the previous one (edit distance, embedding cosine similarity, or a light-touch LLM judge). Set `no_op: true` if the revision changed nothing across all rounds. Chapter 3 says to drop these from the SFT set; you will do so in Part D.

Save the raw trace to `outputs/cai-run/trace.jsonl`.

### Part D — Convert to an SFT dataset

Build `outputs/cai-run/cai_sft/` as a `datasets.Dataset` (saved with `save_to_disk`) containing:

```python
{
  "prompt": str | list,          # match the shape you used in exercise 01
  "response": str,               # the final_response from Part C
  "meta": {
    "source": "cai-run-v1",
    "harm_category": str,
    "expected_response_class": str,
    "principle_ids": list[str],
    "rounds_run": int,
    "generator_model": str,
    "critic_model": str,
    "response_len_tokens": int
  }
}
```

Filters to apply, in order, logging counts to `outputs/cai-run/filter_log.txt`:

1. **Drop no-ops.** Rows where every revision was a no-op contribute no new training signal; per chapter 3, drop them (or alternatively route them to a "validation the generator was already safe on this prompt" file for later inspection).
2. **Length filter.** Drop revised responses shorter than 5 tokens or longer than 1024 tokens (chapter 2's `response_len_tokens` bounds — keep your safety data in the same length envelope as your general SFT data).
3. **Deduplicate by prompt hash** against your exercise-01 SFT set. If a CAI-revised row is a near-duplicate of an existing exercise-01 refusal, keep whichever you prefer but not both.
4. **Chat-template validity check.** Round-trip each `(prompt, response)` through the tokenizer's chat template. Drop any row that fails to serialize (the CAI critique / revision prompts occasionally leak template tokens into the output; this filter catches them).

### Part E — Hand-audit ~100 rows

Sample 100 rows from `cai_sft` (fixed seed) and hand-inspect. For each, decide:

- **Class match.** Does the `final_response` match the `expected_response_class`? A "hard_refusal"-expected prompt where CAI produced a fluent safe-completion is *not necessarily wrong* — the constitution may reasonably have argued for the softer form — but it *is* a signal about your constitution's shape.
- **Style consistency.** Are the refusals in one voice, or does the critic drift between clinical and lecture-y? Length inconsistency is a leading indicator of downstream style drift.
- **Over-refusal risk.** For prompts where the harm was ambiguous, did the critic revise toward blanket refusal ("I cannot discuss X in any way") when a soft refusal or safe completion was warranted? Count these.
- **Honesty issues.** Any responses that dodge the ask with a plausible-sounding fabrication ("I don't know" when the model does know; "this information does not exist" when it does)? Chapter 3's "purely-harmlessness-driven revisions produce polite fabrications" — count these.
- **Constitution-vs-output mismatches.** Cases where the revision violates its own selected principle. Rare but instructive.

Write the audit to `outputs/cai-run/audit.md` with:
- The 5 counts above, out of 100.
- 5 specific examples (with prompt, before-revision, after-revision, and your commentary) — one per axis above.
- A one-paragraph verdict: is the constitution ready for a real run, or does it need principle edits?

If any category above lands above ~20 / 100, **iterate on the constitution and re-run at least the affected subset before shipping the SFT artifact.** This is the "hand-audit ~100 revisions; adjust the principles that produce weird outputs; re-run" cadence from chapter 3.

### Part F — Write the dataset card

`outputs/cai-run/dataset_card.md`. On top of exercise-01's dataset-card template, add CAI-specific sections:

- **Constitution reference.** Link to `constitution.md`; note the count of principles and per-category coverage.
- **Generator / critic provenance.** Model IDs and revisions. If distilling from a stronger critic, disclose it here — some open model licenses restrict downstream use of distilled outputs; verify before shipping.
- **Pipeline configuration.** Round count, sampling temperatures, one-principle-per-round vs all-at-once, batch size.
- **Filter deltas.** From `filter_log.txt`: pre-filter row count, per-filter drop counts, final count. No-op drop rate is the key CAI-specific number.
- **Audit summary.** The five counts from Part E and the verdict paragraph.
- **Blend guidance for exercise-01 mix.** How many CAI rows to blend in; where (SFT-side, not preference-side, for SL-CAI); ratio recommendation. Chapter 3's blend advice: same as any other safe-completion rows, chapter 2's SFT-side mix ratio (2–10% of tokens).
- **Known limitations.** Judge / critic biases you observed; categories where the revisions were weakest; anything worth flagging to a chapter-6 governance reviewer.

## Starter guidance

- **Write the constitution before you touch any code.** The temptation is to script the loop first and iterate the constitution against real outputs. That does work, but you burn a lot of wall-clock on runs that a five-minute pre-review of the principles would have avoided. Write ten principles, sanity-read them, then start generating.
- **The critic can be the same model as the generator on this scale.** Chapter 3 notes both variants work; self-critique is what Bai et al. did originally. If you have GPU room for a stronger critic, use one — but do not block the exercise on it.
- **vLLM makes the difference between 30 minutes and 3 hours.** If your GPU has room for a small vLLM instance, spin one up and serve both models via the OpenAI-compatible endpoint. Chapter 3's compute cost table is not a joke; naive generation dominates the exercise wall-clock.
- **Read 20 traces before you write the audit.** Skim the raw JSONL first. You will notice patterns (a specific principle producing weird outputs, a specific harm category getting over-refused) that formal auditing then quantifies.
- **The no-op rate is diagnostic.** A very high no-op rate (say, > 60%) means your generator model is already handling most of the prompts safely — either your prompts are too easy for this model, or your model's baseline is already strong. A very low no-op rate (say, < 10%) means either the model is genuinely unsafe on most of the prompt set (uncomfortable but useful signal) or the critic is over-eager and revising responses that were already fine.
- **Do not train a model in this exercise.** The output is a dataset. Blending into the safety mix and training on it is a downstream activity (mod-103 SFT exercise re-run on the blended set), left out here so the CAI-specific dynamics do not get lost in a training run.

## Acceptance criteria

You have completed the exercise when:

- `outputs/cai-run/constitution.md` exists with 10–20 principles, each with a rationale line, spanning harmlessness / helpfulness / honesty / voice / (optional) domain principles.
- `outputs/cai-run/prompts.jsonl` contains ~200 red-team prompts across ≥ 6 harm categories, with the decontamination check against exercise-04's eval sets documented.
- `outputs/cai-run/trace.jsonl` has one entry per prompt containing all rounds and metadata; the row count matches `prompts.jsonl`.
- `outputs/cai-run/cai_sft/` loads via `datasets.load_from_disk` and contains the schema in Part D; `filter_log.txt` shows the pre/post counts per filter.
- `outputs/cai-run/audit.md` has counts across the five axes on 100 sampled rows, 5 example traces, and a verdict paragraph.
- `outputs/cai-run/dataset_card.md` covers every section in Part F, including the CAI-specific ones (constitution reference, critic provenance, no-op rate, blend guidance).
- If any audit axis lands above ~20 / 100, the constitution was iterated and the affected subset re-run — with the diff noted in the dataset card.

## Stretch goals

- **RL-CAI half.** For 100 prompts, sample two responses from the SL-CAI-tuned model (or, if you skipped training, from the generator with two random seeds), and run the judge step from chapter 3 (`judge_pair`) using the same critic model with an AI-judge prompt. Produce a `(prompt, chosen, rejected, principle)` preference dataset. This feeds directly into a DPO run on top of exercise 01's preference set — the same shape as exercise 03's output.
- **Cross-judge audit.** Take 100 revisions and re-critique them with a *different* critic model (a different family, ideally: if your critic was Llama, use Qwen; if Qwen, use Mistral). Report agreement on "does this revision improve the response." Below 80% agreement means judge idiosyncrasy is a big fraction of your training signal; chapter 3 flags this.
- **Distillation collapse audit.** If you used a stronger critic, sample 50 revisions where the critic made a nontrivial edit. For each, ask: could the generator model have produced this revision on its own with better prompting? A high "yes" fraction means you are cheaply distilling the critic; a high "no" fraction means the critic is genuinely adding capability the generator lacks. Chapter 3's "distillation collapse" note asks you to be honest about which regime you're in.
- **Multi-turn extension.** Extend the loop to multi-turn attacks — the initial "response" is a 3-turn conversation ending in a harmful assistant turn; the critique and revision target the final turn while preserving the earlier context. Chapter 4 flags multi-turn / gradual jailbreak as a distinct attack class that single-turn SL-CAI does not handle well.
- **Stopping criterion.** Instrument a small eval loop: after every 50 processed prompts, sample-generate on a 20-prompt held-out set and score with Llama Guard 3. Stop the run when the score plateaus. This is chapter 3's "define the stopping criterion up front" turned into instrumentation.
- **Constitution ablation.** Remove each principle one at a time; re-run 50 prompts; measure how many revisions change. Principles whose removal produces no change are either subsumed by others or aren't doing work; consider dropping them. Report which principles were load-bearing.
- **Publish the constitution.** If your organization has an AUP or policy document, commit the constitution as an internal artifact and cite it from the model card. Chapter 3 makes the point that CAI's leverage is that the constitution is a *small human artifact* controlling a *large machine artifact*; owning it as a policy document is what compounds across model versions.
