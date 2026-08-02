# Common Pitfalls and Handoff to Serving / Systems

**Estimated read:** 30 minutes.
**You should be able to, by the end:** recognize and prevent the seven pitfalls that most often invalidate a fine-tuning eval (chat-template mismatch, sampling drift, tokenizer changes, stop-token bugs, prompt caching, few-shot exemplar drift, quantization-changes-the-eval), and describe the interface contract that hands eval artifacts to mod-109 (quantization / serving) and mod-111 (post-training systems / platform).

## Why a pitfalls chapter

Chapters 2–6 built the eval discipline. This chapter is the honest debrief on the ways that discipline is still routinely broken by mistakes at the boundary. Every one of these has caused a "we thought the model regressed but the eval was wrong" incident somewhere; each has a concrete prevention.

If you catch even one of these in your gate before shipping, this chapter has paid for itself.

## Pitfall 1 — Chat-template mismatch

**The failure.** An instruct-tuned model is evaluated without applying its tokenizer's chat template. The model has been trained to produce assistant-turn tokens inside a specific special-token wrapper (`<|im_start|>assistant\n...<|im_end|>`, `<start_of_turn>model\n...<end_of_turn>`, etc.). Fed a raw prompt without that wrapper, the log-likelihoods it assigns to the "gold" continuation are wrong, generations are erratic, and MMLU / ARC / TruthfulQA scores collapse for no capability-related reason.

**Symptoms:** log-likelihood tasks drop 5–15 points against the base model card's reported number; generation tasks produce chat-template artifacts (`<|im_end|>` in the middle of code, hedging preambles ahead of the answer).

**Prevention:**

- Run `--apply_chat_template` on every instruct-tuned model eval.
- Pair with `--fewshot_as_multiturn` so few-shot exemplars are formatted as their own turns.
- Sanity-check with `--write_out` — open the dumped prompts and see whether they actually look like a chat with the expected special tokens.
- On a new model, run the same suite with and without the chat-template flags and compare against the base model's published numbers. If neither matches, keep digging — you're either on a different task version, using a different backend, or hitting a template that the tokenizer itself gets wrong.

**Fixing after the fact:** re-run the eval with the correct flags. There is no post-hoc correction to a score computed with the wrong template.

## Pitfall 2 — Sampling parameter drift

**The failure.** Generation-based evals (GSM8K, HumanEval, IFEval, judged evals) are sensitive to temperature, top-p, top-k, and max_new_tokens. Different engineers, different backends, and different `lm-evaluation-harness` versions default to different sampling. A number reported at greedy is not comparable to a number reported at temperature 0.2.

**Symptoms:** GSM8K scores drift 2–5 points between reruns with no model change; HumanEval pass@1 numbers wildly disagree with the model card because the model card ran 20 samples at T=0.2 and you ran 1 at T=0.

**Prevention:**

- Fix all sampling parameters in the gate manifest (chapter 6). Do not rely on defaults.
- Default to greedy (T=0, top-p=1) for gate runs. Sampling parameters that mirror production go in a separate "production-mirror" eval lane, not the release gate.
- When reproducing a paper's number, match the paper's sampling exactly: number of samples per problem, temperature, top-p, top-k, self-consistency voting or not.

**Reporting rule:** Every generation-task score in your report includes `(T=X, top_p=Y, samples=N)` in the caption. If it does not, the number is unreproducible.

## Pitfall 3 — Tokenizer changes silently invalidate old numbers

**The failure.** You bump the tokenizer between fine-tunes — added special tokens, switched from LLaMA-1's SentencePiece to LLaMA-3's tiktoken-style, added a new chat template. Old eval numbers are computed against different tokenization and are no longer directly comparable to new ones.

**Symptoms:** a fine-tune whose *only* change was adding tool-call special tokens shows a "regression" on MMLU because those tokens interact with the log-likelihood scoring.

**Prevention:**

- Log the tokenizer's sha256 (or version + special-token list) alongside every eval run. Store in the results JSON.
- If the tokenizer changed between adapter N and adapter N−1, flag the comparison in the gate report as "tokenizer changed; delta interpretation limited." Consider re-running N−1 with the new tokenizer for a like-for-like comparison, or accepting the delta as a stepwise re-baseline.
- Adding special tokens is almost always safe for generation tasks and slightly risky for log-likelihood — the added vocabulary shifts embedding weights and can perturb the base representations. Test on a canary eval before proposing a new special token.

## Pitfall 4 — Stop-token bugs

**The failure.** The model generates past its intended stop, either because the task's `until` list does not include the model's chat-template stop token, or because the model was fine-tuned to emit a stop token that the task does not recognize. The excess generation either wastes budget (max_new_tokens burns through) or breaks the scoring (garbage after the answer).

**Symptoms:** GSM8K scores lower than expected because the model emits `Q:` immediately after the answer and the scorer picks up the next question's setup; HumanEval scores plummet because the model appends a chatty "I hope this helps!" that fails syntax parsing; judged evals score responses lower because the response includes chat-template gibberish at the end.

**Prevention:**

- Configure the eval runner to include the tokenizer's `eos_token` and any model-specific chat-template stops in the `until` list for every generation task.
- Sanity-check with `--write_out` on the first run — open 10 dumped responses per task and confirm they end where you expect.
- For HumanEval and similar code tasks, prefer a post-processor that extracts the function definition and ignores trailing prose. Some `lm-evaluation-harness` versions include this; verify for your version.

## Pitfall 5 — Prompt cache and few-shot exemplar drift

**The failure.** Two forms:

- **Prompt cache confounds** — a served backend (vLLM, SGLang) uses a prefix cache for shared few-shot exemplars. Different item ordering within a batch changes cache hits, which does not change semantics but can subtly shift latency-based metrics and, in some edge cases, numerical precision (mixed-precision cache retrieval).
- **Few-shot exemplar drift** — the task samples few-shot exemplars from the dev split. If the sampling depends on a random seed and the seed drifts between runs, the exemplars drift, the score drifts.

**Symptoms:** small, non-reproducible score wobble across reruns of the same eval on the same model; a "regression" of 0.5 pp that disappears when you re-run.

**Prevention:**

- Fix the seed in the manifest. `lm-evaluation-harness` accepts `--seed`; use the same value across runs.
- For log-likelihood tasks, the exemplars are usually deterministic given the seed. For generation tasks with sampling, add the sampling seed too.
- Do not enable server-side prompt caching in a way that mixes across items with different intended few-shot contexts. When in doubt, disable prompt caching for gate runs.

## Pitfall 6 — Contamination-scan false positives

**The failure.** The overlap scan flags rows that look like matches but are not — boilerplate license headers, common phrases ("The following is a multiple-choice question…"), shared quotations, code patterns. Removing them from training does nothing useful; keeping them logged as "contamination" in the model card is misleading.

**Symptoms:** contamination scan reports 10,000 flagged rows, hand-inspection shows 90% are copyright notices and common instruction preambles.

**Prevention:**

- Pre-filter obvious boilerplate (copyright headers, license text, common eval-format phrases) before the scan.
- Always hand-sample at least 50 flagged rows per eval and compute the confirmed-hit rate. Report both raw and confirmed counts in the disclosure.
- Do not delete rows without human review of the highest-severity examples. Blindly removing everything flagged either loses useful training data or convinces you contamination is under control when the real matches are still there.

Mod-102 chapter 3 covered this at build time; this pitfall is when the eval-time scan repeats the same mistake because the pre-filter was not carried forward.

## Pitfall 7 — Quantization changes the eval score

**The failure.** You evaluated the BF16 checkpoint and it passed. Mod-109 quantized it to INT4 for serving. Users see regressions. The eval report says the model was fine — but the eval was against a checkpoint that is not the artifact being served.

**Symptoms:** post-launch quality complaints on a model that gated cleanly; a widening gap between offline eval and A/B production numbers.

**Prevention:**

- The release gate must eval the *served artifact*, not the training-time checkpoint. If quantization is part of the serving pipeline, run the gate against the quantized model too, and require the quantized number to be within a stated tolerance of the full-precision number.
- Chapter 9 (mod-109) covers the standard tolerances (typically ≤1 pp regression on academic tasks for a well-done INT8, ≤2 pp for INT4 with GPTQ/AWQ; larger deltas on generation tasks). Set the tolerance as a hard gate.
- Store both numbers in the release report: full-precision and served-precision. The gap is itself a health metric.

This is the primary handoff to mod-109; the eval infrastructure this module built is what tells you a quantized artifact is still shippable.

## Handoff to mod-109 — quantization and serving

Chapter 9 takes your fine-tuned checkpoint into the serving pipeline. The eval infrastructure you built here is what it depends on. Specifically:

**What mod-109 consumes from this module:**

- **The release-gate manifest.** Chapter 9 re-runs the fast lanes (academic benchmarks, task-specific held-out) against each quantized variant. The manifest is what defines "fast lanes."
- **The tolerance thresholds.** How much regression is acceptable per task on a quantized model? Chapter 9 uses these to decide which quantization method makes it to production.
- **Task-specific held-out set.** Quantization can regress narrowly on specialized capabilities that the academic suite misses. The chapter-3 held-out set is the primary signal here.
- **The judge infrastructure.** Chapter 9 re-runs the judged eval against quantized outputs — quantization can subtly degrade open-ended generation in ways log-likelihood tasks miss.
- **The contamination disclosure.** Quantization does not change contamination — the disclosure carries forward unchanged.

**What mod-109 hands back:**

- **The served artifact hash.** Every quantized artifact gets a hash; the release-gate report references it.
- **A quantization-delta table.** Per-task, full-precision vs. quantized numbers.
- **Serving-latency and cost numbers** that this module does not measure. Those feed a different lane in the release notes.

## Handoff to mod-110 — managed fine-tuning APIs

If your fine-tune runs through OpenAI's, Anthropic's, or Vertex's managed APIs (mod-110), the eval story is *harder*, not easier. You cannot introspect the weights. You cannot run `lm-evaluation-harness` in `hf` backend. The held-out evals (chapter 3) and judged evals (chapter 4) become the *primary* quality signal — because they can be run against any backend that exposes a generation endpoint.

Specifically:

- Lane 1 (academic benchmarks) needs to run against the managed API — use the `openai-chat-completions` or `local-chat-completions` backend to `lm-evaluation-harness`. Slower and more expensive because you pay per token; log-likelihood tasks may be limited or unavailable on chat-only endpoints.
- Lane 2 (task-specific held-out) is your workhorse. Runs against any HTTP endpoint.
- Lane 3 (judged) is workhorse #2. Runs the same regardless of where the model is served.
- Lane 4 (safety) — same story, run against the managed endpoint.

The contamination disclosure gets weaker because you cannot scan the training data — you can only run the model-signal probes (chapter 5, lens 3). Mod-110 discusses the disclosure adaptation.

## Handoff to mod-111 — post-training systems design

Mod-111 is where the eval infrastructure becomes part of a platform: an experiment tracker that stores every eval JSON forever, a scheduler that fires the gate on every proposed release candidate, a UI that surfaces per-item diffs to reviewers, a governance layer that gates production deploys on gate pass.

**What mod-111 consumes from this module:**

- **The gate manifest.** Serves as the platform's declarative spec for "what does gating a release mean."
- **The results JSON schema.** The platform stores every field this module produces; new fields require a schema bump.
- **The per-item score / diff data.** The platform's UI diffs adapter N against adapter N−1 per item.
- **The contamination report.** The platform's data-governance layer stores this alongside dataset lineage.
- **The regression report template.** The platform generates the human-readable report on gate completion; the template is what makes that possible.

**What mod-111 provides that this module does not:**

- **Multi-tenant eval scheduling.** Many teams' gates running concurrently against a shared GPU pool.
- **Historical trend analysis.** MMLU on adapter series over 6 months; task-specific held-out drift over time.
- **Automated regression triage.** The platform can flag "this stratum has regressed 3 releases in a row" earlier than a human comparing single reports would.
- **Compliance / audit trails.** Which manifest gated which deploy; who approved the soft-gate override; when the contamination disclosure was reviewed.

Chapter 6's gate manifest is the API between this module and mod-111. If the manifest is structured cleanly, the platform can consume it directly; if it is ad-hoc scripts, the platform work is much harder.

## Handoff to the fine-tuning loop (backward)

Eval signal also flows *back* to the fine-tuning process. The three most useful patterns:

1. **Per-stratum regression → dataset diagnosis.** A regression on the "billing/dispute" stratum in the task-specific eval means the SFT data may be under-represented in billing-dispute examples. Mod-102's stratified sampling can rebalance; the eval report is the trigger.
2. **Contamination discovery → decontam pipeline.** Chapter 5's finding of new contamination in adapter N means adapter N+1's build pipeline needs an updated exclusion list. Mod-102 chapter 3 owns the fix.
3. **Judge disagreement patterns → rubric revision.** If the anchor set shows the judge systematically differs from humans on a category, either the rubric needs to be sharper or that category needs to go to human-only labeling. Chapter 4 covers the revision workflow.

Every eval regression is either a genuine model quality issue (retrain) or a signal that the training / dataset / rubric process has a bug (fix upstream). Do not accept either as the default; investigate every regression until you know which category it falls into.

## A checklist you can pin

Before you sign off on any eval report:

- [ ] The report cites the harness commit SHA, model artifact hash, tokenizer hash, and judge model + version.
- [ ] Every generation task's caption includes `(T, top_p, samples)`.
- [ ] Every log-likelihood task ran with `--apply_chat_template` (for instruct models) or without (for base models), matching the reference.
- [ ] `--write_out` samples were spot-checked on the first run against a new model.
- [ ] Per-item scores are dumped and available for diffing.
- [ ] Bootstrap CIs and paired-test p-values accompany every headline delta.
- [ ] Contamination scans (fast lanes) have run on the current dataset; slow lanes have run on the release candidate.
- [ ] Judged-eval report includes judge model version, rubric version, position-bias fraction, judge-vs-human κ / ρ, mean response length per model.
- [ ] The served artifact (post-quantization if applicable) is what was gated, not just the training checkpoint.
- [ ] Every soft-gate failure has a written captain review.

If any of these is missing, do not ship — regenerate the missing piece first.

## Summary

- Seven pitfalls silently corrupt fine-tuning evals: **chat-template mismatch**, **sampling drift**, **tokenizer changes**, **stop-token bugs**, **prompt cache and exemplar drift**, **contamination-scan false positives**, and **quantization changes the eval**. Each has a concrete prevention.
- The eval infrastructure hands off cleanly to three downstream modules: **mod-109** re-runs the gate against quantized artifacts (must stay within stated tolerances); **mod-110** relies on held-out and judged evals as primary signal because it cannot introspect managed-API weights; **mod-111** consumes the manifest, JSONs, and per-item diffs as the platform's declarative spec.
- Eval signal also flows *back* to the fine-tuning loop: per-stratum regressions trigger dataset rebalancing (mod-102), contamination discoveries trigger decontam-pipeline updates (mod-102 ch. 3), and judge disagreements trigger rubric revisions (chapter 4 of this module).
- The pre-sign-off checklist is short and non-negotiable. If any item is missing, do not ship — regenerate first.
