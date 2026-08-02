# Feeding Managed Fine-Tunes Through the Same Eval Harness

**Estimated read:** 35 minutes.
**You should be able to, by the end:** run mod-108's release-gate suite against a hosted OpenAI / Anthropic / Vertex / Bedrock fine-tune using `lm-evaluation-harness`'s API backends, reconcile the sampling / chat-template / stop-token settings across backends so the numbers are actually comparable, explain which tasks lose their meaning on a chat-only endpoint (log-likelihood tasks) and what to do about it, control eval cost with batching and the Batch API, and produce a "release artifact" record for a hosted fine-tune that is the exact analog of mod-109's served-artifact hash.

Chapter 6's ADR made a decision. Chapter 7 is what verifies it. The eval gate does not care whether your fine-tune lives on OpenAI or on your own vLLM — but only if you set it up so it doesn't. This chapter is that setup.

## The rule to remember

**The comparison is only meaningful if the eval is the same.** If you eval the self-hosted candidate with `--apply_chat_template --fewshot_as_multiturn --temperature 0`, then evaluate the managed candidate at whatever the default is on some other tool, the resulting leaderboard is not a comparison — it is two different measurements of two different things.

Mod-108 chapter 6 defined the release gate as a declarative manifest with pinned tasks, sampling parameters, chat-template rules, and tolerances. That manifest is the single source of truth. This chapter is about extending the manifest to run against hosted endpoints without changing what "the gate" means.

## The `lm-evaluation-harness` API backends

`lm-evaluation-harness` (the version pinned in mod-108 exercise 01) supports several API-style backends. The three that matter here:

- **`openai-chat-completions`** — for OpenAI direct fine-tunes and for anything that speaks the OpenAI Chat Completions protocol (including a local vLLM in OpenAI-compat mode, Fireworks, Together, and OpenAI-compat gateways over other vendors).
- **`local-chat-completions`** — generic OpenAI-compatible HTTP client; use for self-hosted vLLM / TGI / SGLang endpoints with the OpenAI-compat server enabled, and for compatible gateways.
- **`anthropic-chat-completions` / `anthropic-completions`** — Anthropic direct API.

For Vertex and Bedrock and any HTTP endpoint that is not OpenAI-shape, either use a compatibility gateway (LiteLLM, an internal proxy that speaks OpenAI-in, vendor-out) or write a small custom backend. The compatibility-gateway pattern is what most teams end up on because the OpenAI backend then works everywhere.

<!-- needs-research: verify current `lm-evaluation-harness` version's backend list and the exact `--model_args` schema for each. Backend names and their argument shapes have shifted across `lm-eval` releases; confirm against the pinned version. -->

## Running the OpenAI backend against an OpenAI fine-tune

The exact command pattern:

```bash
export OPENAI_API_KEY=sk-...

lm_eval \
    --model openai-chat-completions \
    --model_args model=ft:gpt-4.1-mini-2025-07-14:acme::support-triage-v3:AbCdEf12,base_url=https://api.openai.com/v1 \
    --tasks mmlu,arc_challenge,truthfulqa_mc2,gsm8k,ifeval \
    --num_fewshot 5 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --gen_kwargs temperature=0,top_p=1 \
    --output_path results/openai-ft/ \
    --log_samples \
    --write_out
```

Two settings that always matter:

- **`--apply_chat_template --fewshot_as_multiturn`** for instruct-tuned models. Same rule from mod-108 chapter 7 pitfall #1. A fine-tuned model on OpenAI is an instruct model.
- **`--gen_kwargs temperature=0,top_p=1`** for gate runs. Sampling parameters that mirror production go in a separate lane, not the gate.

## Running against Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...

lm_eval \
    --model anthropic-chat-completions \
    --model_args model=<your-fine-tuned-claude-model-id> \
    --tasks arc_challenge,gsm8k,ifeval,<your-task-specific-held-out> \
    --gen_kwargs temperature=0,max_tokens=1024 \
    --apply_chat_template \
    --output_path results/anthropic-ft/
```

Two things to know:

- The Anthropic backend supports generation-based tasks natively. **Log-likelihood-based tasks** (see next section) do not port cleanly.
- Not every mod-108 task will have a comparable score on the Anthropic backend. Focus the gate on generation tasks that are semantically identical across backends: your task-specific held-out, IFEval, GSM8K, HumanEval / MBPP, judged evals from mod-108 chapter 4.

## Running against Vertex, Bedrock, and Azure

The pragmatic route: a **compatibility gateway** that presents OpenAI-shape endpoints backed by the vendor's real API. LiteLLM is the common choice; internal proxies are also common. Point `lm-evaluation-harness` at the gateway with the `local-chat-completions` backend:

```bash
lm_eval \
    --model local-chat-completions \
    --model_args model=gemini-2.5-flash-ft-support-triage,base_url=http://litellm.internal:4000/v1,api_key=<gateway-key> \
    --tasks arc_challenge,gsm8k,ifeval,<your-held-out> \
    --gen_kwargs temperature=0,max_tokens=1024 \
    --apply_chat_template \
    --output_path results/vertex-ft/
```

The gateway handles auth, region routing, retry, and per-vendor request-shape conversion; the harness sees a single stable OpenAI-shape endpoint.

The alternative — writing a custom `lm-evaluation-harness` model backend per vendor — is more code, less flexible, and mostly reinvents the gateway. Unless your team has a specific reason to avoid the gateway pattern, use it.

## The log-likelihood gap

This is the single largest correctness problem in cross-backend eval.

**Log-likelihood tasks** — MMLU (as scored by choice log-probability), ARC-Challenge, PIQA, WinoGrande, TruthfulQA (mc2 variant), and others in the log-likelihood family — score by asking the model "what is the log-probability you assign to each of these answer choices?" and picking the argmax. This requires the model to expose logprobs over the vocabulary.

The gap:

- **Self-hosted `hf` backend, vLLM's `--enable-prefix-caching` off, OpenAI direct**: logprobs available (OpenAI exposes top-N logprobs on chat completions; earlier there were more restrictions, verify current API).
- **Anthropic**: logprobs are not exposed on the Claude API in a way that scores full log-likelihood tasks. MMLU-as-log-likelihood is not directly runnable.
- **Vertex Gemini**: logprobs available on some endpoints, not others; verify per model.
- **Bedrock**: varies by base model; verify per base.
- **Chat-only OpenAI-compat gateways (LiteLLM etc.) over vendors that don't expose logprobs**: no logprobs.

<!-- needs-research: verify current logprob availability on each vendor. This has been evolving — OpenAI added logprobs to Chat Completions; Anthropic and Vertex have made changes. Cross-check current API docs before publishing exact claims. -->

The workaround: **MMLU-as-generation.** Reformulate the task so the model *generates* the letter of the correct answer rather than being scored by logprob. `lm-evaluation-harness` supports both variants (`mmlu` and `mmlu_generative` variants exist as task groups). The generative version is slightly noisier (requires stop-token discipline, occasional parsing failures) but portable across every chat backend.

Trade-off: **the two variants produce slightly different scores** on the same model — usually within 1–3 points but the direction is not guaranteed. Your leaderboard cannot mix "MMLU-logprob for self-hosted, MMLU-generative for managed" — pick one and use it for all candidates, or run both and compare within each.

The generalized rule: **for cross-backend comparability, prefer generation-based evals across the board.** IFEval, GSM8K, HumanEval, task-specific held-out, judged eval — all generation-based. This is why mod-108's chapter 3 (task-specific held-out) and chapter 4 (judge) are the "workhorse" evals for this module — they are portable.

## Chat-template reconciliation

The self-hosted model has a tokenizer whose `apply_chat_template` you know. The managed model has a chat template you *do not* control — the vendor applies it server-side. This has three consequences:

1. **`--apply_chat_template` in `lm-evaluation-harness` becomes a no-op or a double-apply, depending on backend.** For OpenAI direct and other chat-completion-style APIs, the vendor applies the chat template server-side; `lm-evaluation-harness`'s `--apply_chat_template` for API backends typically means "format the prompt as chat messages before sending" — which is the right behavior for chat models. Do not confuse this with `hf`-backend `--apply_chat_template` which literally wraps the prompt in the model's chat-template special tokens.
2. **Few-shot exemplars behave differently.** On the `hf` backend, `--fewshot_as_multiturn` inserts each exemplar as its own message pair inside the same conversation. On the API backends, this is the natural behavior. Behavior should be identical if configured correctly — but verify by dumping a prompt (`--write_out`) on each backend and diffing.
3. **The model saw a chat template at training time; verify that template matches production.** For self-hosted models, this is under your control. For managed fine-tunes, the vendor's chat template is what the model was trained with — you cannot change it, and it might differ subtly from any local reproduction. Do not eval with a hand-rolled chat template against a managed model — the vendor's server-side templating is authoritative.

The rule: **let the vendor apply the chat template server-side; do not apply one yourself.** Send `messages` as a list; consume the response as generated text.

## Sampling reconciliation

For fair comparison across backends, sampling parameters must be identical. Concrete list to pin per gate run:

- `temperature` — always `0` for gate; other values for production-mirror lane.
- `top_p` — always `1.0` for gate.
- `top_k` — if supported, pin; several vendors default to some finite `top_k`.
- `max_tokens` — must be large enough that no candidate truncates but small enough to control cost.
- `presence_penalty`, `frequency_penalty`, `stop`, other sampling knobs — pin explicitly or set to identity (`0` for penalties, empty list for stop, then rely on model's built-in EOS).
- `seed` — pin. Some vendors respect it (OpenAI does); some don't. Where a vendor does not respect the seed, note the added noise and take the mean of N runs.
- **Number of samples per problem** — usually 1 for greedy gate, N for self-consistency lanes. Pin per task in the manifest.

A gate-manifest snippet with the sampling block explicit:

```yaml
gate_manifest:
  name: support-triage-release-gate
  version: 2026-08-01
  sampling:
    temperature: 0
    top_p: 1.0
    max_tokens: 1024
    seed: 42
    samples_per_problem: 1
  tasks:
    - name: gsm8k
      backend: local-chat-completions
      metric: exact_match_strict
    - name: ifeval
      backend: local-chat-completions
      metric: prompt_level_strict_acc
    - name: task_held_out_support_triage_v3
      backend: local-chat-completions
      metric: judge_correct_rate
      judge_model: claude-sonnet-4-6
      judge_seed: 42
  candidates:
    - id: self_hosted_llama_3_1_8b_ft_v3
      backend_config:
        base_url: http://vllm.internal:8000/v1
        model: /models/support-triage-v3
    - id: openai_ft_gpt_4_1_mini_v3
      backend_config:
        base_url: https://api.openai.com/v1
        model: ft:gpt-4.1-mini-2025-07-14:acme::support-triage-v3:AbCdEf12
    - id: anthropic_ft_claude_haiku_v3
      backend_config:
        base_url: https://api.anthropic.com/v1
        model: <fine-tuned-claude-model-id>
```

The manifest is what the eval driver iterates over. Every candidate goes through the same tasks with the same sampling. The output is a comparable leaderboard.

## Cost-controlling eval

Eval-through-API is not free. A single MMLU run at 14k questions × ~500 tokens each × N candidates is a real dollar amount at fine-tune-tier pricing. Three levers:

**Lever 1 — Subset intelligently.** Mod-108's fast lane (chapter 6) already exists for this exact reason. Run the full suite once per major release; run the fast lane per candidate.

**Lever 2 — Use the Batch API.** OpenAI, Anthropic, and Bedrock all offer batch modes with substantial discounts (often ~50%) in exchange for async submission and a 24-hour SLA. `lm-evaluation-harness` does not natively speak batch, but wrapping the eval with a script that submits all requests to Batch, waits, and then feeds the responses through the scorer is straightforward. For a full-suite eval that costs $200 in synchronous mode and can wait a day, batch turns it into $100.

**Lever 3 — Cache aggressively.** If you rerun the same eval on the same model, the results should be identical (given fixed seed and greedy sampling). Cache per (candidate_id, task, item_id, sampling_config_hash). On a rerun after a partial failure, cached items short-circuit. Both `lm-evaluation-harness` and the eval driver you build around it can carry a cache.

Combined: a fast-lane eval on the pinned tasks, submitted via Batch, with per-item caching, is roughly 10–20× cheaper than a naive full-suite synchronous eval. This is what makes eval-through-API viable as a routine gate rather than a special event.

## The released-artifact record for a hosted fine-tune

Mod-109 defines a "served artifact hash" — a SHA of the quantized weight file that the release-gate report references. For a hosted fine-tune, there is no weight file to hash — but you still need an artifact identifier that pins what was evaluated.

The equivalent record for a hosted candidate:

```yaml
released_artifact:
  kind: hosted
  vendor: openai
  model_id: ft:gpt-4.1-mini-2025-07-14:acme::support-triage-v3:AbCdEf12
  base_pinned_version: gpt-4.1-mini-2025-07-14
  fine_tune_job_id: ftjob-AbCdEf12345
  training_file_sha256: 3c...        # from chapter 2's manifest
  training_manifest_uri: s3://data/support/ftjob-AbCdEf12345-manifest.yaml
  first_callable_at: 2026-08-01T15:47:33Z
  eval_at: 2026-08-01T18:12:00Z
  eval_gate_manifest_version: 2026-08-01
  eval_results_uri: s3://eval/support-triage-v3/openai-ft/results.json
```

The four load-bearing fields:

- `model_id` — what the request is actually sent to. This is your "artifact hash" equivalent — the thing that identifies a specific model.
- `base_pinned_version` — the version-pinned base. If OpenAI ships a new snapshot, this stays fixed.
- `fine_tune_job_id` — reproducibility. Given this, the eval report can trace back to the training-data hash and the hyperparameters.
- `eval_gate_manifest_version` — pins the eval itself. Same discipline as mod-108 chapter 6.

Store this alongside the self-hosted candidates' analogous records. The release-gate report then presents both on the same table.

## Producing a cross-backend leaderboard

The output of running the manifest above:

| Candidate | GSM8K | IFEval | task_held_out | Judged (mod-108 ch. 4) | $/req (input+output) |
|---|---|---|---|---|---|
| Self-hosted Llama 3.1 8B FT v3 | X.XX | X.XX | X.XX | X.XX | $0.0002 |
| OpenAI ft:gpt-4.1-mini v3 | X.XX | X.XX | X.XX | X.XX | $0.0012 |
| Anthropic ft:claude-haiku v3 | X.XX | X.XX | X.XX | X.XX | $0.0008 |

Rules of thumb for reading this leaderboard:

- **Absolute scores must include CIs.** Mod-108 chapter 3's bootstrap CI machinery applies unchanged; a 1-point delta with overlapping CIs is not a signal.
- **The $/req column is not the same as the total cost.** Chapter 5's four-lane cost model applies. A candidate that is $0.0002/req but requires $2000/month of GPU capacity has a different profile than one that is $0.0012/req with $0 fixed cost.
- **Judged eval quality is subject to judge quirks.** Same rules from mod-108 chapter 4 — position bias, self-preference bias, verbosity bias. Use a judge that is not in the candidate set.
- **Task-specific held-out is your workhorse signal.** Managed models can score suspiciously well on academic tasks (because they've seen them at pre-training) and mediocre on your task-specific set. The held-out set is the primary quality signal for your project.

## Contamination disclosure for hosted models

Mod-108 chapter 5 built the contamination-detection story. For self-hosted models fine-tuned on your data, you scan the training data for eval-set overlap. For hosted models, this scan is unchanged — the training data is still yours to scan.

What changes: **you cannot scan the base model's pretraining data.** OpenAI, Anthropic, and Google do not publish their pretraining corpora. If your eval set is a public benchmark (MMLU, GSM8K, HumanEval), you have to assume the base model has seen it in pretraining and therefore any absolute score is contaminated to some degree.

Two mitigations:

1. **Prefer private held-out evals for the primary quality signal.** Mod-108 chapter 3's task-specific held-out set is not in any pretraining corpus. Regressions on it are meaningful in a way public-benchmark regressions are not.
2. **Use model-signal contamination probes.** Mod-108 chapter 5 lens 3 (perplexity on eval items compared to shuffled variants) works on any model with generation output — including hosted models. Run these probes on public benchmarks; disclose the results in the release-gate report.

The disclosure section for a hosted candidate:

```markdown
## Contamination disclosure — openai_ft_gpt_4_1_mini_v3

### Fine-tuning data
- Scanned for exact / near-exact overlap with all release-gate eval sets: 0 hits.
- Scanned against MMLU, ARC, TruthfulQA, GSM8K, HumanEval: 0 hits.

### Base-model pretraining data
- Not scannable (proprietary). Assume pretraining includes public benchmarks.
- Model-signal probes ran on MMLU / GSM8K: [pass / fail / suspicious per lens 3].
- Primary quality signal: task-specific held-out (private, mod-108 ch. 3 chapter 3).
- Interpretation: absolute academic-benchmark scores are indicative, not authoritative;
  deltas across our fine-tune candidates on the same academic benchmark are still meaningful.
```

Same shape as mod-108 chapter 5's disclosure, adapted to the "we cannot see the base's pretraining" reality.

## Common eval-handoff bugs

**Bug: managed candidate scores 20 points higher on MMLU than self-hosted.**
Likely cause: (a) base model contamination (see above), (b) chat-template mismatch on the self-hosted side (mod-108 chapter 7 pitfall #1), or (c) MMLU-logprob-vs-generative variant mismatch across the two candidates. Diagnose in that order.

**Bug: self-hosted GSM8K matches BF16 self-hosted but managed GSM8K is 10 points lower.**
Likely cause: sampling mismatch. The managed backend defaulted to a non-zero temperature, or the max_tokens was too low and answers were being truncated. Check the `--write_out` sample dumps for truncation and confirm the exact sampling params were sent.

**Bug: eval-through-API bill is 5x what you estimated.**
Likely cause: the input token count was estimated for the base tokenizer, but the actual chat template on the managed side inflates by 20–30% (system prompt + wrapping). Add a 30% headroom to your estimate, then measure actuals on the first eval and re-baseline.

**Bug: the managed candidate refuses ~5% of prompts (safety-filter refusal).**
Likely cause: the base model's safety filter is stricter than your self-hosted base. This is a real quality signal — the managed candidate is not answering questions your product may need answered. Log the refusal rate as a separate metric; do not silently score a refusal as "wrong."

**Bug: eval scores drift by 0.5-2 points between runs of the same manifest on the same managed model.**
Likely cause: even with `temperature=0`, some vendors introduce non-deterministic components (batch scheduling, kv-cache reuse, quantization stochasticity in the served stack). If you cannot make the API deterministic, mean-of-N is the answer — usually N=3 is sufficient.

## Handing off to mod-111

The eval-across-backends leaderboard is one of the inputs mod-111 (post-training systems design) consumes as it builds the model registry. Every candidate — self-hosted or hosted — is one entry in the registry, with:

- Its released-artifact record.
- Its eval-gate result JSON.
- Its per-request cost profile.
- Its compliance one-pager (chapter 5).
- Its ADR (chapter 6).

The registry is what makes "we serve model X for tenant Y" queryable at platform scale. This module produces the individual records; mod-111 turns them into a searchable, governance-enforced platform.

## Summary

- The mod-108 release-gate manifest is the single source of truth for what a fair comparison means. This chapter extends it to hosted candidates without changing what "the gate" is.
- `lm-evaluation-harness` supports API backends (`openai-chat-completions`, `local-chat-completions`, `anthropic-chat-completions`). For vendors without a native backend, use a compatibility gateway (LiteLLM, internal proxy) so the harness sees an OpenAI-shape endpoint everywhere.
- **Log-likelihood tasks do not port cleanly to chat-only endpoints.** Prefer generation-based evals (task-specific held-out, IFEval, GSM8K, HumanEval, judged) for cross-backend comparability. If you must use MMLU cross-backend, use the generative variant across the board — do not mix variants across candidates.
- Sampling parameters must be pinned per the manifest and match across backends. Temperature 0, top-p 1, fixed seed, explicit max_tokens, no penalties. Non-deterministic backends require mean-of-N runs.
- Eval-through-API cost is controlled by the fast-lane subset, the Batch API discount, and per-item caching. Together, ~10-20× cheaper than a naive synchronous full-suite eval.
- The "served artifact hash" for a hosted candidate is the tuple (`model_id`, `base_pinned_version`, `fine_tune_job_id`, `training_file_sha256`, `eval_gate_manifest_version`). Same discipline as mod-109's artifact hash; different fields.
- Contamination disclosure for hosted candidates admits that the base's pretraining data is not scannable. Prefer private held-out evals as the primary quality signal; use lens-3 model-signal probes on public benchmarks and disclose.
- The cross-backend leaderboard is one of the inputs mod-111 uses to build the model registry. Every candidate gets an entry; the registry is what makes multi-vendor operation tractable.
