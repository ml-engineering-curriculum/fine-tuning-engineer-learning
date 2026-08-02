# exercise-03: RoPE vs ALiBi Context-Length Experiment

**Estimated effort:** 3 hours
**Companion chapter:** [`03-positional-encodings.md`](../03-positional-encodings.md)

## Objective

Feel the difference between RoPE and ALiBi at context lengths past their pretrained window, with your own eyes and on your own machine. By the end you will have a short experimental writeup that shows (a) what happens when you run a RoPE model beyond its trained context without scaling, (b) what changes when you apply a scaling recipe, and (c) how an ALiBi model behaves in the same range.

This is a *measurement* exercise. You are not fine-tuning anything — you are running inference under three configurations and documenting the differences quantitatively.

## Prerequisites

- Python 3.10+ with `torch`, `transformers`.
- Enough compute to load a 1B–3B model in half precision. A single 8-GB GPU is enough; CPU works if you are patient.
- You should have read the companion chapter.

Model choices (pick one from each family):

- **RoPE model** — `meta-llama/Llama-3.2-1B` (pretrained at 128k context — for this exercise treat the *effective* trained context as 8k and go past that), or `Qwen/Qwen2.5-1.5B` (32k), or any smaller RoPE model with a shorter trained context that fits your compute.
- **ALiBi model** — `mosaicml/mpt-1b-redpajama-200b-instruct` or a small BLOOM checkpoint (e.g., `bigscience/bloom-560m` or `bloom-1b1`). If neither is easily obtainable, note that you skipped this branch and explain your alternative.

If your chosen RoPE model has a very large trained context, you may want to force a shorter "effective" context by editing `config.max_position_embeddings` on load to something small (say, 2048) so you can go past it meaningfully on modest hardware.

## Requirements

### Part A — Baseline metric

Write a script that computes **perplexity on a long text sample**, run at increasing context lengths. Use any longish public document — a Wikipedia article, a chapter of a Project Gutenberg book, a technical paper's plain text. Pick one document at least 32k tokens long by the model's own tokenizer.

For each context length `L` in `{512, 1024, 2048, 4096, 8192, 16384}` (adjust the top of the list to comfortably exceed your model's pretrained max context):

1. Take the first `L` tokens of your document.
2. Run the model forward and compute per-token cross-entropy on the actual next tokens.
3. Report the *average* NLL over the last 512 tokens of the window — that measures "given a context of `L-512`, how well does the model predict the next 512 tokens?"

You should see a graceful decrease in NLL as `L` grows up to the pretrained maximum (more context helps), then a **sharp increase** past it for an un-scaled RoPE model, and something visibly different for the ALiBi model.

Plot the NLL vs `L` curve (matplotlib, or a saved CSV plus a table in your writeup — the point is the numbers, not the graph library).

### Part B — RoPE, three configurations

Repeat part A on your RoPE model in three configurations, using the same document and the same set of `L` values:

1. **Baseline** — no scaling. `AutoModelForCausalLM.from_pretrained(model_id)`.
2. **Linear position interpolation (PI)** — pass `rope_scaling={"type": "linear", "factor": 4.0}` at load time (adjust factor to match your ratio of target-max to pretrained-max).
3. **Extended-context release** — if your model has a shipped long-context variant (e.g., Llama 3.1's built-in `rope_scaling` block), load that variant with default settings.

Overlay the three curves. Which configuration collapses fastest past the pretrained max? Which holds up best?

### Part C — ALiBi baseline

Run the same NLL-vs-L experiment on your ALiBi model, using its default configuration only. There is no "scaling recipe" here — the point of the experiment is that ALiBi's behavior past pretrained length should look qualitatively different from un-scaled RoPE's.

### Part D — Writeup

In a 400–600 word writeup, answer:

1. Where in the `L` sweep does the un-scaled RoPE model start to visibly degrade? By what factor does NLL rise from the last "in-window" `L` to the first "out-of-window" `L`?
2. Does linear PI recover meaningful quality? By how much? Does it hurt short-context quality (`L < pretrained_max`)? Cite specific numbers.
3. Does the shipped extended-context variant outperform your ad-hoc PI settings? By how much?
4. How does the ALiBi model's curve differ shape-wise from any RoPE curve? What is your interpretation, referencing the mechanics from the companion chapter?
5. **Systems consequence.** If you had to fine-tune your RoPE model to a 4× extended context for a real project, what would you need to change beyond the `rope_scaling` config? Name at least three concerns from the pipeline map in chapter 4 (data mix, VRAM / KV-cache, evaluation, batch shapes, etc.).

## Starter guidance

- Perplexity across a long window is expensive; use `torch.no_grad()` and half precision, and only run the top of the `L` ladder on the largest configuration that fits.
- To do the per-window NLL cleanly: `outputs = model(input_ids, labels=input_ids)`. The returned `loss` is the mean cross-entropy over the whole sequence with the built-in shift. To restrict to the last 512 positions, either slice the logits and labels yourself or run the model twice (once on the prefix, once on the full sequence, then difference the sums — mathematically equivalent, sometimes easier).
- The `rope_scaling` API in `transformers` has evolved. As of the LLaMA 3.1 release the recommended field names include `rope_type`, `factor`, `low_freq_factor`, `high_freq_factor`, and `original_max_position_embeddings`. Check the version of `transformers` you are on and the current `LlamaConfig` docstring — the older `{"type": "linear", "factor": ...}` form still works for pure PI but is less expressive.
- If your ALiBi model choice fails to load (dependency mismatch, license, etc.), document exactly what went wrong and either substitute a different ALiBi checkpoint or run only the RoPE branches and expand the writeup to compensate.
- Save your raw NLL numbers as a CSV so you can re-plot without re-running.

## Acceptance criteria

You have completed the exercise when:

- You have NLL-vs-L curves (or tables) for at least three configurations: RoPE baseline, RoPE with a scaling recipe, and an ALiBi model.
- Your writeup answers all five questions with reference to actual numbers from your runs.
- The systems-consequence question names at least three real concerns beyond the config-flag flip and connects each to a downstream module of this track.
- Your numbers are reproducible — the script, the document you used, and a `requirements.txt` (or equivalent version pin) are checked in alongside the writeup.

## Stretch goals

- Add a **needle-in-a-haystack** micro-benchmark: hide a fact ("The secret code is 8471") at various depths in a long document, then ask the model to recall it. Compare recall rate across your three RoPE configurations and the ALiBi model. This is a much more sensitive test of long-context quality than perplexity alone.
- Repeat the RoPE experiment with the YaRN scaling recipe (there are open community implementations you can wire in). Compare to plain PI.
- Take a RoPE model of your choice, apply a short SFT pass (a few hundred million tokens of long-context data — you may not have the compute; describe the recipe in writing if not), and re-measure. This bridges into mod-107.
- If you happen to have access to a genuine `long-context` variant of your ALiBi model, add a fourth configuration to your comparison and comment.
