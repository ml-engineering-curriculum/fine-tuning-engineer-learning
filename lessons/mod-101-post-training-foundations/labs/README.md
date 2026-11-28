# Labs for mod-101-post-training-foundations (Post-Training Foundations: Transformer Internals for Fine-Tuners)

Guided, step-by-step hands-on sessions. These are shorter and more scripted than the open-ended [`exercises/`](../exercises/) — the labs walk you through specific commands and expected outputs, while the exercises ask you to produce an artifact from scratch.

- [`lab-01-reading-a-base-model-config.md`](lab-01-reading-a-base-model-config.md) — Open the `config.json` of a real open base and map every field to the specific tensor or operation inside the forward pass from chapters 1 and 3. ~45 minutes, no GPU required.
- [`lab-02-special-token-budget-accounting.md`](lab-02-special-token-budget-accounting.md) — Break a two-turn chat into its three token accounts (structural, system prompt, content), reproduce the "end mid-sentence" symptom from truncation, and confirm the `add_generation_prompt` mismatch between training and inference. ~60 minutes, tokenizer only (no weights).
- [`lab-03-reading-rope-scaling-across-releases.md`](lab-03-reading-rope-scaling-across-releases.md) — Read `rope_theta` and `rope_scaling` out of three real configs and match each one to a recipe from chapter 3 (PI, NTK, YaRN, or a vendor-custom scheme). ~45 minutes, no GPU required.
