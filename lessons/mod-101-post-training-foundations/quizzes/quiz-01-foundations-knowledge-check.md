# quiz-01: Post-Training Foundations Knowledge Check

**Format:** 10 multiple-choice questions.
**Covers:** chapters 1–4 of `mod-101-post-training-foundations`.
**How to use:** answer every question before scrolling to the key. One or two sentences of explanation per answer points to the chapter the question is drawn from.

The questions test whether you can *reason* about the material, not whether you can recite phrases from the chapters. Expect distractors that sound plausible if you only skimmed.

---

## Questions

### 1. Which four `nn.Linear` layer names make up the self-attention sublayer inside a Llama-style block?

- A. `attn_in`, `attn_out`, `gate_proj`, `down_proj`
- B. `q_proj`, `k_proj`, `v_proj`, `o_proj`
- C. `c_attn`, `c_proj`, `fc_in`, `fc_out`
- D. `wq`, `wk`, `wv`, `wo` — those are PyTorch's canonical names and Hugging Face aliases them

### 2. You set `peft.LoraConfig(target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])` on a Llama 3 8B base. Which statement is correct?

- A. LoRA adapters are attached to every `nn.Linear` named in the list, on every one of the 32 transformer blocks, while RMSNorms and the LM head remain frozen.
- B. LoRA adapters are attached to the seven named linears plus the final RMSNorm, because `o_proj` and the final norm share parameters.
- C. The adapters target the seven linears, but the LM head is also fine-tuned by default because `peft` always unfreezes `lm_head`.
- D. The seven-module target list only applies to the first and last block; interior blocks fall back to `q_proj` / `v_proj`.

### 3. Why does the chapter warn against applying BitFit to Llama-family bases?

- A. BitFit only updates LayerNorm scales, and Llama uses RMSNorm.
- B. BitFit modifies the LM head by default, which ties back to the embedding table.
- C. BitFit trains only bias parameters, and Llama-family linears have no biases to train.
- D. BitFit is incompatible with grouped-query attention because the biases differ by head.

### 4. On a Llama 3 Instruct tokenizer, which special token is the trained end-of-turn marker that a serving stack should stop generation on?

- A. `<|end_of_text|>`
- B. `<|eot_id|>`
- C. `<|im_end|>`
- D. `</s>`

### 5. A junior teammate sets `tok.pad_token = tok.eos_token` so that `SFTTrainer` can batch examples, but makes no further changes to the data collator. What is the concrete failure mode the chapter warns about?

- A. Gradients will NaN on the first step because EOS has no embedding for padded positions.
- B. The tokenizer will refuse to encode the padding token and raise at runtime.
- C. The model learns to emit EOS everywhere, because every padded position contributes an EOS target to the cross-entropy loss.
- D. The attention mask will be zeroed out, so no tokens contribute to the loss at all.

### 6. You render an SFT example with `apply_chat_template(..., tokenize=False)`, then pass the string into `tok(text)` with its default kwargs before handing it to the trainer. What footgun has you just stepped on?

- A. The chat template was never applied; `tokenize=False` skips it entirely.
- B. Two BOS tokens are now at the front of the sequence, because `apply_chat_template` already prepended one and `tok(text)` added another via `add_special_tokens=True`.
- C. The round-trip decode is lossless, so nothing is wrong — this is the canonical pattern.
- D. The assistant's response is silently stripped because `tokenize=False` discards non-system content.

### 7. A RoPE-based Llama model pretrained at 8k context length is fed a 16k-token prompt at inference, with no `rope_scaling` applied. Why does quality collapse?

- A. The positional-embedding table only has 8192 rows, so positions ≥ 8192 have no vector to look up.
- B. The KV cache overflows at 8192 entries and silently wraps around.
- C. The high-frequency rotations wrap around past the regime the weights were trained on, so the rotated `q` and `k` vectors become unrecognizable to the attention weights.
- D. The causal mask starts leaking future tokens past position 8192.

### 8. Which statement about ALiBi versus RoPE is supported by chapter 3?

- A. ALiBi is a learned positional embedding table added at the input, while RoPE rotates Q and K inside attention.
- B. ALiBi adds a fixed, non-learned, head-specific linear bias to attention scores based on key–query distance; it has strong length extrapolation but is less commonly adopted at scale than RoPE.
- C. ALiBi is a scaling recipe applied on top of RoPE for long contexts; it is not a separate positional encoding scheme.
- D. ALiBi models cannot be fine-tuned past their pretrained length because the slopes `m_h` would have to be relearned.

### 9. In the pipeline map, which of the following is correctly described as a stage the fine-tuner owns end-to-end?

- A. Pretraining the base model on trillions of tokens.
- B. Supervised Fine-Tuning (SFT), where the fine-tuner chooses the dataset, chat template, PEFT config, and loss-masking policy.
- C. Vocabulary design of the tokenizer the base ships with.
- D. The decision to use `cl100k_base` vs. `o200k_base` inside OpenAI's managed training service.

### 10. According to chapter 4, the primary reason DPO became the open-model default for preference optimization over PPO is:

- A. DPO guarantees a strictly lower KL to the reference than PPO.
- B. DPO reframes the RLHF objective as a supervised loss on preference pairs, skipping the separately-trained reward model and PPO loop, at competitive quality for most tasks.
- C. DPO is required by the Hugging Face `trl` library's licensing terms.
- D. DPO uses unpaired "good / bad" signal, which is cheaper to collect than preference pairs.

---

## Answer key

1. **B.** Chapter 1 ("Self-attention, one head at a time") names the four attention linears as `q_proj`, `k_proj`, `v_proj`, `o_proj` — the exact Hugging Face module names on Llama-family models and the canonical LoRA targets. GPT-2-style names in choice C are mentioned as a contrast, not the Llama convention.

2. **A.** Chapter 1's PEFT table lists exactly those seven modules as the "all-linear" QLoRA recipe, and the chapter closes by noting that norms and the LM head are almost never targeted. Choices C and D invent `peft` behavior the chapter does not describe.

3. **C.** Chapter 1's PEFT table footnote states: "Llama-family models have no biases on their linears; BitFit is a poor fit for these bases." The other answers invent constraints the chapter does not make.

4. **B.** Chapter 2 is explicit: on Llama 3 Instruct, `<|eot_id|>` (128009) is the trained turn terminator; `<|end_of_text|>` (128001) is a different token, and stopping only on the latter is a very common serving-stack bug.

5. **C.** Chapter 2's "three tokenization footguns" names this one directly: if padding positions are not masked (`labels[pad_positions] = -100`), every padded slot teaches the model to emit EOS, with no runtime exception to alert you.

6. **B.** Chapter 2 calls this the "double-BOS" footgun: `apply_chat_template` already prepends BOS, so a second tokenizer call with the default `add_special_tokens=True` prepends another. The rule of thumb in the chapter is to pass `add_special_tokens=False` after `apply_chat_template`.

7. **C.** Chapter 3 ("The knob a fine-tuner cares about: `rope_theta` and its scaling") explains that RoPE's highest-frequency rotations wrap within a bounded wavelength budget; past the pretrained range, the rotated Q/K vectors fall outside the regime the weights learned, and quality collapses. Choice A is the failure mode of *absolute* embeddings, not RoPE.

8. **B.** Chapter 3's ALiBi section describes exactly this: a fixed, non-learned, head-specific linear penalty on distance, strong length extrapolation, less commonly adopted at scale than RoPE. The chapter explicitly contrasts ALiBi with RoPE as *separate* schemes, ruling out C.

9. **B.** Chapter 4's "What the fine-tuner controls at each stage" table names SFT as the first fine-tuner-owned stage, with the four specific levers listed in the chapter's SFT section. Pretraining is explicitly called out as upstream and not owned by the fine-tuner.

10. **B.** Chapter 4's preference-optimization section describes DPO as "reframes the RLHF objective as a supervised loss on preference pairs, skipping the reward model" and calls it the "dominant open-model default … cheaper and simpler than PPO with competitive quality." Choice D confuses DPO with KTO, which is the unpaired variant.

---

## If you missed several

- Questions 1–3 missed: re-read [`01-decoder-only-forward-pass.md`](../01-decoder-only-forward-pass.md), specifically the "Inside one block" and "Where PEFT hooks in" sections, then do the shape trace in [`exercise-01`](../exercises/exercise-01-decoder-only-forward-pass-walkthrough.md).
- Questions 4–6 missed: re-read [`02-tokenization-and-chat-templates.md`](../02-tokenization-and-chat-templates.md), with a focus on "Three tokenization footguns," then run [`exercise-02`](../exercises/exercise-02-tokenizer-and-chat-template-audit.md).
- Questions 7–8 missed: re-read [`03-positional-encodings.md`](../03-positional-encodings.md). The interactive feel for RoPE collapse comes from [`exercise-03`](../exercises/exercise-03-rope-vs-alibi-context-length-experiment.md).
- Questions 9–10 missed: re-read [`04-post-training-pipeline-map.md`](../04-post-training-pipeline-map.md) and sketch the pipeline from memory before looking back at the chapter.
