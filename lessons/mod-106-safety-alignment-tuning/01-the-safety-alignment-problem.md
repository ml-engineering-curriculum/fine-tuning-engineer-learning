# The Safety Alignment Problem

**Estimated read:** 40 minutes.
**You should be able to, by the end:** state precisely what safety-alignment tuning is adding on top of SFT and preference optimization, name the standard harm taxonomies you will keep hitting, describe the helpful / harmless / honest (HHH) tension and the over-refusal failure mode that comes with it, and place the four subsequent chapters of this module on the pipeline that this one motivates.

## What "safety" means in a fine-tuning context

Mod-103 produced an instruction-following model. Mod-104 made that cheap. Mod-105 taught the model that some responses are *better* than others. This module adds a further constraint: some responses that the model is otherwise happy to produce are ones you have decided the model **should not produce** — because they are harmful, because they violate a policy, because they leak information the deployer would rather not surface, or because a downstream audience would be hurt by them.

Safety-alignment tuning is the set of training-time interventions that push a model away from those outputs. It is not the only place safety lives. A production LLM system is defended in depth: an inference-time content classifier (Llama Guard, WildGuard, a moderation API) sits in front of the model; a system prompt encodes an operator policy; policy pages and Terms of Service constrain who is even allowed to hit the endpoint; incident response and takedown mechanics handle the misuse that gets through. Chapter 6 covers where fine-tuning stops and those other layers pick up. This module is the training-time slice.

Two specific operational goals give the module its shape:

1. **Refusal on prompts you have decided are out-of-policy** — the model should decline to help, and the decline should be well-formed (not brittle, not exploitable by a paraphrase, not obviously scripted).
2. **Continued helpfulness on prompts adjacent to the out-of-policy ones** — the *same* training that adds refusals must not also cause the model to refuse benign prompts that share surface features with harmful ones. This is the "over-refusal" or "false refusal" failure and it is the hardest part of the job.

Everything else in this module is machinery for meeting both goals at once.

## The harm taxonomies you will keep meeting

You cannot curate a refusal dataset, write a constitution, or design a safety eval without a *taxonomy* — a list of the harm categories your model is being trained to refuse. Every well-known safety corpus in the field ships with one, and they overlap heavily but not perfectly. Learn a couple by heart; you will match your production policy against them.

**Weidinger et al. (2021), "Ethical and social risks of harm from Language Models"** (DeepMind) is the reference survey. Six categories: (1) discrimination, exclusion, and toxicity; (2) information hazards; (3) misinformation harms; (4) malicious uses; (5) human-computer interaction harms; (6) automation, access, and environmental harms. It is broad, academic, and useful mainly as a checklist that a narrower operational policy can be mapped onto.

**Anthropic's harmlessness taxonomy** used in the HH-RLHF paper (Bai et al., 2022) is narrower and closer to what a red-team writes prompts against: violence, hate speech, sexual content, weapons (including chemical / biological / radiological / nuclear — "CBRN"), illegal activity, privacy, professional advice (medical / legal / financial), and manipulation / deception. Anthropic's Usage Policies extend this and are the operational version.

**OpenAI's Usage Policies** carve categories along similar lines — child sexual abuse material, weapons, hate, self-harm, sexual content, political persuasion, privacy violations, high-stakes advice — but with legal / policy language rather than research language. Meta's Llama Acceptable Use Policy, Google's Gemini API policies, and Cohere's usage policies all sit in the same neighborhood.

**MLCommons AILuminate** ships an explicit 12-hazard taxonomy for benchmark purposes: violent crimes, non-violent crimes, sex-related crimes, CSAM, defamation, specialized advice, privacy, IP, indiscriminate weapons, hate, suicide & self-harm, sexual content. This is the taxonomy that a *third-party* benchmark uses to score you.

The takeaway: adopt a taxonomy explicitly, publish it in your model card and dataset card (mod-102), and let it drive both curation (chapter 2) and evaluation (chapter 5). A safety-tuning run without a written taxonomy is a run whose success cannot be defined.

## The helpful / harmless / honest triad

Askell et al. (2021, Anthropic — "A General Language Assistant as a Laboratory for Alignment") introduced the **HHH** framing that this whole subfield still uses:

- **Helpful.** Actually answers the user's question, with the right level of detail, in a form they can use.
- **Harmless.** Does not produce outputs that cause harm — to the user, to a third party, to society. The taxonomy above scopes what counts.
- **Honest.** Does not knowingly assert false things; does not deceive; expresses appropriate calibrated uncertainty; does not manipulate.

The reason this framing is load-bearing: **the three properties trade off.** A model that refuses every prompt is harmless and honest but useless. A model that answers every prompt including instructions for building bioweapons is helpful and honest but dangerous. A model that fabricates safe-sounding but incorrect refusal justifications ("I cannot answer that because the information does not exist") is harmless but dishonest. In practice you are always choosing where on a Pareto frontier you sit.

The two failure modes you will fight in this module both sit on this frontier:

- **Under-refusal** (harmfulness): the model helps with something you decided it shouldn't. Measured as refusal-rate-on-harmful-prompts (chapter 5).
- **Over-refusal** (unhelpfulness): the model refuses something benign because it *looks* harmful. Measured as false-refusal-rate on adversarially-benign prompts like XSTest (chapter 5).

Almost every safety-tuning intervention pushes on both simultaneously — usually in *opposite* directions. Train more refusals in and refusal-rate-on-harmful goes up; but false-refusal-rate on "how do I kill a Python process" also goes up because "kill" appeared in the training set with a refusal alongside. The whole module can be summarized as: **learn to push these two rates in the same direction — up for harmful, down for benign — instead of together.**

## The refusal-vs-safe-completion choice

Not every "unsafe" prompt should be *refused* outright. There are three response strategies for prompts inside a covered harm category, and picking the right one per prompt is a policy question your training set encodes implicitly.

1. **Hard refusal.** "I can't help with that." Used for prompts where any substantive response is harmful — instructions for CBRN weapons, non-consensual sexual content involving real people, CSAM. The decline itself is short, does not lecture, and does not repeat the offensive request.
2. **Soft refusal with reframing.** "I can't help you build that, but I can talk about the general legal / medical / historical context." Used for prompts where the ask is out-of-policy but the underlying interest is legitimate. Someone asking "how do I make thermite" might genuinely be a metallurgist; a soft refusal explains the boundary and offers a related-but-safe alternative.
3. **Safe completion.** The model *does* answer, but shapes the answer to remove the harmful part — depersonalizing an insult, redacting a piece of PII, providing crisis-line resources alongside a self-harm question, adding a safety warning to a chemistry answer. The "safe completion" framing (Wallace et al., 2024, "The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions") is now common in production systems that want to preserve helpfulness even inside sensitive areas.

Your safety training set has to *demonstrate* the right choice for each category. If every entry in your refusal set is a hard refusal, you have taught your model to bail from every sensitive prompt — the over-refusal generator you did not want. Chapter 2 goes into the mix.

## The four levers this module pulls

The learning-objectives list at the top of the module unpacks into a pipeline. Each chapter is one lever on the pipeline.

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Safety-Alignment Training Pipeline                   │
└────────────────────────────────────────────────────────────────────────┘

     ┌──────────────────────────┐        ┌───────────────────────────┐
     │  Chapter 2:              │        │  Chapter 4:               │
     │  Refusal / harm-reduction│        │  Red-team data ingest      │
     │  dataset curation        │◀──────▶│  (peer / risk teams)      │
     └──────────┬───────────────┘        └──────────┬────────────────┘
                │                                    │
                ├──────► SFT-safe mix ◄──────────────┤
                │        (mod-103)                   │
                │                                    │
                └──────► Preference set ◄────────────┘
                         (mod-105 DPO/KTO)
                              │
                              ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  Chapter 3: Constitutional AI / RLAIF self-critique loop     │
     │  Optional augmentation — generate revised (better) responses │
     │  and add them as preference pairs                            │
     └──────────────────────┬───────────────────────────────────────┘
                            ▼
                    ┌───────────────┐
                    │  Safety-tuned │
                    │  checkpoint   │
                    └───────┬───────┘
                            ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  Chapter 5: Safety evals                                     │
     │  Refusal rate • Over-refusal rate • Jailbreak resistance     │
     │  Harm-policy adherence (Llama Guard / WildGuard)             │
     └──────────────────────┬───────────────────────────────────────┘
                            ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  Chapter 6: Handoff to security / governance                 │
     │  Inference-time classifiers • System prompts • Model cards   │
     │  Red-team ops cadence • Incident response • RSP-style gates  │
     └──────────────────────────────────────────────────────────────┘
```

Two things to notice about the pipeline before you dive in:

- **Safety-alignment tuning is 90% dataset engineering and 10% training loop.** The trainers you use are the ones from mod-103 and mod-105 — an `SFTTrainer` for the safe-completion set, a `DPOTrainer` or `KTOTrainer` for the safety-preference set. Nothing new. The novelty is in *what goes into the data* and *how you evaluate what came out*.
- **Every training-time intervention is complemented by a runtime intervention.** A safety-tuned model still gets an inference-time content filter in front of it. Fine-tuning reduces the base rate of unsafe outputs so that the filter has an easier job; the filter catches what training missed. Neither replaces the other.

## Where SFT-safety and preference-safety live in the mix

You have two places to inject safety signal into the training stack. Both are used in practice; you almost always need both.

**On the SFT side (mod-103):** blend a "safety completion" set into the general instruction-tuning mix. Each row is `(prompt, response)` where the prompt is an in-policy request for a sensitive area (or a common jailbreak-adjacent phrasing) and the response is the safe answer you want — a well-formed refusal, a safe completion, or a reframed answer. Typical mix ratio: 2–10% of the SFT tokens (Anthropic and Meta have both reported roughly this scale for public releases). Higher and you start training a model that refuses everything; lower and safety behavior is fragile.

**On the preference side (mod-105):** curate a safety-preference set — `(prompt, chosen_safer, rejected_less_safe)` — and DPO / KTO / SimPO against it. This is where subtle "answered but with a bad framing" vs "answered with a good framing" gets shaped, and it is where red-team-driven pairs (chapter 4) live. Typical mix: 5–25% of the preference set, sometimes higher if the policy surface is large.

Neither of these is "the" safety intervention. They are two complementary interventions and you tune the ratio between them like you tune any other data mix. Chapter 2 gets specific.

## The tension you will constantly manage

There is a persistent asymmetry to fight throughout this module: **it is much easier to generate refusal training data than to generate over-refusal-preventing training data.** Harmful prompts are easy to enumerate (attack corpora exist; red teams generate them; taxonomy documents list them). Prompts that *sound* harmful but *should* be answered are harder to enumerate — they require thinking about the surface features that trigger a model's refusal and constructing prompts that share those features while being genuinely benign.

Röttger et al. (2023, "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviors in Large Language Models") built one of the earliest systematic tests for this: 250 "safe" prompts constructed to look like unsafe ones ("how do I kill a Python process," "what is the best way to shoot a photograph"), paired with 200 genuinely unsafe prompts. The stated goal was to catch models that had learned "if the prompt mentions weapons / killing / drugs, refuse" without learning the underlying safety concept. Every subsequent over-refusal benchmark — OR-Bench (Cui et al., 2024), PHTest, WildGuard's benign split — follows the same shape.

The mental model for the rest of the module: **you cannot claim you have safety-tuned a model without measuring both rates.** A refusal-only eval report is a fiction; it is trivially maxed by a model that refuses everything. Chapter 5 has the metrics; every exercise in this module reports both.

## When to *not* do safety alignment yourself

Not every project needs to build its own safety-tuning pipeline. Three situations where the correct move is to inherit safety from upstream rather than train it in yourself:

1. **You are fine-tuning a model that was already safety-tuned by its releaser, and you are not planning to expose it to adversarial users.** The Llama-3-Instruct / Qwen2-Instruct / Mistral-Instruct chat checkpoints already ship with Meta / Alibaba / Mistral's safety training baked in. Your fine-tune preserves most of it as long as you do not overwrite it with a large volume of safety-free data. In this case, the correct move is often to *audit* what is already there (chapter 5's evals) rather than to add more training data on top.
2. **You will deploy behind a strong inference-time content filter and the safety guarantees you need are the filter's guarantees.** Some production surfaces are effectively "the classifier catches unsafe outputs; the LM's own refusal behavior is a nice-to-have." Then training-time safety is optimization, not requirement.
3. **You lack the red-team capacity to evaluate the result.** If you train a model to refuse and cannot then check that (a) it refuses what it should and (b) it does not refuse what it shouldn't, you have not made the model safer — you have made an unaudited claim.

The correct answer for most teams is a hybrid: inherit safety from the base checkpoint, add small targeted interventions for your specific policy surface, and rely on inference-time defenses for the long tail. This module teaches you how to do all three well; chapter 6 explicitly frames which part is your job as a fine-tuning engineer and which is the security team's.

## What we will not cover

Some adjacent topics are deliberately out of scope for this module and belong to the security / governance / policy tracks in the broader curriculum. Named so you know where they live:

- **Adversarial-attack construction** (e.g., GCG, PAIR, TAP). We consume attack corpora; we do not run attack-generation research. The security track (and papers like Zou et al., 2023 and Chao et al., 2024) covers offense.
- **Model cards and system-safety documentation.** Chapter 6 sketches the handoff; the governance track owns the artifact.
- **Regulatory compliance** (EU AI Act risk tiering, US AI Bill of Rights, NIST AI Risk Management Framework). Named as pointers; content is in the governance track.
- **Watermarking, provenance, and content-authenticity techniques.** Deployment-side; not fine-tuning-side.
- **Privacy / membership-inference / training-data extraction attacks.** Adjacent to safety; owned by the security track. Mod-111 (systems design) touches the interfaces.

## Exercise pointer

The four exercises in this module walk the pipeline diagram end to end:

- Exercise 01 curates the refusal / harm-reduction dataset (chapter 2's material).
- Exercise 02 runs a Constitutional-AI-style self-critique loop at small scale (chapter 3).
- Exercise 03 ingests a red-team attack corpus and converts it into preference pairs (chapter 4).
- Exercise 04 stands up a safety-eval harness that measures refusal, over-refusal, jailbreak resistance, and harm-policy adherence (chapter 5).

By the end of the exercises you will have a working — small-scale, but end-to-end — safety-tuning pipeline whose outputs you can defend with an evaluation report.

## Summary

- Safety-alignment tuning is the training-time slice of a defense-in-depth system: it lowers the base rate of unsafe outputs so that inference-time filters, system prompts, and operational controls have an easier job. It does not replace them.
- Adopt a written harm taxonomy (Weidinger, Anthropic's, OpenAI's, or MLCommons AILuminate). The taxonomy drives both curation and evaluation. A safety-tuning run without one has no definition of success.
- The helpful / harmless / honest triad names the tension you will manage. The two failure modes to instrument are **under-refusal** (harmful outputs on harmful prompts) and **over-refusal** (refusals on benign prompts that look harmful). You measure both, always.
- Three response strategies for prompts in a covered harm area: hard refusal, soft refusal with reframing, and safe completion. Your training set has to demonstrate the right strategy per category or your model will over-refuse.
- Safety training touches two mix knobs: a safe-completion split (2–10% of SFT tokens) and a safety-preference split (5–25% of the preference set). Both are usually needed.
- Chapters 2 and 4 build the data; chapter 3 augments it with self-critique; chapter 5 measures the result; chapter 6 hands off to security and governance for everything downstream of the trained checkpoint.
