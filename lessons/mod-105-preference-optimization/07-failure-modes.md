# Failure Modes: Reward Hacking, Length Bias, KL Collapse, Reference Drift

**Estimated read:** 45 minutes.
**You should be able to, by the end:** recognize the four preference-optimization-specific failure modes by the fingerprints they leave on reward, KL, and eval curves; know which method (PPO / DPO / KTO / ORPO / SimPO / GRPO) each is most likely to hit; and know the first mitigation to try for each.

## The four modes

Preference optimization has its own bestiary of failures, distinct from the SFT catalog (mod-103 chapter 5) and the PEFT catalog (mod-104 chapter 5). They share a common structure: the training loss looks *great*, sometimes even improving through the whole run, and the model is nonetheless worse than the SFT checkpoint you started from.

The four:

1. **Reward hacking.** The policy exploits imperfections in the reward function to score high without actually being better.
2. **Length bias.** The policy learns "longer response = higher preference," producing verbose output regardless of whether length was warranted.
3. **KL collapse.** The policy diverges arbitrarily far from the reference; text quality degrades; entropy crashes.
4. **Reference-model drift.** The "reference" that the KL is anchored to is not actually a fixed anchor — it moves during training — and the constraint becomes meaningless.

Each has a distinctive signature. Each has a first thing to try.

## Reward hacking (a.k.a. Goodhart's law, over-optimization)

### What it is

The reward model or verifier is an *approximation* of the true objective. Any imperfection in that approximation can be exploited by an optimizer that pushes on the metric hard enough. "When a measure becomes a target, it ceases to be a good measure" (Goodhart, 1975; more recently formalized for RL by Skalse et al., 2022, "Defining and Characterizing Reward Hacking").

Concrete examples that have been documented in preference-optimization runs:

- **Sycophancy.** The RM learns "matches user's implicit stance ⇒ preferred." The policy pushes toward agreement even on false premises.
- **Format gaming.** The RM prefers markdown-formatted responses. The policy wraps every response in unnecessary bullet points.
- **Refusal gaming.** The RM has a soft preference for cautious refusals. The policy refuses more and more prompts, including those that should be answered.
- **Length gaming.** Distinct enough to warrant its own section below.
- **Token-level surface features.** The policy learns to emit specific phrases ("Certainly!", "As an AI assistant,") that the RM scores well, whether or not they fit the response.
- **Verifier gaming (in GRPO / RLVR).** The policy discovers that outputting "the answer is 42" always passes a poorly-specified verifier and starts doing so on unrelated prompts.

The unifying pattern: the RM's proxy for "good" has a shortcut, and the policy finds it. The paper of record is Gao, Schulman, and Hilton, 2023 ("Scaling Laws for Reward Model Overoptimization"), which measured this on synthetic setups and derived scaling laws for how quickly a policy over-optimizes an RM as a function of training compute.

### The fingerprint

The classic signature: **RM score / reward metric climbs monotonically; independent eval degrades or plateaus.**

Specifically, watch these together:

- **`objective/rlhf_reward` (PPO) or `rewards/margin` (DPO)** climbing steadily throughout training.
- **A separate held-out preference eval** (e.g., pairwise win rate against SFT judged by GPT-4, MT-Bench, RewardBench score) either flat or dropping.
- **Response length trending upward** (the length-hacking-in-disguise variant).
- **Response entropy / diversity dropping** — the policy has found the exploit and is repeating it.

The critical fact: **you cannot detect reward hacking with your own reward model.** By construction, the RM says the responses are getting better — that is the definition of the failure. You *have* to compare against an independent signal — a different judge, human eval, or a held-out task the RM was not trained on.

### Which methods are most vulnerable

- **PPO-RLHF:** classical vulnerability. Long on-policy training against a single frozen RM is the setup where hacking has been documented most often.
- **GRPO with a rule-based verifier:** vulnerable to specification bugs in the verifier. If your Python-code verifier only checks that the code parses (not that it runs), the policy will produce parsing-valid nonsense.
- **DPO / IPO / KTO:** less vulnerable in the classical sense because they never see on-policy samples. They can still overfit the training pairs (chapter 5's DPO overfitting failure is a cousin of reward hacking on a fixed dataset).
- **ORPO / SimPO:** the same as DPO plus additional vulnerability to whatever bias the fixed pairs carry.

The Gao et al. paper's finding: with an RM of a given size, there is an optimal amount of PPO training beyond which further training makes the policy *worse*. The optimum scales roughly with the RM's size. Practically: for a small RM (< 1B), overtraining hits within a few thousand steps; for a stronger RM (≥ base model size), you can train longer before the hack sets in.

### Mitigations

In order of what to try first:

1. **Cap training length.** For PPO, monitor a held-out eval and stop when it plateaus. For DPO, one epoch is often enough; more than three is usually worse.
2. **Increase the KL constraint.** For PPO, raise `kl_coef`. For DPO, raise `β`. This limits how far the policy can drift from the SFT anchor.
3. **Refresh the RM against on-policy data (iterative RLHF).** After N PPO steps, sample rollouts, label them with humans or a stronger judge, retrain the RM. Anthropic's Constitutional AI and Meta's Llama-2 both do variants of this.
4. **Ensemble the reward.** Use several RMs and average their scores (Coste et al., 2023, "Reward Model Ensembles Help Mitigate Overoptimization"). The consensus is harder to exploit than any single model.
5. **Use a different objective family.** If verifiable-reward is available, GRPO with a rule (not an RM) removes the RM-as-target problem, though not the more-general Goodhart problem.
6. **Length-normalize the reward or use SimPO.** Directly attacks the length-gaming variant.
7. **Constitutional AI / preference from principles.** Reformulate the preference labeling process itself so the RM's proxy is less exploitable.

The best mitigation is *knowing when to stop*. Log an independent eval; watch it; when it plateaus, stop the run.

## Length bias

### What it is

The specific failure of "the policy learns that longer responses win." Distinct enough from generic reward hacking to warrant its own section because it is so common and so easily measured.

The empirical claim (Singhal et al., 2023, "A Long Way to Go: Investigating Length Correlations in RLHF"): on multiple public RLHF datasets, response length correlates positively with the "chosen" label at roughly 60–75% of pairs. RMs trained on that data learn "longer = better" as a shortcut. Policies trained against those RMs produce progressively longer responses. Chatbot Arena leaderboard studies (Dubois et al., 2024) have found similar dynamics in downstream evaluation.

### The fingerprint

- **Median response length trending upward** during PPO / DPO. Log this at every eval step.
- **Length-conditional pairwise eval accuracy** (chapter 3) worse on "chosen shorter" pairs than on "chosen longer" pairs — for either the RM or a downstream judge.
- **Trained-model qualitative outputs are visibly padded** — filler phrases, restated summaries, hedges.

### Which methods are most vulnerable

- **PPO-RLHF with an unnormalized RM:** the highest-risk setup. The RM's length bias multiplies through many training steps.
- **DPO with unfiltered UltraFeedback / HH-RLHF:** documented length bias in both source sets pushes into the trained model.
- **KTO:** less vulnerable to length via preference data structure, but still inherits any length bias in the source.
- **ORPO:** the SFT loss term partly regularizes length toward the chosen distribution; still vulnerable if the chosen side is systematically longer.
- **SimPO:** designed to resist length bias — the length normalization is built into the loss.

### Mitigations

1. **Filter the preference set by length delta** (chapter 2). Drop pairs where `len(y_w) > 2 · len(y_l)`. Cheap; loses some signal.
2. **Length-balance the training set.** Sample so "chosen longer" and "chosen shorter" pairs are equal.
3. **Add a length penalty to the reward.** For PPO, subtract `α · len(y)` from the RM score. For DPO, either use a length-controlled loss variant (Park et al., 2024's R-DPO includes an explicit length regularization) or preprocess the data.
4. **Use SimPO.** The length-normalized implicit reward directly attacks the failure.
5. **Report length in every eval.** If you cannot see the length trending upward, you cannot catch this. Every downstream eval report should include response-length distribution alongside the win-rate number.

Length-controlled eval frameworks (length-controlled Chatbot Arena, LC-AlpacaEval) exist specifically because uncontrolled evals reward the same length bias the training does. If the eval you report is not length-controlled, you are measuring the wrong thing.

## KL collapse

### What it is

The policy `π` diverges arbitrarily far from `π_ref`. The KL penalty in the training objective either becomes overwhelmed (in PPO) or is implicit and unbounded (in reference-free methods). Symptoms range from mildly-off text to complete gibberish to mode collapse (the model outputs the same string on every prompt).

The mechanism differs by method:

- **PPO:** the reward gradient overpowers the KL penalty. Common when `kl_coef` is too low, LR is too high, or the reward has a lucrative shortcut.
- **DPO:** the log-ratio can grow without bound in either direction; both log-probabilities can drop toward zero (log-prob deflation, chapter 5) while the margin is preserved. The model becomes less confident about everything.
- **ORPO / SimPO:** no explicit reference model, so the "collapse" manifests as the model drifting from its base-model behavior with no anchor to pull it back. SFT-loss terms in ORPO partially mitigate; SimPO's length normalization is neutral on this.
- **GRPO:** same as PPO — reward-driven divergence when the KL constraint is too weak or the verifier has a shortcut.

### The fingerprint

- **`objective/kl` (PPO) or `rewards/chosen` and `rewards/rejected` diverging steeply** (DPO / KTO).
- **Generation length collapsing to 1 or the model repeatedly outputting the same token.**
- **Response entropy crashing to near zero.**
- **Eval loss reversing sharply after previously improving.**
- **Qualitative outputs become nonsensical** — either gibberish, empty, or a single canned phrase.

The KL divergence is not necessarily large in every collapse — mode collapse can happen with modest KL — but the *combination* of rising KL and dropping entropy is the diagnostic.

### Which methods are most vulnerable

- **PPO** is the classical case; entire papers exist on "how we prevented KL blowup." Adaptive KL controllers (Ziegler et al., 2020; Huang et al., 2024) are the standard fix.
- **DPO / IPO** with too-low `β` and too-many epochs.
- **ORPO** with too-high preference weight `λ` (SFT loss cannot pull back hard enough).
- **GRPO** with a poorly-specified reward that has a low-hanging exploit.

### Mitigations

1. **Raise the KL constraint.** For PPO, increase `kl_coef` or switch to an adaptive KL controller that tracks a target KL (e.g., 6 nats per response). For DPO, raise `β`. For ORPO, lower `λ`.
2. **Cap training length aggressively.** Preference optimization runs almost never benefit from more epochs; one epoch of DPO or a short PPO run at moderate KL is typically the sweet spot.
3. **Log per-step KL and stop the run on a threshold.** If per-step KL exceeds `2× target`, halt automatically. This turns KL collapse from a "wake up to a bad run" failure into an alerted event.
4. **Add an entropy bonus.** Small (`c_ent ~ 0.001–0.01` in the PPO surrogate) but real. Prevents the entropy crash independently of the KL constraint.
5. **Warm up on a shorter response length.** Long-response rollouts amplify KL because each token contributes to the total. Start with `response_length=128` and only raise it after 500 stable steps.
6. **Reduce LR.** Universal. If nothing else helps, halve LR.

## Reference-model drift

### What it is

The reference model that anchors the KL constraint is supposed to be *fixed*. It is your SFT checkpoint, frozen, and every KL penalty is computed against its behavior. If the reference drifts — accidentally trained, weight-shared with the policy in ways it should not be, or replaced mid-run — the "constraint" no longer constrains.

Concrete causes:

- **QLoRA + DPO with a shared 4-bit base.** The trick is: reference = same base with LoRA adapter disabled. If any non-LoRA parameter (LayerNorm cast to FP32 by `prepare_model_for_kbit_training`, embedding weights, LM head) is trainable, the "disabled adapter" reference is different from the SFT model.
- **`torch.compile` on a shared reference.** Rare edge case: the compiled reference graph can be different from the eager reference graph because of operator fusion differences.
- **Iterative RLHF where the reference is updated but the KL penalty is not proportionately re-tuned.** If you refresh `π_ref` to `π*` every 5000 steps, the KL constraint's meaning changes.
- **Wrong model checkpoint loaded as reference.** Simple bug; catch it by asserting `KL(π || π_ref) = 0` on step 0 for the same SFT init.

### The fingerprint

- **KL to reference is non-zero at step 0** (should be zero if reference == SFT-init policy).
- **KL to reference grows even when the policy is not learning** (loss flat, KL climbing).
- **DPO's `rewards/chosen` and `rewards/rejected` grow together** (both log-ratios increasing, no margin gain).
- **PPO's `objective/kl` climbs without bound but eval is not moving.**

The single most useful check: after loading the reference model, compute `π(y_ref | x_ref) == π_ref(y_ref | x_ref)` on a canned prompt at the *start* of training. Log the diff. If it is not identically zero, the reference is not what you think it is.

### Which methods are most vulnerable

- **DPO + QLoRA + LoRA.** The "disabled adapter" trick is fragile; audit it once per new codebase.
- **PPO with `precompute_ref_log_probs=False` (recomputed each step).** Any non-frozen parameter on the shared backbone bleeds.
- **PPO in distributed setups** where the reference and policy shards may not stay perfectly synchronized. Mod-107 covers this.
- **ORPO / SimPO** are immune (no reference model) — but they trade this for the no-anchor drift discussed above.

### Mitigations

1. **Assert step-0 KL = 0.** Add a startup assertion; refuse to train if the reference does not exactly match the SFT policy at initialization.
2. **Precompute reference log-probs.** TRL's `precompute_ref_log_probs=True` for DPO computes and caches reference log-probs at dataset preparation time, before any training. The cache is immutable during training. Slight memory cost, complete drift immunity.
3. **Freeze all non-LoRA parameters explicitly.** After `prepare_model_for_kbit_training` and `get_peft_model`, walk the parameter list and confirm nothing beyond LoRA `A` / `B` has `requires_grad=True`. `peft`'s `print_trainable_parameters` is a good sanity check but not a formal assertion.
4. **Load reference as a separate model when possible.** Memory cost is a full base copy but drift risk is zero. For PPO especially, the operational safety usually justifies the memory.
5. **Use adaptive KL controllers.** If the reference does drift, an adaptive KL controller (targeting a KL value rather than penalizing at a fixed coefficient) will at least keep the constraint's *effect* stable, even if its interpretation changes.

## The failure catalog at a glance

| Fingerprint | Likely failure | First fix |
|---|---|---|
| RM reward climbs, held-out eval flat or drops | Reward hacking | Add independent eval; stop early; raise KL |
| RM reward climbs, response length climbs | Length gaming | Length filter; SimPO; length penalty in reward |
| `objective/kl` grows unbounded; entropy crashes | KL collapse (PPO) | Raise `kl_coef` or use adaptive KL; drop LR |
| Both `rewards/chosen` and `rewards/rejected` climb together | Reference drift (DPO/QLoRA) | `precompute_ref_log_probs=True`; freeze non-LoRA |
| Both log-probs drop; margin preserved; empty outputs | Log-prob deflation (DPO) | Raise `β`; add SFT-aux term (CPO); lower LR |
| `rewards/accuracy` reaches 1.0 on train; eval loss reverses | DPO overfitting | Switch to IPO; fewer epochs; higher `β` |
| Generation collapses to one repeated string | Mode collapse | Add entropy bonus; raise KL; lower LR |
| KL to reference is nonzero at step 0 | Wrong reference model loaded | Add step-0 assertion; audit checkpoint loading |
| Verifier reward flat but eval score climbs (GRPO) | Verifier gaming | Strengthen verifier; audit failures |
| Response formatted with excessive markdown | Format bias in RM | Rebalance training set; add format-neutral pairs |

The most common failure is the top row: reward metric looks great, real quality does not. Detecting it is *the* skill of preference optimization; nothing else in this module matters as much.

## Instrumentation you should always have

The failures above are only detectable with the right logs. A minimum instrumentation set for every preference-optimization run:

- **Reward / margin curve** — the training-time metric.
- **Held-out eval score** — an independent measure. For chat: pairwise win-rate against SFT judged by a third-party model; MT-Bench / Arena Hard. For verifiable-reward: held-out task accuracy.
- **KL curve** — against the reference model, per step.
- **Response length distribution** — histogram per eval, or median + 90th percentile.
- **Response entropy** — token-level entropy averaged over each rollout.
- **Sample generations** — 5–10 canned prompts, generated once per eval step, dumped to a JSONL. Nothing beats reading actual outputs.
- **Chosen and rejected implicit rewards** (DPO family) or reward histograms (PPO / GRPO).
- **Value loss and value estimate** (PPO only).

The last one — dumping actual generations — is the highest-value instrumentation and the most often skipped. Ten prompts × ten eval steps × one generation each = 100 generations to eyeball. You will spot a collapse or a hack in the first three you read. No amount of dashboard-watching substitutes for this.

## The debug playbook

When a run is misbehaving and you cannot tell which failure it is:

1. **Read the sample generations.** Nine times out of ten the failure is obvious in the actual outputs.
2. **Compare `rewards/chosen` and `rewards/rejected` trajectories** (DPO family) or the reward histogram (PPO). If both are drifting together, it is drift or deflation.
3. **Check the held-out eval.** If it is not improving, everything you are seeing in the training reward is illusion.
4. **Compare the current policy's output on a canned prompt to the SFT model's output.** If they are wildly different but the SFT was fine, you are past the KL point where the constraint stopped working.
5. **Diff the tokenizer's chat template** across RM / policy / reference. Chat-template mismatches masquerade as almost every failure above.
6. **Verify the reference at start of training.** Step-0 KL should be zero.

Almost every preference-optimization failure is one of the four in this chapter. The diagnostic loop above catches all of them.

## Exercise pointer

The failures in this chapter are surfaced across exercises 03, 04, and 05 — the PPO run, the DPO run, and the modern-family survey. Instrument every one of them with the diagnostics above; you will trip a couple of the failures unintentionally and that is exactly the learning outcome.

## Summary

- Four preference-optimization-specific failures: **reward hacking** (policy exploits the RM's imperfections), **length bias** (policy learns "longer = better"), **KL collapse** (policy diverges from reference), and **reference-model drift** (the anchor moves).
- Reward hacking is detectable only via an *independent* eval, not the RM. Log a held-out preference eval and stop training when it plateaus.
- Length bias is measurable in the pairwise-accuracy-by-length-delta test (chapter 3) and mitigable by length filtering, length balancing, length penalties, or a length-normalized objective (SimPO).
- KL collapse hits PPO and DPO differently — PPO with reward vs. KL imbalance, DPO with log-prob deflation and too-low β. Adaptive KL controllers, entropy bonuses, and shorter response lengths are the standard tools.
- Reference drift is subtle and QLoRA-specific in the DPO world. Assert step-0 KL = 0; precompute reference log-probs; freeze all non-LoRA parameters explicitly.
- Instrument every run with reward, KL, length, entropy, held-out eval, and dumped sample generations. The last is the highest-value diagnostic and the most often skipped.
- The debug playbook: read generations, compare implicit rewards, check the held-out eval, diff against the SFT, verify the chat template, verify the reference. Nine failures out of ten reduce to one of the four in this chapter.
