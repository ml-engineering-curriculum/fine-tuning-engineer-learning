# Post-Training as a Platform Property

**Estimated read:** 30 minutes.
**You should be able to, by the end:** name the post-launch obligations that turn a shipped model from an artifact into a running system (monitoring, drift, iteration cadence, retirement); place your team on the four-rung maturity ladder from "we did one fine-tune" to "we operate a fine-tuning platform"; and hand off the vocabulary from this module to the capstone role.

## The model shipped. Now what?

Every prior chapter aimed at "getting the model to production." This chapter is what happens after. A shipped fine-tune is not an artifact — it is a running system with:

- Live traffic that can drift away from the eval distribution.
- Metrics that can regress silently between release-gate runs.
- A rollback window that must be defended.
- A registry entry that must stay accurate.
- A model card that must stay current.
- A retirement horizon that must not be missed.

Post-training as a platform property means the *system of running fine-tunes* is what you operate — not the current release. This chapter is the exit vocabulary.

## What to monitor after launch

Chapter 6 of mod-108 established the release gate. The gate is a *pre-deploy* check. The post-launch monitoring loop is what runs against *served traffic*. Two categories:

### Operational signals (from the serving stack — mod-109 ch7)

- Latency SLOs (TTFT, ITL, E2E).
- Throughput (requests/sec, tokens/sec).
- Error rate (5xx from the model, timeouts, truncations).
- Resource utilization (GPU memory, KV cache pressure, queue depth).
- Cost per million tokens (or per request, or per tenant).

These are runtime team's primary responsibility, but the fine-tuning team must be an alert recipient: a spike in error rate that correlates with a specific fine-tune release is a signal *this* team owns.

### Quality signals (post-deploy eval on live-shaped traffic)

- **Sampled quality re-eval.** Run the mod-108 fast lanes on a schedule (weekly is a starting point, nightly is better) against the served endpoint. Compare to the release-gate baseline. Non-trivial drift is a signal to investigate.
- **Production-traffic quality proxies.** Rates of the safety runtime classifier firing (mod-106 ch6). Rates of JSON-schema validation failures on tool-call outputs. Rates of the retry-with-fallback path being taken.
- **User-observable metrics.** For chat products: user thumbs-down rates, escalation rates (for support), completion rates (for agents), NPS on model-generated outputs where measured.
- **A/B / canary results (if applicable).** If a canary of a candidate release is serving 1–5% of traffic, its per-request quality proxy compared to the incumbent's, aggregated per hour or per day.

The monitoring loop's job is not to be as thorough as the release gate — it is to be *fast enough to catch regressions before they cost real money or trust*.

## The three drift kinds

Chapter 7 of mod-108 named the pitfalls of eval; here we name the post-launch drift kinds.

### Traffic drift

The distribution of prompts served in production shifts away from the distribution the release gate evaluated. Signal: a spike in latency (longer prompts than the calibration set), a jump in tool-call frequency (users doing something new with the assistant), or a shift in the topic mix (measured via a lightweight classifier or a topic-modeling pass on a daily sample).

**Response:** update the task-specific held-out set (mod-108 ex 02) with representative recent traffic, re-run the fast lanes, and refresh the release-gate baseline. If the metric drifted, that's what the next release should target.

### Model / artifact drift

The served model behaves differently over time even though the artifact itself is unchanged. Signal: silent quality drift with no traffic-distribution change. Causes are almost always infrastructure — a serving-stack upgrade changed sampling defaults, a quantization recalibration was re-run on shifted data, a routing rule changed which adapter served which tenant.

**Response:** the served-endpoint quality re-eval catches this. When it fires, the diagnostic is a diff of the deploy history (mod-109 ch7 change log) against the last known-good date.

### Vendor / dependency drift

For managed fine-tunes (mod-110), the underlying model tier can silently update in ways that affect the fine-tune. For open-weight models, a base-model deprecation announcement changes the rollback horizon.

**Response:** subscribe to vendor changelogs. Chapter 6 of mod-110 warned about the vendor deprecation clock; this chapter's monitoring adds the *reactive* half. Every managed fine-tune's registry entry pulls in the vendor's changelog for the base; a change triggers a re-run of the fast lanes.

## Iteration cadence

A shipped fine-tune is rarely the final one for its product. The iteration cadence — the rhythm of "propose → gate → release → monitor → propose again" — is what turns fine-tuning from a project into a practice.

Common cadences:

- **Quarterly.** Slow, safe, low-throughput products. Most internal tooling. Enough time for the safety review, the data collection, and the release-review meeting to be handled thoughtfully.
- **Monthly.** Steady-state product with an active roadmap. Fine-tuning is one of many workstreams; the model is not the bottleneck.
- **Weekly or on-demand.** High-velocity products where the fine-tune is on the critical path (LLM-native products, agent products, fast-moving verticals). Requires the platform to be more mature (all of chapter 5's automation live).

The cadence is a *platform* property, not a per-project one. Attempting weekly cadence on a platform that requires a two-week release-review meeting will collapse; attempting quarterly cadence on a product that expects weekly changes will lose to competitors.

## Cost and capacity management

At platform scale, cost management moves from "did this project stay under budget" to "which projects are earning their fine-tune investment." A running system needs:

- **Cost per artifact per month.** GPU-hours for training + hosting rent for managed + serving cost for self-hosted, per registry entry. Attributable to the sponsoring product.
- **Cost per delta.** For each release, dollars spent (training + eval) divided by the primary metric's improvement. A project consistently spending $50k per 1 pp of metric movement is a candidate for the ADR revisit (chapter 4).
- **Capacity forecast.** GPU-hours for training runs in the next quarter, GPU-hours for evals, judge API tokens for gates. If capacity is tight, the platform decides which projects run when.
- **Retirement backlog.** Old models still in the registry consume storage and (for managed) hosting rent. A weekly retirement pass (chapter 5) keeps the backlog under control.

The cost dashboard is a first-class registry-derived artifact. Each registry entry contributes rows; the dashboard aggregates.

## The maturity ladder

Post-training practice has recognizable rungs. Not every organization needs the top rung; many are fine at rung 2. Naming the rungs helps a team place themselves and decide what to invest in.

### Rung 1 — "We did one fine-tune"

- Single fine-tune, single product.
- No proposal, no ADR — just a Colab notebook and a story.
- Ad-hoc eval, no gate.
- No registry; the model is somewhere on someone's laptop or a random S3 path.
- No rollback plan.

**When rung 1 is fine:** research demonstrations, personal projects, a one-off internal tool with no ongoing dependency.

**Signal to move up:** the fine-tune is now getting traffic, someone asks "what data was this trained on?", or a second product wants one too.

### Rung 2 — "We have a repeatable fine-tuning workflow"

- Proposal (chapter 1) written for each project.
- Base-model rubric (chapter 2) run per project.
- Dataset-budget document (chapter 3) written.
- Release-gate report (mod-108 ch6) produced per release.
- Model card (chapter 6) authored per release.
- Rollback runbook (chapter 5) exists per release.
- The workflow is *documented*, not necessarily *automated*.

**When rung 2 is fine:** most internal fine-tuning teams; one or two products; releases every 1–3 months.

**Signal to move up:** you're running the same manual steps for the fifth time and the cost of manual is exceeding the cost of automation.

### Rung 3 — "We operate a fine-tuning pipeline"

- Registry (chapter 5) is live and required.
- Release-gate is a CI job, not a manual run.
- Promotion transitions are PR-driven.
- Rollback runbook is verified quarterly by dry-runs.
- Post-launch monitoring runs on a schedule.
- ADR chain is version-controlled and searchable.

**When rung 3 is fine:** small platform team supporting 3–10 products; monthly or weekly cadences achievable.

**Signal to move up:** you're supporting more than 10 products, or you have both self-hosted and managed fine-tunes, or governance / regulatory obligations require formal audit trails that manual workflows cannot support.

### Rung 4 — "We operate a fine-tuning platform"

- Registry is a system, not a Git repo.
- Multi-tenant routing (chapter 5) at scale.
- Cross-vendor releases (self-hosted + managed) unified in the registry.
- Cost / capacity forecasting is a first-class dashboard.
- Governance is a workflow, not a meeting — with the meeting itself as one step.
- The team producing fine-tunes is distinct from the team operating the platform.

**When rung 4 is fine:** large orgs with 20+ fine-tunes in flight; regulated environments; multi-region deployments.

**Signal that rung 4 is over-engineered:** if the platform's overhead exceeds the value it provides for the number of models in flight, drop back to rung 3. Platforms are appropriate to scale; over-platforming a small operation is a form of waste.

## The retirement conversation

Every model retires eventually. Retirement is:

- **Operational:** the artifact stops serving; managed fine-tunes are deleted; storage is moved to cold or expires per data-retention policy.
- **Registry:** the entry moves to `status: retired` (chapter 5), stays in the history for audit.
- **Product:** the sponsor and downstream product teams are notified in advance (weeks to months, depending on cadence).
- **Data:** the training data may or may not be retained depending on the data-retention policy in the dataset card (mod-102 ch7).
- **Governance:** any commitments made in the model card (e.g., "we will monitor for X for 12 months") are honored to their horizon or explicitly ended.

Retirement is often skipped ("we'll deal with it later"). Skipping it is how registries grow to hundreds of entries, most of which nobody remembers, most of which cost a small ongoing sum, and one of which is a compliance issue nobody's noticed.

## Post-launch incidents and how they feed the next release

An incident on a production fine-tune is a signal generator. Every incident should produce:

- **The rollback** (chapter 5).
- **An incident report** (24 hours; standard postmortem template).
- **A gate delta.** What could the release gate have caught but didn't? Add the missed case to the task-specific held-out set (mod-108 ex 02).
- **A card delta.** Update §6 (Known Limitations) of the model card even for the rolled-back release. The retired version's limitations are part of the history.
- **An ADR delta.** If the incident revises the fine-tune-vs-alternative decision (chapter 4), or the base selection (chapter 2), or the dataset budget (chapter 3), update the ADR chain with a new version.

The incident is the platform's most valuable teacher. If nothing changes after an incident, the incident is going to happen again.

## What you hand off to the capstone role

The Fine-Tuning Engineer track ends here. The natural next step — the *capstone role* — is running an actual fine-tuning platform end-to-end for a real product surface. Everything you produced in this module is what you hand off:

- The **proposal template** (chapter 1) — used by every new project.
- The **base-selection rubric and scorecard** (chapter 2) — reused across the product line.
- The **dataset-budget document** (chapter 3) — refined into a per-product template.
- The **ADR templates and chain** (chapter 4) — searchable so decisions do not get relitigated.
- The **release-gate manifest, registry schema, and rollback runbook** (chapter 5) — the operational core.
- The **model card and release-review process** (chapter 6) — the governance interface.
- The **monitoring, drift, iteration cadence, and maturity vocabulary** (this chapter) — the running-system layer.

Each of these is a *reusable* artifact. The point of the module is not that you can produce them once; the point is that you can produce them, together, as the operating documents of a fine-tuning practice that other engineers can join, other teams can consume, and other reviewers can trust.

## Where the track ends and the practice begins

Every module in this track — mod-101 through mod-110 — was, at its core, a *how-to* module. How does an autoregressive decoder work; how do you engineer an SFT dataset; how do you run SFT, LoRA, DPO, PPO, safety tuning, distributed training, evaluation, quantization, managed fine-tuning. Mod-111 is a *how-to-run-the-how-tos-at-scale* module. It composes the outputs of the others into a platform.

The practice starts when you stop doing this by memory and start doing it by process. The proposal is a template, not an exercise. The rubric is a shared spreadsheet, not a per-project document. The registry is a required system, not a nice-to-have. The gate is a CI job, not a manual command. The card is a required review artifact, not a post-hoc write-up. The retirement is a weekly cron, not an interrupt.

When those seven things are in place, you have a fine-tuning platform. When they are not, you have a fine-tuning project. Both have their moments; a career in fine-tuning eventually needs the former.

## Summary

- A shipped fine-tune is a running system, not an artifact. Post-launch obligations include monitoring (operational and quality), drift response (traffic, model / artifact, vendor / dependency), iteration cadence, cost / capacity management, and retirement.
- Drift kinds are three: traffic drift (users doing new things), artifact drift (silent quality change from infra), and vendor / dependency drift (base or vendor changed underneath). Each has a signal and a response.
- Iteration cadence is a platform property, not a per-project one. Common cadences: quarterly (slow, safe), monthly (steady state), weekly / on-demand (high-velocity). Match cadence to platform maturity.
- Cost management moves from "did this project stay under budget" to "which projects are earning their investment." Cost-per-artifact and cost-per-delta are first-class registry-derived numbers.
- The maturity ladder has four rungs: (1) we did one fine-tune, (2) we have a repeatable workflow, (3) we operate a pipeline, (4) we operate a platform. Not every team needs rung 4; over-platforming is a form of waste.
- Retirement is where registries silently fail. A weekly retirement pass keeps the registry honest and the hosting rent low.
- Incidents feed the next release: rollback, incident report, gate delta, card delta, ADR delta. An incident that changes nothing is going to happen again.
- The Fine-Tuning Engineer track ends here. The capstone role — operating a fine-tuning platform end-to-end — is what this module trains you to enter. The seven artifacts (proposal, rubric, budget, ADR, gate + registry + rollback, model card + review, monitoring / cadence / retirement) are what you carry into that role.
