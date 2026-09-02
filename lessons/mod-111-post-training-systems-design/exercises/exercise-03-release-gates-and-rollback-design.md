# exercise-03: Release Gates and Rollback Design

**Estimated effort:** 4 hours
**Companion chapters:** [`05-release-gates-registry-rollback.md`](../05-release-gates-registry-rollback.md), [`06-model-cards-and-release-review.md`](../06-model-cards-and-release-review.md), [`07-post-training-as-platform-property.md`](../07-post-training-as-platform-property.md)

## Objective

Extend mod-108's `release_gate.v3.yaml` into a **full release-and-registry manifest** for a multi-adapter product, design the **adapter / model registry schema** and the promotion rules, and author the **rollback runbook** that a stranger on call at 3 AM can execute safely. Land the artifact bundle that a real platform team would keep versioned in Git.

The point is platform machinery, not code. **You are not deploying anything.** A reviewer six months from now — or an on-call engineer at 3 AM — should be able to open your artifacts and either follow them successfully or challenge them on a specific step, not on interpretation.

## Prerequisites

- Chapter 5 read in full; chapter 6 read for the release-review-meeting output; chapter 7 skimmed for the post-launch context that the registry must survive.
- Mod-108 exercise 05 completed *or* an equivalent `release_gate.v3.yaml` you can extend. If you did not do mod-108, this exercise still runs — you will author the `release_gate.v3.yaml` from chapter 6's template as part of Part A, but expect that step to take an extra hour.
- Familiarity with YAML, Markdown, and Git-based review workflows. No live GitHub / GitLab account required — this exercise is authored locally.
- No GPU required. No API budget required.

## Requirements

### Part A — Pick and specify the product

In `product.md`, define exactly **one** product for which this platform machinery will operate. The registry and runbook only make sense with a real subject. Pin:

- **Product name and version at hand.** E.g., `acme-support` and the release you are gating (e.g., `v3.1`).
- **Adapter chain.** At least **two named adapters** on top of one base — chapter 5's multi-adapter setup. Common shapes:
  - `sft-<task>` + `dpo-<task>` + `safety-<version>` on Llama or Qwen.
  - `sft-<task>` + `safety-<version>` (Stack A + safety adapter) — the minimum multi-adapter shape.
  - Two task-specific adapters routed by tenant / feature flag (chapter 5's "multi-adapter routing at platform scale" section).
- **Base model.** Pinned as in exercise 01 (`<org>/<name>` @ revision `<sha>`, tokenizer + chat-template hashes).
- **Serving stack and target.** Chapter 5's `serving` sub-tree — mod-109 vocabulary: vLLM 0.6.x, TGI, SGLang, or an equivalent. Include the deployed-endpoints list (staging + production).
- **Prior release.** The registry entry that this release supersedes and rolls back to. Give it a version (`v3.0`); pin its adapter hashes; this is the `rollback_target` in the registry schema.
- **Tenancy / residency constraints.** Any hard overrides (single-tenant, per-tenant routing, EU-only) that will shape the registry entry's `compliance` block and the routing rule.

One page. Everything downstream anchors on it.

### Part B — Extend `release_gate.v3.yaml`

Author `release_gate.v3.yaml`. It **extends** the mod-108 chapter 6 manifest, not replaces it. Required extensions on top of mod-108's four-lane manifest:

1. **Multi-adapter-aware lanes.** Every lane runs against the *composed serving artifact* — base + adapter chain + quantization — not against the SFT checkpoint in isolation. Specify how the composition is materialized for the gate run (adapters merged on top, adapters loaded live via `peft.load_adapter`, or the vLLM multi-LoRA path from mod-104 ch3). Pin the exact composition mode.
2. **Per-tenant lanes if the product routes by tenant.** For the multi-tenant routing case from Part A, add a `per_tenant_lanes` block listing which tenants get gated separately. Chapter 5's "multi-adapter routing at platform scale" — a per-tenant regression on an experimental adapter should not block the promotion of the shared production chain.
3. **Composition-hash pins.** In addition to mod-108's eight reproducibility pins, add: `composition_hash` (deterministic hash of base + adapter chain + quantization), `serving_stack_version`, `runbook_version`, and the `verdict_document_path` where the release verdict will be stored.
4. **Cross-card AND rule.** Chapter 5's "three cards are AND-ed" rule. Add a `verdict_composition` block that names:
   - The mod-108 release-gate report path.
   - The mod-106 safety report card path.
   - The mod-102 dataset card path.
   - The rule: "verdict is `APPROVE` iff all three pass their respective thresholds; a pass on one does not compensate for a fail on another."
5. **Safety-lane concreteness.** If the safety lane in your mod-108 manifest was `deferred_to: mod-106`, resolve it here. Point at a specific red-team eval set with a hash; pin the mod-106 report card's threshold. Chapter 5: a gate without safety is a gate that failed to name safety as a requirement.
6. **Gate versioning.** Bump the manifest version to `v3.1` (or higher if you have already extended it). Every threshold change is a manifest bump; document what changed vs. mod-108's `v3` in a comment or a companion `manifest_diff.md`.

Every threshold and pin has a one-line comment or a companion `manifest_rationale.md` entry justifying it — chapter 5's "no magic numbers." The manifest is a contract.

### Part C — Design the registry schema

Author `registry_schema.md`. Two deliverables:

1. **The schema itself.** Chapter 5's YAML skeleton, extended for your Part A product. Cover: `id`, `product`, `status`, `owner`, `artifact` (kind + base + adapters + quantization + serving), `release_verdict`, `lineage`, `compliance`, `timeline`. If your product is multi-adapter with a router (chapter 5), include the `adapter_router` block.
2. **The schema properties in prose.** For each top-level block, one paragraph on:
   - Who writes it (which team, which artifact).
   - Who reads it (deploy pipeline, rollback script, audit report, cost dashboard, mod-108 gate runner).
   - What its invariants are (chapter 5's "every hash real, every URI resolves"; "lineage is bidirectional"; "managed and self-hosted use the same schema").
3. **Two filled example entries.** Author `registry/<product>/v3.0.yaml` (the prior release, `status: production` at the point in time when v3.1 is a candidate) and `registry/<product>/v3.1.yaml` (the new candidate, `status: candidate`). Every field populated; every hash a plausible placeholder (`5c3f9e2b...`); every URI structurally realistic.

Chapter 5's "why not just a spreadsheet" section — the schema is what makes the platform operable; the entries are what make the schema real.

### Part D — Author the promotion rules

Author `promotion.md`. For each of chapter 5's five transitions, write:

- **candidate → staged**
- **staged → production**
- **production → rolled-back**
- **production → retired**
- **rolled-back → retired** (or whatever your team's re-promotion / cleanup path is)

For each transition:

1. **Gate.** The precondition the transition requires (e.g., "all mod-108 hard gates pass"; "captain sign + governance sign + verdict = APPROVE").
2. **Rule.** The mechanical action (opens a PR, updates a field, runs a runbook).
3. **Reviewers.** Who signs. Chapter 5's minimum: peer FT engineer + safety for candidate → staged; captain + governance for staged → production.
4. **Additional artifacts.** What must be attached to the PR (verdict draft, verdict signed, incident report, retirement rationale).
5. **Effect.** What the transition causes in production (staging deploy, production traffic, previous-artifact status shift, cold-storage move).

Chapter 5's promotion rule table is the summary shape; your `promotion.md` is the expanded version with rationale per transition. Include the two-column consolidated table too, for at-a-glance reference.

Include one **worked promotion** — narrate v3.1 moving from candidate to production for your Part A product, referencing your actual registry entries and manifest. Chapter 5's process becomes concrete when it has a subject.

### Part E — Author the rollback runbook

Author `rollback_runbook.md` following chapter 5's runbook template. Sections:

1. **When to trigger this runbook.** Specific SLO / safety / data-integrity thresholds that make rollback the correct response. Chapter 5's four defaults; refine to your product.
2. **Before you roll back.** Pre-condition checks. Chapter 5's "confirm the incident is not caused by an upstream dependency"; "confirm the rollback target is healthy." Real health-check URLs, real registry commands.
3. **Perform the rollback.** Exact commands. Either kubectl-based (with a real deployment spec + set-image command) or a control-plane CLI (`acmectl deploy rollback --to v3.0`, chapter 5). If your team does not have a control plane, pick one paradigm and stick with it; do not present two options as "up to you."
4. **Verify.** How on-call confirms traffic is served by the rollback target — canary prompt comparison, response fingerprint, per-endpoint health check.
5. **Update the registry.** Which entries change status; which timestamps get recorded (`rolled_back_at` on v3.1, `production_at` re-recorded on v3.0 or a `re_promoted_at` if your schema supports it).
6. **After the rollback.** #incidents post, page the owner, do-not-delete-the-artifact rule (chapter 5: evidence), 24-hour incident report requirement.
7. **Rollback window.** The horizon during which this runbook applies. Chapter 5's default 30 days; state your product's actual number and defend it.
8. **When rollback WON'T help.** Chapter 5's failure modes: upstream dependency degradation, data-integrity issue present in both artifacts, safety incident rooted in base model. Escalation path for each.
9. **Dry-run cadence.** Chapter 5: verify quarterly. Author the dry-run command (or dry-run runbook) and log the last successful dry-run date. The runbook without a dry-run is a runbook nobody has confirmed works.

Every command block is copy-paste-able. Every variable (registry ID, endpoint URL) is either a real value or a placeholder marked `<...>` with a comment. Chapter 5's discipline: the on-call engineer at 3 AM does not read the surrounding prose; they scan the commands.

### Part F — Author the release verdict document

Author `verdict.md` for v3.1's promotion, following chapter 5's release-verdict template. The verdict document is the composed decision that gates the staged → production transition and lands in the registry entry.

Fill the seven sections:

1. **Identity.** Product, artifact ID, base + adapter chain + quantization + serving + model-card version.
2. **Release-gate report.** Reference the mod-108 chapter 6 report. Summarize hard-gate outcomes (all pass / one fail), soft-gate outcomes, contamination scan verdict, statistical treatment.
3. **Safety report card.** Reference the mod-106 chapter 5 report. Refusal rate per axis, harmful-completion, capability-gain, explicitly-not-tested list.
4. **Dataset card summary.** Reference the mod-102 chapter 7 dataset card. Sources with license verdicts, synthetic-vs-human breakdown, contamination scans, PII scan verdict.
5. **Provenance chain.** Chapter 5's SFT run ID + git SHA + config hash chain, all the way to the rollback target.
6. **Decision.** `APPROVE` / `APPROVE_WITH_DISCLOSURE` / `BLOCK`. If `APPROVE_WITH_DISCLOSURE`, verbatim disclosure text + compensating controls (chapter 5: "runtime classifier bumped to strict for 2 weeks").
7. **Sign-off.** Named signers with dates (yourself, peer, safety, governance, captain — placeholders as `<pending>` are fine; the point is the section exists).

If any of the composed cards are not real for your exercise (e.g., you have no live mod-106 output), synthesize a plausible summary and mark it `<!-- synthesized for exercise; real card is required for a real release -->`. Chapter 5's discipline: the verdict is the artifact even when the inputs are placeholders.

### Part G — Directory layout

Final tree:

```
outputs/exercise-03/
├── product.md
├── release_gate.v3.yaml
├── manifest_rationale.md   # or inline comments in the YAML
├── registry_schema.md
├── registry/
│   └── <product>/
│       ├── v3.0.yaml
│       └── v3.1.yaml
├── promotion.md
├── rollback_runbook.md
├── verdict.md
└── sources.md              # every URL cited, with retrieval date
```

`sources.md` records every URL for a cited standard (chapter 5 references NIST AI RMF, ISO/IEC 42001, EU AI Act GPAI provisions; if you cite any external doc, record it here) with retrieval date.

## Starter guidance

- **Do not skip Part A.** A generic product produces a generic runbook. Pick a real-shaped product with a real adapter chain — chapter 5's `acme-support` example is a fine model; adapt it to your track's mods 103 / 104 / 105 / 106 outputs.
- **The manifest is the contract; treat it as first-class.** Chapter 5's "the registry is the platform's system of record" applies to the manifest too. Do not scatter the thresholds across your runbook and the promotion doc; the manifest holds them and the other docs cite them.
- **Write the runbook out loud.** Read the rollback runbook aloud, top to bottom, pretending to be an on-call engineer who has never touched this product. Every place you say "you would… uh…" is a place the runbook fails. Chapter 5's discipline: strangers at 3 AM.
- **Do the dry-run cadence in Part E.** A runbook without a dry-run schedule is a runbook that will fail the first time it runs. Chapter 5's quarterly verification; log a plausible last-dry-run date and the command that produced it.
- **Managed and self-hosted use the same schema.** Chapter 5's payoff of mod-110. If your Part A product is managed, the `artifact.kind: managed` block from chapter 5's example is the shape. If your product mixes self-hosted and managed (chapter 5: "cross-vendor releases"), author two registry entries with a shared `product` field.
- **Hash placeholders are fine but structured.** `5c3f9e2b...` is acceptable; `<hash>` is not. The point is that a reader can pattern-match a hash and see it flow between the manifest, the registry, the verdict, and the runbook.
- **The verdict is what gets signed, not the model-card.** Chapter 6 owns the model card; chapter 5's verdict is what makes the promotion legal. Do not conflate the two documents in this exercise.
- **Do NOT put real production endpoint URLs, credentials, or customer IDs in any artifact.** Chapter 5's registry lives in Git; your exercise output does too. Placeholders like `https://api.<internal>/support/v3.1` are the pattern.

## Acceptance criteria

You have completed the exercise when:

- `product.md` pins the six product / release fields (name + version, adapter chain, base, serving stack + endpoints, prior release, tenancy / residency).
- `release_gate.v3.yaml` extends the mod-108 manifest with composition-hash pins, cross-card AND rule, resolved safety lane, and per-tenant lanes if applicable; every threshold has a rationale.
- `registry_schema.md` describes the schema in prose and produces two filled example entries (v3.0 and v3.1) with plausible hashes and lineage.
- `promotion.md` covers all five transitions with gate / rule / reviewers / artifacts / effect, plus a worked v3.1 candidate-to-production narration.
- `rollback_runbook.md` follows chapter 5's structure end-to-end, has copy-paste-able commands, names the rollback window, includes the "when rollback won't help" section, and records a dry-run cadence.
- `verdict.md` follows chapter 5's seven-section template and names a specific decision.
- `sources.md` records every citation.

## Stretch goals

- **Add a multi-tenant routing rule.** Chapter 5's `adapter_router` block. Author a registry entry that routes an internal-testers tenant to an experimental adapter and everyone else to the shared chain. Extend `promotion.md` to argue that the experimental-adapter tenant's regression should not block promotion of the shared chain.
- **Add a cross-vendor release.** Chapter 5's "cross-vendor releases" section. Author a second registry entry (`v3.1-managed.yaml`) that ships the same product's fine-tune through a mod-110 managed backend for a specific tenant, and extend the rollback runbook with a per-backend section.
- **Draft the release-review meeting agenda.** Chapter 6's 60-minute agenda applied to your Part A product. Author `release_review_agenda.md`; run through it mentally with the manifest, verdict, and runbook open; note where the presenter would field a hostile question.
- **Model an incident and the rollback trace.** Author `incident-<date>.md` — a plausible incident 20 days into v3.1's production window that triggers the runbook. Trace the runbook end-to-end. Where does the runbook stumble? Amend it. Chapter 5's discipline: the runbook is refined by the incidents it survives.
- **Wire the promotion rules to CI.** Add a `.github/workflows/registry.yaml` (or equivalent) that runs the manifest verification job (hash checks, URI resolution) on every PR that touches `registry/`. This is chapter 5's automated verification.
- **Add an audit script.** Author `audit.py` that walks the registry (Git-log-based or filesystem-walk-based) and produces a report: "what artifacts were live during window [T0, T1], which shipped without a signed verdict, which had an unresolved license blocker at ship time." Chapter 5's "the commit history is the audit log" — this is what makes the audit log useful.
- **Draft the schema for the retirement job.** Chapter 5's retirement pass. Author `retirement.md` describing the criteria (rollback window elapsed AND no outstanding rollback demand AND no downstream references) and a runbook for the platform team's weekly retirement pass. Managed fine-tunes get deleted via `client.models.delete()` — mod-110 ch5; author that call site.
