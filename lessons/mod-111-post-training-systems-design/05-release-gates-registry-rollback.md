# Release Gates, Candidate Selection, Adapter Registry, and Rollback

**Estimated read:** 45 minutes.
**You should be able to, by the end:** compose mod-108's `release_gate.v3.yaml` with mod-106's safety report card and mod-102's dataset card into a per-release verdict document; design an adapter / model registry schema that indexes both self-hosted checkpoints and managed `ft:...` IDs with lineage back to source data; specify the release-candidate promotion rules; and author a rollback runbook that a stranger on call at 3 AM can execute safely.

## The platform layer

Mod-108 gave you a release gate that decides "is this checkpoint good enough to deploy?" Mod-104 gave you named adapters and multi-adapter serving. Mod-106 gave you the safety report card. Mod-109 gave you the serving runbook. Mod-110 gave you the managed-fine-tune model ID. This chapter is where those individual artifacts stop being per-project and become platform primitives — a *release contract*, a *registry*, a *promotion rule*, a *rollback runbook*.

The through-line: **artifacts you already produce, indexed and orchestrated so a stranger can operate them.** The platform does not invent new artifacts; it makes the existing ones queryable, promotable, and reversible.

## The full release verdict document

The release-gate report from mod-108 chapter 6 is one input to the release verdict. It is not the whole verdict. The whole verdict pulls in three more artifacts and produces a single deploy/block/degraded decision that a captain signs.

Structure:

```markdown
# Release verdict — <product> <artifact-id>
**Verdict:** APPROVE | APPROVE_WITH_DISCLOSURE | BLOCK
**Signed by:** <release captain>
**Date:** YYYY-MM-DDTHH:MM:SSZ

## 1. Identity
- Product: <name>
- Artifact ID (registry key): <e.g., acme-support-v3.1>
- Artifact hash (self-hosted) or vendor model ID (managed)
- Base model + revision
- Adapter chain (LoRA layers, safety adapter): each with hash
- Quantization method + hash (mod-109)
- Serving runbook version (mod-109 ch7)
- Model card version (chapter 6 of this module)

## 2. Release-gate report
Verbatim link + summary of the mod-108 ch6 report:
- Hard-gate outcomes (all pass / one fail / etc.)
- Soft-gate outcomes
- Contamination scan verdict
- Statistical treatment used (CI method, paired test)

## 3. Safety report card
Verbatim link + summary of the mod-106 ch5 report:
- Refusal rate on harm taxonomy (per axis)
- Harmful completion rate
- Capability-gain vs. base (dual-use)
- Explicitly-not-tested attacks list

## 4. Dataset card summary
Verbatim link + summary of mod-102 ch7 dataset card:
- Sources with license verdicts
- Synthetic vs. human breakdown
- Contamination scans against gate eval sets
- PII scan verdict

## 5. Provenance chain
- SFT training run ID (mod-103) + git SHA of training code + config hash
- DPO / preference-optim run ID (mod-105) + hash
- Safety adapter ID + hash (mod-106)
- Quantization job ID (mod-109) + hash
- Serving artifact hash (or managed fine-tune ID)
- Rollback target: <prior artifact ID>

## 6. Decision
- Verdict + rationale
- Disclosures required for APPROVE_WITH_DISCLOSURE (verbatim, to be pasted
  in the change log)
- Compensating controls if verdict is APPROVE_WITH_DISCLOSURE (e.g., "runtime
  classifier bumped to strict for 2 weeks")

## 7. Sign-off
- Fine-tuning owner: <name> + date
- Safety reviewer: <name> + date
- Governance reviewer: <name> + date
- Release captain: <name> + date + verdict
```

The verdict document is the input to the registry write. Nothing gets promoted to `production` in the registry without it.

## Composing the three cards

Three artifact families feed the verdict, each authored by a different owner:

- **Release-gate report** (mod-108 ch6) — owned by the fine-tuning engineer. Numbers on academic benchmarks, task-specific held-out, judged pairwise, and safety. Structured per `release_gate.v3.yaml`.
- **Safety report card** (mod-106 ch5) — owned by safety. Four axes: refusal on harm taxonomy, harmful-completion, capability-gain (dual-use), explicitly-not-tested.
- **Dataset card** (mod-102 ch7) — owned by the data engineer. Sources, licenses, synthesis disclosure, PII, contamination, mix ratios.

The verdict composes them. Each card has its own thresholds and its own owner; the verdict adjudicates the intersection.

The three-card composition has one non-obvious rule: **a "pass" on one card does not compensate for a "fail" on another.** A model can have perfect safety numbers but a dataset card with an unresolved license blocker — that model does not ship. A model can have a permissive dataset and a good release-gate report but a safety-report regression on the harm taxonomy — that model does not ship. The three are AND-ed, not averaged.

## The adapter / model registry

The registry is the platform's system of record for every fine-tune that exists — trained, evaluated, in-review, in-production, rolled-back, or retired. Everything in this chapter and the next composes on top of the registry. Getting its schema right is the highest-leverage step in the whole module.

### Registry schema (minimum viable)

Each entry is a YAML file (or a row in a small relational database if you outgrow files). The file is the source of truth; anything else — dashboards, CI hooks, deploy CRDs — reads it.

```yaml
# registry/acme-support/v3.1.yaml
id: acme-support-v3.1
product: acme-support
status: production            # candidate | staged | production | rolled-back | retired
created: 2026-08-14T15:22:11Z
owner: L. Ortega

# Identity
artifact:
  kind: self-hosted            # self-hosted | managed
  base:
    model: meta-llama/Llama-3.1-8B-Instruct
    revision: e7c5b8d9...      # HF revision sha
    tokenizer_hash: 8f2a1c3e...
    chat_template_hash: 3a7b9d1e...
  adapters:
    - name: sft-billing
      hash: 5c3f9e2b...
      base: <link back to same base>
      training_run_id: acme-sft-3f19
      config_hash: 9d2b7c1a...
    - name: dpo-support
      hash: 2b8a1e9f...
      training_run_id: acme-dpo-b2e0
      config_hash: 4e2f1c8b...
    - name: safety-v3
      hash: 8a1c7b4e...
      training_run_id: acme-safety-9c73
      config_hash: 2f8a3b1e...
  quantization:
    method: awq
    hash: 1e9b4a2c...
    calibration_dataset: s3://acme-cal/v2.jsonl
  serving:
    stack: vllm
    version: 0.6.2
    runbook: runbooks/acme-support-v3.1.yaml    # mod-109 ch7 template
    artifact_uri: s3://acme-models/support/v3.1/
    deployed_endpoints:
      - production: https://api.acme.internal/support/v3.1
      - staging: https://staging.acme.internal/support/v3.1

# For managed:
# artifact:
#   kind: managed
#   vendor: openai
#   model_id: ft:gpt-4.1-mini-2025-07-14:acme::9zTk
#   fine_tune_job_id: ftjob-xyz
#   created_at: 2026-08-10T09:14:00Z

# Verdict
release_verdict:
  document: verdicts/acme-support-v3.1.md
  verdict: APPROVE
  captain: J. Park
  signed: 2026-08-14T14:55:03Z
  gate_report: results/acme-support-v3.1/gate-v3.json
  safety_report: safety/acme-support-v3.1.md
  dataset_card: datasets/acme-support-v3.1.md
  model_card: cards/acme-support-v3.1.md

# Lineage
lineage:
  parent: acme-support-v3.0    # previous release
  rollback_target: acme-support-v3.0
  supersedes: []
  superseded_by: null

# Metadata for compliance
compliance:
  data_residency: us-east-1
  redistribution_intent: internal-only
  license_summary: License trace verified in dataset card, section 3.
  pii_scan: passed
  contamination_scan: passed
  known_limitations: See model card §7.

# Timeline
timeline:
  candidate_at: 2026-08-12T10:00:00Z
  staged_at: 2026-08-13T14:30:00Z
  production_at: 2026-08-14T15:22:11Z
  rolled_back_at: null
  retired_at: null
```

### Schema properties that matter

- **Every hash is real, every URI resolves.** A registry entry whose `hash` fields do not match the artifact on disk is worse than no registry — it lies. Run a `verify` job on registry write that re-hashes and compares.
- **Every adapter, dataset, and job is a first-class object with its own ID.** Not "SFT run for v3.1" — `acme-sft-3f19`, a stable ID that any other registry entry can point to. The same SFT run may be composed with different DPO runs into two different production artifacts.
- **Lineage is bidirectional.** `parent` points backwards; `supersedes` and `superseded_by` maintain the DAG of releases. Rollback consults `rollback_target`; audits walk the whole chain.
- **The registry is versioned in Git.** Every change to a registry entry is a commit with a reviewer. `status: production` is a merge to `main`; `status: rolled-back` is a merge with the incident report attached.
- **Managed and self-hosted use the same schema.** `kind: self-hosted` vs. `kind: managed` differs in the `artifact` subtree only. This is the platform-level payoff of the mod-110 pattern.

### Why not just a spreadsheet

A spreadsheet gets you started; a spreadsheet is not a registry. Two properties a spreadsheet cannot provide:

- **Reviewability.** Every registry change should be a PR that a peer signs. Spreadsheets get edited in place without a diff.
- **Machine consumability.** The registry is what the deploy pipeline reads to route traffic, what the rollback script reads to find the previous artifact, what the audit report reads to enumerate what's live. YAML files in Git are diff-able, review-able, and machine-readable. Spreadsheets are one of those three at best.

Some teams use a proper artifact registry (MLflow, Weights & Biases model registry, a custom service). Those work — the schema above is the *content*, not the *transport*. Whatever transport, the schema is what makes the platform operable.

## Release-candidate promotion rules

Promotion is the process by which an artifact moves through states in the registry: **candidate → staged → production**. Each transition has a rule.

### candidate → staged

- **Gate:** all mod-108 hard gates pass (mod-108 ch6 vocabulary).
- **Rule:** the fine-tuning engineer opens a PR that adds the registry entry with `status: candidate` and attaches the release-verdict document draft.
- **Reviewers:** peer fine-tuning engineer, safety reviewer.
- **Merge criterion:** two reviews and CI green (verify hashes, verify links).
- **Effect:** the artifact is deployed to the staging endpoint. Runtime team can smoke-test.

### staged → production

- **Gate:** all mod-108 hard gates AND all soft gates pass, OR release captain approves the specific soft-gate failures with disclosure.
- **Rule:** the release captain signs the verdict document. The PR updates `status: staged` → `status: production` and records `production_at`.
- **Reviewers:** governance reviewer, release captain (captain sign is required).
- **Merge criterion:** captain sign + governance sign + verdict-document status is `APPROVE` or `APPROVE_WITH_DISCLOSURE`.
- **Effect:** the artifact receives production traffic. The prior production entry moves to `status: production-previous` (still ready for instant rollback) or `status: retired` (past the rollback window).

### production → rolled-back

- **Gate:** an on-call incident occurs and the on-call decides to roll back.
- **Rule:** on-call runs the rollback runbook (below). The registry entry's `status` moves to `rolled-back`, the prior production entry's `status` moves back to `production`, and `rolled_back_at` is recorded. An incident report is required within 24 hours.
- **Effect:** production traffic is served by the prior artifact within the rollback window.

### production → retired

- **Gate:** an artifact has been superseded for at least the rollback window (default 30 days) and no rollback demands are outstanding.
- **Rule:** platform team runs a periodic retirement pass (weekly). Retired artifacts remain in the registry (immutable history) but their `artifact_uri` may be moved to cold storage. Managed fine-tunes are `client.models.delete()`-d to stop hosting rent (mod-110 ch5).
- **Effect:** the artifact is no longer a rollback target; historical record is preserved.

## The promotion rule table

Consolidated for a single glance:

| Transition | Gate | Reviewer(s) | Additional artifact |
|---|---|---|---|
| new → candidate | mod-108 fast lanes pass | fine-tuning owner | training run summary |
| candidate → staged | all hard gates pass | peer FT eng + safety | verdict draft |
| staged → production | all hard + soft (or disclosure) | captain + governance | verdict signed |
| production → rolled-back | incident | on-call | incident report within 24h |
| production → retired | rollback window elapsed | platform team | none (automatic) |

Every transition is a registry commit. The commit history is the audit log.

## The rollback runbook

Rollback is the operation you hope never runs and prepare for anyway. A rollback that requires expert intervention at 3 AM is a rollback that fails. Write the runbook to be executable by an on-call engineer who has never touched the fine-tune before.

The runbook is a Markdown file, versioned alongside the release verdict:

```markdown
# Rollback runbook — <product> <artifact-id>

## When to trigger this runbook
Any of:
- Production p95 quality-SLO breach for 10+ minutes.
- Safety incident (harmful completion rate spikes above <threshold>).
- Data-integrity incident (PII leak reported).
- Manual escalation from Support Product owner.

## Before you roll back
1. Confirm the incident is not caused by an upstream dependency (retrieval
   service, tool API, chat gateway). Check <dashboard-URL>. If any upstream
   is degraded, page that team FIRST — a rollback won't help.
2. Confirm the rollback target is healthy. From registry, resolve
   `rollback_target`:
     $ acmectl registry get acme-support-v3.1 --field rollback_target
     -> acme-support-v3.0
   Check the target's staging endpoint returns a smoke-test 200:
     $ curl -sf https://staging.acme.internal/support/v3.0/health

## Perform the rollback
3. Flip the router. Two supported paths:

   a. **kubectl-based:** update the deployment spec to point at the target
      artifact URI.
        $ kubectl -n support set image deploy/support-inference \
            model=<REGISTRY>/acme-support/v3.0
      Wait for rollout to complete (~2 min):
        $ kubectl -n support rollout status deploy/support-inference

   b. **acmectl-based:** for products behind the internal control plane:
        $ acmectl deploy rollback acme-support --to v3.0

4. Verify traffic is served by the rollback target. Compare response
   fingerprint on a canary prompt against the v3.0 staging endpoint:
     $ acmectl compare-response acme-support production v3.0

5. Update the registry:
     $ acmectl registry set acme-support-v3.1 status rolled-back
     $ acmectl registry set acme-support-v3.0 status production

## After the rollback
6. Post to #incidents with the rollback timestamp and the rolled-back
   artifact ID.
7. Page the fine-tuning owner: <owner-name>, phone <on-call phone-tree>.
8. Do not delete or overwrite the rolled-back artifact — it becomes evidence
   in the incident report.
9. Incident report due within 24 hours: root cause, whether the release-gate
   should have caught it, and (if yes) what changes to the gate.

## Rollback window
Rollback to v3.0 is supported until <production_at of v3.1 + 30 days>. After
that, v3.0 is retired and the rollback path is to v2.4 (see that runbook).

## When rollback WON'T help
- If the incident is caused by an upstream dependency (retrieval, tools),
  rollback of the model will not resolve it.
- If the incident is a data-integrity issue rooted in the base model or the
  training data (present in both v3.1 and v3.0), rolling back to v3.0 will
  reproduce the incident. Escalate to safety lead + platform lead and
  consider a maintenance-mode kill switch (route to a canned response) until
  a fix ships.

## Rollback dry-run
Every quarter, run this runbook in dry-run mode (`acmectl deploy rollback
acme-support --to v3.0 --dry-run`) and confirm the rollback would succeed.
Log the dry-run in the change log (mod-109 ch7).
```

The runbook belongs in Git alongside the registry entry. It is verified quarterly. It is the artifact mod-111 exercise 03 asks you to author.

## The rollback window

Every production artifact has a *rollback window*: the maximum time after supersession during which the artifact remains eligible as a rollback target. Default 30 days; tune to the product.

- **Managed fine-tunes.** The rollback window has an ongoing hosting cost (mod-110 ch5). Retire actively after the window closes.
- **Self-hosted checkpoints.** Storage is cheap; the constraint is compatibility drift (base model deprecation, serving-stack version drift). Retire when either becomes a risk.
- **Regulatory retention.** Some jurisdictions require artifact retention for a longer horizon (e.g., 12 months for regulated use). Keep the artifact in cold storage past the operational rollback window — it is no longer a rollback target, but it exists for audit.

The registry entry's `timeline` block records `production_at`, `superseded_by` (which triggers the rollback window's start), and `retired_at`.

## Multi-adapter routing at platform scale

Mod-104 chapter 3 introduced multi-adapter routing — one base, many named adapters, per-request routing. At platform scale this becomes a *tenant / feature-flag / A-B* concern.

The registry supports it by allowing an artifact to reference *many adapters* on the same base and by naming the routing rule:

```yaml
# registry/acme-support/v3.1-multi.yaml
id: acme-support-v3.1-multi
artifact:
  kind: self-hosted
  base: <same base>
  adapter_router:
    kind: header-based
    header: X-Acme-Tenant
    rules:
      - tenant: "*"
        adapter_chain: [sft-billing, dpo-support, safety-v3]
      - tenant: "internal-testers"
        adapter_chain: [sft-billing-experimental, dpo-support, safety-v3]
```

The registry entry names the router, and the deploy pipeline configures the serving stack accordingly. Chapter 3 of mod-104 already covered the on-GPU mechanics; this chapter's contribution is that the routing rule is a *registered platform artifact*, reviewed, versioned, and rolled back like any other.

## Cross-vendor releases (self-hosted + managed side by side)

Some products serve both self-hosted and managed variants — a self-hosted fine-tune for the majority of traffic, a managed fine-tune for a specific tenant with a residency constraint, or vice versa. The registry handles this by treating each as a separate entry with a shared `product` field and separate `deployed_endpoints`.

The release verdict then evaluates each independently against the gate — the mod-108 harness runs against both backends (mod-110 ch7) — and both must pass their respective gates before either promotes to production. The rollback runbook has a per-backend section.

The pattern is not "one artifact, two backends"; it is "two artifacts with a shared product." Trying to unify them under one ID loses the ability to reason about backend-specific incidents.

## What the platform automates and what stays manual

Not everything above needs to be a human step. What automates well:

- **Registry hash verification** on every write.
- **Gate-report parsing** and hard-gate check.
- **Promotion PR generation** on candidate creation.
- **Rollback dry-run** on a schedule.
- **Retirement pass** on a schedule.
- **Managed-fine-tune deletion** at the retention window end (mod-110 ch5).

What should stay manual:

- **Signing the release verdict.** A human decides "approve" or "approve with disclosure"; the automation gates whether the decision *can* be made, not what the decision *is*.
- **Rolling back.** The runbook automates the mechanics; the decision to trigger the runbook is on-call's, not a monitor's. (Auto-rollback on quality-SLO breach is possible; do it only when the SLO is very well-calibrated and the rollback target is genuinely safe. Erring on the side of manual is fine in most orgs.)
- **Retiring within the rollback window.** Never automatic; always requires the "no outstanding rollback demand" confirmation.

The distinction preserves the human judgment where it matters and removes it where it doesn't.

## Six anti-patterns

- **The "temporary" registry entry.** An artifact deployed as `status: production` "just for a demo" without a verdict. It goes into the registry the same way any production artifact does, or it doesn't go into production. No exceptions; every exception erodes the platform.
- **Rollback that requires re-training.** If your rollback path requires re-running the previous release's SFT, you don't have a rollback — you have a re-run. Keep prior artifacts materialized within the rollback window.
- **A registry that lags production.** Traffic served by an artifact not in the registry (or in the registry with the wrong hash) is invisible to audits, incident response, and cost dashboards. Deploy pipelines must fail closed if the artifact is not registered.
- **Promotion rules that live in someone's head.** Two engineers with different opinions on "is this ready for prod?" will make different calls. Write the rules; each transition is a checklist, not a judgment call.
- **Governance signed once, then bypassed.** The captain / governance sign is per-release, not per-project. A change to the SFT dataset that ships as v3.2 requires the same sign-offs as v3.1's original release.
- **Silent retirements.** Retiring an artifact that another product still points at (routing rule, cross-references, or historical citation) breaks downstream. Retirement passes should refuse to retire when references exist.

## Summary

- The release verdict is a per-release document that composes the mod-108 release-gate report, the mod-106 safety report card, the mod-102 dataset card, the mod-109 serving runbook, and the model card into one signed APPROVE / APPROVE_WITH_DISCLOSURE / BLOCK decision.
- The three cards are AND-ed: a pass on one does not compensate for a fail on another. Each card has its own owner; the verdict adjudicates the intersection.
- The registry is the platform's system of record. Every trained artifact — self-hosted checkpoint or managed `ft:...` — is a YAML file in Git with a schema that pins base, adapters, quantization, serving, verdict, lineage, compliance, and timeline.
- The registry entry's schema is uniform across self-hosted and managed; the `artifact.kind` field differs. This is the payoff of the mod-110 pattern.
- Promotion has explicit rules per transition (candidate → staged → production → rolled-back / retired). Every transition is a registry commit with reviewers; the commit history is the audit log.
- The rollback runbook is executable by a stranger at 3 AM: pre-conditions, verification, exact commands, post-conditions, escalations, and "when rollback won't help" section. Verified quarterly.
- The rollback window is a per-product knob (default 30 days). Managed fine-tunes are actively deleted after the window to stop hosting rent (mod-110 ch5).
- Multi-adapter routing and cross-vendor releases are supported by the registry — each is a named artifact with a routing rule or a shared product ID. No special-cased shortcuts.
- Automate registry verification, gate parsing, promotion PR generation, dry-runs, and retirement passes. Keep the release-verdict signature and the rollback trigger manual.
- Six anti-patterns to guard against: temporary registry entries, rollback-by-retraining, registry-that-lags-production, promotion-rules-in-heads, one-time-governance, silent retirements. Each erodes the platform from a different direction.
