# exercise-02: Managed vs. Self-Hosted Cost and Privacy Matrix

**Estimated effort:** 2 hours
**Companion chapters:** [`05-cost-privacy-ip-residency.md`](../05-cost-privacy-ip-residency.md), [`06-closed-frontier-vs-open-base.md`](../06-closed-frontier-vs-open-base.md), [`03-vendor-tour.md`](../03-vendor-tour.md)

## Objective

For **one concrete deployment scenario** of your choice, build the defended matrix that compares fine-tuning on your own self-hosted stack (mod-109) against fine-tuning on at least **two** managed vendors on **cost**, **data residency**, **IP / weight ownership**, and **privacy / retention / provider-training** posture. Land the matrix in a decision memo that a platform, finance, and compliance reviewer can each sign off on — the artifact bundle a real project would attach to a `managed-vs-self-hosted-<project>` ADR.

The point of this exercise is reasoning, not code. You will not train anything. You WILL cite the vendor's own docs for every claim you make and pin the retrieval date of every price and policy — the whole exercise is worth nothing if a reviewer cannot audit where each number came from.

## Prerequisites

- Chapters 5 and 6 read; chapter 3's vendor tour skimmed.
- Access to **current** public docs for every vendor you compare — pricing page, DPA / data-processing terms, regional availability page, and the fine-tuning docs. You will not use gated / NDA'd info; everything here should be reproducible from public sources.
- Your mod-109 chapter 7 self-hosted cost sheet (or an equivalent per-token cost estimate for the base model on the SKU your team actually runs). If you did not do mod-109 exercise 04, use a defensible published cost calculator (e.g., a cloud-vendor GPU instance page + a published vLLM throughput number for the base you'd run) and note the source.
- A spreadsheet or Markdown-table environment. No API keys required.
- No API budget required — this exercise spends zero dollars.

## Requirements

### Part A — Pick and specify the deployment scenario

In `scenario.md`, define exactly **one** deployment. All later math is anchored on these numbers, so pick honestly:

- **Product context.** One paragraph: what the fine-tune is for, who calls it, what quality bar it must meet.
- **Traffic profile.** Monthly requests, mean input tokens per request, mean output tokens per request, request-rate variance (steady, bursty, batch-nightly). Pick numbers that are realistic for the product; if you are inventing them, say so and pick something between 10k and 100M requests/month.
- **Data classification.** What is in the prompts and responses at rest? Pick the strictest label the product would touch (PII / PHI / confidential business / public). This drives Part C.
- **Regional constraint.** Where must the data be processed? (US only, EU only, "no China," "must stay in AWS `eu-west-1`," "no constraint.") Pick one that mirrors a real product; do not default to "none" unless the product genuinely has none.
- **Latency and availability SLO.** p50 / p99 latency target; monthly uptime target. Rough is fine.
- **Time horizon.** How long will this fine-tune be in production? (3 months, 1 year, 3 years, indefinite.) Amortization of training and switching cost depends on this.

Write it in one page. This is the input to every subsequent part.

### Part B — Pick the three candidates

In `candidates.md`, name the three (or more) candidates you will compare, and for each write two sentences on **why it is a plausible choice for this scenario** (not why it is the winner — that is the point of the exercise).

Required composition:

- **Exactly one self-hosted candidate.** The mod-109 stack from your track: a specific open base + adapter method + serving stack + SKU (e.g., "Llama 3.3 70B Instruct + LoRA rank 16 + vLLM + 2× H100 SXM on-demand"). Pin the specific SKU.
- **At least two managed candidates**, drawn from the seven vendors in chapter 1. Constraints:
  - At least one must be a **closed-frontier** direct-API vendor (OpenAI direct, Anthropic direct, Vertex Gemini). This is the "you cannot self-host this base at all" leg of the comparison.
  - At least one must be a **multi-provider control plane** (AWS Bedrock or Azure AI Foundry), OR a vendor with an **open-weight escape hatch** (Mistral La Plateforme, a Bedrock Llama fine-tune). This gives you a portability contrast.
- If your scenario's regional constraint eliminates a vendor outright (e.g., "data must stay in EU" and the vendor doesn't offer an EU region for fine-tuning), still name it in `candidates.md` with a one-line "eliminated at Part C by residency — retained for contrast in the matrix."

### Part C — Fill in the four-quadrant matrix

Author `matrix.md` with **one row per candidate** and **one column per quadrant**. This is the deliverable's core artifact.

The columns:

1. **Cost.** Monthly total from chapter 5's four-lane formula:
   - Training (amortized: `training_cost / amortization_months`, using Part A's time horizon).
   - Hosting rent (per-fine-tune or per-Provisioned-Throughput-hour, converted to monthly).
   - Inference (`requests × (in_tokens × in_price + out_tokens × out_price)`).
   - Egress + storage + logs (may be $0 for direct-vendor; nonzero for Bedrock / self-hosted).
   - **Show your work.** Every dollar in the cell must be traceable to a formula and a cited price.
   - For self-hosted, use your mod-109 chapter 7 amortized-hardware number (or a documented substitute), and be honest about GPU utilization — a $50k/year H100 that runs at 20% utilization is $250k/year of effective cost.
2. **Data residency.** For this candidate:
   - Which region does training happen in? Which region does inference happen in? (Cite the vendor's regional-availability page.)
   - Does the vendor offer a hard residency contract for your Part A regional constraint? (Y/N + citation.)
   - Any cross-region hops in default configuration? (E.g., Bedrock model artifacts replicating across regions.)
   - Where does the training JSONL live at rest, and for how long? (Cite the vendor's retention policy.)
3. **IP / weight ownership.** For this candidate:
   - Can you download the fine-tuned weights? (Y/N.)
   - Who owns the fine-tuned model per the DPA? (Cite the specific clause.)
   - What happens to the fine-tuned model on account cancellation? (Cite the deletion policy.)
   - What happens to the fine-tuned model on base-model deprecation? (Cite deprecation policy.)
   - Portability: what is the cost, in engineering-days, of moving to a different vendor after this one? (Estimate; explain the estimate.)
4. **Privacy / retention / provider-training.** For this candidate:
   - Is your training data used by the vendor to improve their own foundation models? (Y/N + citation. Default posture in 2026 across major vendors is "no" for fine-tuning workloads, but verify per vendor.)
   - Retention window for the training JSONL after the job completes.
   - Retention window for inference logs (and any opt-out flag).
   - Certifications relevant to your scenario: SOC 2 Type II, ISO 27001, HIPAA BAA availability, GDPR sub-processor list, EU AI Act general-purpose AI provider status. Only include what applies to Part A's data classification.
   - If any of the above requires a specific SKU (e.g., "HIPAA requires the Enterprise tier" or "PT is required for VPC endpoints"), the cost cell in column 1 must reflect that SKU.

Every cell that would say "unknown" must instead say `<!-- needs-research: <specific question> -->` with a link to the doc page you looked at. Do not paper over gaps.

### Part D — Write the decision memo

Author `decision-memo.md`. Audience: your platform lead, your finance partner, and your compliance reviewer will each read this in ten minutes. Sections:

1. **Recommendation.** Which candidate wins for this scenario. One paragraph. No hedging.
2. **Why (the axis that decided).** Chapter 6's five-axis framework applied to Part A's scenario. Point at the one or two axes that pushed the decision — usually capability ceiling and cost, sometimes residency and IP. If the axes conflict, explain the trade you made and why.
3. **The runner-up and what would flip it.** Which candidate would be second choice, and what specific change to the scenario (traffic 10× higher, latency SLO 2× tighter, residency requirement drops, an incident) would flip you to it. This is the exit-trigger discipline from chapter 6.
4. **The cost delta at scale.** A single sentence: "At 10× the traffic in Part A, the winner's monthly cost changes from $X to $Y; the runner-up changes from $A to $B; the crossover would happen at Z requests/month." Every finance reviewer wants this number.
5. **Compliance sign-off checklist.** For every certification your scenario's data classification requires (HIPAA BAA, GDPR DPA, SOC 2 report), one line: "Vendor: signed / requestable / not offered — evidence link." Compliance reviewer signs here.
6. **Assumptions and open questions.** Every assumption in the arithmetic (utilization, traffic growth, retention on the current SKU, current pricing) with a stability estimate (stable, likely to change, verified expiring in N months).

Keep it under 1500 words. If it is longer, it is not a memo, it is a report; the reviewer will not read it.

### Part E — Sensitivity analysis

Author `sensitivity.md`. Take the winning cost cell from Part C and vary each of the following one at a time. Report the new monthly total in a small table:

- Traffic × 0.1, × 10, × 100.
- Inference token premium changes (e.g., vendor doubles the fine-tune-tier premium — check chapter 2's "premium over base" discussion).
- Hosting rent changes (vendor changes hosting SKU, PT hourly rate).
- Utilization on the self-hosted stack changes (drops from 60% to 20%, rises to 90%).

Two paragraphs on which sensitivity dominates. This is the "how brittle is this recommendation" question your platform lead cares about.

### Part F — Directory layout

Final tree:

```
outputs/exercise-02/
├── scenario.md
├── candidates.md
├── matrix.md
├── decision-memo.md
├── sensitivity.md
└── sources.md          # every URL cited across the above, with retrieval date
```

`sources.md` is the audit trail. Every URL that appears in any of the other five files must appear here with `<url> — <what it was cited for> — retrieved YYYY-MM-DD`. If a URL is behind a paywall or requires login, note that — a reviewer needs to know they cannot reproduce the citation without the right access.

## Starter guidance

- **Do Part A first, and pick a real scenario.** A vague "some kind of RAG chatbot" scenario produces a vague matrix. Pin the traffic, the region, the data class, the SLO — most of the analysis falls out of those numbers.
- **Pin the retrieval date of every price and policy.** In `sources.md`, `retrieved 2026-09-02` after every URL. Prices shift in weeks; a matrix with no dates is worth nothing in three months.
- **The mod-109 self-hosted number is the anchor.** If your team does not have one, use a documented public calculator (an inference-cost calculator from a reputable source with the specific base model and your SKU) and cite it. Do not eyeball GPU cost.
- **Do the "10×" and "0.1×" cost delta before writing the recommendation.** The number that decides at the current volume often does not decide at 10× volume; the memo needs to acknowledge the crossover.
- **Do NOT commit any real customer data or internal projections.** The exercise's scenario should be a defensible, plausible, but nonspecific product; if your organization has a real product that fits, generalize the traffic and the data class so nothing sensitive lands in a public repo.
- **Cite the DPA clause, not the marketing page.** For every retention / training-on-your-data / weight-ownership claim, the citation must point at the vendor's data processing addendum, service terms, or fine-tuning-specific doc — not a blog post or a homepage. If only the marketing page exists, mark it `needs-research`.
- **Read chapter 5's arithmetic once before the matrix and once after.** The four-lane model is easy to forget the "egress + storage + logs" lane on; that lane is where Bedrock and self-hosted often creep up.

## Acceptance criteria

You have completed the exercise when:

- `scenario.md` pins the six scenario dimensions (product context, traffic, data class, region, SLO, time horizon).
- `candidates.md` lists at least three candidates including one self-hosted and two managed, with the composition constraints in Part B satisfied.
- `matrix.md` has one row per candidate and four columns (cost / residency / IP / privacy) fully populated, with every price traceable to a formula and every policy traceable to a cited doc page.
- `decision-memo.md` includes all six sections, comes in under 1500 words, and names a specific winning candidate — not "it depends."
- `sensitivity.md` reports the winning cost under at least four variations and identifies the dominant sensitivity.
- `sources.md` lists every URL cited across the deliverable with retrieval date.
- No cell in `matrix.md` says "unknown" — every gap is a `needs-research` with a specific question.

## Stretch goals

- **Add a hybrid candidate.** A **serverless LoRA on open weights** vendor (Fireworks, Together, Replicate, or a Bedrock Llama fine-tune) as a fourth candidate. Chapter 6's discussion of the hybrid path applies. Does it win any cell of the matrix outright? Would you actually pick it, given the vendor-lock-in on the training pipeline but portability on the weights?
- **Add a residency-constrained variant.** Take Part A's scenario and re-run the whole matrix with the regional constraint changed from "US only" to "EU only" (or vice versa). How many candidates drop out? Does the recommendation flip?
- **Model the migration.** Write a one-page `migration.md` that walks through what it would take to switch from the winning candidate to the runner-up: which artifacts port, which need to be rebuilt, how many engineering-days, what the traffic-cutover plan looks like. This is what an eventual "vendor exit" plan will look like in practice.
- **Wire the matrix to a spreadsheet.** Author `matrix.xlsx` or `matrix.csv` with the same cells as `matrix.md`, plus a `sensitivity` tab that recomputes automatically as you change traffic and price assumptions. This is what a finance partner will actually work in.
- **Contrast against a "no fine-tune" baseline.** Add a fifth candidate: "few-shot prompting on the base closed model, no fine-tune." What does its monthly total look like? Is the fine-tune actually paying for itself in the current scenario, or is prompt engineering the productive answer? Chapter 6's capability-probe discipline applies.
