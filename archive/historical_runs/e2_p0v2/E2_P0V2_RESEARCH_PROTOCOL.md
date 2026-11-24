# E2-P0v2 Research Protocol

**Status:** `PLANNING_ONLY_OWNER_REVIEW_REQUIRED`

## Scope

This protocol studies whether off-the-shelf general-purpose decision models can replace specialized rerankers for Top-20-to-Top-5 evidence selection. Primary claims concern evidence identity retention, native inference feasibility, and serving quality–latency–cost trade-offs. It does not claim private-enterprise deployment, factual correctness from reader proxies, or equivalence of distinct native mechanisms.

Historical Track A, Track B, HERB H1, Track C/R2, and E2-M0 results are previously inspected evidence. They remain preserved and are explanatory or planning evidence, not untouched confirmatory evidence for new policies. E2-M0's annotation-conditioned results are: Track A conditional n=1,490, quota oracle 1,196, Clef-G5 observed 689; Track B conditional n=330, quota oracle 318, Clef-G5 observed 206.

## Experimental settings

- **Setting A, common-information:** same frozen query, candidate identities, canonical document view, per-document truncation policy, and stable order wherever the model can represent the view. Infeasible context is reported, never silently shortened.
- **Setting B, native optimized:** each approved model uses its documented native batching, chunking, context and readout. All preparation, tokenization, queueing, physical model work and failures are charged. Results are separate from Setting A.
- Workload: `(canonical_nonempty_question, frozen_Top20_candidates) -> selected_Top5`.
- Primary study: off-the-shelf/zero-shot. Fine-tuning, Jev and paid evaluation are separate and unauthorized here.

## Cohorts

Track A (3,767) and Track B (500, 443 identity-resolved eligible) are historical/follow-up populations. HERB H1 (1,514; 815 answerable/699 unanswerable) is citation diagnostic only. A new confirmatory holdout is not selected: `HOLDOUT_SELECTION_BLOCKED_PENDING_OWNER_REVIEW`. A/B/HERB cannot be relabeled as untouched.

## Hypotheses and analysis

H1 is absolute noninferiority on complete annotated support retention versus a predeclared specialist, with lower paired CI bound greater than `-epsilon`; epsilon is owner-gated. H2a changes candidate context, H2b holds one globally comparable score vector fixed while changing selection policy, and H2c estimates conditional evidence gain exploratorily without gold. H3 tests prespecified workload interactions. H4 tests complete selection latency, p95, sustainable QPS, GPU work and cost.

All model claims are conditioned on valid native adapters, frozen model revisions, exact input hashes, no gold leakage, and explicit missing/error handling.

## Mandatory freeze gates

Before any new TEST inference, owner approval is required for the exact holdout/revision/sample, primary contrast or CAL-only selection algorithm, epsilon, power/sample size, multiplicity, numeric p95 SLO/load regime, core checkpoint roster/revisions, and isolated GPU execution window.

`E2_P0V2_SPEC_READY_FOR_OWNER_REVIEW` is a planning verdict only; it is not scientific freeze or execution authorization.
