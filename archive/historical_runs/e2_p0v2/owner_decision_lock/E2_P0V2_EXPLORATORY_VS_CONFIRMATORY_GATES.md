# E2-P0v2 Exploratory vs Confirmatory Gates

## Governing separation

| Track | Purpose | Allowed population | Required before execution | Current status |
|---|---|---|---|---|
| M1 exploratory | Explain why Clef/Laya policies differ and test mechanism hypotheses | Previously inspected frozen Track A/B populations only | Native interfaces, fixed contracts, authenticated IDs/views, M1 analysis plan, GPU authorization | `M1_PARTIAL_BLOCKED` |
| N confirmatory | Test replacement/noninferiority on a new holdout | Fresh sealed holdout only | Independence, primary contrast, epsilon, power, multiplicity, revisions, serving criteria, separate authorization | `N_BLOCKED` |

## M1 prerequisites

1. Verify Clef-Flash pointwise and native joint binary readouts against official/local adapter behavior.
2. Freeze one checkpoint revision, canonical nonempty question normalization, candidate IDs, candidate order, prompt/view hashes, score-contract version, and token budget.
3. Use the historical Track A and Track B populations only as exploratory follow-up: Track A `N_all=3,767`, Track B identity-resolved `N=443`; report Track A and Track B separately. Retrieval-conditional metrics retain `1,490` and `330` denominators.
4. Validate candidate alignment, duplicate rejection, score direction, tie handling, truncation, raw response hashes, actual input hashes, and physical work telemetry.
5. Obtain a bounded exclusive A100 execution window and owner approval. This phase does not provide that approval.

## N-only prerequisites

- Exact fresh release/revision, sample size, source-document independence, sealed gold and leakage audit.
- One predeclared H1 model-policy pair or a fully specified CAL-only selection algorithm.
- Approved absolute epsilon, alpha, power, sample-size and multiplicity family.
- Verified model/checkpoint revisions and native adapters.
- Numeric p95 SLO and arrival-rate workload, plus production quality/cost boundary.

Unresolved N requirements do not have to block an owner-approved M1, but they prevent any confirmatory claim or P0v2 freeze.
