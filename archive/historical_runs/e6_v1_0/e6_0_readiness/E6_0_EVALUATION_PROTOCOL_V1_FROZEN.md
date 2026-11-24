# DecisionRAG E6 Evaluation Protocol v1 (CPU-frozen proposal)

Status: `OWNER_REVIEW_REQUIRED`; downstream inference is not authorized by E6-0.

## Primary hierarchy

1. Historical anchor: PPLX minus Jina, equal-source-group-weighted set-aware CSSR-U, only where a validated complete-support dataset exists.
2. Compact decision-model comparisons: Liquid, StartLux and Amos only after native Top-5 contract validation; PPLX remains the historical anchor.
3. Specialist controls: Jina, Qwen3 4B, BGE v2-m3 and Qwen3 0.6B on identical candidate sets and native contracts.

## Statistical contract

Source-group bootstrap: 5,000 resamples, seed 731, two-sided percentile 95% CI. Report equal-group macro and query-weighted sensitivity separately. Do not pool incompatible datasets or endpoints. No model selection or endpoint revision after outcomes.

## Missingness and failure policy

Unattempted, timeout, invalid output, native-contract failure and infrastructure failure are separate counts. No silent retry beyond a predeclared retry policy. Missing outputs remain in the denominator for unconditional metrics; conditional metrics name their eligibility denominator.

## Factual-answer boundary

Answer F1/EM are deterministic reference proxies. Independent factual correctness requires blinded human or separately validated adjudication; it is not inferred from reference agreement.

## Confirmation gate

E5-F/E6 confirmation remains locked. Prospective confirmation requires a new source-disjoint reservation, frozen primary endpoint/effect margin, power calculation, independent answer validation where claimed, and owner approval.

## Current amendment status

The current DocHop-QA/GaRAGe inputs require the endpoint amendment in `E6_0_ENDPOINT_AMENDMENT_PROPOSAL.md`; no E6-1 execution may start before approval.
