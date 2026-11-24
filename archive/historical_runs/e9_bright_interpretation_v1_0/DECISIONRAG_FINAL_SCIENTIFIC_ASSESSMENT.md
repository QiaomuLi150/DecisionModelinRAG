# DecisionRAG final scientific assessment

## Scope

This assessment concerns whether existing general-purpose decision models can replace specialist rerankers. It does not pool incompatible endpoints or turn exploratory results into confirmation. Historical E6-3 empty-passage timing is superseded by E6-3A; E7/E8 order behavior is behavioral evidence, not architectural causal evidence.

## RQ1 — Ranking capability

**Conclusion: Supported with limitations that no universal replacement advantage is established.** Historical E2–E8 evidence is heterogeneous: E3 reports dataset-specific policy differences, E6 reports mixed SciFact/GaRAGe exploratory results, and E9 BRIGHT provides an external ranking workload. On the frozen BRIGHT operational endpoint, PPLX−Jina was -0.098372 with domain-bootstrap CI [-0.15503154829573165, -0.0434499211615929]; Liquid−Qwen common-valid results are in the machine-readable comparison table. These are workload- and contract-specific comparisons, not model-family causality.

## RQ2 — Native inference reliability

**Conclusion: Supported with limitations.** E9 records 773 PPLX context-limit failures and two Qwen allocator-OOM failures. E7/E8 independently document contract-specific PPLX context failures and candidate-order behavior. Native interface and full-context feasibility are part of replacement suitability, not a nuisance to be removed retrospectively.

## RQ3 — Downstream RAG utility

**Conclusion: Not evaluated on BRIGHT; inconclusive overall.** E3–E5 contain evidence-retention and lexical/reference-overlap proxies, but no independent factual adjudication. BRIGHT reader status is `BRIGHT_READER_NOT_EVALUABLE`. No factual correctness, grounding, or answer-utility claim is made here.

## RQ4 — Practical replacement suitability

**Conclusion: No defensible replacement advantage is demonstrated.** PPLX's original operational result is lower than Jina and includes substantial native infeasibility. Common-valid analyses are selected-success diagnostics, not all-query effects. Fallback results are post-hoc and not free because a failed native request can consume compute and latency. E9 resource distributions use different GPUs and exclude model load, so they are descriptive only; they do not establish production QPS, cost savings, or matched latency superiority.

## Status vocabulary

- Ranking capability: **SUPPORTED_WITH_LIMITATIONS** for workload-specific differences; universal superiority: **NOT_ESTABLISHED**.
- Native reliability: **SUPPORTED_WITH_LIMITATIONS**.
- Downstream factual utility: **NOT_EVALUATED/INCONCLUSIVE**.
- Practical replacement: **NO_DEFENSIBLE_ADVANTAGE_ESTABLISHED**.

Authoritative claim-level evidence and source paths are in `DECISIONRAG_E2_E9_AUTHORITATIVE_EVIDENCE_LEDGER.csv`.
