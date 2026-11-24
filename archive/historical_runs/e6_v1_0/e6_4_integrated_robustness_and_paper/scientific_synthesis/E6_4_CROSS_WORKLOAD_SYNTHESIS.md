# E6-4 cross-workload synthesis

SciFact and GaRAGe are separate tasks and are not pooled. E6-4 is an exploratory candidate-order mechanism study, not a confirmatory test.

## Authenticated quality anchors

- SciFact historical E6-2 R3 PPLX−Jina operational binary nDCG@5: +0.017400, 95% CI [-0.010549, +0.045281], n=277; corrected Top-20 judged-positive feasibility 254/277.
- GaRAGe historical E6-2 R5 PPLX−Jina binary evidence-relevant nDCG@5: -0.015802, 95% CI [-0.022765, -0.008966], n=1,397; source-document independence not established.
- These endpoints are workload-specific and are not evidence of factual correctness.

## E6-4 robustness

See `mechanism_scoring/E6_4_PRIMARY_STATISTICS.json` and `mechanism_scoring/E6_4_RELEVANCE_SENSITIVITY.csv`. Structural changes under reversal/permutation demonstrate behavioral order sensitivity only; they do not identify architecture-level causality.

## Deployment interpretation

The corrected E6-3A serving records support bounded selector diagnostics. They do not establish sustainable production QPS, dollar cost, or full end-to-end RAG latency. Amos full-content context failures remain an explicit deployment limitation.
