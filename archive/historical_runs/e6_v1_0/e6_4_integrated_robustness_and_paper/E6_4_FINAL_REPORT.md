# DecisionRAG E6-4 — Final report

## Gate

`E6_4_READY_WITH_LIMITATIONS`

## Execution

- Four frozen policies: pplx_decider, startlux_decision_4b, jina_v3_5, qwen3_reranker_0_6b.
- CAL populations: 100 SciFact + 100 GaRAGe.
- Candidate orders: canonical, reversed, seeded permutation (seed 731).
- Planned and executed attempts: 2,400 / 2400.
- Labels were loaded only after native outputs were committed.

## Corrected serving integrity

E6-3A confirmed and corrected the empty-candidate-input defect. Original E6-3 SciFact serving claims are invalid for full-content comparison and remain preserved. Corrected E6-3A records are bounded selector diagnostics; retrieval, persistence and reader generation are outside the timed boundary.

## Robustness and quality

See `mechanism_inference/E6_4_ORDER_STABILITY_SCORECARD.csv`, `mechanism_scoring/E6_4_RELEVANCE_SENSITIVITY.csv`, and `mechanism_scoring/E6_4_PRIMARY_STATISTICS.json`. SciFact and GaRAGe are not pooled. Order sensitivity is behavioral evidence only and does not identify architecture-level causality.

## Reader and deployment boundary

The E6-3A reader endpoint is a normalized token-F1/reference-overlap proxy; factual correctness is not verified. Amos has 56/200 corrected SciFact context-limit failures. No sustainable QPS or monetary production cost was measured.

## Scientific decision

The record is ready for external scientific review with limitations. A genuinely independent prospective confirmation is required before stronger replacement, equivalence, noninferiority, factual-correctness, or production-economics claims. E5-F and E6-F remain unopened.
