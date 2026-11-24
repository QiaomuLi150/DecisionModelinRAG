# DecisionRAG revised tables

The machine-readable tables are authoritative: `E6_4_CROSS_WORKLOAD_METRICS.csv`, `E6_4_ORDER_STABILITY_SCORECARD.csv`, `E6_4_RELEVANCE_SENSITIVITY.csv`, and the E6-3A corrected serving scorecard. SciFact and GaRAGe are not pooled.

## Table 1 — Model and native policy

| Policy | Revision | Native readout |
|---|---|---|
| PPLX Decider | `14c9d901...` | candidate-aligned choice probabilities |
| StartLux Decision | `9c0a32e9...` | native typed choice probabilities |
| Jina v3.5 | `e8a93f33...` | native listwise relevance scores |
| Qwen3 Reranker 0.6B | `e61197ed...` | native cross-encoder scores |

## Table 2 — Endpoint contracts

SciFact uses sparse binary qrel nDCG@5. GaRAGe uses fixed-pool binary `evidence_relevant` nDCG@5 with explicit no-positive handling. Neither is factual answer correctness.

## Tables 3–7

Use the CSV scorecards and provenance manifest. Every table is traceable to an authenticated historical or E6-4 artifact; no noninferiority margin or production cost is claimed.
