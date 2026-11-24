# DecisionRAG E6-2 R4D — Frozen GaRAGe Native Selector Inference

## Final gate: `E6_2_R4D_SELECTIONS_FROZEN_READY`

- Frozen exploratory pools: 1,397; candidates per pool: 15; scheduled model-query attempts: 11,176.
- Attempted: 11176; valid selections: 11176; failed: 0; unexecuted: 0.
- Quality scoring: NOT RUN; R5 was not started.
- RESERVED content, annotations, answers and citation judgments: not loaded.
- Jev/paid calls: 0/0.

## Model accounting

| model | scheduled | attempted | valid | failed | unexecuted | status |
|---|---:|---:|---:|---:|---:|---|
| pplx_decider | 1397 | 1397 | 1397 | 0 | 0 | COMPLETE |
| liquid_d1_3b | 1397 | 1397 | 1397 | 0 | 0 | COMPLETE |
| startlux_decision_4b | 1397 | 1397 | 1397 | 0 | 0 | COMPLETE |
| amos_1_small_2b | 1397 | 1397 | 1397 | 0 | 0 | COMPLETE |
| jina_v3_5 | 1397 | 1397 | 1397 | 0 | 0 | COMPLETE |
| qwen3_reranker_4b | 1397 | 1397 | 1397 | 0 | 0 | COMPLETE |
| bge_reranker_v2_m3 | 1397 | 1397 | 1397 | 0 | 0 | COMPLETE |
| qwen3_reranker_0_6b | 1397 | 1397 | 1397 | 0 | 0 | COMPLETE |

## Interpretation

Selections are quality-blind native Top-15→Top-5 outputs. No nDCG, Precision, Recall, Hit, grounding, or answer metric was calculated. Latency and memory are diagnostic native-selector measurements only; they do not establish production QPS, dollar cost, or matched serving superiority.

Forward accounting: 13595 observed or contract-derived physical calls; 1772 valid calls remain NOT_MEASURED because their internal forward boundary was not exposed. No unobserved zero was converted into a measured zero.

R4C source-independence limitations remain: 1,701 observed-content components, largest size 336, 768/2,329 rows in non-singleton components, absent authoritative source IDs, and no sealed reservation attestation.

## Verification

R4D focused tests: 6/6 passed. Full project regression suite: 251/251 passed. See E6_2_R4D_TEST_AND_REGRESSION_REPORT.md.

## Artifact integrity

All R4C governing input hashes are rechecked before merge. The combined registry is immutable for the next offline scoring stage; no historical R0–R4C result was modified. Raw append-only model chunks remain preserved.
