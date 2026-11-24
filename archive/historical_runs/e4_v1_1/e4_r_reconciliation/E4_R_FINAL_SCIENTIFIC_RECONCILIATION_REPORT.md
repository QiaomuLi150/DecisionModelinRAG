# DecisionRAG E4-R Final Scientific Reconciliation Report

## Gate

`E4_R_OWNER_REVIEW_READY`

Audit completion is not scientific confirmation. E4-5 remains `CONFIRMATORY_DATA_NOT_READY`. This report used CPU/offline artifact reads only: no GPU, checkpoint, HTTP/API, Jev, paid API, reader regeneration, training, retrieval rebuild, multimodal evaluation, or holdout outcome access.

## R0/R1 reconciliation

The E2-M1 raw score, selected-ID and case-metric artifacts independently reproduce the historical Track A +14.295 pp conditional and Track B +20.000 pp conditional Clef joint-minus-pointwise contrasts. The E4-3 raw score/selection traces independently reproduce K10 -4.854 pp on common feasible n=206, K20 -4.333 pp on n=300, and K20-minus-K10 interaction 0.000 pp on n=206.

The exact-query/candidate/view overlap is zero. Both studies use the same pinned Clef revision and named binary objectives, but their candidate origins, view lineage, population denominators, and selection contexts differ. Verdict: `COMPARABLE_BUT_HETEROGENEOUS`; no implementation mismatch or K-only causal reversal is established.

## E4-3 audit

The frozen E4-3 cohort has 300 representative and 80 diagnostic questions. K10 complete-support feasibility is 206/300 and 47/80; K20 is 300/300 and 80/80 by construction of the frozen candidate cohort. The K20 ceiling is selection-conditioned, not a retrieval guarantee.

## Representative scorecard

| Policy | K | CSSR-U | CSSR-C | Ceiling | Any-gold Top-5 |
|---|---:|---:|---:|---:|---:|
| clef_pointwise_binary | K10 | 0.363333 | 0.529126 | 0.686667 | 0.923333 |
| clef_pointwise_binary | K20 | 0.306667 | 0.306667 | 1 | 0.866667 |
| clef_joint_binary | K10 | 0.33 | 0.480583 | 0.686667 | 0.896667 |
| clef_joint_binary | K20 | 0.263333 | 0.263333 | 1 | 0.85 |
| liquid_native_joint | K10 | 0.37 | 0.538835 | 0.686667 | 0.916667 |
| liquid_native_joint | K20 | 0.36 | 0.36 | 1 | 0.9 |
| jina35_native | K10 | 0.466667 | 0.679612 | 0.686667 | 0.966667 |
| jina35_native | K20 | 0.536667 | 0.536667 | 1 | 0.966667 |
| pplx_decider_native | K10 | 0.51 | 0.742718 | 0.686667 | 0.976667 |
| pplx_decider_native | K20 | 0.536667 | 0.536667 | 1 | 0.97 |
| bge_pointwise | K10 | 0.433333 | 0.631068 | 0.686667 | 0.926667 |
| bge_pointwise | K20 | 0.393333 | 0.393333 | 1 | 0.903333 |

## Dependence

The representative cohort contains 25 source components; the largest contains 261/300. Leave-largest-out leaves 39 representative queries and is descriptive only. Component IDs are authentic E3-D connected source groups, but independence across groups is not guaranteed.

## Operational evidence

| Policy | K | Valid | Physical requests | Instrumented forwards | Reused | p50 shared ms | p95 shared ms | Tier |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| clef_pointwise_binary | K10 | 380 | 380 | 380 | 0 | 1417.32 | 1534.94 | SHARED_GPU_DIAGNOSTIC |
| clef_pointwise_binary | K20 | 380 | 380 | 380 | 0 | 2716.96 | 2877.23 | SHARED_GPU_DIAGNOSTIC |
| clef_joint_binary | K10 | 380 | 380 | 380 | 0 | 1407.95 | 1491.54 | SHARED_GPU_DIAGNOSTIC |
| clef_joint_binary | K20 | 380 | 380 | 380 | 0 | 2675.41 | 2799.87 | SHARED_GPU_DIAGNOSTIC |
| liquid_native_joint | K10 | 380 | 380 | 0 | 0 | 383.843 | 436.394 | SHARED_GPU_DIAGNOSTIC |
| liquid_native_joint | K20 | 380 | 0 | 0 | 380 | NOT_MEASURED | NOT_MEASURED | REUSED_NO_TIMING |
| jina35_native | K10 | 380 | 380 | 380 | 0 | 336.729 | 364.444 | SHARED_GPU_DIAGNOSTIC |
| jina35_native | K20 | 380 | 0 | 0 | 380 | NOT_MEASURED | NOT_MEASURED | REUSED_NO_TIMING |
| pplx_decider_native | K10 | 380 | 380 | 380 | 0 | 2095.99 | 2176.31 | SHARED_GPU_DIAGNOSTIC |
| pplx_decider_native | K20 | 380 | 0 | 0 | 380 | NOT_MEASURED | NOT_MEASURED | REUSED_NO_TIMING |
| bge_pointwise | K10 | 380 | 380 | 380 | 0 | 85.6989 | 98.6914 | SHARED_GPU_DIAGNOSTIC |
| bge_pointwise | K20 | 380 | 0 | 0 | 380 | NOT_MEASURED | NOT_MEASURED | REUSED_NO_TIMING |

E4-3 matched isolated timing is blocked. E3-C and E2-S/P-R2 isolated measurements are preserved as nonmatched historical tiers. No production QPS, dollar cost or latency Pareto claim is made.

## Publication paths

1. **Exploratory empirical paper now:** center native decision-model versus specialist evidence selection, within-model mechanism controls, negative E4-2/E4-3 findings, reader-proxy limitations, source-group threats, and explicit timing gaps. It must avoid confirmatory or factual-accuracy language.
2. **Narrow prospective confirmation:** acquire genuinely new source-document-disjoint data, timestamp and hash a reservation before any outcome inspection, predeclare one primary joint-versus-pointwise contrast and effect margin, use group-aware power, independently verify answer correctness, and measure matched isolated serving. No existing E2/E3/E4 inspected cohort can be retimestamped into that holdout.

## Final limitations

The evidence does not identify universal operating superiority, factual answer accuracy, sustainable production economics, or confirmatory generalization. The reviewer packet and claims matrix preserve positive, negative, inconclusive, summary-only and blocked results separately.
