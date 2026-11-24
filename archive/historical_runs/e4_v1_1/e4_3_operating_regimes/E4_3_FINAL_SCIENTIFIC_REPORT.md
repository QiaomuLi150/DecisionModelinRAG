# DecisionRAG E4-3 Final Scientific Report

## Gate

`E4_3_OWNER_REVIEW_READY` with quality evidence qualified by shared-GPU timing blockage. The new exploratory cohort is frozen at **380** questions (representative=300, diagnostic=80). No holdout, Jev, paid API, training, reader regeneration, new retrieval or new model family was used.

## Findings

The report files and scorecard preserve separate K10/K20 CSSR-U, CSSR-C, feasibility ceilings, structural Top-5 limits and valid-denominator accounting. The Clef mechanism rows report matched K-specific effects and the K10-feasible D20−D10 interaction. Clef joint is not presumed superior; negative, null and unavailable cells remain visible.

## Quality scorecard

CSSR-U is unconditional over valid questions; CSSR-C is conditional on all gold-support IDs being present in the K-candidate view. The ceiling is the fraction of questions for which complete support is present. `any_gold_top5` is a descriptive Top-5 any-support rate.

| Policy | K | Cohort | Valid n | CSSR-U (num/den) | CSSR-C (num/den) | Complete-support ceiling | Any-gold Top-5 | Failures |
|---|---:|---|---:|---|---|---:|---:|---:|
| clef_pointwise_binary | K10 | representative | 300 | 0.36333333333333334 (109/300) | 0.529126213592233 (109/206) | 0.6866666666666666 | 0.9233333333333333 | 0 |
| clef_pointwise_binary | K10 | diagnostic | 80 | 0.2 (16/80) | 0.3404255319148936 (16/47) | 0.5875 | 0.95 | 0 |
| clef_pointwise_binary | K20 | representative | 300 | 0.30666666666666664 (92/300) | 0.30666666666666664 (92/300) | 1.0 | 0.8666666666666667 | 0 |
| clef_pointwise_binary | K20 | diagnostic | 80 | 0.175 (14/80) | 0.175 (14/80) | 1.0 | 0.925 | 0 |
| clef_joint_binary | K10 | representative | 300 | 0.33 (99/300) | 0.48058252427184467 (99/206) | 0.6866666666666666 | 0.8966666666666666 | 0 |
| clef_joint_binary | K10 | diagnostic | 80 | 0.2 (16/80) | 0.3404255319148936 (16/47) | 0.5875 | 0.9 | 0 |
| clef_joint_binary | K20 | representative | 300 | 0.2633333333333333 (79/300) | 0.2633333333333333 (79/300) | 1.0 | 0.85 | 0 |
| clef_joint_binary | K20 | diagnostic | 80 | 0.1625 (13/80) | 0.1625 (13/80) | 1.0 | 0.8625 | 0 |
| liquid_native_joint | K10 | representative | 300 | 0.37 (111/300) | 0.5388349514563107 (111/206) | 0.6866666666666666 | 0.9166666666666666 | 0 |
| liquid_native_joint | K10 | diagnostic | 80 | 0.3125 (25/80) | 0.5319148936170213 (25/47) | 0.5875 | 0.9125 | 0 |
| liquid_native_joint | K20 | representative | 300 | 0.36 (108/300) | 0.36 (108/300) | 1.0 | 0.9 | 0 |
| liquid_native_joint | K20 | diagnostic | 80 | 0.2375 (19/80) | 0.2375 (19/80) | 1.0 | 0.8625 | 0 |
| jina35_native | K10 | representative | 300 | 0.4666666666666667 (140/300) | 0.6796116504854369 (140/206) | 0.6866666666666666 | 0.9666666666666667 | 0 |
| jina35_native | K10 | diagnostic | 80 | 0.2625 (21/80) | 0.44680851063829785 (21/47) | 0.5875 | 0.9875 | 0 |
| jina35_native | K20 | representative | 300 | 0.5366666666666666 (161/300) | 0.5366666666666666 (161/300) | 1.0 | 0.9666666666666667 | 0 |
| jina35_native | K20 | diagnostic | 80 | 0.3375 (27/80) | 0.3375 (27/80) | 1.0 | 0.9875 | 0 |
| pplx_decider_native | K10 | representative | 300 | 0.51 (153/300) | 0.7427184466019418 (153/206) | 0.6866666666666666 | 0.9766666666666667 | 0 |
| pplx_decider_native | K10 | diagnostic | 80 | 0.3875 (31/80) | 0.6595744680851063 (31/47) | 0.5875 | 0.9875 | 0 |
| pplx_decider_native | K20 | representative | 300 | 0.5366666666666666 (161/300) | 0.5366666666666666 (161/300) | 1.0 | 0.97 | 0 |
| pplx_decider_native | K20 | diagnostic | 80 | 0.35 (28/80) | 0.35 (28/80) | 1.0 | 0.975 | 0 |
| bge_pointwise | K10 | representative | 300 | 0.43333333333333335 (130/300) | 0.6310679611650486 (130/206) | 0.6866666666666666 | 0.9266666666666666 | 0 |
| bge_pointwise | K10 | diagnostic | 80 | 0.2875 (23/80) | 0.48936170212765956 (23/47) | 0.5875 | 0.95 | 0 |
| bge_pointwise | K20 | representative | 300 | 0.3933333333333333 (118/300) | 0.3933333333333333 (118/300) | 1.0 | 0.9033333333333333 | 0 |
| bge_pointwise | K20 | diagnostic | 80 | 0.3 (24/80) | 0.3 (24/80) | 1.0 | 0.925 | 0 |

## Clef mechanism contrasts

| Contrast | K | Matched n | Estimate | Bootstrap 95% low | Bootstrap 95% high | Status |
|---|---|---:|---:|---:|---:|---|
| clef_joint_minus_pointwise | 10 | 206 | -0.04854368932038835 | -0.11650485436893204 | 0.019417475728155338 | MEASURED |
| clef_joint_minus_pointwise | 20 | 300 | -0.043333333333333335 | -0.1 | 0.01 | MEASURED |
| interaction_D20_minus_D10 | 10_to_20 | 206 | 0.0 | -0.05825242718446602 | 0.05825242718446602 | MEASURED |

The representative joint-minus-pointwise estimates are negative at both K=10 and K=20, with intervals crossing zero. The K20-minus-K10 interaction is evaluated on the identical K10-feasible population and does not support a claim that joint inference improves with larger K.

## Native work and observed execution telemetry

| Policy | K | Valid n | Physical-call n | Exact-reuse n | Forward calls | Observed p50 ms | Observed p95 ms | Peak GB | GPU IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| clef_pointwise_binary | K10 | 385 | 385 | 0 | 385 | 1417.319829 | 1537.964961 | 18.98053216934204 | 1 |
| clef_pointwise_binary | K20 | 385 | 385 | 0 | 385 | 2717.929911 | 2879.349345 | 20.199169635772705 | 1 |
| clef_joint_binary | K10 | 385 | 385 | 0 | 385 | 1407.945109 | 1491.721834 | 18.574839115142822 | 1 |
| clef_joint_binary | K20 | 385 | 385 | 0 | 385 | 2675.205623 | 2800.813223 | 19.307617664337158 | 1 |
| liquid_native_joint | K10 | 385 | 385 | 0 | 0 | 383.842846 | 437.037027 | 6.337667942047119 | 1 |
| liquid_native_joint | K20 | 385 | 0 | 385 | 0 | None | None | None | 1 |
| jina35_native | K10 | 385 | 385 | 0 | 385 | 336.413336 | 364.541472 | 1.6320314407348633 | 1 |
| jina35_native | K20 | 385 | 0 | 385 | 0 | None | None | None | 0 |
| pplx_decider_native | K10 | 385 | 385 | 0 | 385 | 2096.232876 | 2176.67333 | 49.1799373626709 | 1 |
| pplx_decider_native | K20 | 385 | 0 | 385 | 0 | None | None | None | 5 |
| bge_pointwise | K10 | 385 | 385 | 0 | 385 | 85.398603 | 98.691419 | 1.173083782196045 | 1 |
| bge_pointwise | K20 | 385 | 0 | 385 | 0 | None | None | None | 0 |

These are non-isolated native-call diagnostics. They are not matched end-to-end Top-K-to-Top-5 isolated latency and must not be converted to sustainable QPS or dollar cost.

Specialist comparisons are policy-bundle comparisons, not architecture effects. Jina, BGE, Liquid and PPLX differ in native objective and context view. Historical K20 specialist rows are reused only where exact input/model/contract hashes match; K10 is native work. The 27B PPLX cell is blocked wherever local K10 feasibility could not be established.

## Latency and replacement

Matched isolated selector latency is blocked because approved GPUs were not isolated. Consequently no production QPS, cost saving, or latency Pareto claim is supported. The 1 s/3 s replacement boundary is `NOT_MEASURED`.

## Reviewer conclusion

This exploratory evidence can diagnose candidate-size, redundancy, support-cardinality and context-footprint interactions, but it cannot establish production replacement. The central question—whether Clef joint beats Clef pointwise and is competitive with Jina/BGE after latency—remains qualified by the exact measured scorecard, blocked timing branch and dominant source-group concentration.
