# DecisionRAG E6-2 R5 — Independent GaRAGe Relevance Evaluation

## Final gate

`E6_2_R5_SCIENTIFIC_RESULTS_READY_WITH_LIMITATIONS`

R4D's frozen quality-blind native selections were authenticated and scored offline on the 1,397-sample EXPLORATORY GaRAGe population. No model inference or external service call was made in R5.

## Accounting

| item | count |
|---|---:|
| EXPLORATORY samples | 1,397 |
| model policies | 8 |
| model-sample prediction records | 11,176 |
| no-positive K15 samples | 64 |
| unresolved relevance-label samples | 1 |
| RESERVED rows exported to scorer | 0 |

## Primary PPLX versus Jina

PPLX mean binary nDCG@5: **0.871522**. Jina: **0.887324**. PPLX−Jina: **-0.015802**, 95% query-bootstrap CI **[-0.022765, -0.008966]**, B=5000, seed=731.
Paired outcomes: PPLX-only 160; Jina-only 242; ties 995. No-positive samples remain in the denominator.

The exploratory exact-content component sensitivity is [-0.022213, -0.007207], but it is not source-independent inference because the complete R4C membership cannot be reconstructed without opening RESERVED content.

## Model scorecard

See `E6_2_R5_MODEL_QUALITY_SCORECARD.csv` for all eight models, the fixed-prefix control, positive-only sensitivity, precision/recall/hit diagnostics and resource metadata.

## Fixed-prefix control

The control is `ORIGINAL_FIXED_POOL_PREFIX5`, not a natural retrieval baseline. Its mean nDCG@5 is **0.738936**. Prefix and model comparisons are paired on the same K15 pools.

## Cross-workload interpretation

SciFact's historical PPLX−Jina effect was +0.017400 with CI [-0.010549, +0.045281] on 277 natural Top-20 queries; GaRAGe's result is -0.015802 with the CI above on 1,397 fixed K15 passage pools. These are cross-workload comparisons, not pooled confirmation.

## Scientific boundary

This establishes Level 1 structural validity and Level 2 within-dataset exploratory quality only. Source-independent generalization, downstream factual answer correctness, and production replacement economics remain unestablished. E6-3 is not started.

## Provenance and tests

All governing R0/R3/R4C/R4D manifest hashes were independently verified. R5 performs zero new neural forwards, retrieval calls, reader calls, Jev calls or paid API calls. See the test report and provenance manifest.
