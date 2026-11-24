# DecisionRAG v3.6 follow-up report

## Status

**PARTIAL.** Stage F CPU corrections are complete. The Jina wrapper repair produced valid outputs on budgetable requests, but the prespecified exact-list agreement gate failed (1126/1352 = 83.284%; set agreement 1276/1352 = 94.379%), so all Jina rerun quality/timing claims remain unverified.

N1 is complete at 27,680 records per session (55,360 total); each session has 27,414 valid and 266 failed records. N2 session C contains the required Jina/PPLX anchors and local specialists. The separate Drex-v2 GPU-4 branch is native and successful on completed HTTP requests but has no same-session anchors, so it is not a matched replacement comparison. The Winnow GPU-4 branch has native requests and same-session anchors, but the pre-outcome protocol receipt was not frozen before its anchor phase, so it remains descriptive/partial. N3 is blocked: the available transfer data use frozen hybrid RRF first-stage slates, while the follow-up requires an exact BM25-compatible source artifact.

No paid API calls, protected holdout access, new retrieval, training or historical-artifact modification occurred.

## Interpretation

The corrected PPLX quality frontier can be described using the Stage F whole-domain intervals and both fallback/no-fallback bases. Serving conclusions require the completed N1 traces and same-session anchors; no production QPS or monetary claim is made. Transfer to EnterpriseRAG-Bench or MuSiQue is not established by this follow-up.
