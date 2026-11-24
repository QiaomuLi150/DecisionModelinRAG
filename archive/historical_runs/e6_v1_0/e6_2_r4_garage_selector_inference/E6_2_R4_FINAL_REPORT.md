# DecisionRAG E6-2 R4 — GaRAGe Selection and SciFact Integrity Audit

## Final gate

`E6_2_R4_SCIFACT_INTEGRITY_BLOCKED`

The independent SciFact audit found a material R3 reporting defect, so R4 stops before GaRAGe selector inference. The exact-K15 row count is not silently promoted to an exploratory evaluation population.

### SciFact audit

- Frozen exploratory queries: 277
- Correct Top-20 feasibility: 254/277 (0.916968)
- R3-reported Top-20 feasibility: 277/277 (incorrect field/reporting result)
- PPLX–Jina nDCG@5 Δ: 0.017400
- 95% CI: [-0.010549, 0.045281]
- Corpus-ID mismatches: 0
- R1 qrel-access flags: 0

The primary nDCG@5 result reproduces exactly. The feasibility defect is isolated to R3 feasibility reporting; R3 artifacts remain unchanged and the correction is documented separately.

### GaRAGe audit

The release contains 2366 rows and 2329 exact-K15 pools with aligned arrays. It has no stable source-document IDs and no prior approved GaRAGe partition. An outcome-blind exact-K15 cohort proposal was recorded but not committed.

### Inference boundary

All eight models were excluded before execution: attempted 0, valid 0, failed 0, excluded 8. No GPU was used, no model was loaded, no selector output was fabricated, and no GaRAGe quality metric was computed.

E6-2 R5 is blocked until the R3 feasibility erratum is owner-reviewed and a GaRAGe exploratory partition is separately approved.
