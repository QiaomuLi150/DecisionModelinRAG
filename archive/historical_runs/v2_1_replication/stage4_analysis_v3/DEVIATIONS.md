# V3 consolidated deviations

1. The exact BM25-only Top-20 input contract was unavailable for both benchmarks. The owner-approved versioned V2-HYBRID-RRF amendment was used; the original blocked Stage-0 artifacts remain preserved. Approval was given in the working session; it is not recorded in the repository.
2. On the amended slates, the trace key `bm25_order` is actually the hybrid BM25-top100 + dense-top100 + RRF@60 first-stage order; no BM25-only claim is made.
3. Native PPLX non-original orders were not run under the frozen Stage-2 contract and are explicitly `NOT_RUN` in every v3 table.
4. The first Stage-3 non-original cohort selection was invalid because it did not filter to original order. It was stopped, retained, and excluded; the corrected dedicated session is the only serving result.
5. Determinism rechecks were not present in historical artifacts and were therefore run post hoc on the frozen original-order inputs using seed 20261022. They are diagnostic, not a new quality experiment.
6. Query-level resampling remains correct because both label-independent group searches produce one group per query, mean group size 1.0, below the cluster rule.
7. Stage-3 all-request and completed-request bases coincide because all 3,600 requests succeeded. No fallback packaging time was added.
8. The shareable package excludes raw traces, passage text, process/device receipts and internal trace paths; hashes and record counts are retained.
9. Empty/zero-row legacy v2 exports are superseded by v3 populated exports; any intentionally empty concept is represented as `NOT_RUN` or `UNAVAILABLE` and explained in `README_V3.md`.
