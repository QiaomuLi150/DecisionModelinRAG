# DecisionRAG v2.1 controlled replication — final Stage-5 closure

## Overall status

`COMPLETE` under the owner-approved `V2-HYBRID-RRF` versioned input amendment. The original exact-BM25-only contract remains historically preserved and blocked; this report does not relabel the amended run as an exact BM25 replication.

## Populations and integrity

Enterprise contains 500 requests, 443 labeled and 330 reachable from the frozen Top-20. MuSiQue contains 2,167 labeled requests, 555 reachable from the frozen Top-20. Every canonical policy/order cell is accounted for. Labels were joined only after raw traces were frozen; native traces contain no labels, retries or fill-in. The original blocked Stage-0 artifacts, partial traces and the invalid non-original Stage-3 attempt remain preserved.

## Primary and secondary quality contrasts

On Enterprise, PPLX two-stage minus independently executed matched two-stage Jina is +0.1016 (95% CI [0.0700, 0.1354]) in the original order and +0.1324 ([0.1038, 0.1625]) averaged over the three non-original orders. On MuSiQue the same contrasts are -0.0203 ([-0.0309, -0.0097]) and -0.0129 ([-0.0223, -0.0034]). These are FS@5 results and must not be numerically compared with BRIGHT nDCG@5.

## Native failures and order sensitivity

Native PPLX had 485 Enterprise context-limit failures and only 15/500 valid original-order requests (3.0%); the non-original native cells were intentionally not run under the frozen Stage-2 contract. MuSiQue native PPLX was valid on the original order, while its non-original cells were likewise not run. Budgeted and two-stage policies were 100% valid in the executed cells. The score tables and length scans are in the derived exports; no failure was silently converted to a successful selection.

## Serving

The dedicated serial timing session contains 3,600 requests with zero failures. Jina two-stage stayed below a 2x p95 ratio to same-session Jina full on both datasets (Enterprise 1.192 [1.140, 1.270]; MuSiQue 1.490 [1.403, 1.551]). PPLX budget and PPLX two-stage exceeded 2x by large margins (Enterprise 6.475 and 13.830; MuSiQue 11.314 and 17.275). These are serial matched-session measurements, not sustainable QPS or monetary cost estimates.

## Conclusions

Transferred: the matched two-stage contrast is positive on Enterprise and negative on MuSiQue under the amended frozen workload; native PPLX context failures reappear on the longer Enterprise inputs; Jina two-stage satisfies the illustrative serving ratio check. Divergent: the preferred selector is workload-dependent, and PPLX's quality and serving results do not transfer uniformly. Unverified: exact BM25-only replication, factual answer correctness, production QPS and monetary savings.

## Deviations

1. The exact frozen BM25-only Top-20 input artifacts were unavailable for both benchmarks.
2. An owner-approved versioned amendment used preserved hybrid BM25-top100 + dense-top100 + RRF@60 Top-20 slates.
3. Native PPLX non-original orders were not required by the frozen protocol and remain `MISSING_OR_INCOMPLETE`.
4. An invalid Stage-3 non-original selection was stopped, retained, and excluded after the original-order filter check.
5. Quality release-check fields not produced by the evaluator are marked `?` in the derived table rather than inferred.

## Artifacts

See `FINAL_GATE_V2.json`, `FINAL_INTEGRITY_TEST_REPORT_V2.json`, `RAW_TRACE_SHA256_MANIFEST_V2.json`, `stage4_analysis_v2_hybrid/export/`, `stage3_serving_v2_hybrid/STAGE3_SERVING_REPORT.md`, `replication_table_v2.csv`, and `replication_table_v2.tex`.
