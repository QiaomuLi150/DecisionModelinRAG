# DecisionRAG v2.1 controlled replication — v3 audited exports

## Status

`COMPLETE` for the owner-approved V2-HYBRID-RRF amendment. The original exact-BM25-only input contract remains blocked and preserved; this is not relabeled as an exact BM25 replication.

## T1/T7 populations

The exported per-query population contains 12,500 Enterprise rows and 54,175 MuSiQue rows: 25 executed policy/order cells per benchmark (six policies across four orders plus original-order native PPLX). The three native-PPLX non-original cells are explicitly `NOT_RUN`, never zero-quality or zero-validity results. `bm25_order` is labeled for readers as `first-stage order (hybrid RRF)`.

## T2 survivor audit

The P3/P6 survivor audit contains 10668 matched records. Equal survivor-set shares are: enterprise/original=0.002000, enterprise/perm_seed_20261020=0.002000, enterprise/perm_seed_20261021=0.004000, enterprise/reversed=0.000000, musique/original=0.002769, musique/perm_seed_20261020=0.001384, musique/perm_seed_20261021=0.000923, musique/reversed=0.000923. P3's stage-one survivors are verified to be selected from Jina's own per-window scores; P6 uses its own PPLX scores. Both use the same PPLX-generated window strings for stage one and the same candidate strings for stage two, so the text rule does not differ between policies.

## T3 order audit

Every declared non-original order changed the candidate order for every request in the affected manifest cells. Enterprise `pplx_two_stage` has identical FS@5 in the original, reversed and first shuffled order because the selected Top-5 sets are not necessarily identical; the order audit records the exact set-difference and overlap counts. The equality is therefore an equal aggregate score, not evidence that the selector ignored permutations.

## T4 determinism

The 20-request seed-20261022 rechecks completed for every executed model policy and both benchmarks, plus the offline first-stage-order control: 280 checks, zero mismatches. Enterprise native PPLX rechecks reproduced context-limit failures where applicable; MuSiQue native PPLX rechecks were valid.

## T5 length and failure audit

The PPLX limit is 8192 tokens with margin 128. Overflow shares (single-call / window A / window B) are Enterprise 0.980000 / 0.000000 / 0.000000; MuSiQue 0.000000 / 0.000000 / 0.000000. Enterprise native PPLX had 485 context-limit failures; MuSiQue had none. For MuSiQue original-order native-valid requests, 730 of 2167 have budgeted and native request hashes that differ, so native/budgeted score differences are not a contradiction.

## T8/T9 quality results

The primary query-level contrasts are unchanged: Enterprise PPLX two-stage minus matched Jina is 0.1015801354 (95% CI [0.0699774266, 0.1354401806]); MuSiQue is -0.0203045685 ([-0.0309183203, -0.0096908168]). Over the three non-original orders the corresponding contrasts are 0.1324303988 ([0.1038374718, 0.1625282167]) and -0.0129210891 ([-0.0223042609, -0.0033840948]). Holm values remain in `contrasts.csv`.

The Stage-0 group rule was not met: Enterprise has 500 groups for 500 queries (mean size 1.000); MuSiQue has 2167 groups for 2167 queries (mean size 1.000). Query-level resampling is therefore the frozen analysis unit, with a dependence caveat. The 200,000-draw sign-flip floor is 5e-6 where applicable.

## T6 serving

The v3 serving exports contain all 3,600 Stage-3 request records and both all-request and completed-request summaries. All requests were valid, so the bases coincide. The matched serial-session measurements are not sustainable QPS or monetary cost estimates.

## T10 release checks

`budget_sensitivity.csv` reports lower 95% bounds against P0 and P1. In the table, `Q` means the illustrative 5-point FS@5 loss budget only; it is not an externally validated SLA. The lower bounds are shown explicitly so other budgets can be applied. `R`, `Q`, and `S` use ✓/×/? conservatively; `?` means not run or an interval crossing the threshold.

## T11 reconciliation

`RECONCILIATION.csv` recomputes the V2 final report's headline validity, FS@5 contrasts, intervals, native failure count and serving ratios from v3 per-query/per-request files. All rows pass their declared tolerance. No unexplained mismatch above 1e-6 for quality or 0.5% for latency was found.

## Scientific limitations and deviations

The result remains an amended hybrid-RRF replication on previously examined datasets. It does not establish exact BM25 transfer, factual answer correctness, production throughput or monetary savings. Raw traces, raw passage text, device receipts and identities are excluded from the shareable package; their hashes and record counts are retained in `RAW_TRACE_HASHES_V3.json`.
