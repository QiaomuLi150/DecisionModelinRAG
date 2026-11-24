# DecisionRAG v2.1 replication exports v3

Status: **COMPLETE** for the approved hybrid-RRF amendment; exact BM25-only replication remains blocked and preserved.

## Export counts

- Enterprise `perquery.csv`: 12,500 rows; MuSiQue: 54,175 rows.
- Expected count: 25 executed cells per benchmark (24 required run cells plus original-order native PPLX), excluding three native-PPLX non-original cells marked `NOT_RUN`.
- Stage-3 serving: 3,600 raw-derived request rows; 3,600 valid.
- Every CSV has a header and at least one row except no intentional empty CSV remains in the v3 export.

## Answers to the reviewer questions

1. **Permutation effect:** every declared permutation changed the candidate order for every affected manifest request. Enterprise PPLX two-stage's equal FS@5 across the original, reversed and first shuffled order is an aggregate equality; the order audit gives selected-set differences and mean overlaps, so it is not interpreted as identical selections.
2. **Survivors:** P3/P6 equal-set shares are `enterprise/original=0.002000, enterprise/perm_seed_20261020=0.002000, enterprise/perm_seed_20261021=0.004000, enterprise/reversed=0.000000, musique/original=0.002769, musique/perm_seed_20261020=0.001384, musique/perm_seed_20261021=0.000923, musique/reversed=0.000923`. P3 uses its own Jina-selected survivors: the raw stage records' per-window selected-ID union matches P3's stage-one survivor list for every audited record. P6 independently selects its own survivors. Both use shared PPLX-generated window strings for stage one and the selected candidate strings for stage two.
3. **Overflow:** see each benchmark's `length_scan.csv`. Enterprise native full-text single-call overflow is 0.980000; MuSiQue is 0.000000. Window overflows are reported separately and are zero or lower as shown. Enterprise's 485 native context-limit failures align with the full-text overflow population; MuSiQue native PPLX has no such failures. MuSiQue budgeted/native request hashes differ for 730/2167 native-valid original-order requests.
4. **Resampling unit:** the cluster rule was not met. Enterprise and MuSiQue each have one label-independent group per query with mean group size 1.0, so all primary/secondary contrasts remain query-resampled. Holm values are unchanged. Exact sign-flip p-values cannot be smaller than 5e-6 when 200,000 draws are the declared cap; the v3 binary/three-order audit uses the equivalent integer-weight calculation.
5. **Quality budgets:** `budget_sensitivity.csv` gives the lower 95% bounds and pass/fail decisions against P0 and P1 at 2, 5 and 10 points. The table's `Q` symbol refers specifically to the illustrative 5-point budget. `?` is reserved for not-run or threshold-crossing cases; no missing values are rendered as zero.
6. **Reconciliation:** `RECONCILIATION.csv` recomputes the V2 headline contrasts, intervals, native failure count and serving ratios from v3 per-query/per-request data. All rows pass tolerance. No unexplained mismatch remains.

## Hygiene and provenance

The v3 package contains only derived, path-free exports, reports and hashes. It excludes raw traces, raw passage text, internal trace paths, device receipts, GPU UUIDs and process listings. `RAW_TRACE_HASHES_V3.json` records trace keys, SHA-256 values and record counts only. `CONTENTS.md` records the hash and row count of every packaged file.
