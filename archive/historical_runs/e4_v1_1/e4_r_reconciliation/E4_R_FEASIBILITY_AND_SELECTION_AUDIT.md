# E4-R Feasibility and Selection Audit

The E4-3 cohort contains 300 representative and 80 diagnostic-enriched exploratory questions. The diagnostic cohort is not pooled into representative rates.

## Retrieval ceilings

- K10 complete-support feasibility recomputes to 206/300 representative and 47/80 diagnostic.
- K20 complete-support feasibility is 300/300 and 80/80 in these frozen slates. This is a property of the selected exploratory cohort and original Top-20 candidate construction, not a general retrieval guarantee.
- A query with gold support absent from the candidate view is separated from selector failure through the CSSR-U/C denominators.
- No query has more than five distinct necessary supports in the frozen metadata; structural Top-5 infeasibility is therefore not observed, but document aliases and support identity remain source-data assumptions.

## Alignment

- Valid score/selection rows checked: 4560; score-sort mismatches: 0; frozen candidate-order mismatches: 0.
- Stable sort rule audited: descending score, original candidate rank, canonical ID.
- E4-3 native errors are preserved in the raw casebook; only valid full-cohort rows enter the recheck.

## Stratification

Support cardinality, frozen redundancy bands, and the deterministic candidate-length proxy are recomputed. Dataset-origin stratification is `NOT_ESTIMABLE` beyond MultiHop because the E4-3 cohort contains only MultiHop.

## Interpretation

The K20 ceiling must not be generalized outside this selected cohort. The paired CSSR-C contrast is conditional on common K-specific candidate feasibility; the unconditional CSSR-U contrast uses all valid queries. Neither is a factual answer-accuracy measure.

