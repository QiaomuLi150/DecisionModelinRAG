# DecisionRAG E6-2 R0 Final Scientific Readiness Report

## Gate

`E6_2_R0_RETRIEVAL_FREEZE_REQUIRED`

The protocol, endpoint distinctions, model contracts, missingness policy and statistical plan are ready for owner review. The primary SciFact benchmark cannot begin because no authentic, frozen, gold-blind SciFact Top-20 retrieval slates were found. The E6-1 deterministic SHA-256 candidate views are not substituted.

## Historical status

- E5 QASPER anchor is preserved: 1,502 questions, 587 groups, PPLX 306 vs Jina 299 complete-support successes, +0.5405 percentage points, 95% CI [-0.3586, +1.4805] pp; frozen slate hash `be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7`.
- E6-1/E6-1B hashes and adapter traces were audited without inference. Their technical feasibility interpretation is retained.
- Historical E6-1B baseline reported focused 5/5 and full 203/203 regression tests. This R0 adds no model calls.

## Dataset readiness

SciFact local release audit: 5,183 documents, 1,109 queries, 919 train qrels, 339 test qrels, 1,258 total qrel rows; joins passed. It is eligible for natural relevance only after the missing retrieval freeze.

GaRAGe local release audit: 2,366 rows, 2,329 exact K15 pools, unique sample IDs and aligned arrays. It is eligible as a fixed-pool graded-grounding secondary; source-group bootstrap is unavailable.

## Model admission

The eight-model roster is frozen. Native Top-5 adapters are syntactically validated on CAL fixtures. PPLX, Liquid, Jina, StartLux and Amos are order-sensitive in the small reversal diagnostic; Amos has six K20 context failures and unmeasured physical forwards. These conditions are carried into the future benchmark and are not quality claims.

## Source independence and holdout

SciFact split metadata are CAL 555 / EXPLORATORY 277 / RESERVED 277 over 572 qrel identity components, seed 731. Reserved raw content and labels remained unopened. No OS-level sealed-vault boundary exists, and near-duplicate independence is not proven; source-independent confirmation claims remain out of scope.

## Exact remedy

Owner-authorize a separate retrieval-only preparation for the 277-query SciFact EXPLORATORY cohort (or a predeclared broader admitted cohort), using a fixed documented retriever/index and no qrel access. Freeze, hash and audit Top-20 slates first. Only then authorize E6-2 selector quality inference. Do not rerun E6-1, do not open RESERVED, and do not begin E6-3/E6-4/E6-5.
