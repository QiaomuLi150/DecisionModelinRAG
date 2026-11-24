# DecisionRAG v3.6 complete follow-up closure

## Final scientific gate

`PARTIAL_T1_REPLICATION_LIMIT_AND_N3_BLOCKED`

The v3.6 follow-up completed its admissible CPU corrections, full-population timing, T2 frontier analysis, Jina contract repair, and the newly frozen N2-E repair. It does not support a complete three-replicate T1 serving claim, exact-BM25 transfer claim, production-QPS claim, monetary-cost claim, or factual-answer-quality claim.

All results are append-only. No historical E2–v3.6 artifact was overwritten, no protected holdout was opened, and no paid/API call was made.

## Stage and gate inventory

| Area | Result | Complete output |
|---|---|---|
| Stage 0 frozen-input preflight | PASS | 1,384 BRIGHT queries; L=8192 identity receipt passed |
| Stage 1 original quality | PASS | 31,832 per-query rows; offline label join; known-positive nDCG@5 |
| Follow-up Stage F CPU corrections | PASS | normalization, accounting, hash and export corrections |
| Follow-up Jina diagnostic rerun | BLOCKED initially | 1,126/1,352 exact-list agreement; below 99% gate |
| Versioned Jina contract repair | PASS for repaired historical comparison | 1,352/1,352 exact list, set, score-vector and request-hash matches; 169 physical batch forwards |
| N1 full-population timing | PASS | 55,360 records across two complete sessions; each 27,414 valid and 266 failed |
| Original N2 | PARTIAL | 9,720 records across sessions C/D/E; Drex lacked same-session anchors; original Winnow branch lacked pre-outcome receipt |
| N2-E repaired rerun | PASS_WITH_RESOURCE_CAVEAT | 3,240 records; 360 queries × 3 policies × 3 attempts; all four shards reconciled |
| T2 frontier and CPU statistics | PASS | Three complete frozen C120 sessions; 9,360 policy records |
| GPU-isolated solo C120 diagnostic | PASS | Completed, but not a whole-host-idle experiment |
| N3 transfer | BLOCKED | Exact BM25-only artifact unavailable; available transfer is hybrid RRF |
| Stage 4 optional capacity/optimization | NOT_RUN | No sustainable-QPS or monetary-cost result |

## Historical quality results

The original Stage 1 population is 1,384 BRIGHT queries. The endpoint is all-request known-positive nDCG@5, normalized by `min(gold_count, 5)`; invalid outcomes score zero. It is a ranking metric, not downstream answer correctness.

| Policy | Valid rate | Mean nDCG@5 |
|---|---:|---:|
| `jina_full` | 100.00% | 0.1844 |
| `pplx_single_4096` | 99.21% | 0.1875 |
| `pplx_single_8192` | 99.71% | 0.1928 |
| `pplx_two_stage_4096` | 99.21% | 0.2013 |
| `pplx_two_stage_8192` | 99.78% | 0.2108 |
| `jina_two_stage_8192` | 99.71% | 0.1790 |
| `liquid_d1_default` | 100.00% | 0.1476 |
| `winnow_native` | 79.62% | 0.1599 |
| `drex_v2_native` | 87.57% | 0.1781 |

The original matched-text quality contrasts favored PPLX at several budgets, but they remain exploratory and do not establish downstream answer quality or replacement superiority.

## Serving and frontier results

### T1 full-population serving

Two complete five-policy sessions were available; the third namespace contained Jina-only records. The complete-session pooled p95 ratios were:

| Policy | Complete sessions | p95 ratio vs Jina full | 95% interval | Verdict |
|---|---:|---:|---:|---|
| `jina_pplx_budget` | 3 | 0.201 | [0.130, 0.250] | pass |
| `jina_two_stage_pplxwindow` | 2 | 0.597 | [0.396, 0.741] | pass |
| `pplx_budget` | 2 | 1.945 | [1.260, 2.423] | inconclusive |
| `pplx_two_stage` | 2 | 5.615 | [3.676, 6.917] | fail |

The PPLX budgeted selector is borderline at the declared 2× p95 threshold. Two-stage PPLX is clearly slower in the available matched sessions.

### T2 frontier

T2 used three complete frozen C120 sessions and is a separate timing population from T1. Key results:

- PPLX two-stage L=8192: quality delta +0.02640 versus Jina full, 95% CI [0.01765, 0.03591]; p95 ratio 5.91, 95% ratio CI [3.82, 8.46] — fails the 2× serving criterion.
- PPLX single L=4096: quality delta +0.00311, 95% CI [-0.00856, 0.01134]; p95 ratio 0.969, 95% CI [0.608, 1.405] — passes the separate quality, reliability and serving checks used for that frontier point.
- PPLX budget L=8192: quality delta +0.00839, 95% CI [-0.00254, 0.01707]; p95 ratio 2.098, 95% CI [1.312, 3.055] — timing inconclusive.
- Jina two-stage L=4096 and L=8192 pass the declared T2 R/Q/S checks, with p95 ratios 0.263 and 0.602.

T1 and T2 are not pooled into one serving estimate. No sustainable QPS, host-wide cost, non-inferiority or causal parameter-scaling claim is supported.

## N2-E repaired result

The new frozen repair contains 3,240 label-blind records from 360 BRIGHT queries, using three attempts for each of `jina_full`, `pplx_budget` and `winnow_native`.

| Policy | Valid attempts | Valid rate | Mean nDCG@5 |
|---|---:|---:|---:|
| `jina_full` | 1,080/1,080 | 100.00% | 0.17018 |
| `pplx_budget` | 1,077/1,080 | 99.72% | 0.17573 |
| `winnow_native` | 483/1,080 | 44.72% | 0.07410 |

Using the prespecified 40,000-draw, 12-domain bootstrap:

- PPLX budget − Jina full: +0.00555, 95% CI [-0.01246, +0.02310], sign-flip p=0.6553. This is inconclusive.
- Winnow native − Jina full: -0.09608, 95% CI [-0.12413, -0.06912], sign-flip p=0.0083. This is dominated by 597/1,080 native failures and is not a clean model-quality comparison.

The finalizer’s gold join was corrected to use the authoritative BRIGHT `(domain, query_id)` mapping. The invalid GPU5 fill-in, wrong-slice and fatal-hash attempts are preserved and excluded.

## Jina contract-repair history

The initial follow-up diagnostic rerun failed the exact-list gate at 1,126/1,352 (83.284%) despite 1,276/1,352 (94.379%) set agreement. The versioned contract repair then corrected the historical-field serialization and restored the original batch-8 `score_many` path. It reproduced all 1,352 replayable successful historical rows exactly in selected lists, sets, score vectors and request hashes. This repairs the historical comparison contract; it does not silently rewrite the original traces.

## N3 blocker

N3 was not executed. The required exact unchanged BM25-only transfer artifact was not found. The available v2.1 transfer population uses hybrid RRF retrieval and is not an interchangeable protocol. No proxy result or cross-protocol transfer claim was substituted.

## Integrity and resource status

- Full project regression: 302 collected, 302 passed, 0 failed, CUDA disabled.
- Raw traces retained with SHA-256 and record counts.
- Label joins performed offline after raw output freeze.
- No protected holdout access.
- No paid/API calls.
- GPU 3–6 were used for the repaired N2-E shards and released afterward.
- N2-E quality conclusions are usable; some shard latency/allocator observations are descriptive because a model allocation remained resident while the Winnow server was active. They are not isolated serving-cost evidence.

## Authoritative artifact groups

- Original v3.6: `results/v3_6_essential/`
- Follow-up v1: `results/v3_6_essential/followup_v1/`
- Jina contract repair: `results/v3_6_essential/followup_v2_contract_repair/`
- N2-E repair: `results/v3_6_essential/followup_v2_n2e_repair_final/`
- Shareable N2-E package: `decisionrag_v3_6_n2e_repair_final.zip`

