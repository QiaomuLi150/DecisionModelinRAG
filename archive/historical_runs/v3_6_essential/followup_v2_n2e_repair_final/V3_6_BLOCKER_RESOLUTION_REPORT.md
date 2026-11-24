# DecisionRAG v3.6 blocker-resolution closure

## Final gate

`N2_E_REPAIRED: PASS_WITH_RESOURCE_CAVEAT`

`N3_BM25_TRANSFER: BLOCKED`

This is an append-only v3.6 follow-up. Historical v3.6 artifacts were not overwritten.

## N2-E repaired execution

The repaired primary record set contains exactly 3,240 label-blind records: 360 frozen BRIGHT questions × 3 policies × 3 attempts. GPU 3, GPU 4 and GPU 6 use the clean shards from `followup_v2_n2e_repair_attempt2`; GPU 5 uses the corrected, non-overlapping shard from `followup_v2_n2e_repair_attempt5_gpu5`. All four shards contain 810 records and together cover 360 unique canonical IDs with no duplicate `(policy, canonical_id, attempt)` keys.

Labels were joined only in the CPU closure. Every retained raw record has `labels_joined=false` and `input_label_blind=true`. The closure also corrected the gold join to use the authoritative BRIGHT `(domain, query_id)` mapping rather than the numeric suffix of the internal canonical ID.

| Policy | Records | Valid attempts | Valid rate | Mean nDCG@5 (failures=0) |
|---|---:|---:|---:|---:|
| `jina_full` | 1,080 | 1,080 | 100.00% | 0.17018 |
| `pplx_budget` | 1,080 | 1,077 | 99.72% | 0.17573 |
| `winnow_native` | 1,080 | 483 | 44.72% | 0.07410 |

The 40,000-draw, 12-domain bootstrap contrasts are:

| Contrast | Delta | 95% interval | Sign-flip p | Interpretation |
|---|---:|---:|---:|---|
| PPLX budget − Jina full | +0.00555 | [-0.01246, +0.02310] | 0.6553 | Inconclusive; no evidence of a reliable difference |
| Winnow native − Jina full | -0.09608 | [-0.12413, -0.06912] | 0.0083 | Lower observed utility, but heavily affected by native failures |

The three PPLX failures were `RuntimeError: UNBUDGETABLE_QUERY_OR_OVERHEAD`; all 597 Winnow failures were `HTTPError: HTTP Error 400: Bad Request`. Failure records remain in the metric population and score zero as predeclared.

## Excluded attempts

The following attempts remain preserved but are not included in the final metric set:

1. `followup_v2_n2e_repair`: freeze-only plus an implementation/path-resolution failure before valid raw outcomes.
2. `followup_v2_n2e_repair_attempt2/shards/gpu5`: 867 records instead of 810 because the original fatal-error handler synthesized fill-in rows; excluded from all analysis.
3. `followup_v2_n2e_repair_attempt3_gpu5`: completed the wrong frozen slice (duplicate of the GPU-4 slice); excluded.
4. `followup_v2_n2e_repair_attempt4_gpu5`: correct slice but an unhandled `None` iteration in request hashing caused a fatal PPLX run after 57 records; excluded.

The final GPU-5 rerun corrected both implementation defects and has 810 records, including ordinary explicit failure records only.

## Resource qualification

The quality/validity closure is usable. On some shards, a model allocation remained resident while the Winnow server was active, so latency and allocator measurements are descriptive rather than isolated matched-serving evidence. This report does not promote them to a production cost or throughput claim. GPUs 3–6 were checked before use and were released after completion; no unrelated process was terminated.

## N3 status

N3 remains blocked. The exact frozen BM25-only transfer artifact required by the protocol is not present. The available v2.1 transfer population uses hybrid RRF retrieval and is not an interchangeable substitute. No N3 metric was generated or promoted.

## Reproducibility and provenance

- Frozen repair protocol SHA-256: `c2c9390e0268347532d439b7fba1c13f95f5881b9352bf13dbc8cecfb729456c`
- GPU-5 repair input-manifest SHA-256: `c2737623a539cf96ab1cfe7b673adfc6ea031e58560845430b7224414855e860`
- Final merged raw trace SHA-256: `f2bc39d684faf8880c16bfbc2b7ccfb9b9c9856d5ba3e16247f229fcd0390a3a`
- Final per-query CSV SHA-256: `49a027868cb7ca79de5995abf2126403bc2730c8afca2e51bf97648aa1e998c6`
- Final contrasts CSV SHA-256: `6039e307e59fa02fd2326ba04c455d02c5cd914172baef1e7b84aa1b7acb0c62`
- Full regression: 302 passed, 0 failed, CUDA disabled.
- Paid/API calls: 0. Protected holdout access: none.

Machine-readable summary: `N2E_SUMMARY.json`.

