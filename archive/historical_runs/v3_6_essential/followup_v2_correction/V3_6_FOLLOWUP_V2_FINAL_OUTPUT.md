# DecisionRAG v3.6 Follow-up v2 — Final Output

**Date:** 2026-10-24  
**Scope:** C1–C5 corrective follow-up  
**Final gate:** `V3_6_FOLLOWUP_V2_PARTIAL_BLOCKED`

## Executive result

The authorized C1, C2, C3 and C5 work is complete and preserved in the append-only correction namespace. C4 clean matched timing was not executed because the dedicated-GPU telemetry gate did not pass. Therefore this follow-up is not a full milestone pass: quality and diagnostic results are available, but clean C4 serving claims remain blocked.

No KEEP artifacts were rerun or replaced. The KEEP set was preserved: Stage F `quality_v2`, N1 full-population PPLX/Jina ratios, determinism and leave-one-domain-out analyses, per-domain deltas, and the existing N2 mxbai/Qwen ratios.

## Gate summary

| Stage | Status | Final disposition |
|---|---|---|
| C1 — N3 frozen confirmation | DONE | All four budgets were scanned and evaluated; raw outputs were frozen before labels were joined and the 40,000-draw bootstrap was completed. |
| C2 — Jina disagreement diagnostic | DONE | Historical disagreement was quantified; two fresh fp32 batch-1 runs agreed exactly at list and set level. |
| C3 — Winnow/Drex repair | DONE / NOT COMPARABLE | Original context settings were used, but both policies remained below the 99% validity threshold for latency comparison. |
| C4 — clean timing sessions | BLOCKED / NOT RUN | Dedicated-GPU telemetry did not satisfy the clean-session gate. No C4 timing result is claimed. |
| C5 — frontier reconstruction | DONE | Frontier rebuilt offline from existing quality and N1 serving inputs; it does not substitute for C4. |

## C1 — Frozen transfer-set confirmation

### Input-length scan

The CPU scan used the frozen v2.1 hybrid transfer slates and the PPLX tokenizer. No budget met the 99% hash-identical no-truncation skip rule, so all four budgets were evaluated.

| Budget L | Enterprise identical/full | MuSiQue identical/full | All transfer questions identical/full |
|---:|---:|---:|---:|
| 2048 | 0/500 (0.0000) | 0/2,167 (0.0000) | 0/2,667 (0.0000) |
| 4096 | 0/500 (0.0000) | 223/2,167 (0.1029) | 223/2,667 (0.0836) |
| 6144 | 0/500 (0.0000) | 1,079/2,167 (0.4979) | 1,079/2,667 (0.4046) |
| 8192 | 0/500 (0.0000) | 1,662/2,167 (0.7670) | 1,662/2,667 (0.6232) |

### Native trace accounting

The C1 join authenticated 21,336 raw records and produced 21,336 per-query records. Labels were joined only after raw output freezing. The analysis used 40,000 bootstrap draws with seed 731.

| Model | Budget coverage | Validity |
|---|---:|---:|
| Jina matched-text | 5,334/5,334 records | 100% |
| PPLX matched-text, L=2048/4096 | 5,334/5,334 records | 100% |
| PPLX matched-text, L=6144/8192 | 5,025/5,334 records | 94.21%; 309 `ValueError` failures |

### C1 quality results

Metrics below are the joined known-positive nDCG@5 values from `c1_summary.csv`; the paired contrast is PPLX minus Jina from `c1_contrasts.csv`. The reported Q field is the prespecified diagnostic comparison only; S is `NOT_MEASURED` for this quality run.

| Dataset | L | PPLX nDCG@5 | Jina nDCG@5 | PPLX validity | Jina validity | PPLX−Jina delta | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Enterprise | 2048 | 0.58697 | 0.33227 | 100.0% | 100.0% | +0.25469 | [+0.22189, +0.28859] |
| Enterprise | 4096 | 0.62381 | 0.43291 | 100.0% | 100.0% | +0.19090 | [+0.16065, +0.22190] |
| Enterprise | 6144 | 0.63899 | 0.48446 | 100.0% | 100.0% | +0.15452 | [+0.12368, +0.18613] |
| Enterprise | 8192 | 0.24095 | 0.52190 | 38.2% | 100.0% | −0.28094 | [−0.32875, −0.23279] |
| MuSiQue | 2048 | 0.47325 | 0.52236 | 100.0% | 100.0% | −0.04911 | [−0.05824, −0.03992] |
| MuSiQue | 4096 | 0.48306 | 0.53876 | 100.0% | 100.0% | −0.05571 | [−0.06435, −0.04707] |
| MuSiQue | 6144 | 0.48464 | 0.54420 | 100.0% | 100.0% | −0.05956 | [−0.06808, −0.05110] |
| MuSiQue | 8192 | 0.48644 | 0.54757 | 100.0% | 100.0% | −0.06113 | [−0.06980, −0.05256] |

The Enterprise L=8192 PPLX result is not a quality-only success claim because validity falls to 38.2%. Its failure burden is retained in the denominator and is visible in the raw traces.

## C2 — Jina disagreement diagnostic

The historical frozen-versus-corrected comparison remains below the 99% agreement gate:

- 1,130/1,384 list-identical requests (81.79% over all records; the earlier valid-only report was approximately 83.3%).
- 1,280/1,384 set-identical requests (92.49% over all records; the earlier valid-only report was approximately 94.4%).
- 254 list differences and 104 set differences were preserved.
- The first-20 difference casebook is included. Score differences alone do not identify dtype, padding, batching or truncation as the cause.

Two fresh Jina runs were executed with the requested fp32 scoring head and batch-size-1 semantics:

- Run 1: 1,380 valid / 1,384 total.
- Run 2: 1,380 valid / 1,384 total.
- Fresh list agreement: 1,384/1,384.
- Fresh set agreement: 1,384/1,384.
- Maximum fresh-run nDCG@5 spread: 0.
- Fresh-run mean nDCG@5: 0.1768753 for both runs.

The fresh-run agreement does not retroactively replace the historical frozen result; it only shows that the new controlled execution was internally deterministic.

## C3 — Winnow and Drex-v2 repair

The reruns used the original context settings and preserved server launch information and five HTTP-400 response bodies per policy.

| Policy | Requests | Valid | Failed | Failure rate | Latency verdict |
|---|---:|---:|---:|---:|---|
| Winnow | 1,080 | 483 | 597 HTTP 400 | 55.28% | `NOT COMPARABLE` |
| Drex-v2 | 1,080 | 966 | 114 HTTP 400 | 10.56% | `NOT COMPARABLE` |

Both valid rates are below the prespecified 99% threshold. No latency ratio is reported for either policy.

## C4 — Clean timing status

C4 was not run. The required clean-session conditions included dedicated-GPU telemetry, one model per GPU, Jina full-text anchors at the start and end of every session, and matched request-level timing. The available telemetry did not pass that gate, so no timing result was fabricated and the earlier mixed-GPU N2 numbers were not relabeled as clean C4 evidence.

## C5 — Frontier reconstruction

The frontier was rebuilt offline from:

- the existing v3.6 `quality_v2` per-query and summary files;
- the existing N1 pooled serving ratios;
- the frozen BRIGHT Top-20 retrieval slates.

The result contains 15 policy rows, uses 40,000 bootstrap draws with seed 731, and applies the prescribed R/Q/S rules. Because C4 was blocked, frontier serving values remain inherited historical measurements and must not be interpreted as newly matched clean-session measurements.

## Integrity and provenance

### Raw trace hashes

| Artifact | SHA-256 |
|---|---|
| C1 Jina L2048/4096 | `6c6e44b5e519019b714fe693f9eaf3029ff54b6a422d483b2bfa947b8320bfc1` |
| C1 Jina L6144/8192 | `44be3d5b63789e9fd0f57a7c607128bed32d505f200d50655c1d4cd4f0600cf2` |
| C1 PPLX L2048/4096 shard 0 | `c40eb11b73cf8b709770375a5d867118318c5a59b84539d3e9d2474fad752b32` |
| C1 PPLX L2048/4096 shard 1 | `3a074a922080e18ea69ad0600d10c4003f535b229a28c379fc9826529217b2c9` |
| C1 PPLX L6144/8192 shard 0 | `7f5587fdfe9797aac27c2b097ffeb69e07c3d4d10f983d806105db15e1481cc5` |
| C1 PPLX L6144/8192 shard 1 | `55b8521299e6ae18aced04a6d4cbfe9e4b362a348f7796ed883182317ffbeced` |
| C1 PPLX L6144/8192 shard 2 | `2cf7d922b475b1f26dcabc93dde3d5d1d3aefb814a7d7ce8259c789446c4a45c` |
| C2 fresh Jina run 1 | `ee0f41505613b1e5eb22aad134762be5a7e4b832241421839ccd9ca73efb0975` |
| C2 fresh Jina run 2 | `0b3c74a10b7cc4ad404978d9b7e4ae5149f491753e7d706e2cbf78965d01778e` |
| C3 Winnow raw trace | `e2918f3af6843b88152147f1f43a9b34da8f656d6e8535a1cf30c253a0d27502` |
| C3 Drex-v2 raw trace | `04cda8bac5a8a1ea284a720b439e3b8741c49da0b103b995cccb031c7ba58836` |
| C5 frontier CSV | `ba9b07a4d9a1b8c9a4f3d3b144f5f4400b4983c720959ee1670c24b1ab1c6371` |

### Tests and package verification

- All `scripts/v3_6_correction_*.py` files pass Python compilation.
- C1 join completed with `status: PASS`, 21,336 raw rows and 21,336 per-query rows.
- C2, C3 and C5 receipts report `status: PASS` for their scoped work.
- The refreshed results ZIP passed `unzip -t` integrity verification.
- Refreshed ZIP size: 73,427,136 bytes.
- Refreshed ZIP SHA-256: `0a9ce3f095818dd89393044efcc495f126101286cdb2aa1f3b8c311f2da61966`.
- No original historical result was overwritten or deleted.

## Artifact index

Primary correction namespace:

`results/v3_6_essential/followup_v2_correction/`

Key outputs:

- `C1_transfer_confirmation/c1_perquery.csv`
- `C1_transfer_confirmation/c1_summary.csv`
- `C1_transfer_confirmation/c1_contrasts.csv`
- `C1_transfer_confirmation/C1_JOIN_SUMMARY.json`
- `C2_jina_diagnostic/C2_OFFLINE_SUMMARY.json`
- `C2_jina_diagnostic/C2_FRESH_RUN_SUMMARY.json`
- `C3_server_repairs/winnow/TRACE_META.json`
- `C3_server_repairs/drex/TRACE_META.json`
- `C5_frontier_v2/frontier_v2.csv`
- `C5_frontier_v2/FRONTIER_RECEIPT.json`

Shareable package:

`results/v3_6_essential/DecisionRAG_v3_6_followup_v2_correction_available_results_v2.zip`

## Final interpretation

The follow-up resolves the C1 long-context execution, C2 fresh determinism diagnostic, C3 repair attempt and C5 offline frontier reconstruction. It does not establish clean matched serving performance because C4 was blocked by the telemetry gate. The record therefore supports reporting the quality, failure and determinism findings above, while any new C4-based latency or production-efficiency conclusion remains unsupported.
