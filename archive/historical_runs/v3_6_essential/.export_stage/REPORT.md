# DecisionRAG v3.6 Essential Verification — Final CPU Closure

Overall status: **PARTIAL**.

This closure preserves the original sandbox CUDA failures and the initial Jina two-stage adapter failures. Corrected GPU traces are versioned under the GPU3/GPU6 namespaces; no raw trace was overwritten.

CPU/offline integrity checks passed, including the full repository regression suite (302/302), raw-trace hash reconciliation, and path-free export hygiene.

## Gate status

- Stage 0 frozen-input preflight: **PASS** (1,384 BRIGHT queries; L=8192 identity receipt passed).
- Stage 1 quality: **PASS** (31,832 per-query rows; labels joined only in CPU analysis).
- Stage 2 serving: **PARTIAL_2_OF_3_COMPLETE_REPLICATES** (two complete five-policy T1 sessions; the third namespace contains Jina-only records).
- Stage 3 frontier/statistics: **PARTIAL_PENDING_T2_AND_OPTIONAL**.
- Stage 4 optional capacity/optimization: **NOT_RUN**.

## Quality results

All-request known-positive nDCG@5 is normalized by min(gold_count,5); invalid inference outcomes score zero. These are ranking metrics, not downstream answer correctness.

| Policy | n | Valid % | Mean nDCG@5 |
|---|---:|---:|---:|
| drex_v2_native  | 1384 | 87.57% | 0.1781 |
| jina_full  | 1384 | 100.00% | 0.1844 |
| jina_single 2048 | 1384 | 94.94% | 0.1221 |
| jina_single 4096 | 1384 | 99.21% | 0.1637 |
| jina_single 8192 | 1384 | 97.69% | 0.1738 |
| jina_two_stage 2048 | 1384 | 97.11% | 0.1471 |
| jina_two_stage 4096 | 1384 | 99.21% | 0.1761 |
| jina_two_stage 8192 | 1384 | 99.71% | 0.1790 |
| liquid_d1_default  | 1384 | 100.00% | 0.1476 |
| pplx_single 2048 | 1384 | 94.94% | 0.1747 |
| pplx_single 4096 | 1384 | 99.21% | 0.1875 |
| pplx_single 8192 | 1384 | 99.71% | 0.1928 |
| pplx_two_stage 2048 | 1384 | 97.11% | 0.1854 |
| pplx_two_stage 4096 | 1384 | 99.21% | 0.2013 |
| pplx_two_stage 8192 | 1384 | 99.78% | 0.2108 |
| winnow_budget  | 1384 | 75.51% | 0.1426 |
| winnow_native  | 1384 | 79.62% | 0.1599 |

### Matched-text contrasts

| Contrast | Delta | 95% CI | Holm p | Positive domains |
|---|---:|---:|---:|---:|
| pplx_single_1024-jina_single_1024 | 0.0216 | [0.0162, 0.0287] | 0.0146484375 | 9/12 |
| pplx_single_2048-jina_single_2048 | 0.0526 | [0.0441, 0.0632] | 0.00244140625 | 12/12 |
| pplx_single_3072-jina_single_3072 | 0.0339 | [0.0249, 0.0429] | 0.01171875 | 11/12 |
| pplx_single_4096-jina_single_4096 | 0.0238 | [0.0142, 0.0329] | 0.060546875 | 9/12 |
| pplx_single_6144-jina_single_6144 | 0.0165 | [0.0083, 0.0264] | 0.060546875 | 9/12 |
| pplx_two_stage_2048-jina_two_stage_2048 | 0.0383 | [0.0292, 0.0473] | 0.009765625 | 10/12 |
| pplx_two_stage_4096-jina_two_stage_4096 | 0.0252 | [0.0168, 0.0339] | 0.02685546875 | 10/12 |

## Matched serving results

The complete-policy T1 results are available for two isolated sessions (slots A and B in derived exports). Slot C contains the full Jina block only; it is not treated as a complete five-policy replicate. Ratios are same-session p95(policy)/p95(Jina full), with cluster-bootstrap intervals over C360 IDs and all-request failure timing included.

| Session | Policy | p50 ms | p95 ms | p95 ratio | 95% interval | Verdict |
|---|---|---:|---:|---:|---:|---|
| T1_A | jina_full | 300.7 | 1222.4 | 1.000 | [1.000, 1.000] | pass |
| T1_A | jina_pplx_budget | 162.4 | 245.6 | 0.201 | [0.130, 0.250] | pass |
| T1_A | jina_two_stage_pplxwindow | 333.8 | 729.7 | 0.597 | [0.394, 0.741] | pass |
| T1_A | pplx_budget | 1733.7 | 2377.2 | 1.945 | [1.257, 2.417] | inconclusive |
| T1_A | pplx_two_stage | 3574.6 | 6860.3 | 5.612 | [3.653, 6.906] | fail |
| T1_C | jina_full | 294.7 | 1223.5 | 1.000 | [1.000, 1.000] | pass |
| T1_C | jina_pplx_budget | 164.8 | 245.9 | 0.201 | [0.132, 0.251] | pass |
| T1_B | jina_full | 293.5 | 1221.5 | 1.000 | [1.000, 1.000] | pass |
| T1_B | jina_pplx_budget | 162.7 | 245.4 | 0.201 | [0.133, 0.251] | pass |
| T1_B | jina_two_stage_pplxwindow | 332.9 | 731.1 | 0.599 | [0.397, 0.745] | pass |
| T1_B | pplx_budget | 1734.9 | 2382.0 | 1.950 | [1.296, 2.439] | inconclusive |
| T1_B | pplx_two_stage | 3583.8 | 6880.0 | 5.632 | [3.692, 6.948] | fail |

### Pooled available-session view

| Policy | Available complete sessions | p95 ratio | 95% interval | Verdict |
|---|---:|---:|---:|---|
| jina_full | 3 | 1.000 | [1.000, 1.000] | pass |
| jina_pplx_budget | 3 | 0.201 | [0.130, 0.250] | pass |
| jina_two_stage_pplxwindow | 2 | 0.597 | [0.396, 0.741] | pass |
| pplx_budget | 2 | 1.945 | [1.260, 2.423] | inconclusive |
| pplx_two_stage | 2 | 5.615 | [3.676, 6.917] | fail |

## Plain-language answers

1. **G1:** Budgeted PPLX (`P_8192`) is about 1.95× full Jina in both complete sessions. The available-session interval crosses 2×, so the serving verdict is **inconclusive**, not a pass or failure. The 1.5× and 3× sensitivity rows are computed in `serving_threshold_sensitivity.csv`; no claim is made from an absent third PPLX session.
2. **G2:** Winnow, Drex and Liquid were not newly timed in this v3.6 runner; their quality records are historical controls only. No new same-session serving claim is made for them.
3. **G3:** On quality alone, PPLX two-stage at L=4096 and L=8192 exceeds full-text Jina's mean nDCG@5 in this exploratory BRIGHT population; the complete three-check frontier is not finalized because the required replicated/T2 timing and quality-difference intervals are incomplete. PPLX two-stage is clearly slower than Jina in the available serving sessions.
4. **Two-stage:** At the matched L=2048/4096 quality points, PPLX two-stage exceeds matched-text Jina two-stage in the per-query analysis; this is an exploratory result, not a confirmatory release claim.
5. **Tuning Jina:** Lower budgets reduce Jina quality and often validity; Jina two-stage is more reliable at 8192 than its single-call budgeted variant, but no complete frontier release decision is asserted.
6. **Interference:** The three namespace names do not constitute a valid concurrent three-GPU interference experiment. `solo_vs_concurrent.csv` is explicitly `NOT_RUN`.
7. **Parameters:** The available timing records show the 27B PPLX model has materially higher p95 latency than the 0.6B Jina model; this is descriptive and not a causal parameter-scaling law.
8. **Optional capacity:** Stage 4 was not run; no sustainable QPS or monetary cost claim is made.
9. **Manuscript implication:** describe the result as an exploratory BRIGHT sensitivity/serving study, state that budgeted PPLX is borderline at the 2× p95 threshold, and disclose incomplete T2/solo/three-complete-session coverage.

## Confirmed, changed, unverified

- **Confirmed:** frozen BRIGHT input identity; L=8192 reuse identity; corrected native Stage1 traces; two complete primary T1 sessions; raw failures retained; no label-blind serving leakage.
- **Changed:** the original one-session serving interpretation is superseded by a versioned two-complete-session plus Jina-only third-session record; the initial two-stage Jina adapter failure is superseded only for the repaired policy trace.
- **Unverified:** the full three-complete-replicate serving gate, solo interference, T2 budget frontier, historical-model matched timing, determinism rechecks, and optional concurrency/optimization experiments.

## Artifact pointers

- `stage0_preflight/` — frozen input/protocol receipts.
- `stage1_quality_v2/export/quality/` — quality per-query, summary and contrast exports.
- `stage2_serving/export/` — path-free derived serving exports.
- `stage2_serving/`, `stage2_serving_gpu6/`, `stage2_serving_gpu3_r3/` — internal raw traces and provenance.
- `stage3_analysis/export/frontier/` — partial release-check/frontier outputs.
- `RAW_TRACE_SHA256_MANIFEST.json` — final recursive artifact hash manifest.
