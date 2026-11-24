# DecisionRAG v1.6 D2 Matched Timing Report

Status: `D2_COMPLETE`

The frozen 120-ID cohort was evaluated three times per policy, one request at a time, in one host-capable Python session on dedicated GPU3. The old blocked D2 protocol and gate are preserved; this report is the versioned continuation after the GPU3 gate cleared.

## Frozen execution

- Policies: jina_full, jina_two_stage_full, jina_two_stage_pplxwindow, jina_pplx_budget, pplx_budget, pplx_two_stage
- Records: 2,160 (360 per policy)
- Valid: 2,148; failures: 12
- Labels joined: no
- Protected holdouts/API calls: no/zero
- Timing boundary: view preparation, native calls, score parsing, ID mapping and Top-5 packaging; model loading excluded and reported in the session metadata.
- Jina and PPLX model families were loaded sequentially in the same process because both cannot be resident on an 80-GiB device; all requests used GPU3.

## Timing and ratio table

| policy | p50 ms | p95 ms | p95 / Jina-full | 95% ratio CI | Pr(ratio ≤ 2) | alloc p95 MiB | valid/attempted |
|---|---:|---:|---:|---|---:|---:|---:|
| jina_full | 286.381 | 1171.476 | 1.0000 | [1.0000, 1.0000] | 1.0000 | 4432.377 | 360/360 |
| jina_two_stage_full | 350.395 | 1115.910 | 1.0301 | [0.8430, 1.2247] | 1.0000 | 2638.974 | 360/360 |
| jina_two_stage_pplxwindow | 323.302 | 729.539 | 0.6393 | [0.3938, 0.8977] | 1.0000 | 1951.463 | 357/360 |
| jina_pplx_budget | 151.613 | 245.224 | 0.2114 | [0.1346, 0.3115] | 1.0000 | 1946.185 | 357/360 |
| pplx_budget | 1606.550 | 2373.671 | 2.0486 | [1.3052, 3.0249] | 0.3301 | 51345.796 | 357/360 |
| pplx_two_stage | 3465.436 | 6782.089 | 5.8838 | [3.7039, 8.4437] | 0.0000 | 51323.595 | 357/360 |

## Failure accounting

Every failed request remains in the raw trace. The 12 failures are the same frozen `robotics:89` cohort record across the relevant policies: three exact D1 PPLX-source failures for `jina_two_stage_pplxwindow`, three unavailable E11 budget views for `jina_pplx_budget`, three unavailable E11 budget views for `pplx_budget`, and three dynamic-budget failures for `pplx_two_stage`. No retry or fill-in was used.

## Fallback basis

The all-request ratio basis adds the previously measured, frozen CPU BM25 Top-5 fallback time to failed request times only. That fallback trace is not a new inference run and is hash-recorded in `D2_RAW_TRACE_SHA256_MANIFEST_V2.json`. Successful request times are unchanged. The fallback measurement covers list extraction and first-five packaging, not a fresh BM25 search or model work.

## Limitations

- GPU3 was isolated at the recorded preflight (4 MiB used, 81,149 MiB free, 0% utilization); the session was not randomized across time and model families were sequentially loaded.
- Native runtimes do not expose a portable forward counter; `n_calls` and native-call stage records are retained, while physical forward counts are not claimed.
- These are matched selector timings, not sustainable production QPS or monetary cost estimates.
- The ratio bootstrap is a 10,000-draw query-cluster bootstrap with seed 731 over 120 IDs and three attempts per ID.
