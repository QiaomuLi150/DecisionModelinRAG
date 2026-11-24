# R1 completion

Gate: `R1_COMPLETE_WITH_PPLX_FAILURES_RETAINED`

The frozen 120-query cohort was run three times per policy with one request at a time on GPU3. The exact counts and measurements are in `r1_serving_gate_v2/export/serving_r1.csv`.

| policy | attempts | valid | p50 ms | p95 ms | peak allocated p95 MiB |
|---|---:|---:|---:|---:|---:|
| pplx_budget | 360 | 357 | 1608.9815 | 2378.9435 | 51345.013 |
| pplx_two_stage | 360 | 357 | 3461.2010 | 6786.6465 | 51322.8462 |
| jina_full | 360 | 360 | 285.6645 | 1130.2588 | 1471.226 |
| jina_pplx_budget | 360 | 357 | 150.7860 | 244.7325 | 1947.4386 |

The three PPLX-related failures per policy are retained; no fallback was used. PPLX model load was approximately 26037 ms and Jina load approximately 1471 ms, reported separately from request latency. The marginal-cost columns record the prespecified fixed non-selection decision request interleaved with each PPLX selection: PPLX budget marginal p50 1485.032 ms and marginal peak allocation 1072.666 MiB.
