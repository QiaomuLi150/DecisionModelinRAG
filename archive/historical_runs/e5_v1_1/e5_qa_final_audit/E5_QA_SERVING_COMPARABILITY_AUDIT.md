# E5-QA Serving Comparability Audit

The E5-D raw traces support a same-GPU, concurrency-one, warmed complete-selector comparison. They do not support production QPS, dollar cost, or load-capacity claims.

| policy | n | p50 ms | p95 ms | p95/p50 | p95≤1s | p95≤3s | physical forwards | peak allocator bytes | actual input hashes |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|
| pplx_decider | 200 | 1320.085241 | 1616.622862 | 1.225 | False | True | 200 | 53188925952 | 200 |
| jina35 | 200 | 62.244847 | 88.790322 | 1.426 | True | True | 200 | 1551129600 | 200 |

GPU UUID: `<REDACTED_GPU_UUID>`. Timing boundary: `wrapper preparation + native scoring + Top-5 sorting + packaging`.

PPLX was measured before Jina; no randomized/alternating repeated-order data exist, so time drift and thermal/order confounding cannot be corrected.

Peak allocator bytes are model-framework allocator observations, not total physical GPU memory. PPLX used 53,188,925,952 bytes and Jina 1,551,129,600 bytes in the frozen summaries. No monetary conversion was attempted.
