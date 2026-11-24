# E5-D Final Serving Scientific Report

Status: `D_ANALYSIS_PASS`.

Both policies were measured at concurrency 1 on the same authorized GPU5 UUID, with 12 CAL smoke requests, 20 warmups, and one timed native complete-selector request for each representative question. The measured boundary includes wrapper input preparation, native scoring, Top-5 sorting and output packaging.

| policy | assigned | valid | p50 ms | p95 ms | p95 <=1s | p95 <=3s | physical forwards | peak allocated |
|---|---:|---:|---:|---:|---|---|---:|---:|
| pplx_decider | 200 | 200 | 1320.0852405 | 1616.62286185 | False | True | 200 | 53188925952 |
| jina35 | 200 | 200 | 62.2448475 | 88.79032159999997 | True | True | 200 | 1551129600 |

GPU5 was selected because it was compute-idle at the pre-load gate. GPU6 was occupied and was not substituted. Before/after snapshots and unrelated process lists are retained in the per-policy summaries. No process was terminated.

No sustainable QPS, monetary cost, noninferiority or production replacement claim is made.
