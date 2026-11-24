# DecisionRAG E3-C — C5 Owner-Review Closure

Verdict: `E3_C_OWNER_REVIEW_READY`

Scientific status: `E3_C_PASS_WITH_LIMITATIONS`. C0-C4 and C6 remain valid; C5 is now `C5_PASS_GPU0_ISOLATED`. GPU0 passed the isolation guard before each model, no unrelated process was interrupted, and it was empty after cleanup. The complete 300-query Top-20→Top-5 task was physically measured for both validated native policies.

## Measured C5 results

| Model | n | valid | load ms | warmup ms | p50 ms | p95 ms | p99 ms | 1s hit | 3s hit | peak allocated MiB | forwards |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| liquid_d1_3b | 300 | 300 | 12029.11 | 2182.99 | 172.16 | 649.71 | 700.25 | 1.0000 | 1.0000 | 7780.0 | 300 |
| zerank_2 | 300 | 300 | 3893.28 | 814.55 | 307.58 | 970.19 | 1066.20 | 0.9700 | 1.0000 | 10556.4 | 900 |

Full machine-readable request traces are in `E3_C_SELECTOR_LATENCY_RAW.jsonl`; quality results remain in `E3_C_DATASET_QUALITY_SUMMARY.csv` and `E3_C_PRIMARY_PAIRED_STATISTICS.csv`. The timing study is serialized single-client only. It does not establish sustainable QPS, kernel-seconds, or monetary cost. E3-0 remains `E3_0_HOLDOUT_RESERVATION_PENDING_METADATA`.

No fresh holdout, Jev, paid API, training, multimodal experiment, retrieval rebuild or production-service change occurred.
