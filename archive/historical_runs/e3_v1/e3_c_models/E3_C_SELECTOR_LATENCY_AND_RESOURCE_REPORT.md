# E3-C C5 Selector Latency and Resource Report

Status: `C5_PASS_GPU0_ISOLATED`.

The previously blocked C5 gate was completed on approved physical GPU0 after a pre-run guard found no GPU0 compute process and at least 70 GiB free. The workload is the immutable 300-query E2-S/P cohort, with exact question hashes and ordered Top-20 candidate IDs checked before timing. Models were run one at a time; warmup and model-load time are reported separately. Each timed request includes canonical input construction, tokenization/preprocessing, native inference, physical forwards, score extraction and deterministic Top-5 postprocessing.

| Model | n | valid | load ms | warmup ms | p50 ms | p95 ms | p99 ms | 1s hit | 3s hit | peak allocated MiB | forwards |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| liquid_d1_3b | 300 | 300 | 12029.11 | 2182.99 | 172.16 | 649.71 | 700.25 | 1.0000 | 1.0000 | 7780.0 | 300 |
| zerank_2 | 300 | 300 | 3893.28 | 814.55 | 307.58 | 970.19 | 1066.20 | 0.9700 | 1.0000 | 10556.4 | 900 |

The 1-second and 3-second columns are descriptive single-client scenario hit rates, not sustainable-QPS claims. `serialized_throughput_rps` is reported in the machine-readable summary only and is not production capacity. No GPU-seconds or dollar cost is inferred. The prior blocked artifacts were copied to versioned `*_PRE_C5_BLOCKED` files before this completion.

GPU isolation, memory, exact workload hashes, post-run guards and model revisions are recorded in `E3_C_C5_GPU0_ISOLATION_AUDIT.json`. Jev and paid API calls remain zero.
