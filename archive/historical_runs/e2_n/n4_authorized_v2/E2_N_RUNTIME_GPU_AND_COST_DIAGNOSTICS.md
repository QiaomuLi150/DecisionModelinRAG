# E2-N Runtime, GPU and Cost Diagnostics

These are selector-scoring measurements, not end-to-end production latency or a monetary invoice. Clef full traces are compatible E2-M1 traces reused with source hashes; BGE and Laya are fresh E2-N traces. GPU active-seconds and dollar rates were not measured consistently, so cost is `NOT_MEASURED`.

| Policy | n | p50 ms | p95 ms | Mean ms | Measured selector seconds | Input-token p50 | Input-token p95 | Physical forwards | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bge_pointwise | 4267 | 20.248819 | 27.839646 | 20.346194559409422 | 86.817212185 | 2775 | 4429 | 4267 | NOT_MEASURED |
| laya_pointwise | 4267 | 145.335503 | 156.558084 | 147.25294436700258 | 628.328313614 | NOT_AVAILABLE | NOT_AVAILABLE | 21335 | NOT_MEASURED |
| clef | 8534 | 1195.797493 | 2385.233657 | 1379.9477409022732 | 11776.47402086 | 6560 | 9923 | 8534 | NOT_MEASURED |
| qwen_direct | 4267 | 138.893602 | 167.974933 | 140.82225291492853 | 600.888553188 | 3998 | 4998 | 4267 | NOT_MEASURED |
| qwen06 | 4267 | 96.138834 | 116.464828 | 98.75540155589407 | 421.38929843899996 | 4052 | 5164 | 4267 | NOT_MEASURED |
| qwen4 | 4267 | 350.463335 | 423.147 | 356.41836100187487 | 1520.837146395 | 4052 | 5164 | 4267 | NOT_MEASURED |
| gte | 4267 | 26.343856 | 37.066503 | 28.524160772439654 | 121.712594016 | 2644 | 3870 | 4267 | NOT_MEASURED |
| minilm | 4267 | 14.349141 | 18.949711 | 14.878655635106632 | 63.487223595 | 2527 | 3876 | 4267 | NOT_MEASURED |
| jina35 | 4267 | 71.147352 | 98.799647 | 72.66866412561518 | 310.077189824 | NOT_AVAILABLE | NOT_AVAILABLE | 4267 | NOT_MEASURED |
| nimble | 4267 | 1107.193882 | 1688.06262 | 1230.9787326496366 | 5252.5862522160005 | NOT_AVAILABLE | NOT_AVAILABLE | 4267 | NOT_MEASURED |
| intern_decision | 4267 | 1126.4218343421817 | 2092.431896366179 | 1331.4077540349986 | 5015.41300944984 | 6408 | 6875 | 3767 | NOT_MEASURED |
| pplx_decider | 4267 | 1621.894107 | 2592.822396 | 1691.724214241856 | 7218.58722217 | NOT_AVAILABLE | NOT_AVAILABLE | 4267 | NOT_MEASURED |

No Jev or paid API calls occurred. Existing host services and unrelated GPU workloads were not restarted or interrupted.
