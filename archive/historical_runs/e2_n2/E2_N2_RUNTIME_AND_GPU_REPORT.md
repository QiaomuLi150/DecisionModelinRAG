# E2-N2 Runtime and GPU Report

Status: `PASS_WITH_LIMITATIONS`

This report summarizes authenticated E2-N physical traces; it is not a new serving run. Current N2 packaging used zero GPUs. The source traces used GPU IDs recorded per row and were completed under prior authorized E2-N execution.

| Policy | Valid timed rows | p50 ms | p95 ms | Source GPU IDs |
|---|---:|---:|---:|---|
| `bge_pointwise` | 4267 | 20.249 | 27.840 | `6` |
| `gte` | 4267 | 26.344 | 37.067 | `0` |
| `intern_decision` | 3767 | 1126.422 | 2092.432 | `6` |
| `jina35` | 4267 | 71.147 | 98.800 | `0` |
| `laya_pointwise` | 4267 | 145.336 | 156.558 | `0` |
| `minilm` | 4267 | 14.349 | 18.950 | `0` |
| `nimble` | 4267 | 1107.194 | 1688.063 | `6` |
| `pplx_decider` | 4267 | 1621.894 | 2592.822 | `0` |
| `qwen06` | 4267 | 96.139 | 116.465 | `6` |
| `qwen4` | 4267 | 350.463 | 423.147 | `0` |
| `qwen_direct` | 4267 | 138.894 | 167.975 | `0,6` |

The live GPU snapshot was used only for resource observation. GPU0 and GPU5 were free at inspection; GPUs1/2/4/6/7 had resident allocations and were not preempted. No resident service was restarted or modified. This is not a production-QPS or SLO study, and no monetary GPU cost is inferred.
