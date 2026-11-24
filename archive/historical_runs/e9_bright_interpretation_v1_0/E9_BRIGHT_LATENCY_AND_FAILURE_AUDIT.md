# E9-BRIGHT latency and failure audit

No new serving benchmark was run. All values below are recomputed from frozen native records.

## Measurement boundary

The E9 runner timer starts after a CUDA synchronization and peak-memory reset, immediately before request construction/native scoring, and ends after score extraction and Top-5 packaging. Model loading and candidate lookup are outside the timer. Input token counts are reported only where the frozen records actually contain them.

## Stratified distributions

| Model | Success n | Success p50 ms | Success p95 ms | Failure n | Failure p50 ms | Failure p95 ms | All p50 ms | All p95 ms | Input-token observations | p95 memory MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pplx_decider | 611 | 1764.997916 | 2677.1878895 | 773 | 55.793531 | 98.77215899999999 | 82.879796 | 2419.5117656499997 | 0 | 51334.747314453125 |
| jina_v3_5 | 1384 | 382.141571 | 1422.2969771999979 | 0 | None | None | 382.141571 | 1422.2969771999979 | 1384 | 4497.824902343747 |
| liquid_d1_3b | 1384 | 602.7278779999999 | 1734.2709429499998 | 0 | None | None | 602.7278779999999 | 1734.2709429499998 | 1384 | 9210.712890625 |
| qwen3_reranker_0_6b | 1382 | 322.25375399999996 | 1505.1584703000003 | 2 | 16191.298291 | 29036.988318999996 | 323.2234465 | 1524.2218250999988 | 0 | 33618.5220703125 |

PPLX's approximately 82.9 ms all-attempt median is therefore not a successful-inference latency claim: 773 attempts were rejected by the native 8,192-token question-branch limit. Qwen0.6B had two allocator-OOM failures. Different GPUs were used (PPLX GPU 1, Liquid GPU 3, Jina GPU 5, Qwen GPU 6 in the frozen run manifests), so these results do not support matched-hardware latency, sustainable QPS, or monetary-cost conclusions.

Input-token completeness is model-dependent: PPLX and Qwen records do not provide a usable token-count field for all attempts, so missing token statistics remain NOT_VERIFIABLE_FROM_FROZEN_RECORDS rather than imputed.
