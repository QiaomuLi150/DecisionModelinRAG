# V1.5 C1 GPU3 Offline Analysis

Status: COMPLETE_WITH_SERIAL_AND_BATCHED_QUALITY_POPULATIONS

The frozen BRIGHT Top-20 manifest was validated before the gold manifest was opened. All 21 new Jina policy/order shards contain 1,384 records and preserve 20 canonical candidate IDs per query. Batched traces are valid for selection-quality analysis but their elapsed time is not isolated-request latency.

## Primary contrasts

| contrast | n | query effect | domain effect | 95% domain-bootstrap CI | raw sign-flip p | Holm p |
|---|---:|---:|---:|---|---:|---:|
| pplx_two_stage - jina_two_stage_full (original) | 1384 | 0.023928 | 0.024859 | [0.010896, 0.040104] | 0.004883 | 0.014648 |
| pplx_two_stage - jina_two_stage_full (mean six shuffled) | 1384 | 0.019266 | 0.019522 | [0.004715, 0.035860] | 0.035156 | 0.070312 |
| pplx_two_stage - jina_two_stage_pplxtext (original) | 1384 | 0.031802 | 0.033218 | [0.018442, 0.049095] | 0.000977 | 0.003906 |
| jina_two_stage_full - jina_full (original) | 1384 | 0.002475 | 0.001870 | [-0.002017, 0.005468] | 0.358398 | 0.358398 |

## Measurement boundary

Serial rows report complete selector elapsed milliseconds. Batched rows report batch elapsed milliseconds and `timing_valid=false`; they are not used as isolated p50/p95 latency claims. GPU3 was released after inference.

## Artifacts

- `export/c1_perquery.csv`
- `export/c1_summary.csv`
- `export/c1_domain_summary.csv`
- `export/c1_contrasts.csv` and `c1_contrasts.json`
- `export/c1_order_stability.csv`
- `export/C1_GOLD_JOIN_RECEIPT.json`
- `export/C1_PROVENANCE_MANIFEST.json`
