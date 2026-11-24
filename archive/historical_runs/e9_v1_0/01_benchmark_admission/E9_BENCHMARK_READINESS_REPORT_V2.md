# E9 TREC RAG 2024 fallback admission

Gate: `E9_BENCHMARK_ADMISSION_BLOCKED`

The 2024 fallback was checked once after TREC 2025 failed before model-outcome inspection. Its exact 301-topic test-aligned published candidate artifact contains complete Top-20 text views, but the prespecified judged-coverage floor is not met.

| Admission quantity | Value |
|---|---:|
| official topics | 301 |
| pool rows | 301 |
| candidate-complete topics | 301 |
| topics with positive qrel in Top-20 | 84 |
| median judged fraction in Top-20 | 0.0 |

## Integrity boundary

- qrels were used only for fallback admission and were not placed in model inputs.
- No selector output or quality outcome was inspected.
- The candidate file is a published multi-stage RankZephyr pipeline pool, so the narrow conditional post-reranked-pool interpretation would apply if it had passed coverage.
- No further candidate source search or benchmark substitution was performed.
