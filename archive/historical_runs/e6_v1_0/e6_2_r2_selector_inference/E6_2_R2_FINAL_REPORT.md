# DecisionRAG E6-2 R2 Native Selector Inference

## Gate: `E6_2_R2_SELECTIONS_FROZEN_WITH_EXCLUSIONS`

This run executed native selector inference only on the frozen, gold-blind SciFact EXPLORATORY Top-20 slates. No relevance metrics were calculated.

## Population and isolation
- 277 frozen questions × 8 models = 2,216 scheduled attempts; actual attempts: 1939.
- Candidate slate hash set is identical across models: one frozen R1 slate hash per query.
- qrels, gold labels, reference answers, RESERVED content, readers and R3 outputs were not loaded.
- No fallback to retrieval Top-5 or another model was used.

## Model execution

| Model | Attempts | Valid | Failed | Success p50 ms | Success p95 ms | Status |
|---|---:|---:|---:|---:|---:|---|
| pplx_decider | 277 | 277 | 0 | 1721.988865 | 1727.696602 | VALIDATED |
| liquid_d1_3b | 277 | 277 | 0 | 328.893755 | 352.3829332 | VALIDATED |
| startlux_decision_4b | 0 | 0 | 0 | None | None | MODEL_LOAD_BLOCKED |
| amos_1_small_2b | 277 | 194 | 83 | 1957.8583310358226 | 2379.084634082392 | VALIDATED_WITH_EXCLUSIONS |
| jina_v3_5 | 277 | 277 | 0 | 217.385127 | 343.8695444 | VALIDATED |
| qwen3_reranker_4b | 277 | 277 | 0 | 613.105536 | 962.7216275999999 | VALIDATED |
| bge_reranker_v2_m3 | 277 | 277 | 0 | 460.695177 | 668.8252814 | VALIDATED |
| qwen3_reranker_0_6b | 277 | 277 | 0 | 200.819996 | 305.2726675999999 | VALIDATED |

## Accounting limitations
- Physical forwards are retained only when a native wrapper exposed positive observed hook events; otherwise the value is `NOT_MEASURED`.
- Latencies are selector-attempt diagnostics, not E6-3 serving throughput or production cost measurements.
- Decision-model probability sorting is a derived Top-5 policy, not joint-set optimization.

## R3 boundary
Native selections and explicit failures are committed. R3 may consume this bundle for offline relevance scoring only after separate owner authorization.

## Test and provenance closure

- Focused R2 tests: **11/11 passed**.
- Full authoritative regression: **225/225 passed**.
- All new artifact hashes are recorded in `E6_2_R2_PROVENANCE_MANIFEST.json`.
- StartLux remains explicitly excluded because its native adapter could not load in the active environment (`ModuleNotFoundError: causal_conv1d`); no replacement or fallback was used.
