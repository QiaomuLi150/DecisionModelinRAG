# DecisionRAG E6-3 R1 — Paired Downstream RAG Reader Evaluation

## Final gate

`E6_3_R1_READER_OUTPUTS_FROZEN_READY`

Reader outputs were frozen without answer-quality scoring. The R1 run used the authenticated resident Qwen3-VL-8B-Instruct service and the exact frozen R5 selector outputs.

## Population and accounting

- Representative primary cohort: **200** questions.
- Disagreement diagnostic cohort: **60** questions.
- No-positive diagnostic cohort: **64** questions.
- Unique union: **313** questions.
- Five logical evidence-policy arms: **1565** unique policy-question slots.
- Physical fresh reader calls: **1565**; failures: **0**; application-cache reuse: **0**.

| policy | logical slots | successful calls | failures |
|---|---:|---:|---:|
| pplx_decider | 313 | 313 | 0 |
| jina_v3_5 | 313 | 313 | 0 |
| startlux_decision_4b | 313 | 313 | 0 |
| qwen3_reranker_0_6b | 313 | 313 | 0 |
| original_fixed_pool_prefix5 | 313 | 313 | 0 |

## Structural validity only

Every successful response preserves raw text, response JSON, raw-response hash, actual model-input hash, selected canonical IDs and deterministic parsed answer/citation/abstention fields. The R1 parser does not correct outputs or invent citation IDs. No answer-reference agreement, citation relevance, grounding, abstention quality or factual correctness was computed.

## Limitations

The resident vLLM service does not expose per-request physical forward counts or definitive GPU attribution. GPU snapshots and service identity are preserved separately. R4C/R5 GaRAGe source independence remains limited, and the 64 no-positive cases are not certified globally unanswerable. The representative cohort is the only population eligible for the primary R2 contrast; diagnostic cohorts must remain separate.

## Next gate

The frozen output package is ready for separately authorized E6-3 R2 offline scoring. E6-3 R2, E6-4 and E6-5 were not started. No Jev or paid API calls occurred.
