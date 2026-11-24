# E2-N2 Native Smoke Report

Status: `PASS_WITH_LIMITATIONS`

N2-1 evidence is reused from authenticated E2-N native smoke traces. No smoke request was repeated in N2 because the exact checkpoint revisions, native adapters, frozen CAL items, and raw trace provenance are unchanged. The resident Qwen service path remains non-promoted; the direct Qwen checkpoint trace is the promoted evidence.

| Model | Revision | Status | Rows | Physical-forward rows | Source |
|---|---|---|---:|---:|---|
| bge | NOT_RECORDED | PASS | 10 | NOT_RECORDED | E2_N_N1_BGE_SMOKE_SUMMARY.json |
| clef | NOT_RECORDED | PASS | 20 | NOT_RECORDED | E2_N_N1_CLEF_SMOKE_SUMMARY.json |
| Alibaba-NLP/gte-reranker-modernbert-base | f7481e6055501a30fb19d090657df9ec1f79ab2c | PASS | 10 | 10 | E2_N_N1_GTE_SMOKE_SUMMARY.json |
| internlm/Intern-Decision-4B | 0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd | PASS | 10 | 10 | E2_N_N1_INTERN_DECISION_SMOKE_SUMMARY.json |
| jinaai/jina-reranker-v3.5 | e8a93f33f0b22108f8c2364f8484ce3422552fbc | PASS | 10 | 10 | E2_N_N1_JINA35_SMOKE_SUMMARY.json |
| laya | NOT_RECORDED | PASS | 10 | NOT_RECORDED | E2_N_N1_LAYA_SMOKE_SUMMARY.json |
| cross-encoder/ms-marco-MiniLM-L6-v2 | 233902d25c440f23af6f7d6e94d2946bac0bee0a | PASS | 10 | 10 | E2_N_N1_MINILM_SMOKE_SUMMARY.json |
| bespokelabs/Bespoke-Nimble-9B-v2 | 4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe | PASS | 10 | 10 | E2_N_N1_NIMBLE_SMOKE_SUMMARY.json |
| perplexity-ai/pplx-decider-v1.1-27b | 14c9d90137962d300f3de95633f7b6535e80c5ce | PASS | 10 | 10 | E2_N_N1_PPLX_DECIDER_SMOKE_SUMMARY.json |
| Qwen/Qwen3-Reranker-0.6B | e61197ed45024b0ed8a2d74b80b4d909f1255473 | PASS | 10 | 10 | E2_N_N1_QWEN06_SMOKE_SUMMARY.json |
| Qwen/Qwen3-Reranker-4B | 22e683669bc0f0bd69640a1354a6d0aebcfeede5 | PASS | 10 | 10 | E2_N_N1_QWEN4_SMOKE_SUMMARY.json |
| Qwen/Qwen3-VL-Reranker-2B | 4bd860ac4f15ad1897a214615cccc700f8f71818 | PASS | 10 | NOT_RECORDED | E2_N_N1_QWEN_NATIVE_TELEMETRY_SUMMARY.json |
| qwen | NOT_RECORDED | PASS_WITH_LIMITATIONS | 10 | NOT_RECORDED | E2_N_N1_QWEN_SMOKE_SUMMARY.json |

Physical inference remains genuine for the promoted local-checkpoint traces. The source GPUs and exact trace hashes are recorded in the execution manifest; current N2 packaging allocated no GPU.
