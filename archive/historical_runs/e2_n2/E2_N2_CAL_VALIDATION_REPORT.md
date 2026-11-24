# E2-N2 CAL Validation Report

Status: `PASS_WITH_LIMITATIONS`

N2-2 is complete by authenticated reuse of the exact E2-N CAL traces. Configurations were frozen before the source N3 run; no outcome-based tuning was performed and no gold labels entered inference. Clef has separate pointwise and native-joint rows. Historical stale text claiming missing new checkpoints is superseded by the current local-revision and trace-hash inventory; the raw source report is preserved unchanged.

| Model | Status | Rows | OK | Physical | Input hashes | Source |
|---|---|---:|---:|---:|---:|---|
| bge | PASS | 30 | 30 | 30 | 30 | E2_N_N2_BGE_CAL_SUMMARY.json |
| clef | PASS | 60 | 60 | 60 | 60 | E2_N_N2_CLEF_CAL_SUMMARY.json |
| Alibaba-NLP/gte-reranker-modernbert-base | PASS | 30 | 30 | 30 | NOT_RECORDED | E2_N_N2_GTE_CAL_SUMMARY.json |
| internlm/Intern-Decision-4B | PASS | 30 | 30 | 30 | NOT_RECORDED | E2_N_N2_INTERN_DECISION_CAL_SUMMARY.json |
| jinaai/jina-reranker-v3.5 | PASS | 30 | 30 | 30 | NOT_RECORDED | E2_N_N2_JINA35_CAL_SUMMARY.json |
| laya | PASS | 30 | 30 | 30 | 30 | E2_N_N2_LAYA_CAL_SUMMARY.json |
| cross-encoder/ms-marco-MiniLM-L6-v2 | PASS | 30 | 30 | 30 | NOT_RECORDED | E2_N_N2_MINILM_CAL_SUMMARY.json |
| bespokelabs/Bespoke-Nimble-9B-v2 | PASS | 30 | 30 | 30 | NOT_RECORDED | E2_N_N2_NIMBLE_CAL_SUMMARY.json |
| perplexity-ai/pplx-decider-v1.1-27b | PASS | 30 | 30 | 30 | NOT_RECORDED | E2_N_N2_PPLX_DECIDER_CAL_SUMMARY.json |
| Qwen/Qwen3-Reranker-0.6B | PASS | 30 | 30 | 30 | NOT_RECORDED | E2_N_N2_QWEN06_CAL_SUMMARY.json |
| Qwen/Qwen3-Reranker-4B | PASS | 30 | 30 | 30 | NOT_RECORDED | E2_N_N2_QWEN4_CAL_SUMMARY.json |
| Qwen/Qwen3-VL-Reranker-2B | PASS | 30 | 30 | 30 | NOT_RECORDED | E2_N_N2_QWEN_DIRECT_CAL_SUMMARY.json |

Jev calls: `0`; paid API calls: `0`; downloads during N2 packaging: `0`.
