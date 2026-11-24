# E2-N2 Checkpoint and License Audit (N2-0)

Status: `N2_0_COMPLETE_AWAITING_EXPLICIT_E2_N2_AUTHORIZATION`

This is an offline inventory. No checkpoint was loaded and no model inference, network request, GPU allocation, service request, retrieval, indexing, Jev, paid API, or fresh-holdout access occurred.

Exact proposed roster entries inspected: **12**; local snapshots present: **12**; absent: **0**.

| Model | Revision | Local snapshot | Adapter source | License/card evidence | Context evidence |
|---|---|---|---|---|---|
| `bespokelabs/Bespoke-Nimble-9B-v2` | `4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe` | `True` | `True` | `present; card=present` | `{"schema_config.json:max_length": 8192, "tokenizer_config.json:model_max_length": 262144, "training_summary.json:max_length": 8192}` |
| `internlm/Intern-Decision-4B` | `0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd` | `True` | `True` | `present; card=present` | `{"config.json:text_config.max_position_embeddings": 262144, "tokenizer_config.json:model_max_length": 262144}` |
| `perplexity-ai/pplx-decider-v1.1-27b` | `14c9d90137962d300f3de95633f7b6535e80c5ce` | `True` | `True` | `present; card=present` | `{"config.json:text_config.max_position_embeddings": 262144, "tokenizer_config.json:model_max_length": 262144}` |
| `Qwen/Qwen3-Reranker-0.6B` | `e61197ed45024b0ed8a2d74b80b4d909f1255473` | `True` | `True` | `not found locally; card=present` | `{"config.json:max_position_embeddings": 40960, "tokenizer_config.json:model_max_length": 131072}` |
| `Qwen/Qwen3-Reranker-4B` | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` | `True` | `True` | `not found locally; card=present` | `{"config.json:max_position_embeddings": 40960, "tokenizer_config.json:model_max_length": 131072}` |
| `jinaai/jina-reranker-v3.5` | `e8a93f33f0b22108f8c2364f8484ce3422552fbc` | `True` | `True` | `not found locally; card=present` | `{"config.json:max_position_embeddings": 131072, "tokenizer_config.json:model_max_length": 131072}` |
| `Alibaba-NLP/gte-reranker-modernbert-base` | `f7481e6055501a30fb19d090657df9ec1f79ab2c` | `True` | `True` | `not found locally; card=present` | `{"config.json:max_position_embeddings": 8192, "tokenizer_config.json:max_length": 512, "tokenizer_config.json:model_max_length": 8192}` |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | `233902d25c440f23af6f7d6e94d2946bac0bee0a` | `True` | `True` | `not found locally; card=present` | `{"config.json:max_position_embeddings": 512, "tokenizer_config.json:model_max_length": 512}` |
| `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | `True` | `True` | `not found locally; card=present` | `{"config.json:max_position_embeddings": 8194, "tokenizer_config.json:model_max_length": 8192}` |
| `Qwen/Qwen3-VL-Reranker-2B` | `4bd860ac4f15ad1897a214615cccc700f8f71818` | `True` | `True` | `not found locally; card=present` | `{"config.json:text_config.max_position_embeddings": 262144, "tokenizer_config.json:model_max_length": 262144}` |
| `Cloudflare/clef-flash` | `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` | `True` | `True` | `present; card=present` | `{"config.json:text_config.max_position_embeddings": 262144, "tokenizer_config.json:model_max_length": 262144}` |
| `convaiinnovations/laya` | `7b928d828b7b0e022f929d9bd2e44165aa270148` | `True` | `True` | `not found locally; card=present` | `{}` |

A local snapshot is not evidence that native inference has passed. License fields are reported as local-file evidence only; no external model-card or license verification was performed in N2-0.

## Adapter compilation

`PASS`

```json
{
  "status": "PASS",
  "files": [
    {
      "path": "scripts/e2_m1_experiment.py",
      "status": "PASS",
      "stderr": ""
    },
    {
      "path": "scripts/e2_n_intern_decision.py",
      "status": "PASS",
      "stderr": ""
    },
    {
      "path": "scripts/e2_n_nimble.py",
      "status": "PASS",
      "stderr": ""
    },
    {
      "path": "scripts/e2_n_pplx.py",
      "status": "PASS",
      "stderr": ""
    },
    {
      "path": "scripts/e2_n_qwen_direct.py",
      "status": "PASS",
      "stderr": ""
    },
    {
      "path": "scripts/e2_n_specialist_runner.py",
      "status": "PASS",
      "stderr": ""
    },
    {
      "path": "scripts/typed_model_audit.py",
      "status": "PASS",
      "stderr": ""
    }
  ]
}
```
