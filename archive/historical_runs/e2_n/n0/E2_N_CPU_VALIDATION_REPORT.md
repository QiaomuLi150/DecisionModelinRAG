# E2-N CPU Validation Report

Execution scope: CPU-only static and synthetic checks. No checkpoint was loaded; no model, GPU, network, Qdrant, service, Jev, paid API, retrieval, indexing, or holdout request was made.

Focused N0 checks: **22 PASS, 0 FAIL**.
Existing repository regression suite: **PASS**, 65 tests collected (CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 /usr/bin/python3 -m pytest -q); see the machine-readable execution manifest for the captured tail.

| Check | Status | Detail |
|---|---|---|
| `top5_member_and_cardinality` | `PASS` | ['d01', 'd02', 'd03', 'd04', 'd05'] |
| `stable_original_rank_tie_break` | `PASS` | ['d01', 'd02', 'd03', 'd04', 'd05'] |
| `duplicate_candidate_rejection` | `PASS` | synthetic duplicate detected before scoring |
| `out_of_slate_rejection` | `PASS` | synthetic selected ID is outside frozen slate |
| `nonfinite_score_rejection` | `PASS` | NaN is invalid |
| `score_direction_declared` | `PASS` | higher_is_more_relevant required for comparable independent/joint binary scores |
| `categorical_denominator_is_local` | `PASS` | LOCAL_CATEGORICAL probabilities cannot be pooled across quartets |
| `top5_exactly_five` | `PASS` | five unique in-slate IDs |
| `nonempty_question` | `PASS` | 'What is the answer?' |
| `blank_question` | `PASS` | '' |
| `actual_input_hash_determinism` | `PASS` | 31a3da6a939fca24ab58595e5b7611ccba553e2bdc022a198c41f820bf9539a8 |
| `gold_blind_input_schema` | `PASS` | synthetic input contains no gold fields |
| `frozen_top20_contract` | `PASS` | E2-N uses existing immutable Track A/Track B Top-20 slates; no retrieval/index build |
| `required_trace_fields_declared` | `PASS` | actual_input_sha256,cache_reuse,candidate_ids,gold_in_inference_input,gpu_id,model_id,physical_forward_calls,query_id,response_hash,revision,score_contract,status |
| `physical_forward_count_not_api_count` | `PASS` | schema carries physical_forward_calls independently |
| `cache_provenance_required` | `PASS` | cache_reuse plus exact-input provenance required for any reuse |
| `gold_leakage_fail_closed` | `PASS` | gold_in_inference_input must be false |
| `selection_module_exists_and_parses` | `PASS` | ./src/decisionrag/selection.py |
| `selection_module_has_no_network_client_literal` | `PASS` | static inspection only; historic M1 script is not executed |
| `selection_is_model_neutral` | `PASS` | ['hashlib', 'json', 're', 'unicodedata', 'random', '__future__', 'dataclasses', 'typing'] |
| `historic_m1_adapter_exists_and_parses` | `PASS` | ./scripts/e2_m1_experiment.py |
| `historic_m1_adapter_has_no_network_client_literal` | `PASS` | static inspection only; historic M1 script is not executed |

## Local checkpoint inventory

Exact local snapshots were inspected by revision and file manifest; model weights were not loaded. Large weight files retain their Hugging Face content-addressed blob names rather than being rehashed during this CPU-only preflight.

| Model | Revision | Snapshot bytes | Files |
|---|---|---:|---:|
| `Cloudflare/clef-flash` | `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` | 19,083,378,312 | 17 |
| `convaiinnovations/laya` | `7b928d828b7b0e022f929d9bd2e44165aa270148` | 2,372,594,929 | 39 |
| `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | 2,293,259,337 | 7 |
| `Qwen/Qwen3-VL-Reranker-2B` | `4bd860ac4f15ad1897a214615cccc700f8f71818` | 4,271,056,557 | 20 |

No newly proposed exact official checkpoint was found locally. Existing nearby checkpoints (for example MiniLM-L2/L4 and sentence-transformers MiniLM-L6) were not substituted for the exact proposed models.
