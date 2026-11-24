# E7 main native-contract reference report

The primary four policies were executed through the independent E7 native path. It constructs full candidate text, hashes the native payload, reads the model-native response, restores scores to the submitted candidate positions, and applies a local stable Top-5 mapping.

## Contract findings

- `pplx_decider`: PASS_WITH_LIMITATIONS; 330/400 valid in E7-B; unique canonical Top-5=True.
- `startlux_decision_4b`: PASS; 400/400 valid in E7-B; unique canonical Top-5=True.
- `jina_v3_5`: PASS; 400/400 valid in E7-B; unique canonical Top-5=True.
- `qwen3_reranker_0_6b`: PASS; 400/400 valid in E7-B; unique canonical Top-5=True.

The PPLX Top-5 policy is a derived sort of one categorical choice probability per candidate, not a learned joint five-set objective. Jina index responses are restored by returned original position. Ties are resolved by submitted position and then ID. Missing `text` is a hard error in the E7 path.
