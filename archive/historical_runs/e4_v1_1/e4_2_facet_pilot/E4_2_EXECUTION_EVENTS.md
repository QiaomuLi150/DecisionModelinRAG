# E4-2 Execution Events

## Pre-pilot full-view smoke failure

The initial 12-case smoke attempt used the unbounded E3-D source text view. Jina produced CUDA out-of-memory messages on shared physical GPU 2 before a scientific row was accepted. The process was interrupted; no answer or selection from that attempt is used.

## Interpreter diagnostic

One preliminary full-phase launch used the restricted system Python, which had no Torch installation. It produced `MODEL_LOAD_ERROR` rows before model loading and no inference. That diagnostic trace is preserved as `E4_2_NATIVE_SCORING_RAW_failed_system_python.jsonl` and is excluded from the accepted raw trace. The accepted pilot used the project shared CUDA-enabled environment.

The protocol was then frozen as `common_word_capped_256_v1`, applied identically to all five policies. Historical E3-D baseline reuse is rejected unless the complete E4-2 candidate-view/input hashes match this contract. All baselines will therefore be re-inferred under the harmonized view. This is a pre-pilot context-feasibility correction, not an outcome-driven change.
