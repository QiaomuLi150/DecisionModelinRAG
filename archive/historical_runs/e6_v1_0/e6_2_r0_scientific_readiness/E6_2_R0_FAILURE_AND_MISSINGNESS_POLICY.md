# E6-2 R0 Failure and Missingness Policy

Each query-policy attempt has one terminal status: `success`, `context_limit`, `timeout`, `oom`, `invalid_output`, `service_error`, `missing_input`, or `not_run`. A valid success must contain exactly five unique canonical IDs in the submitted candidate view. Invalid or missing output is retained in unconditional denominators and reported by category; it is not silently dropped.

Amos's six observed K20 context-limit failures are preserved. No truncation, fallback, adaptive prompt or decoder change is authorized in R0. Physical-forward counts marked `NOT_MEASURED` remain missing, not zero and not imputed.

SciFact no-positive and sparse-judgment cases are retained. Conventional unjudged-as-zero nDCG is labeled as an evaluator convention, with judged-coverage sensitivity. GaRAGe UNKNOWN/empty labels do not become negative support labels.
