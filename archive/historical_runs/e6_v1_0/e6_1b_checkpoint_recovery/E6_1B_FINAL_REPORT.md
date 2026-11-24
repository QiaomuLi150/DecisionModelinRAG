# DecisionRAG E6-1B Checkpoint Recovery and Native Adapter Validation

## Gate: `E6_2_READY_WITH_EXCLUSIONS`

This milestone validates native selection contracts and input-feasibility limits. It is not a ranking-quality benchmark.

## Frozen-input authentication
- E6-1 CAL rows: 120 unique SciFact questions.
- Canonical E6-1 input manifest SHA-256: `a1b006a79456aeba818fe3faf0059a56c1eb1c6adaf47483cac665845097f318` (expected `a1b006a79456aeba818fe3faf0059a56c1eb1c6adaf47483cac665845097f318`).
- Historical E5 Top-20 slate SHA-256: `be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7`.
- qrels, gold labels, reader outputs and reserved holdouts were not loaded.
- Exact per-model attempt schedule: 10 smoke + 240 CAL + 6 reversal diagnostics = 256; K15=125, K20=131.

## Model scorecard

| model | valid / attempts | K15 | K20 | K15 p50/p95 ms | K20 p50/p95 ms | order match | physical forwards | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| startlux_decision_4b | 256 / 256 | 125 | 131 | 654.1719678789377 / 779.7900665551424 | 785.9190693125129 / 1078.5173222422593 | 0/3 | 256 | VALIDATED |
| amos_1_small_2b | 250 / 256 | 125 | 125 | 335.3657405823469 / 414.0055062249303 | 470.82613594830036 / 555.0189603120089 | 0/3 | NOT_MEASURED | VALIDATED_WITH_CONTEXT_FAILURES |

## Contract and feasibility interpretation
- Both recovered checkpoints use official native typed categorical choice readouts. Candidate probabilities are aligned to the submitted option IDs and Top-5 is produced by deterministic descending-probability sorting with original-order tie breaking.
- This is a single-choice readout followed by Top-5 sorting, not joint-set optimization.
- StartLux: 256/256 native outputs valid. Its bundled checker tests a stale Transformers marker; direct binding inspection and CUDA kernel smoke calls verified causal-conv1d and FLA dispatch. The documented guard override was recorded; no reference PyTorch fallback was used.
- Amos: 250/256 valid. Six scheduled attempts exceeded the official decoder's 16,384-token strict limit, covering five distinct query IDs; `scifact:100` failed in both the smoke and CAL K20 attempts. They were retained as explicit failures; no truncation or retry with altered inputs occurred. Amos used the official local decoder path, rotations=1, and no state cache.
- The six Amos failures are context-feasibility exclusions, not model-quality failures. The five distinct affected query IDs are listed in `E6_1B_FAILURES.csv`.
- Amos native API attempts are recorded for every scheduled request, but the official DecoderEngine invokes its underlying decoder through an internal path not observed by the runner's top-level forward hook. Therefore its physical-forward count is `NOT_MEASURED`, not zero; the 250 successful native calls establish a lower bound on completed model forwards.

## Inference accounting
- Load time, peak allocated/reserved GPU memory, native API calls, observed hook events and K15/K20 valid counts are in `E6_1B_INFERENCE_ACCOUNTING.csv`.
## Order diagnostics
- Exact set matches, Jaccard overlap and changed-selection counts are in `E6_1B_ORDER_DIAGNOSTICS.csv`; canonical ordering remains the primary protocol.

## Readiness
- `E6_2_READY_WITH_EXCLUSIONS` is recommended because Amos is native-contract valid where feasible but has six K20 context-limit failures among 131 scheduled K20 attempts (five distinct frozen query IDs).
- StartLux may be included; Amos may be included only with its prespecified context-feasibility accounting, or excluded from any E6-2 population that requires complete K20 coverage.
- No quality conclusion is authorized from this CAL feasibility work.

## Restrictions verified
- E6-2 was not started. No Jev, paid API, training, reader generation, retrieval optimization or reserved-holdout access.
- GPU 5 was used sequentially after UUID/occupancy audit and released after each model; no unrelated process was evicted.

## Failure accounting
- Failed scheduled attempts: 6.
- Distinct affected query IDs: 5 (scifact:100, scifact:1005, scifact:1161, scifact:1196, scifact:124).
- All failures were official strict context-length rejections before model forward; no timeout, OOM, invalid choice schema, duplicate selection, or fallback was observed.
