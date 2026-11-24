# DecisionRAG E6-1 Native Adapter Validation

## Gate: `E6_1_CONDITIONAL_OWNER_REVIEW_READY`

This milestone validates native selection contracts only. It is not a ranking-quality benchmark and no qrels, gold labels, E5 outcomes, readers, Jev or paid APIs were used.

## Frozen CAL protocol
- Questions: 120 SciFact CAL records, seed 731.
- Candidate construction: deterministic SHA-256 order over the full public SciFact corpus; qrels were not loaded.
- Candidate counts: K=15 and K=20; exact Top-5 output required.
- Historical E5 Top-20 slate hash recorded: `be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7`.
- Frozen input manifest hash: `a1b006a79456aeba818fe3faf0059a56c1eb1c6adaf47483cac665845097f318`.

## Model results

| Model | checkpoint | valid / attempts | Top15 | Top20 | p50 ms | p95 ms | forwards | order-set checks | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pplx_decider | True | 256 / 256 | 125 | 131 | 1701.8271445 | 1741.91456 | 256 | 0/3 | VALIDATED_WITH_ORDER_SENSITIVITY |
| liquid_d1_3b | True | 256 / 256 | 125 | 131 | 300.7221085 | 338.23189325 | 256 | 0/3 | VALIDATED_WITH_ORDER_SENSITIVITY |
| startlux_decision_4b | False | 0 / 1 | 0 | 0 | None | None | 0 | 0/0 | BLOCKED_MISSING_CHECKPOINT |
| amos_1_small_2b | False | 0 / 1 | 0 | 0 | None | None | 0 | 0/0 | BLOCKED_MISSING_CHECKPOINT |
| jina_v3_5 | True | 256 / 256 | 125 | 131 | 183.055657 | 259.62515525 | 256 | 0/3 | VALIDATED_WITH_ORDER_SENSITIVITY |
| qwen3_reranker_4b | True | 256 / 256 | 125 | 131 | 561.6329935 | 770.7012012499999 | 643 | 3/3 | VALIDATED |
| bge_reranker_v2_m3 | True | 256 / 256 | 125 | 131 | 403.8609575 | 547.2926055 | 643 | 3/3 | VALIDATED |
| qwen3_reranker_0_6b | True | 256 / 256 | 125 | 131 | 183.9206045 | 258.957725 | 643 | 3/3 | VALIDATED |

## Contract interpretation
- PPLX uses its native categorical choice readout; probabilities are candidate-aligned and normalized by the checkpoint implementation.
- Liquid uses one native `system_one` choice over named candidate options. This validates a candidate-aligned readout, not cross-head calibration or set-complementarity.
- Jina uses its native listwise `rerank` interface and maps returned indices back to canonical IDs.
- Qwen3 and BGE use their official pairwise/cross-encoder scoring interfaces; raw higher-is-better scores are used only within each candidate view.
- All outputs are checked for exactly five unique IDs inside the submitted candidate list. Missing or invalid outputs remain failures.
- Physical forwards: PPLX/Liquid were counted with registered top-level forward hooks. Jina is recorded as one native listwise call per selector; CrossEncoder counts are derived as ceil(K/8) from the official batch contract because the internal hook boundary was not exposed.

## Candidate-order diagnostic
- Candidate-order handling was tested by reversing K=20 slates for three CAL queries per locally available model.
- pplx_decider: 0/3 set-level shuffle checks passed.
- liquid_d1_3b: 0/3 set-level shuffle checks passed.
- jina_v3_5: 0/3 set-level shuffle checks passed.
- qwen3_reranker_4b: 3/3 set-level shuffle checks passed.
- bge_reranker_v2_m3: 3/3 set-level shuffle checks passed.
- qwen3_reranker_0_6b: 3/3 set-level shuffle checks passed.
- PPLX, Liquid and Jina are therefore contract-valid for canonical ordered inputs but order-sensitive in this diagnostic; their selection output must retain the frozen candidate ordering.
- Pairwise specialists passed all three set-level shuffle checks in this pilot.

## Conditional blockers
- Missing local checkpoints: startlux_decision_4b, amos_1_small_2b.
- StartLux and Amos were not downloaded or replaced; their native Top-5 contracts remain unvalidated.
- CAL slates are feasibility inputs, not quality evidence; no model policy is admitted to E6-2 on CAL performance.
- Any adapter failures, OOMs, timeouts or order-sensitivity observations are preserved in `E6_1_NATIVE_TRACES.jsonl`.
- GPU 5/6 UUIDs and pre-load memory were captured before execution; the host rejected the initial index-based compute-app query, so a supported UUID-based audit is preserved in `E6_1_GPU_APP_AUDIT.json`. It listed no compute app on either approved GPU; no process was evicted and no resident service was modified.

## Prohibitions verified
- E6-2 and later stages not started.
- No training, reader generation, retrieval optimization, Jev, paid API, or reserved-holdout access.
- Historical artifacts were not overwritten.
