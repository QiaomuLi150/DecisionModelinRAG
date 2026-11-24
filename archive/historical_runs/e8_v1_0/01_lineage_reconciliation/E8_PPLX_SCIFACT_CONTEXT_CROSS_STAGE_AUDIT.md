# E8 PPLX SciFact cross-stage context audit

## Authenticated E7 facts

The E7 cohort contains 40 SciFact CAL questions. PPLX has 130 valid rows and 70 failures over the five E7 conditions, corresponding to 14 query-level full-text overflow cases. E8 retains those 14 query IDs and adds `scifact:418` under the extra rank-tag prefix, giving 15 query-level overflow cases and 30 failed C/D rows. The failure accounting is explicit and no truncation repair was applied.

## Reconciliation boundary

E7 uses the exact 40-question CAL cohort, full authenticated candidate text and the pinned 8,192-token native contract. Earlier E6 populations are not assumed to overlap merely because they use the `scifact:` namespace. Exact query/payload parity is only asserted where a native input hash is present; earlier records without payload hashes cannot establish token or serialization parity. The per-query comparison table is `E8_PPLX_QUERY_INPUT_PARITY.csv`.

Observed E7 PPLX status counts: {"('VALID', 'none')": 330, "('INFERENCE_ERROR', 'context_limit')": 70}. Earlier E6 reports with higher feasibility are therefore classified conservatively as a combination of **COHORT_ONLY / INPUT_VIEW_DIFFERENCE / TOKENIZATION_OR_CONTEXT_CONFIG_DIFFERENCE / UNRESOLVED**, pending exact payload parity; this E8 audit does not overwrite unavailable physical outcomes.

Earlier artifact inventory and hashes:
- {'path': 'results/non_jev/v2/e6_v1_0/e6_2_r2_selector_inference/E6_2_R2_NATIVE_TRACES.jsonl', 'sha256': '7a162a9aa6c4bd6ad02bec1699d13269c4151dc6564e25dcd852c95618ce5751', 'scifact_row_hits': 1939}
- {'path': 'results/non_jev/v2/e6_v1_0/e6_2_r2_selector_inference/E6_2_R2_NATIVE_SELECTIONS.jsonl', 'sha256': 'd05b22c141129556c497b001aa4321b01449cef41bb785143833d03b6ef7fcd1', 'scifact_row_hits': 1939}
- {'path': 'results/non_jev/v2/e6_v1_0/e6_4_integrated_robustness_and_paper/E6_4_NATIVE_SELECTIONS.jsonl', 'sha256': '48f263788b27d576d164a3c560777fc2a477012d1aaa6864e1035e40b080e850', 'scifact_row_hits': 1200}
- {'path': 'results/non_jev/v2/e6_v1_0/e6_3a_integrity_and_manuscript/serving_audit/E6_3A_CORRECTED_SERVING_PER_REQUEST.csv', 'sha256': '893ed4c85e9249c2aa7c3782caf56cfd0d13280a886677069470513bb99928b2', 'scifact_row_hits': 1608}

## Remaining unknowns

Earlier native token counts, effective max-model-len, tokenizer revision and overflow behavior are not uniformly present in all E6 artifacts. No contradiction is inferred from aggregate denominators alone. The E7 8,192-token failures remain valid for the E7 CAL full-text contract.
