# E4-2 CAL and Smoke Validation Report

Status: `PASS` for the pre-pilot native contract and context-feasibility gates.

The frozen CAL cohort contains 30 questions and 150 policy validations. The 12-question smoke subset is committed as phase `smoke` (60 rows); the remaining 18 CAL-only questions are committed as phase `cal` (90 rows). All five policies used the same 20-candidate common view and emitted five selected IDs.

Phase/status counts: {"cal:VALID": 90, "full:VALID": 1250, "smoke:VALID": 60}

The initial unbounded-view Jina OOM is retained in `E4_2_EXECUTION_EVENTS.md`; it produced no accepted scientific row. The final `common_word_capped_256_v1` contract was frozen before accepted pilot outcomes and applied to every policy.

No Jev/paid inference, training, new model family, multimodal work or reserved-outcome access occurred.
