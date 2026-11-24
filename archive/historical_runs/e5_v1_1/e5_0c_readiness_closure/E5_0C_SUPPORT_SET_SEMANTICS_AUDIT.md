# E5-0C QASPER Support-Set Semantics Audit

## Scope and access

The audit inspected all 703 permitted CAL records and all 2,028 permitted exploratory records structurally; raw semantic reconstruction succeeded for 2728 records and 3 CAL records lacked a permitted source-paper paragraph manifest and remain metadata-only. Raw QASPER values were materialized only for source papers represented in the permitted CAL/exploratory corpus; reserved source-paper values were skipped by a top-level structural parser and were not decoded or exported. No reserved question, answer, support ID, slate or outcome was accessed.

## CAL classification

Among the 519 E5-0 mapping-valid CAL questions, the reconstructed classes are:

```json
{"MULTIPLE_ALTERNATIVE_SETS": 53, "PARTIAL_OR_NONEXHAUSTIVE": 95, "SINGLE_DESIGNATED_SET": 371}
```

These classes are overlapping diagnostics, not mutually exclusive truth categories. The raw wrapper/evidence structure includes answer-wrapper multiplicity, alternative mapped support sets, duplicate evidence, non-text evidence sentinels, and exact-text mapping failures. The complete per-query audit is in `E5_0C_ANNOTATION_MAPPING_CASES.csv`.

The E5-0 mapper deterministically uses the first mapped answerable support set. It does not form a union across answer wrappers. Reconstructed first-set hashes matched the E5-0 CSV for `True` rows with no mismatch. The audit therefore confirms the frozen mapping implementation, but it does not establish that the first set is exhaustive or factually sufficient.

## Scientific interpretation

The CSSR-U predicate is defensible only as **retention of the frozen designated support set**, not as answer correctness, source-groundedness, or exhaustive evidence recall. Multiple/partial/non-text annotations must remain visible. A future pre-outcome amendment could define a set-aware predicate as success when at least one prespecified valid support set is contained in Top-5, but it must not be silently substituted into E5-A.

Primary endpoint status: `PRIMARY_ENDPOINT_NOT_VALID_FOR_PRIMARY_E5_A_UNTIL_SET_AWARE_AMENDMENT`.
