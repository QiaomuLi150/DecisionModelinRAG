# DecisionRAG v1.6 validation report

Status: `PASS`

| check | result |
|---|---|
| d2_exact_2160_rows | PASS |
| d2_unique_policy_id_attempt | PASS |
| d2_six_policy_counts | PASS |
| d2_no_labels | PASS |
| d2_failure_rows_retained | PASS |
| d2_export_perrequest_schema | PASS |
| d2_export_ratio_schema | PASS |
| d2_protocol_preoutcome | PASS |
| d2_raw_hash_manifest | PASS |
| d1_input_lock | PASS |
| d1_determinism | PASS |
| historical_zip_preserved | PASS |
| v2_zip_hygiene | PASS |
| python_compile | PASS |

The checks are CPU/offline and do not access labels or protected holdouts. The prior v1.6 package hash was checked unchanged; the completed v2 package passed the export-hygiene scan.
