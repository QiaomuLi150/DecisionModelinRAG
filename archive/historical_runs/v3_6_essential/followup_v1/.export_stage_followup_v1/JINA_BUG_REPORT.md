# Jina L8192 Correction Audit

Status: diagnostic evidence recorded; full correction rerun is still evaluated by the follow-up gate.

## Historical behavior

- Frozen historical trace: `results/v3_6_essential/stage1_quality/raw/jina_single_8192__8192_reuse.jsonl`
- Rows: 1384; rows containing an error/status failure: 32.
- The historical trace contains 32 explicit `TypeError: cannot unpack non-iterable NoneType object` records, all in the domain set `['robotics']`.
- Those records have no persisted model-return payload and no latency value. Their candidate-ID lists are empty in the failed records. The trace therefore cannot distinguish a native `None` from a wrapper/normalization failure after the native call.
- The historical trace hash is recorded in `jina_pilot_audit.json`; the original trace is unchanged.

## Instrumented pilot

- Pilot rows: 120 across 6 policies and 20 frozen requests.
- Explicit native return type was checked before unpacking. `None` returns: 0; error rows: 0.
- Observed return types: `{'list': 60, 'None': 60}`. Two-stage records use the explicit stage-level path and do not claim a single native return type.
- Labels were not joined and protected holdouts were not accessed.

## Repair interpretation

The corrected wrapper preserves model-return type, `None` detection, elapsed time and error class before candidate-ID mapping. This resolves the historical observability defect without asserting that all historical failures were native-model failures. The full six-policy rerun and its gate remain authoritative for replacement of the old Jina records.

Machine-readable case inventory: `export/quality_v2/jina_bug_cases.csv`.
