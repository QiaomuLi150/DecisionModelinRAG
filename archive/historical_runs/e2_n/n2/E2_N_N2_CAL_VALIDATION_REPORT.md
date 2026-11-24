# E2-N N2 CAL Freeze Report

Status: `PASS`

Thirty existing CAL queries were evaluated with fixed, predeclared local configurations. No gold labels were loaded into inference, no model replacement or outcome-based tuning was performed, and no cache reuse occurred.

| Model | Expected rows | Valid rows | Physical evidence | Input hashes | Gate |
|---|---:|---:|---:|---:|---|
| bge | 30 | 30 | 30 | 30 | PASS |
| clef | 60 | 60 | 60 | 60 | PASS |
| laya | 30 | 30 | 30 | 30 | PASS |

Clef intentionally has 60 rows because each of the 30 queries has one pointwise and one native-joint trace. The initial derived summary briefly used 30 as the expected row count; this finalizer corrected that denominator without changing the raw trace.

Qwen is not promoted: its local service returned aligned scores but did not expose physical-forward telemetry. New candidate models remain blocked because checkpoint downloads were not authorized and no local snapshots were present.

Jev calls: `0`; paid API calls: `0`.
