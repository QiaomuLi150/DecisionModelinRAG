# v3.6 Stage-3 Test and Integrity Report

- Full regression: **302 collected, 302 passed, 0 failed**.
- Stage-3 targeted integrity checks: **13 passed, 0 failed**.
- T2 raw traces: 9,360 records across T2-A/B/C and B6.2 solo T2-D.
- All valid outputs have five unique candidate IDs; invalid outputs remain in
  the raw trace with their error class and elapsed time.
- Serving traces were label-blind; paid API calls were zero; protected
  holdouts were not accessed.
- GPU3 and GPU6 were verified released after inference; foreign GPU processes
  were not terminated.

The machine-readable details and raw-trace hashes are in
`TEST_AND_INTEGRITY_REPORT_STAGE3.json` and
`stage3_analysis/T2_RAW_TRACE_MANIFEST.json`.
