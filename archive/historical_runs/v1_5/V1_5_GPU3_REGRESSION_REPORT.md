# v1.5 GPU3 Continuation Regression Report

- Focused artifact checks: **7/7 PASS** (21 C1 shards × 1,384 rows, C4 1,384 rows, C3 476 events, label-blind raw traces, package presence).
- Python compilation: **PASS** for all new/modified continuation runners and analyzers.
- Full suite under the configured ML environment (`MultiModalRAG/venv`): **PASS**; no failing tests were reported.
- Default system-Python full suite: 3 dependency-gated failures remain unchanged from the project baseline (`pyarrow` unavailable for two SciFact audit tests; `torch` unavailable for one StartLux binding test).
- Archive sanitization grep: **PASS**, no forbidden absolute paths, user names, host names or process listings.
- GPU3 closure: **PASS**, post-run 4 MiB used, 0% utilization, no owned compute process.
