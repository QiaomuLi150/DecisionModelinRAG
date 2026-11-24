# v1.3 Test Report

- Shared ML virtual environment: **PASS**, 302/302 pytest tests passed.
- System Python: **PARTIAL**, 299/302 passed and 3 failed. Two failures require the already-cached SciFact Arrow audit dependency `pyarrow`; one requires `torch` for the StartLux runtime import. These are environment dependency failures, not v1.3 result failures.
- v1.3 offline R2/R3/R6 analyses: **PASS**; exact frozen-population row and uniqueness checks are in `V1_3_FINAL_STATUS.json`.
- v1.3 adapter scripts: **PASS** for Python compilation after the recorded R6 fixes.
- Raw records retain label-blind markers; all label joins occur in offline analysis scripts.
- No Jev/paid API calls, training, retrieval rebuild or protected-holdout access.
