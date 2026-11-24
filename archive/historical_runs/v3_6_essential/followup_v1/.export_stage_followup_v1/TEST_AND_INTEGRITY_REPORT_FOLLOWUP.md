# v3.6 follow-up test and integrity report

## Regression suite

- Designated project environment: `MultiModalRAG/venv` with CUDA disabled for the test process.
- Result: **302 passed, 0 failed**.
- Warnings: expected CPU/Triton fallback and deprecation warnings in the StartLux runtime test.
- System-Python diagnostic: 3 failures caused by missing `pyarrow` and `torch`; this interpreter was not used for the final verdict.

## Integrity checks

- Frozen input manifests and protocol files were read without modification.
- Raw trace files remain outside the export package; only SHA-256 and record-count metadata are exported.
- Label joining remains offline; the serving traces contain no relevance labels.
- No protected holdout was accessed and no paid/API call was made.
- Stage 1 Jina selection-gate failure remains explicitly blocked rather than promoted.
- N2 Drex and Winnow limitations remain explicitly reported; no unmatched branch is treated as a primary comparison.
