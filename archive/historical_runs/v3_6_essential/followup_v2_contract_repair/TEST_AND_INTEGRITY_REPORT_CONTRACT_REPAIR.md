# v3.6 Contract-Repair Test and Integrity Report

## Targeted checks

- Python compilation of the three repair scripts: PASS.
- Frozen BRIGHT SHA-256 and E11 budget-manifest SHA-256: PASS.
- 1,384 unique canonical IDs: PASS.
- 1,352 exact historical request hashes: PASS.
- 1,352 exact score vectors and selected lists: PASS.
- 32 historical failures retained without retry: PASS.
- Raw output label-blindness and protected-holdout firewall: PASS.
- GPU4 post-run release check: PASS (8 MiB used, 0% utilization at the recorded check).

## Full regression

The designated `MultiModalRAG/venv` CPU/offline regression suite completed with **302 passed, 0 failed**. Existing expected Triton/deprecation warnings were non-fatal.

## Integrity interpretation

The repaired result is append-only and does not overwrite the original v3.6 package or its failed-gate report. The exact-list gate is promoted only in `GATE_SUMMARY_V2.json`; N3 and N2-E remain separately blocked/partial.
