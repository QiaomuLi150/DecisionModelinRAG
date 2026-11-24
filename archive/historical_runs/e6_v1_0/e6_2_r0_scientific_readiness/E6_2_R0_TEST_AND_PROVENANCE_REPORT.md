# E6-2 R0 Test and Provenance Report

## Executed scope

- CPU/offline only; no model loading, inference, embedding, retrieval construction, GPU, Jev, paid API, reader or reserved-content access.
- Independent local SciFact Arrow counts and joins passed.
- Independent local GaRAGe row, ID and grounding-array checks passed.
- Historical E5 QASPER Top-20 SHA-256 was checked against the frozen value.
- Six historical provenance manifests were re-read and their declared entries checked where paths were resolvable; any failure is listed in the machine-readable report.
- E6-1B trace/accounting records were audited without rerunning inference.

## Regression status

The prior E6-1B execution recorded focused 5/5 and full 203/203 in the designated shared environment. This R0 adds no model behavior and does not relabel that historical result as a newly executed model benchmark. The R0 readiness script and its output hashes are the new audit evidence.

## Gate

`E6_2_R0_RETRIEVAL_FREEZE_REQUIRED` — the only remaining scientific prerequisite is an independently authorized, authentic frozen SciFact Top-20 retrieval slate preparation.
