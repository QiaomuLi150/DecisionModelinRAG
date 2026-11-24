# E3-F0 Integrity and Provenance Audit

Status: `PASS_WITH_LIMITATIONS`.

The package reads authenticated E2/E3 manifests, scorecards, reports and the existing E3-D-R metadata-only holdout audit. It does not read reserved answers, qrels, evidence text, model outputs or fresh benchmark outcomes. Historical E3-D and E3-D-R manifests were rehashed; their source bytes remain unchanged. File integrity proves byte preservation, not scientific validity.

Observed reconciliation: historical population 4,267; Track A conditional n=1,490; Track B conditional n=330; E3-D external selector population 900 MultiHop + 109 FinanceBench; full selector rows 6,054; reader rows 2,154. E3-D-R matched reader denominator is 250 MultiHop, not 900. FinanceBench has 109 reader rows per policy and incomplete/non-exhaustive evidence annotations.

This is a CPU/offline package. No GPU/model/HTTP/paid/Jev work or E3-F execution occurred. Current git state and exact input hashes are in `E3_F0_SOURCE_ARTIFACT_INDEX.json` and the final provenance manifest.
