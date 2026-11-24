# DecisionRAG v3.6 Targeted Audit — Final Report

## Final gate

`V3_6_TARGETED_AUDIT_PARTIAL_BLOCKED`

## A1–A3 historical Jina audit

The actual v1.5/v1.6 original-order pair contains 1384 common queries, 1069 Top-5 list differences and 789 Top-5 set differences. This does not match the previously stated 197-query premise. The cited nDCG values are reproduced from existing exports: v1.5 = 0.1789703393 and v1.6 D1 = 0.1855466324; observed difference = 0.0065762931.

For the first 30 differing queries, the earliest evidence-bounded difference is a window serialized-input/content-hash mismatch. The raw text strings themselves were not retained, so the audit exports exact request hashes where available and does not claim a raw-text comparison. Score-level, dtype, padding, batch and tokenizer causes remain unidentified.

The existing C2 fp32/batch-1 pair is reused: both runs have 1,380 valid rows and exact list agreement. Its mean nDCG is retained as a diagnostic only; it does not reproduce either historical row and does not identify a cause.

### Paper handling

Retain the v1.5 frozen E11 input-text result and the v1.6 D1 matched-text control as separate protocol-labeled rows. Do not silently replace 0.17897 with 0.18555 or attribute the gap to a setting that was not recorded.

## B1–B2 transfer timing

The new clean timing covers 42672 target request records over the available frozen cohort. The required 2,500 MuSiQue population cannot be executed from the repository: the exact frozen R1 artifacts contain 2,167 MuSiQue slates, leaving 333 missing candidate slates. This is a hard coverage blocker, not an outcome-based exclusion.

Per-request and per-session p95 exports are in `B_transfer_timing/export_d2_latency_perrequest.csv` and `B_transfer_timing/export_d2_latency_ratios.csv`. Each worker/session uses its own 120-record (60 start + 60 end) full-text Jina anchor, with query-ID bootstrap (40,000 draws, seed 731); ratios are explicitly `NOT_COMPARABLE` below 99% validity.

## Integrity

Raw timing records are label blind, historical inputs are preserved, and the execution events document the two runner failures that were corrected before canonical traces were finalized. GPUs 3–6 were the only devices used for the authorized timing workers; no other process occupied them during the runs.

## Artifacts

- `STATUS.md`
- `A_twostage_jina/twostage_jina_diff.csv`
- `A_twostage_jina/twostage_jina_first30_causes.csv`
- `A_twostage_jina/A2_SETTINGS_AUDIT.json`
- `A_twostage_jina/A2_EXISTING_RUN_SUMMARY.csv`
- `B_transfer_timing/PROTOCOL.json`
- `B_transfer_timing/export_d2_latency_perrequest.csv`
- `B_transfer_timing/export_d2_latency_ratios.csv`
- `B_transfer_timing/EXECUTION_EVENTS.json`
- `RAW_AND_EXPORT_SHA256_MANIFEST.json`
