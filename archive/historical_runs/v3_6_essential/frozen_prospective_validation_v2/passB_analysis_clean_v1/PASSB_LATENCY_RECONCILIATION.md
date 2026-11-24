# CQADupStack v2 Pass-B Latency Reconciliation

Status: `PASSB_STRICT_CLEAN_LATENCY_ANALYSIS_COMPLETE`

This report uses only label-blind Pass-B traces. No qrels were read and no model was loaded.

## Coverage

- Frozen latency cohort: 2000 query IDs.
- Pass 1 logical records: 8120 (split across the preserved GPU3 partial and GPU4 continuation files).
- Pass 2 records: 8120 (single GPU3 session).
- Pass 1 validation: {'PASS': 8120}; Pass 2 validation: {'PASS': 8120}.
- Pass 2 request-level failures: none.

## Latency interpretation

All request latencies, including the failed request's elapsed time, are retained. No BM25 fallback latency was substituted because the native runner did not execute a fallback path.
Both passes are complete single-GPU GPU3 sessions with clean pre-run snapshots, matched policy views, complete anchors and no request-level failures. These are the primary latency results for the strict S gate. Allocator warning lines observed on stderr are retained as a telemetry limitation; they did not create failed request records.

## Integrity

- Cross-pass input-hash mismatches: 0.
- Reverse-order mismatches: 0.
- Cross-pass selected-ID mismatches among jointly valid requests: 0.
- Label joins: none; qrels: not read by this analysis.

## Artifacts

- `PASSB_PERREQUEST.csv` — label-blind per-request latency and status records.
- `PASSB_LATENCY_SUMMARY.csv` — p50/p95/p99, validity, failure and memory summaries by pass and policy.
- `PASSB_LATENCY_RATIOS.csv` — 10,000-draw query-cluster bootstrap p95 ratios, including pooled serial-pass sensitivity.
- `PASSB_ANCHOR_SUMMARY.csv` — start/end Jina anchor summaries.
- `PASSB_INTEGRITY.json` — coverage, order, input-hash and preservation checks.
- `PASSB_RAW_MANIFEST.json` — SHA-256 and record counts for all source traces.
