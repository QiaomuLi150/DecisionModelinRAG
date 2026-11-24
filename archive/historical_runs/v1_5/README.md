# DecisionRAG v1.5 Control-Experiment Evidence

The v1.5 run is a partial owner-review closure. C2 is complete from sealed
historical traces using CPU-only analysis; C1, C3 and C4 are blocked because
the dedicated-A100 preflight gate did not pass. No new model inference,
external service call, process termination, or historical-artifact mutation
occurred in v1.5.

## Main artifacts

- `V1_5_FINAL_REPORT.md` — scientific conclusions and blockers.
- `V1_5_STATUS.json` — machine-readable gate and test status.
- `V1_5_PROVENANCE_MANIFEST.json` — hashes and record counts.
- `V1_5_REGRESSION_REPORT.md` — integrity checks.
- `c1_two_stage_specialist/` — frozen protocol and dedicated-GPU blocker.
- `c2_latency_uncertainty/` — complete CPU-only latency/fallback exports.
- `c3_concurrency/` — frozen protocol and dedicated-GPU blocker.
- `c4_winnow_rerun/` — frozen R5-derived protocol and dedicated-GPU blocker.

The shareable ZIP contains only v1.5 protocols, status/export files and
sanitized machine-readable outputs. Historical raw traces remain in their
original locations and are represented by relative paths, SHA-256 hashes and
record counts only.
