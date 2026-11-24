# E2-N Authenticity and Source Preservation (N0)

N0 is offline only. No physical model calls, checkpoint loads, GPU forwards,
network requests, service calls, retrieval calls, index builds, Jev calls, or
paid API calls occurred.

Preserved source evidence:

- E2-M0, E2-P0v2, E2-M1, Track A/B/C, and HERB artifacts were read-only inputs.
- Existing E2-M1 raw traces and final artifacts were not rewritten.
- No fresh confirmatory holdout labels or sidecars were opened.
- Model snapshots are referenced by exact local HF revision and file/blob
  inventory in the N0 execution manifest.

For N1-N4, raw response bytes must be hashed before normalization, actual input
and candidate-view hashes must be recorded, and physical forward/pass counts
must not be inferred from API-call counts. Missing evidence fails closed.
