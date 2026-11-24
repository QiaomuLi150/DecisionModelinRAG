# E6-3A Timing-Boundary Audit

## Original E6-3 boundary

The implementation constructs the candidate ID list and document lookup outside the timed region. It then synchronizes CUDA, starts a wall-clock timer, invokes the adapter, synchronizes again, stops the timer, and performs Top-5 sorting after the stop. Adapter-native token capping/serialization and model scoring are inside the adapter call; dataset lookup, candidate-view construction, final sorting and CSV/JSON persistence are outside the measured interval. Thus the historical report's prose describing a complete request boundary was broader than the executable timing boundary.

## E6-2 boundary

E6-2 records the adapter's native-call duration after its own synchronization. Its input view is reconstructed from the corpus as `title + newline + text`; the historical E6-2 Amos call therefore sees the full native K=20 view and correctly exposes 83 context-limit failures at the 16,384-token contract.

## Consequence

The empty-view defect changes both computation and context feasibility, so the original E6-3 p50/p95 values cannot be used as full-content selector latency. Identical GPU UUIDs do not repair this mismatch. The corrected run is versioned separately and reports its actual boundary explicitly; failed requests remain in the denominator and no inverse-latency QPS or monetary cost is inferred.
