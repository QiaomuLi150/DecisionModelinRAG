# E3-C Selector Latency and Resource Report

Status: `TIMING_BLOCKED_SHARED_GPU`.

The prespecified 300-query E2-SP workload is frozen and its manifest hash is recorded in the protocol. GPU1 had unrelated resident compute processes at preflight, and GPUs0/2/4 were also occupied. The SDD requires exclusive or genuinely contention-free A100 timing for promoted complete Top-20-to-Top-5 latency. No unrelated process was interrupted, so isolated p50/p95/p99, 1-second SLO status, and 3-second SLO status are `NOT_MEASURED_ISOLATION_BLOCKED` rather than substituted with shared timings. Model load and serving costs are likewise `NOT_MEASURED` for the isolated contract.
