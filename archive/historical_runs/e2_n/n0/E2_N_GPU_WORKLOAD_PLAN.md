# E2-N GPU Workload Plan (N0 only)

Status: `PLANNING_ONLY`; no GPU reservation or inference occurred in N0.

## Current observed capacity

The read-only `nvidia-smi` snapshot at N0 execution is recorded in
`E2_N_EXECUTION_AND_PERMISSION_MANIFEST.json`. GPU use is not authorized by
this plan. Existing services and unrelated workloads must remain untouched.

## Workload sizing

The frozen historical population is 4,267 questions (3,767 Track A + 500 Track
B), each with 20 candidates. A pointwise policy implies 85,340 candidate
scores; a native K=20 joint policy implies 4,267 slate calls. CAL smoke/freeze
must be bounded and completed before any historical run.

As a non-binding historical reference only, E2-M1 measured Clef pointwise mean
1323.26846852824 ms and native joint mean 1436.6270132763066 ms per query on one A100
under its own authorization. These figures are not forecasts for other models,
views, precision, batching, or contention.

## Proposed sharding after approval

- N1: 8--16 deterministic CAL requests per feasible model, isolated per
  approved GPU, with short/long input and schema-edge cases.
- N2: CAL-only bounded configuration freeze; no historical TEST adaptation.
- N3: one restart-safe shard per model/policy and immutable raw trace index.
- N4: offline-only independent evaluator and paired bootstrap (5,000 reps,
  fixed documented seed).

The owner authorization must name GPU IDs, isolation and concurrency limits,
download/network permission, checkpoint scope, and a wall-time/VRAM ceiling.
27B models require a separate feasibility/access decision. The plan never
authorizes GPU use, downloads, service restarts, or unlimited execution.
