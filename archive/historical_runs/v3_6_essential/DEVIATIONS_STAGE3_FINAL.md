# v3.6 Stage-3 Deviations and Limits

1. Only GPUs 3 and 6 were available without touching foreign workloads. T2
   sessions were therefore scheduled sequentially on GPU3/GPU6/GPU3. This is
   a declared hardware reduction, not an unreported resource substitution.
2. The B6.2 solo process used GPU3 in isolation from the project’s other
   Stage-3 processes, but unrelated host processes were not terminated; its
   measurements are not a whole-host-idle causal estimate.
3. T2 all-request ratios add the measured label-free Top-5 packaging fallback
   only to failed rows. Raw traces are preserved unchanged; completed-only
   ratios are also reported.
4. Quality is the existing BRIGHT known-positive nDCG@5 endpoint. The timing
   cohort is C120 x 3 attempts. Their different populations and the remaining
   T1 replication gap limit cross-stage claims.
5. No Stage-4 experiment, new retrieval, reader inference, Jev request or
   paid API call was performed.
