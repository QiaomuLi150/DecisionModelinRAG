# DecisionRAG v3.6 Stage 3 Completion Report

## Gate

`stage3 = PASS_T2_FRONTIER_AND_CPU_STATS`. The overall v3.6 status remains
`PARTIAL_T1_REPLICATION_LIMIT`: the earlier T1 five-policy replication has two
complete primary sessions plus a Jina-only third session. Stage 4 was not run.

## Completed work

- T2 A/B/C: three complete frozen C120 sessions, one request at a time, three
  attempts per ID, with Jina and PPLX policies split across sessions and the
  declared Jina-full/PPLX-budget anchors.
- B6.2: one GPU-isolated solo C120 session with Jina-full, Jina single L8192,
  and PPLX budget; 1,080 records completed.
- CPU closure: 40,000-draw fixed-domain quality bootstrap and 10,000-draw
  canonical-ID p95-ratio bootstrap, both seed 731; all-request and completed
  analyses retained; failure elapsed time plus the measured label-free
  packaging fallback is used only in the all-request basis.

## Principal findings

The Stage-1 BRIGHT quality endpoint is known-positive nDCG@5 over 1,384
queries. PPLX two-stage L8192 has mean 0.21077 versus Jina-full 0.18437,
delta 0.02640 with 95% CI [0.01765, 0.03591], but its T2 p95 ratio is
5.91 with 95% ratio CI [3.82, 8.46], so it fails the declared 2x serving
criterion. PPLX single L4096 has mean 0.18748, delta 0.00311 with CI
[-0.00856, 0.01134], and p95 ratio 0.969 with CI [0.608, 1.405]; it passes
the three separate R/Q/S checks. PPLX budget L8192 has quality delta 0.00839
with CI [-0.00254, 0.01707] and p95 ratio 2.098 with CI [1.312, 3.055],
therefore its timing decision is inconclusive rather than a pass. Jina
two-stage L4096 and L8192 pass R/Q/S, with p95 ratios 0.263 and 0.602.

These are separate quality and C120 timing populations and do not establish
production QPS, whole-host cost, non-inferiority, or factual answer quality.
The GPU-isolated solo run is a diagnostic comparison, not a globally idle
host experiment.

## Integrity

Serving traces remained label-blind, protected holdouts were not accessed,
paid API calls were zero, and the append-only raw trace manifest records the
hash and record count of every T2 phase trace. See `stage3_analysis/` exports.
