# DecisionRAG v1.5 Control Experiments — Final Scientific Report

## Gate

`V1_5_OWNER_REVIEW_READY_PARTIAL`

The CPU-only C2 analysis is complete. C1, C3 and C4 are explicitly blocked at
their dedicated-A100 preflight gate; no model calls were made for those stages.
Historical E2–v1.3 artifacts were read only and were not modified.

## Frozen protocol and integrity

- BRIGHT Top-20 source: `results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl`
- Top-20 SHA-256: `c304b2c35f445153181098d3a899271db6fc653b44aea50186c6efaeec8e6ea2`
- C1 population: 1,384 queries, original BM25 order plus reversed and five seeded permutations.
- C2 serving population: 120 questions × 3 attempts = 360 requests per historical policy.
- C3 routing population: 50 frozen prompts, 60-second warmup and 300-second measurement cells.
- C4 source protocol: R5 v2, SHA-256 `5ea8ea34e9906f6965c3452c76a712caee3e549c52d40f6d26e8c3205d2a3e02`.
- PPLX revision: `14c9d90137962d300f3de95633f7b6535e80c5ce`.
- Jina v3.5 revision: `e8a93f33f0b22108f8c2364f8484ce3422552fbc`.
- Qwen3-Reranker-0.6B was reused only as an inherited E12 historical trace in C2; no new Qwen call was made.

## C2 completed analysis

C2 reconciled 3,960 historical request records across 11 policy/session
combinations and measured 360 CPU-only BM25 Top-5 fallback records. The
fallback p50 was 0.001352 ms and p95 was 0.002846 ms. Failed requests retain
their recorded request latency plus this fallback cost; requests without a
recorded latency would remain untimed rather than being imputed. In this
historical input all 3,960 records had an elapsed-time value.

The exact machine-readable outputs are in
`c2_latency_uncertainty/export/`:

- `latency_perrequest.csv` — normalized request-level records.
- `fallback_cost.csv` — 360 label-free fallback measurements.
- `latency_policy_summary.csv` — all-request-timed and completed-only p50/p95.
- `latency_ratios.csv` — 10,000-draw query-cluster bootstrap ratios, seed 731.
- `C2_RAW_TRACE_MANIFEST.json` — source hashes and record counts.

### Historical R1 timing summary

| Policy | Completed | p50 all timed (ms) | p95 all timed (ms) | p95 ratio vs Jina full | 95% bootstrap CI | Pr(ratio ≤ 2) |
|---|---:|---:|---:|---:|---:|---:|
| Jina full | 360/360 | 285.665 | 1,130.259 | 1.000 | — | — |
| Jina PPLX-budget | 357/360 | 150.786 | 244.732 | 0.217 | [0.135, 0.313] | 1.000 |
| PPLX budget | 357/360 | 1,608.981 | 2,378.944 | 2.105 | [1.313, 3.051] | 0.243 |
| PPLX two-stage | 357/360 | 3,461.201 | 6,786.646 | 6.005 | [3.870, 8.525] | 0.000 |

The corresponding completed-only PPLX two-stage ratio is 5.975 with CI
[3.808, 8.619]. Thus the historical R1 evidence supports a p95 ratio above
2 for PPLX two-stage versus Jina full, but it does not answer the new C1
question about PPLX two-stage versus Jina two-stage full.

### Historical E12/O3 and Liquid timing summary

| Session/policy | Completed | p50 all timed (ms) | p95 all timed (ms) |
|---|---:|---:|---:|
| Jina v3.5 | 360/360 | 290.633 | 1,169.588 |
| Winnow12 | 297/360 | 2,394.131 | 5,374.397 |
| Drex DLM8 | 180/360 | 222.632 | 1,049.275 |
| Kev9 | 174/360 | 69.106 | 818.722 |
| Qwen3-Reranker-0.6B | 357/360 | 302.509 | 1,225.017 |
| mxbai-v2 | 360/360 | 365.976 | 1,330.156 |
| Liquid d1-3B | 360/360 | 1,122.292 | 3,479.833 |

Failure rates remain part of the interpretation: Winnow 63/360, Drex
180/360, Kev 186/360 and Qwen0.6B 3/360 failed in the inherited E12 traces.
The all-request p95 ratios therefore must not be read as completed-request
throughput or as a substitute for a new matched timing session.

## Required v1.5 questions

1. **Does PPLX two-stage beat Jina two-stage full on BM25 or shuffled order?**
   Not measured. C1 is blocked by the dedicated-GPU gate; no quality claim is
   made from the historical R1 traces.
2. **Does the two-stage procedure help Jina?** Not measured in v1.5. The
   proposed Jina two-stage versus Jina full contrast remains pending C1.
3. **Is the PPLX p95 ratio CI entirely above 2?** For the inherited R1 PPLX
   two-stage versus Jina full comparison, yes: all-request CI [3.870, 8.525].
   This is not the unmeasured C1 contrast.
4. **How much does routing p95 rise under concurrency?** Not measured. C3 is
   blocked; no concurrency or sustainable-QPS claim is made.

## C1/C3/C4 blockers

No inspected GPU satisfied the frozen requirement of one dedicated A100 with
no resident workload for the complete experiment. Sharing or evicting a
resident workload would invalidate the matched timing design, so the stages
were not started. The blocker records contain no model outputs and no process
identifiers. No processes were terminated and no production service was
changed.

## Scientific limitations

- C2 is a historical-trace reconciliation, not a new matched-GPU experiment.
- E12 policy traces use a separate historical session and hardware topology;
  they are not a substitute for C1/C3 matched measurements.
- CPU fallback timing measures only list extraction and first-five packaging;
  it excludes BM25 search/index access and all model inference.
- Failed requests are retained. Untimed failures would be excluded from
  latency percentiles rather than imputed.
- The bootstrap is query-clustered over the 120-question cohort and uses
  10,000 draws with seed 731; it does not establish a production SLO.

## Final interpretation

The only new completed v1.5 evidence is the offline C2 accounting. It confirms
that the inherited PPLX two-stage selector was much slower than inherited Jina
full in R1, with a p95 ratio whose bootstrap interval was entirely above 2.
The central v1.5 control questions—whether PPLX two-stage beats Jina
two-stage full, whether Jina two-stage helps, and how routing scales under
concurrency—remain unanswered until a valid dedicated-A100 window is
available. This is a partial scientific closure, not a positive or negative
verdict on the proposed new controls.
