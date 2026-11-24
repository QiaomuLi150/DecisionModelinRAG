# v1.3 R3 — PPLX two-stage selection

Status: `COMPLETE_LABEL_JOIN_AND_ANALYSIS_NONDEDICATED_SERVING`

The full BRIGHT population was evaluated in three deterministic non-overlapping
shards after a slower single-worker attempt was stopped before raw commit. The
three committed shards cover exactly 1,384 unique canonical IDs. Native
selection was label-blind; gold labels were joined only by the offline analysis.

| Policy | Attempts | Valid | Failed | Valid-row nDCG@5 | All-attempt nDCG@5 |
|---|---:|---:|---:|---:|---:|
| PPLX two-stage (two 10-way windows, then 10-way final) | 1,384 | 1,381 | 3 | 0.211230 | 0.210772 |
| PPLX E11-A budgeted single pass | 1,384 | 1,380 | 4 | 0.193320 | 0.192761 |
| Jina full historical baseline | 1,384 | 1,384 | 0 | 0.184369 | 0.184369 |

Relative to the historical PPLX budgeted policy, the two-stage query-weighted
delta is `+0.018011`; the equal-domain delta is `+0.019434`, with the
40,000-draw whole-domain bootstrap interval `[+0.006771,+0.032125]` and exact
2^12 sign-flip two-sided p=`0.017578`. Relative to Jina, the query-weighted
delta is `+0.026403`; the equal-domain delta is `+0.026729`, with bootstrap
interval `[+0.012837,+0.042339]` and exact sign-flip p=`0.004395`.

These are exploratory frozen-BRIGHT results, not a confirmation or a claim of
production superiority. R3 ran on shared GPUs with resident processes, so its
`total_ms` values are not dedicated selector latency and are not used to claim
an SLO, QPS or cost advantage. The R1 dedicated timing gate remains blocked.

Raw shard hashes:

- `multipass_raw_start0_n462.jsonl`: `ff6bcddd0b4fd81035b27c1f7d200fcd174d5014b18f18785bbc756062d50434`
- `multipass_raw_start462_n461.jsonl`: `30151a17338eb9d075d869fdfc5483a48e74a9e7892644819d9c1d664b095b61`
- `multipass_raw_start923_n461.jsonl`: `e83c3c13d563270e53eb0842aedc9039bc447b9a5157e6fc42391a94f0b6dd42`

Machine-readable outputs:

- `export/multipass_perquery.csv`
- `export/multipass_summary.csv`
- `export/multipass_bootstrap.json`
- `export/multipass_provenance.json`

