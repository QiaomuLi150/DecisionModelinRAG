# D2 completion receipt

Status: `D2_COMPLETE_OWNER_REVIEW_READY`

The originally blocked D2 gate was cleared by a versioned continuation after a
direct preflight found GPU3 isolated: 4 MiB used, 81,149 MiB free and 0%
utilization, with no running compute process. The pinned Jina v3.5 and PPLX
Decider revisions were evaluated in one host-capable Python session. Jina was
loaded for the four Jina policies, released, and PPLX was then loaded for the
two PPLX policies; this was required to fit both checkpoints on one A100 and
does not mix model-family request execution.

The frozen schedule was policy-major, one request at a time, 120 canonical
IDs, three attempts, six policies. It produced 2,160 rows: 2,148 valid and 12
failures. The failures were retained and classified; no retries, fallback
selection, label joins or holdout access occurred during native inference.

The all-request p95 ratio basis adds only the previously measured frozen CPU
BM25 Top-5 packaging time to failed request times. It gives:

| policy | p50 ms | p95 ms | p95 ratio vs Jina full | 95% ratio CI | Pr(ratio ≤ 2) |
|---|---:|---:|---:|---|---:|
| jina_full | 286.381 | 1171.476 | 1.0000 | [1.0000, 1.0000] | 1.0000 |
| jina_two_stage_full | 350.395 | 1115.910 | 1.0301 | [0.8430, 1.2247] | 1.0000 |
| jina_two_stage_pplxwindow | 323.302 | 729.539 | 0.6393 | [0.3938, 0.8977] | 1.0000 |
| jina_pplx_budget | 151.613 | 245.224 | 0.2114 | [0.1346, 0.3115] | 1.0000 |
| pplx_budget | 1606.550 | 2373.671 | 2.0486 | [1.3052, 3.0249] | 0.3301 |
| pplx_two_stage | 3465.436 | 6782.089 | 5.8838 | [3.7039, 8.4437] | 0.0000 |

Intervals are 10,000-draw query-cluster bootstrap intervals, seed 731, over
the 120 IDs and three attempts per ID. D2 does not support a production-QPS
or monetary-cost claim. The native runtimes expose call-stage counts, but not
a portable physical-forward counter.

The complete raw trace is hash-committed in
`D2_RAW_TRACE_SHA256_MANIFEST_V2.json`; the compact machine-readable exports
are in `export/`. The old blocked protocol and resource receipt remain
unchanged as historical evidence.
