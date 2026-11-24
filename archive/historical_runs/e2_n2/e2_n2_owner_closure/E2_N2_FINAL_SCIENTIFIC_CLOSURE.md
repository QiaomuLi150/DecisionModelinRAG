# E2-N2-R Final Scientific Closure

Verdict: `E2_N2_OWNER_ACCEPTED_WITH_LIMITATIONS`
Next authorized gate: `E2_SP_READY_FOR_GPU_AUTHORIZATION`

This closure is offline-only. It preserves all prior E2-N2, E2-N, M0, P0v2, M1, Track A/B/C, and HERB artifacts. It does not authorize or perform E2-S/P serving, fresh-holdout access, training, downloads, Jev, paid inference, model requests, or service changes.

## Closure gates

| Check | Result |
|---|---|
| Frozen population and 20-candidate identity alignment | PASS |
| Source trace hashes and execution-manifest hashes | PASS |
| Frozen Top-5 selection reproduction | PASS |
| Independent quality recomputation | PASS_WITH_LIMITATIONS |
| Independent paired bootstrap recomputation | PASS_WITH_LIMITATIONS |
| CPU regression suite | PASS (65/65; command: PYTHONPATH=. pytest -q) |
| No new inference / no Jev / no paid calls | PASS |

## Independently recomputed quality highlights

CSSR is a complete-support selection proxy, not independently verified reader factual accuracy. Track A and Track B remain separate.

| Cohort | Dataset | Policy | n scored | n eligible | n conditional | CSSR-U | CSSR-C | Any-gold coverage | Blocked |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| track_a | hotpot | jina35 | 800 | 800 | 598 | 0.7275 | 0.9732 | 0.9888 | 0 |
| track_a | musique | jina35 | 2167 | 2167 | 555 | 0.1998 | 0.7802 | 0.9229 | 0 |
| track_a | wiki2 | pplx_decider | 800 | 800 | 337 | 0.4113 | 0.9763 | 0.9962 | 0 |
| track_b | enterprise | pplx_decider | 500 | 443 | 330 | 0.6862 | 0.9212 | 0.7400 | 0 |
| track_a | __equal_dataset_macro__ | pplx_decider | 3767 | 3767 | 1490 | 0.4375 | 0.8888 | 0.9693 | 0 |
| track_b | __equal_dataset_macro__ | pplx_decider | 500 | 443 | 330 | 0.6862 | 0.9212 | 0.7400 | 0 |
| track_b | enterprise | intern_decision | 0 | 0 | 0 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 500 |

The independent recomputation should match the frozen summary values at floating-point tolerance for all rows that have a valid scored population. Intern-Decision's enterprise row is intentionally `n scored=0` because all 500 of its native views are context-blocked; this is an explicit feasibility limitation, not a zero-quality claim.

## Priority paired comparisons

The paired statistic is left-policy complete-support success minus right-policy success. Intervals use 5,000 query-level bootstrap resamples, seed 731, with the frozen exploratory convention. They are not confirmatory significance claims.

| Cohort | Population | Comparison | n | Difference | 95% CI | Wins/Losses/Ties |
|---|---|---|---:|---:|---|---|
| track_a | unconditional_eligible | jina35_minus_retrieval5 | 3767 | 0.136714 | [0.125564, 0.148659] | 539/24/3204 |
| track_a | unconditional_eligible | jina35_minus_bge | 3767 | 0.036368 | [0.029201, 0.043536] | 170/33/3564 |
| track_a | unconditional_eligible | nimble_minus_bge | 3767 | -0.002124 | [-0.011680, 0.007168] | 159/167/3441 |
| track_a | unconditional_eligible | pplx_decider_minus_retrieval5 | 3767 | 0.132732 | [0.121051, 0.144412] | 540/40/3187 |
| track_a | unconditional_eligible | pplx_decider_minus_bge | 3767 | 0.032387 | [0.023892, 0.041147] | 200/78/3489 |
| track_a | retrieval_conditional | jina35_minus_retrieval5 | 1490 | 0.345638 | [0.319463, 0.371812] | 539/24/927 |
| track_a | retrieval_conditional | jina35_minus_bge | 1490 | 0.091946 | [0.073826, 0.110738] | 170/33/1287 |
| track_a | retrieval_conditional | nimble_minus_bge | 1490 | -0.005369 | [-0.028859, 0.018792] | 159/167/1164 |
| track_a | retrieval_conditional | pplx_decider_minus_retrieval5 | 1490 | 0.335570 | [0.309396, 0.362416] | 540/40/910 |
| track_a | retrieval_conditional | pplx_decider_minus_bge | 1490 | 0.081879 | [0.060403, 0.103356] | 200/78/1212 |
| track_b | unconditional_eligible | jina35_minus_retrieval5 | 443 | 0.036117 | [-0.002257, 0.074492] | 43/27/373 |
| track_b | unconditional_eligible | jina35_minus_bge | 443 | 0.056433 | [0.022573, 0.090293] | 41/16/386 |
| track_b | unconditional_eligible | nimble_minus_bge | 443 | 0.074492 | [0.036117, 0.112867] | 55/22/366 |
| track_b | unconditional_eligible | pplx_decider_minus_retrieval5 | 443 | 0.128668 | [0.097065, 0.162528] | 62/5/376 |
| track_b | unconditional_eligible | pplx_decider_minus_bge | 443 | 0.148984 | [0.112867, 0.185102] | 72/6/365 |
| track_b | retrieval_conditional | jina35_minus_retrieval5 | 330 | 0.048485 | [0.000000, 0.096970] | 43/27/260 |
| track_b | retrieval_conditional | jina35_minus_bge | 330 | 0.075758 | [0.033333, 0.121212] | 41/16/273 |
| track_b | retrieval_conditional | nimble_minus_bge | 330 | 0.100000 | [0.048485, 0.151515] | 55/22/253 |
| track_b | retrieval_conditional | pplx_decider_minus_retrieval5 | 330 | 0.172727 | [0.127273, 0.218182] | 62/5/263 |
| track_b | retrieval_conditional | pplx_decider_minus_bge | 330 | 0.200000 | [0.151515, 0.248485] | 72/6/252 |
| track_a | unconditional_eligible | pplx_decider_minus_jina35 | 3767 | -0.003982 | [-0.010884, 0.002920] | 86/101/3580 |
| track_a | retrieval_conditional | pplx_decider_minus_jina35 | 1490 | -0.010067 | [-0.028188, 0.008054] | 86/101/1303 |
| track_b | unconditional_eligible | pplx_decider_minus_jina35 | 443 | 0.092551 | [0.063205, 0.124153] | 47/6/390 |
| track_b | retrieval_conditional | pplx_decider_minus_jina35 | 330 | 0.124242 | [0.084848, 0.166667] | 47/6/277 |

## Authenticity and provenance conclusions

- E2-N2 reused authenticated E2-N traces under the recorded E2-N authorization manifest and added no model requests in the N2 namespace.
- The normalized rows contain model revisions, actual input hashes, raw-response hashes, candidate-view hashes, physical-forward flags, and score contracts. The exact source-trace hashes match the E2-N2 manifest.
- Physical forward totals are verified, but frozen artifacts do not record API/service request counts, per-request timestamps, queue intervals, or all native internal selection operations. Those remain `UNVERIFIABLE` and are not converted into latency or monetary claims.
- Laya's five recorded forward batches per query explain the forward/evidence difference; it does not establish five API calls.
- Model-specific tokenization and native context behavior mean comparisons are common-slate exploratory comparisons, not matched compute-budget or identical-input-format experiments.
- No gold labels entered inference inputs, the fresh confirmatory holdout was not accessed, Jev calls are zero, and paid calls are zero.

## Remaining limitations and next decision

1. The 500 Intern-Decision enterprise rows need a separately authorized context-feasibility decision or adapter experiment; they must remain blocked in historical quality summaries.
2. E2-S/P can now be prepared as an executable contract, but it is not executed by this closure. GPU exclusivity, model roster, arrival rates, p95 SLO, and downstream-reader cohort still require explicit owner authorization.
3. Downstream answer quality must be measured with a fixed reader and independently validated references; selection CSSR and nonempty answers must not be reported as factual accuracy.

Amendment proposed: `NONE`. No frozen experimental protocol or prior result was changed.
