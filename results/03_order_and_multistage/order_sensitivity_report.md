# v1.3 R2 — Candidate-order sensitivity

Status: `COMPLETE_LABEL_JOIN_AND_ANALYSIS`

The frozen BRIGHT population contains 1,384 queries and seven order conditions
(original, reverse and permutation seeds 1–5). Original-order scores are
authenticated historical traces; the six non-original conditions are the
label-blind native traces in `raw/`. Qwen0.6B is the authenticated pointwise
order-safe reuse control and has no new physical model calls.

| Policy | Rows | Valid | Failed | Mean nDCG@5 over orders | SD across orders | Top-5 set change vs original |
|---|---:|---:|---:|---:|---:|---:|
| PPLX budget | 9,688 | 9,660 | 28 | 0.181557 | 0.006665 | 97.39% |
| Jina full | 9,688 | 9,688 | 0 | 0.184740 | 0.001613 | 77.91% |
| Qwen0.6B | 9,688 | 9,674 | 14 | 0.168171 | 0.000409 | 5.21% |

The order-sensitivity statistic is exploratory and is not a quality claim by
itself. The PPLX and Jina native implementations expose scores whose candidate
position changes can alter the selected Top-5; the Qwen pointwise reuse control
is much more order-stable.

For the prespecified PPLX-minus-Jina comparison, query-weighted delta is
`+0.008392`, while the equal-domain-weighted delta is `+0.007295`. The 40,000
draw whole-domain bootstrap 95% interval is `[-0.013747, +0.028233]`; the exact
2^12 domain sign-flip two-sided p-value is `0.53125`. This does not establish a
PPLX advantage. The Qwen-versus-Jina query-weighted delta is `-0.015947`, with
whole-domain bootstrap interval `[-0.034218, -0.002128]`; its exact domain
sign-flip p-value is `0.063965`.

All labels were joined only in this offline analysis. Raw records retain
`labels_joined: false`; failures remain in the all-attempt denominator and were
not silently removed.

Machine-readable outputs:

- `export/order_perquery.csv`
- `export/order_scores.csv`
- `export/order_summary.csv`
- `export/order_borda.csv`
- `export/order_bootstrap.json`

