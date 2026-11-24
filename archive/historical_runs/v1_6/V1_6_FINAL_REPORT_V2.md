# DecisionRAG v1.6 final continuation report

Gate: `V1_6_D1_D2_COMPLETE_OWNER_REVIEW_READY`

## D1 quality answer

Yes. On the frozen BRIGHT known-positive nDCG@5 endpoint, PPLX still beat
two-stage Jina when Jina was given the exact recoverable per-window strings
that PPLX received. The PPLX-minus-Jina query-weighted delta was 0.025226
(95% CI [0.012333, 0.041851]) on the original order and 0.021345 (95% CI
[0.007313, 0.037994]) averaged over six shuffled orders. The D1 run had 9,688
records, 9,667 valid records and 21 propagated PPLX-source failures; the
20-record determinism recheck was exact.

## D2 serving answer

Yes. Both matched Jina two-stage variants passed the preregistered p95 ≤ 2×
full-text-Jina criterion on the 120-ID, three-attempt cohort. With the
all-request basis (failed request time plus the previously measured frozen CPU
BM25 Top-5 packaging time), the p95 ratio for `jina_two_stage_full` was 1.0301
(95% cluster-bootstrap CI [0.8430, 1.2247]); for
`jina_two_stage_pplxwindow` it was 0.6393 (95% CI [0.3938, 0.8977]). Both had
Pr(ratio ≤ 2) = 1.0 over 10,000 query-cluster bootstrap draws, seed 731.

The PPLX selector policies did not pass the same operational bar: `pplx_budget`
was 2.0486 [1.3052, 3.0249], with Pr(ratio ≤ 2) = 0.3301, and
`pplx_two_stage` was 5.8838 [3.7039, 8.4437], with probability 0.0.

## Execution integrity

D2 produced all 2,160 required request records, with 2,148 valid and 12
retained failures. It ran sequentially on isolated GPU3, with no retries, label
joins, protected-holdout access or paid API calls. Jina and PPLX were loaded
sequentially in one host-capable session because both checkpoints cannot be
resident on one 80-GiB A100. Model loading is reported separately from request
latency. The final GPU check showed the process released and 4 MiB used.

The native runtimes expose stage call counts but not a portable physical
forward counter. These are matched selector timings—not sustainable QPS,
monetary cost, or a claim that one model is globally superior.

The prior blocked D2 protocol, resource gate and package remain unchanged. The
completed append-only exports are in `d2_matched_timing/`; the completed
sanitized package is `decisionrag_v1_6_continuation_exports_v2.zip`.
