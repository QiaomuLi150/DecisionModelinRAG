# DecisionRAG v1.3 follow-up experiments

## Current gate

`PARTIAL_PREPARATION_BLOCKED_FOR_INFERENCE`

The append-only protocols and label-blind manifests are frozen. No model inference, model-service request, qrel/answer read, or paid API call was made by this preparation run. The timing gate requires a dedicated A100-SXM4-80GB with no resident process; the current hardware snapshot shows resident unrelated processes on every GPU.

Prepared artifacts are listed in `README.json`. R1 and R4 require a dedicated GPU. R2/R3/R5/R6 require safe native execution and, for R6, a native tokenizer feasibility result before sub-slate commitment. No result CSV is created until actual inference outputs exist.

## Final v1.3 execution status

The preparation-only gate above is superseded by the append-only execution records
below. No historical artifact was overwritten.

| Experiment | Status | Completed evidence |
|---|---|---|
| R1 serving gate | `BLOCKED` | Dedicated-GPU requirement unavailable; 0 timed requests |
| R2 order sensitivity | `COMPLETE` | 1,384 × 7 conditions for PPLX, Jina and Qwen control; raw failures retained |
| R3 multi-pass PPLX | `COMPLETE_PARTIAL_TIMING` | 1,384 two-stage selections, 1,381 valid; shared-GPU quality only |
| R4 CRAG reader window | `BLOCKED` | 512/1,024 reader outputs and valid dedicated reader window unavailable |
| R5 Winnow budget | `BLOCKED` | Exact pinned runtime binary unavailable |
| R6 LeetCode trim | `PARTIAL` | Kev/Drex native and budgeted outputs plus Jina native controls; Jina-on-budget blocked |

### Headline results

- R2 all-order known-positive nDCG@5: PPLX 0.181557, Jina 0.184740,
  Qwen0.6B 0.168171. The PPLX-minus-Jina equal-domain exploratory delta is
  +0.007295, with whole-domain bootstrap 95% interval
  [-0.013747,+0.028233] and exact 2^12 sign-flip p=0.53125.
- R3 two-stage PPLX all-attempt nDCG@5 is 0.210772 versus historical Jina
  0.184369 and single-pass PPLX 0.192761. The two-stage versus Jina
  equal-domain delta is +0.026729, interval [+0.012837,+0.042339], exact
  sign-flip p=0.004395. This is exploratory quality evidence only; no
  dedicated latency or production-cost conclusion is valid.
- R6 native Kev and Drex trail matched Jina on valid paired native subsets;
  budgeted Jina controls are `NOT_MEASURED` because their exact input strings
  were not persisted. See the R6 report for denominators.

### Claim changes and limitations

- The budgeted-PPLX p95 ≤ 2× Jina criterion is `NOT_MEASURED`: R1's dedicated
  timing gate did not pass. Shared-GPU elapsed times are not substituted.
- R2 shows substantial order-dependent Top-5 set changes for PPLX and Jina;
  the primary PPLX-minus-Jina sign is positive by point estimate but uncertain
  under the prespecified domain-aware interval and sign-flip test.
- The LeetCode native deficit persists for Kev and Drex on the 142-query
  diagnostic cohort. Trimmed model-versus-Jina budget contrasts cannot be
  claimed because matched Jina budget strings are absent.
- No Jev/paid API call, protected-holdout access, training, retrieval rebuild,
  or manuscript edit occurred.

See `TEST_REPORT.md`, `RAW_TRACE_SHA256_MANIFEST.json`, and
`V1_3_FINAL_STATUS.json` for the machine-readable closure.
