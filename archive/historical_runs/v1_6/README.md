# DecisionRAG v1.6 continuation

Overall status: `D1_COMPLETE_D2_COMPLETE`

## D1 — exact PPLX-window Jina quality

- Status: complete; 9,688 request/order records attempted, 9,667 valid and 21
  expected propagated source failures.
- Primary original-order delta PPLX − Jina: 0.025226; 95% CI
  [0.012333, 0.041851]; Holm p 0.0078125.
- Six-shuffled-order mean delta: 0.021345; 95% CI [0.007313, 0.037994];
  Holm p 0.01953125.
- Determinism: 20/20 exact selected-ID and score-trace matches.
- No deviations from the frozen quality contract. D1 used shared GPUs 4–7 in
  parallel quality workers; this is not a serving-timing measurement.

## D2 — matched serving timing

- Status: complete after a versioned continuation. The original D2 gate was
  blocked and remains preserved in `d2_matched_timing/PROTOCOL.json` and
  `D2_RESOURCE_GATE.json`; GPU3 later passed the dedicated gate before the
  continuation run.
- Frozen cohort: 120 IDs × 3 attempts × 6 policies = 2,160 records; 2,148
  valid and 12 failures. No retries or fill-in were used.
- D2 used one host-capable Python session, one request at a time, and GPU3
  (`<REDACTED_GPU_UUID>`) with 81,149 MiB free and 0%
  utilization at preflight. The model process was released after completion;
  the final direct check showed 4 MiB used and no running process.
- All-request p95 ratios versus `jina_full`, using the pre-existing frozen BM25
  fallback timing only for failed requests: `jina_two_stage_full` 1.0301
  [0.8430, 1.2247], `jina_two_stage_pplxwindow` 0.6393 [0.3938, 0.8977],
  `jina_pplx_budget` 0.2114 [0.1346, 0.3115], `pplx_budget` 2.0486
  [1.3052, 3.0249], and `pplx_two_stage` 5.8838 [3.7039, 8.4437].
- `pplx_budget` does not meet the preregistered p95 ≤ 2× criterion under this
  all-request basis (`Pr(ratio ≤ 2)=0.3301`); `pplx_two_stage` does not
  (`0.0000`). Both Jina two-stage variants do under this timing endpoint.
- D2 is a matched selector-timing result, not sustainable production QPS or a
  monetary cost estimate. Native runtimes do not expose portable forward
  counters; the trace records native call counts and stage input hashes.

## Artifacts

- D1 artifacts remain under `d1_perwindow_jina/` and are unchanged.
- D2 versioned artifacts are under `d2_matched_timing/`, including
  `PROTOCOL_V2.json`, `D2_RESOURCE_GATE_V2.json`, the raw-trace hash manifest,
  the raw trace, per-request export, ratio export and timing report.
- The earlier package
  `decisionrag_v1_6_continuation_exports.zip` is preserved unchanged. The
  completed continuation package is
  `decisionrag_v1_6_continuation_exports_v2.zip`.
