# DecisionRAG v1.6 D2 — Matched Timing Gate

Status: `D2_BLOCKED_DEDICATED_GPU_UNAVAILABLE`

D2 was not started. The frozen protocol requires one dedicated A100 with no
resident process before or after the serial 120-ID × 3-attempt session. The
required GPU3 gate was checked and failed: an active resident compute service
was present, with 74,287 MiB allocated, 6,865 MiB free and 98% utilization.

No D2 model call, retry, timing row, ratio estimate or cost claim was produced.
No unrelated process was terminated or disturbed. The absence of D2 evidence
is explicit; D1 quality results must not be used as timing substitutes.

The exact gate receipt is `d2_matched_timing/D2_RESOURCE_GATE.json`, and the
frozen timing contract is `d2_matched_timing/PROTOCOL.json`.
