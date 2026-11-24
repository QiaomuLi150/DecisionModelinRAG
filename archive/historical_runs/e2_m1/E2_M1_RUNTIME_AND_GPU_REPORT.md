# E2-M1 Runtime and GPU Report

```json
{
  "gpu_active_seconds": "NOT_MEASURED_DIRECTLY",
  "gpu_audit": {
    "authorized_gpu_ids": [
      0,
      1,
      2,
      4
    ],
    "checkpoint": {
      "local_snapshot": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Cloudflare--clef-flash/snapshots/8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
      "model_id": "Cloudflare/clef-flash",
      "revision": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3"
    },
    "precision": "bfloat16",
    "selected_gpu_ids": [
      0
    ],
    "selected_gpu_reason": "GPU0 had approximately 81 GiB free and no compute process at inspection; GPUs1/2 host active services, GPU4 was busy, and other devices were not authorized by the attached task.",
    "service_restarts": 0,
    "telemetry_note": "Actual GPU utilization/memory samples are recorded per inference call by torch; device inventory was inspected outside the sandbox before this run.",
    "unrelated_services_interrupted": false
  },
  "jev_calls": 0,
  "joint": {
    "mean_ms": 1436.6270132763066,
    "n": 4267,
    "p50_ms": 1206.837661,
    "p95_ms": 3012.230583,
    "total_s": 6130.08746565
  },
  "joint_physical_forward_calls": 4267,
  "paid_api_calls": 0,
  "pointwise": {
    "mean_ms": 1323.26846852824,
    "n": 4267,
    "p50_ms": 1192.743014,
    "p95_ms": 2281.5717,
    "total_s": 5646.38655521
  },
  "pointwise_physical_forward_calls": 4267,
  "selection_latency_scope": "model scoring only; exploratory, not production SLO validation"
}
```

These are exploratory selector-mechanism timings, not maximum sustainable production capacity and not validation of the proposed 1-second SLO. GPU0 was used because it was the only owner-authorized device observed genuinely free; GPUs1/2 host active services and GPU4 was busy. No unrelated service was interrupted.
