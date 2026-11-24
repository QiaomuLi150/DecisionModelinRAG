# E2-M1 GPU Resource Audit

```json
{
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
}
```

No unrelated service was stopped or restarted. Only GPU0 was selected from the authorized set because it was genuinely free at inspection; GPUs1/2 host active services and GPU4 was busy.
