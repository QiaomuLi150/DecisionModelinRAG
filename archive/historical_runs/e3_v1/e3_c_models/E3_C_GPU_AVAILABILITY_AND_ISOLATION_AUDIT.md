# E3-C GPU Availability and Isolation Audit

Approved GPUs: `[0, 1, 2, 4]`. Selected native quality device: GPU `1`.

Pre-run snapshot:
```json
{
  "approved": [
    {
      "index": 0,
      "memory_free_mib": 81147,
      "memory_total_mib": 81920,
      "memory_used_mib": 6,
      "name": "NVIDIA A100-SXM4-80GB",
      "utilization_percent": 0,
      "uuid": "<REDACTED_GPU_UUID>"
    },
    {
      "index": 1,
      "memory_free_mib": 80577,
      "memory_total_mib": 81920,
      "memory_used_mib": 575,
      "name": "NVIDIA A100-SXM4-80GB",
      "utilization_percent": 0,
      "uuid": "<REDACTED_GPU_UUID>"
    },
    {
      "index": 2,
      "memory_free_mib": 26367,
      "memory_total_mib": 81920,
      "memory_used_mib": 54785,
      "name": "NVIDIA A100-SXM4-80GB",
      "utilization_percent": 0,
      "uuid": "<REDACTED_GPU_UUID>"
    },
    {
      "index": 4,
      "memory_free_mib": 24570,
      "memory_total_mib": 81920,
      "memory_used_mib": 56583,
      "name": "NVIDIA A100-SXM4-80GB",
      "utilization_percent": 0,
      "uuid": "<REDACTED_GPU_UUID>"
    }
  ],
GPU process-metadata field omitted from public release.
  "devices": [
    {
      "index": 0,
      "memory_free_mib": 81147,
      "memory_total_mib": 81920,
      "memory_used_mib": 6,
      "name": "NVIDIA A100-SXM4-80GB",
      "utilization_percent": 0,
      "uuid": "<REDACTED_GPU_UUID>"
    },
    {
      "index": 1,
      "memory_free_mib": 80577,
      "memory_total_mib": 81920,
      "memory_used_mib": 575,
      "name": "NVIDIA A100-SXM4-80GB",
      "utilization_percent": 0,
      "uuid": "<REDACTED_GPU_UUID>"
    },
    {
      "index": 2,
      "memory_free_mib": 26367,
      "memory_total_mib": 81920,
      "memory_used_mib": 54785,
      "name": "NVIDIA A100-SXM4-80GB",
      "utilization_percent": 0,
      "uuid": "<REDACTED_GPU_UUID>"
    },
    {
      "index": 3,
      "memory_free_mib": 20689,
      "memory_total_mib": 81920,
      "memory_used_mib": 60463,
      "name": "NVIDIA A100-SXM4-80GB",
      "utilization_percent": 0,
      "uuid": "<REDACTED_GPU_UUID>"
    },
    {
      "index": 4,
      "memory_free_mib": 24570,
      "memory_total_mib": 81920,
      "memory_used_mib": 56583,
      "name": "NVIDIA A100-SXM4-80GB",
      "utilization_percent": 0,
      "uuid": "<REDACTED_GPU_UUID>"
    },
    {
      "index": 5,
      "memory_free_mib": 67630,
      "memory_total_mib": 81920,
      "memory_used_mib": 13523,
      "name": "NVIDIA A100-SXM4-80GB",
      "utilization_percent": 47,
      "uuid": "<REDACTED_GPU_UUID>"
    },
    {
      "index": 6,
      "memory_free_mib": 6293,
      "memory_total_mib": 81920,
      "memory_used_mib": 74859,
      "name": "NVIDIA A100-SXM4-80GB",
      "utilization_percent": 99,
      "uuid": "<REDACTED_GPU_UUID>"
    },
    {
      "index": 7,
      "memory_free_mib": 28930,
      "memory_total_mib": 81920,
      "memory_used_mib": 52223,
      "name": "NVIDIA A100-SXM4-80GB",
      "utilization_percent": 0,
      "uuid": "<REDACTED_GPU_UUID>"
    }
  ],
  "status": "ok"
}
```

GPU1 is not exclusive: two unrelated resident processes were observed. No unrelated process was stopped, slowed, restarted or evicted. Native C0-C4 quality inference is therefore shared-workload evidence. C5 isolated timing is `TIMING_BLOCKED`; no shared-GPU latency is promoted as a hardware-equal baseline. GPU5 was not used because it is outside the E3-C approved set.
