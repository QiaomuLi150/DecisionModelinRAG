# E2-N2 GPU Resource Audit (N2-0)

Status: `OBSERVE_ONLY`; no device was claimed.

The following is a read-only device observation. It is not an execution authorization and does not justify using GPUs 0, 1, 2, 4 or any other device. Existing services and unrelated workloads remain untouched.

```json
{
  "status": "UNAVAILABLE",
  "command": "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits",
  "stdout": "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.",
  "stderr": ""
}
```

Required owner authorization before N2-1: exact GPU IDs, isolation/concurrency limits, checkpoint/download scope, network permission, wall-time and VRAM ceilings, and whether the 27B model is included.
