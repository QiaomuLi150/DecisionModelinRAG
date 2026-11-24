# E5-A GPU Authorization Record V1

**Date:** 2026-10-14  
**Status:** `AUTHORIZED_FOR_E5_A_PRECHECK`  
**Owner authorization:** Current conversation approval.

## Selected devices

The E5-A precheck may use the following currently idle devices:

| Device | UUID | Current process state at authorization capture |
|---:|---|---|
| 5 | `<REDACTED_GPU_UUID>` | no compute process; 8 MiB allocated; 0% utilization |
| 6 | `<REDACTED_GPU_UUID>` | no compute process; 4 MiB allocated; 0% utilization |

Devices 0/1/2/4 were not selected for this run because the authorization
capture found active workloads: training processes on 0/1 and resident vLLM
engines on 2/4. No process will be terminated or preempted.

## Execution limits

- This record authorizes only the E5-A native selector precheck/evaluation
  covered by the approved E5-A protocol.
- No Jev, paid API, training, new checkpoint family, reader generation,
  reserved-holdout access or production-service modification is authorized.
- Before timed inference, the runner must re-query process ownership and free
  memory on the selected UUIDs. If either device is no longer idle or safe,
  execution must stop and record `E5_A_GPU_RESOURCE_BLOCKED`.
- The runner must record device UUID, model revision, physical calls, peak
  memory, utilization samples, errors and input/output hashes.
