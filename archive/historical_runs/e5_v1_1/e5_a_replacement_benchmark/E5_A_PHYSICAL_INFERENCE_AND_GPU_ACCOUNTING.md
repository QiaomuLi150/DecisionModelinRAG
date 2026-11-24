# E5-A Physical Inference and GPU Accounting

## Recorded work

- H0 CAL smoke: PPLX 12/12 valid; Jina 12/12 valid.
- A0 primary: PPLX 1502/1502 valid, 1502 recorded physical forward calls; Jina 1502/1502 valid, 1502 recorded physical forward calls.
- A0 cache reuse flags: PPLX 0; Jina 0.
- Summed per-request selector durations: PPLX 1943.192s; Jina 111.990s. These are sums of request-level observations, not wall-clock throughput or isolated production latency.

GPU5 (`<REDACTED_GPU_UUID>`) hosted PPLX and GPU6 (`<REDACTED_GPU_UUID>`) hosted Jina for the recorded runs. Before/after device and process snapshots are retained in the H0/A0 summaries. Existing unrelated processes were not terminated. Shared-device occupancy visible in snapshots means these measurements are not a matched isolated cost or serving benchmark.

This benchmark reports selector inference work only; it does not claim matched serving p95/QPS, dollar cost, or deployment economics.
