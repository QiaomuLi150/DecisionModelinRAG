# E2-P0v2 Serving SLO Proposal

**Status:** `PROPOSED_NOT_APPROVED`

## Initial target

Use a **1.0-second p95 Top-5 selection latency** as an initial candidate SLO. This is a proposal only; it must be accepted or replaced by the owner before serving evaluation. The target applies to warm, complete selection work for K=20 candidates, not full retrieval-plus-reader latency.

## Required workload

- One exclusive NVIDIA A100 80GB reservation per timed local model; no unrelated model service on the device.
- Cold start excluded; record startup separately.
- Warmup under the declared batch/concurrency policy.
- Realistic frozen question and document-length distribution; record tokenization, truncation, overflow and candidate count.
- Concurrency-1 attribution followed by an open-loop increasing arrival-rate sweep until the first predeclared p95/SLO failure or instability criterion.
- Report queue wait, model compute, serialization, complete selection latency, physical forwards, batch size, GPU utilization/memory, timeouts, retries and failures.
- Application cache disabled. Internal/model-server KV/cache behavior logged separately.
- Full RAG latency is reported separately and is not substituted for selection latency.

## Capacity and cost

The offline package makes no runtime claim. Later execution must report measured admitted/completed QPS, GPU-seconds/query, active time, memory and supported local cost scenarios. If active GPU seconds are unavailable, use `NOT_MEASURED` and disclose scenario assumptions; do not claim monetary savings from zero API fees.
