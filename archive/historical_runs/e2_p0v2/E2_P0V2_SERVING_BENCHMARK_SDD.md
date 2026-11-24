# E2-P0v2 Serving Benchmark SDD

This is a future design, not an execution authorization.

## Measurement boundary

Measure preprocessing/tokenization, queue wait, model compute, serialization, complete Top-5 selection latency, and full RAG latency separately. Selection latency includes all native calls, batching, repeated fields, context assembly and error handling required to emit five IDs. Full RAG latency additionally includes retrieval and reader only when explicitly evaluated.

## Controlled workload

Use nonempty canonical questions and frozen Top-20 identities. Reserve an exclusive or demonstrably equivalent A100 80GB allocation per local model; no unrelated VLM service may share the timed device. Warm up under a predeclared policy, then run concurrency-1 attribution and an open-loop increasing arrival-rate sweep. Concurrency 1/10 alone is insufficient. Continue until the first predeclared p95 failure or stability criterion.

Report p50/p95 with uncertainty, admitted/completed QPS, backlog, timeouts, retries, physical forwards, batch size, GPU active seconds, memory, cache reuse, hashes and failure rates. Application cache must be disabled unless the experiment explicitly measures cache behavior; model-server/internal cache is logged separately.

## Owner-gated economics

A numeric p95 selection SLO and arrival-rate target are **NOT FROZEN**. Recommend initially considering a 2-second p95 selection SLO at a representative steady-state load, but do not use that value without owner approval. A deployment cost threshold is also **NOT FROZEN**; report measured GPU-seconds and supported rates before claiming savings. Jev is a separate remote API experiment and cannot be equated with local A100 time.
