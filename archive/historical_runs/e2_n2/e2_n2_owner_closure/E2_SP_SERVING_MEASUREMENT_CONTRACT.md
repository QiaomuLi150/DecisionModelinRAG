# E2-S Serving Measurement Contract

Status: `READY_FOR_SEPARATE_AUTHORIZATION`; no serving run performed.

Measure the complete client-visible request from admission to final response, while recording disjoint stage intervals for queue wait, query encoding, candidate retrieval, view assembly, reranker queue wait, model computation, serialization, and client receipt. Do not add component medians to substitute for measured end-to-end latency.

For each policy use the same frozen query/slate cohort, candidate order, score direction, Top-5 rule, warm-up policy, cache-disabled application state, and exclusive hardware entitlement. Record p50/p90/p95, valid/error counts, throughput, input/output tokens, physical forwards, GPU ID, peak VRAM, utilization samples, and raw request/response hashes. Repeat randomized interleaved blocks only after owner approval. GPU cost is a model `active_seconds × approved_rate` and must be reported as `NOT_MEASURED` until a rate and accounting boundary are approved.
