# DecisionRAG E2-S/P Execution SDD

Status: `READY_FOR_SEPARATE_AUTHORIZATION`; not executed by E2-N2-R.

## Scope

E2-S measures production-serving economics for the frozen reranker roster. E2-P measures downstream RAG/reader consequences. The tracks remain separate and use only historical CAL/DEV material until separately authorized. No fresh confirmatory holdout, Jev, or paid API is permitted by this document.

## Frozen workload

One frozen query with its immutable Top-20 candidate slate is passed to one reranker, producing a Top-5. Candidate IDs, original ordering, query bytes, model input bytes, and raw response bytes must be hashed before normalization. Application-level caching is disabled. Reusable indexes may remain resident; any server/internal cache must be recorded separately.

## E2-S serving measurements

Measure end-to-end request latency and separately timestamp: request admission, queue wait, query encoding, candidate retrieval/view construction, reranker queue wait, reranker compute, response serialization, and client receipt. Report p50/p90/p95, valid n, failures, physical forwards, input/output tokens, peak VRAM, GPU utilization, and arrival-rate throughput. Use exclusive owner-authorized A100 hardware for each matched block; do not infer cost from utilization alone.

No arrival-rate curve, concurrency, or one-second p95 SLO is frozen here. These must be explicitly authorized and locked before execution. No historical Track C latency is promoted as an E2-S/P result.

## E2-P downstream validation

Use the same reader checkpoint, prompt, decoding, evidence formatting, and answer/citation schema across policies. Use a prespecified CAL/DEV stratified cohort. Report independently verified factual correctness, evidence/citation completeness, abstention, latency, and cost. Reference-answer EM/F1 are secondary proxies; nonempty answers and CSSR are not factual correctness.

## Stop conditions

Stop on any cache leak, missing raw-input hash, candidate-order drift, GPU contention, unapproved holdout access, service restart, model revision mismatch, or paid/JeV call. Preserve all traces and label the block.
