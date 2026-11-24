# DecisionRAG E6-2 R1 Final Retrieval-Freeze Closure

## Gate: `E6_2_R1_RETRIEVAL_FROZEN_READY`

The authorized retrieval-only run produced 555 CAL and 277 exploratory Top-20 SciFact slates, plus 277 deterministic Top-5 prefixes. It issued 376 physical requests to the pre-existing Qwen3-VL-Embedding-2B service at revision `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`, with zero failed batches.

The retrieval contract was BM25 Top-100 plus exact-cosine dense Top-100 fused with RRF(60), yielding 20 unique canonical `scifact:<_id>` candidates per query. Qrels, support labels, answers, selectors, readers and RESERVED content were not accessed. No new GPU process was started and no process was evicted.

Focused R1 tests passed 6/6. The full authoritative suite passed 214/214 in the shared ML environment. The system-Python comparison was 212/214 only because that interpreter lacks `pyarrow`; this is recorded as an environment limitation, not an R1 retrieval failure.

R1 is complete. E6-2 selector inference remains a separate owner authorization and was not executed.
