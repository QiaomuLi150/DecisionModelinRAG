# DecisionRAG E6-2 R1 Final Retrieval-Freeze Report

## Gate: `E6_2_R1_GPU_AUTHORIZATION_PENDING`

R1 completed the authorized CPU-only preparation. Authentic SciFact corpus and partition metadata were validated; the historical BM25 contract was mechanically checked on all 555 CAL queries with deterministic repeated rankings. Dense embeddings, dense Top-100 retrieval, RRF fusion and Top-20 slate generation were not run because the R1 specification requires fresh device-specific GPU authorization and none was supplied.

## Authenticated population

- Corpus: 5,183 unique canonical documents; corpus Arrow hash is recorded in the audit JSON.
- Public queries: 1,109; CAL 555; EXPLORATORY 277; RESERVED 277 metadata-only.
- Source metadata: 572 qrel identity components, seed 731.
- Historical E5 Top-20 hash: `be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7`; it is QASPER and is not substituted for SciFact.

## CPU validation

BM25 `lowercase_regex`, `k1=1.2`, `b=0.75`, Top-100 was run twice on CAL only. The rank digests matched and no relevance metric was computed. Dense and RRF stages remain pending.

## Scientific status

Frozen exploratory SciFact Top-20 slates: **0/277**. Retrieval Top-5 baseline: **not ready**, because it is derived from the missing frozen Top-20 slate. No empty success files were created for these blocked operations.

The next authorized action is a fresh owner-approved GPU UUID/resource preflight, followed by the already frozen dense/index/RRF protocol. No reranker, decision model, reader, Jev, paid API or RESERVED access is authorized by this result.
