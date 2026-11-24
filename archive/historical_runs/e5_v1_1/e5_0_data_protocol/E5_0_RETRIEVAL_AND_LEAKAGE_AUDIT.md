# E5-0 Retrieval and Leakage Audit

Natural slates: 1502; manifest SHA-256: `be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7`. Candidate generation was retriever-only and records `gold_in_request=false`, `qrels_accessed_during_generation=false`. Reserved questions were never passed to the embedding service. The local exact dense matrix was used rather than changing the production Qdrant collection.

The implementation has no code path consuming support IDs, answers or downstream scores during retrieval. Gold joins remain physically separate. A synthetic unit-test fixture covers qrel permutation invariance; live query slates were frozen before any benchmark PPLX/Jina calls. The only PPLX/Jina calls in E5-0 were the separately labelled synthetic interface smoke records.
