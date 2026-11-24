# DecisionRAG E6-2 R1 Authorized Retrieval-Freeze Report

## Gate: `E6_2_R1_RETRIEVAL_FROZEN_READY`

The owner’s current instruction authorized use of idle/available GPUs. R1 used the already-running host embedding service at `http://service.example.invalid/v1/embeddings`; no new GPU process was started and no unrelated process was evicted.

- Corpus: 5183 canonical SciFact documents.
- CAL validation slates: 555.
- EXPLORATORY frozen Top-20 slates: 277/277.
- EXPLORATORY retrieval Top-5 baselines: 277.
- Embedding model revision: `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`.
- Dense embeddings: exact cosine over the full corpus.
- Fusion: BM25 Top-100 + dense Top-100 + RRF(60), 20 unique IDs.
- qrels/support/answers/selectors/readers: not accessed.
- RESERVED content: not accessed.

The Top-20 slates are now fixed inputs for every future selector. R1 is complete; E6-2 selector inference remains a separate owner decision.
