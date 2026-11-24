# E9 TREC RAG 2025 benchmark admission

Gate: `E9_BENCHMARK_ADMISSION_BLOCKED`

The candidate source contains 105 query rows and 105 candidate-complete Top-20 topics. The downloaded artifact is the published RankQwen AG retrieval file, not a verified BM25 first-stage file; all downstream claims use `CONDITIONAL_SELECTION_OVER_PRERANKED_POOL`.

| Admission quantity | Value |
|---|---:|
| official topics | 105 |
| pool rows | 105 |
| candidate-complete topics | 105 |
| topics with positive qrel in Top-20 | 22 |
| median judged fraction in Top-20 | 0.0 |

## Provenance

- Topics and graded qrels were downloaded from the official TREC RAG 2025 endpoints and hashed in `E9_TREC25_SOURCE_HASHES.json`.
- Candidate JSONL was obtained from the published `castorini/ragnarok_data` LFS media object. The raw GitHub URL is only an LFS pointer.
- The candidate pool remains label-blind to all model selectors. qrels were used only for the pre-outcome admission audit.
- No TREC model outcomes were inspected before this admission decision.

## Limitations

- The documented BM25 Top-100 artifact was not downloadable at the checked repository path; the admissible study is a post-reranked-pool replacement experiment, not an immediate-after-retrieval reranker study.
- TREC qrels are used as graded passage relevance, not reader factual correctness.
