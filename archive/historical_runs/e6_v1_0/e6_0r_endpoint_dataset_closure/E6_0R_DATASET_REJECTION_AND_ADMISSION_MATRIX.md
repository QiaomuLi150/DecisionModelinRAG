# E6-0R Dataset Rejection and Admission Matrix

| Candidate | Actual audit | Natural complete-support | Natural relevance | Fixed-pool grounding | Decision |
|---|---|---:|---:|---:|---|
| DocHop-QA | 11,379 rows; 3–5 supplied contexts; no answer/evidence fields; no bundled source corpus | NO | NO | NO (no labels) | `NOT_ADMISSIBLE_FOR_CURRENT_ENDPOINTS` |
| GaRAGe | 2,366 rows; K=5–15; 2,329 K=15; per-passage grounding labels; no source IDs | NO | NO | YES, conditional K=15→5 | `FIXED_POOL_GROUNDING_ADMISSIBLE_CONDITIONAL` |
| BEIR/SciFact | 1109 queries; 5183 text corpus rows; 1258 binary qrels | NO | YES, sparse qrels | NOT NEEDED | `PRIMARY_NATURAL_RELEVANCE_ADMISSIBLE_CONDITIONAL` |

Primary E6 amended workload: **SciFact natural relevance**. Secondary grounding workload: **GaRAGe exact K=15 supplied pool**. These endpoints must not be pooled. CSSR-U remains historical QASPER-only and unchanged.
