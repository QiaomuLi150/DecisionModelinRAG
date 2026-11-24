# E6-0R Corpus Reconstructability Audit

## Decision

The defensible primary amended workload is **BEIR/SciFact natural document relevance**, with `nDCG@5`/`Recall@5`-style qrel-aware endpoints. It has an independently stored 5,183-row text corpus, 1,109 nonempty queries and exact qrel-to-query/document joins. Its qrels are sparse and binary, so it is **not** a complete-support CSSR workload.

DocHop-QA has real source references and context text but does not package the searchable PMC corpus or any answer/evidence annotation fields. Its context list is therefore a supplied context view, not a natural retrieval corpus.

GaRAGe has real grounding annotations and answer fields, but its rows contain supplied per-question passage pools of 5–15 passages, not a natural searchable corpus. The exact-K=15 subset is a defensible secondary fixed-pool grounding workload (`15 -> 5`), never `20 -> 5` natural retrieval.

### Counts

- DocHop-QA: 11379 rows, 9250 PMC IDs, no full corpus release in the audited file, no answer fields, no evidence-label fields.
- GaRAGe: 2366 rows; 2329 exact-15 supplied pools; no stable source-document IDs in grounding entries.
- SciFact: 1109 queries, 5183 corpus units, 1258 qrel rows covering 1109 queries.

The SciFact workload is conditionally exploratory because adjacent historical RAG artifacts mention SciFact. It is not represented as a fresh independent confirmation set.
