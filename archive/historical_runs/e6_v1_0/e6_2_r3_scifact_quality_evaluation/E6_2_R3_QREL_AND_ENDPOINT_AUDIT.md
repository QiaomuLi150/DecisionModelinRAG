# Qrel and endpoint audit

- Eligible population: 277 frozen EXPLORATORY SciFact queries.
- qrel rows materialized: 308 (exact query-ID-filtered exploratory projection only).
- Queries with positive qrels: 277.
- Total positive judgments: 308.
- Primary endpoint: binary nDCG@5, PPLX minus Jina.
- Zero-IDCG rule: nDCG is 0.0 when a query has no eligible positive judgment; such cases are counted, not omitted.
- Unjudged documents are not verified negatives; they receive zero gain only as required by the conventional binary ranking calculation.
- Source groups are descriptive qrel-connected components. R0 marks source-group bootstrap as NOT_DEFENSIBLE, so query bootstrap is primary.

The shared Arrow qrel files were authenticated and iterated with an exact exploratory query-ID filter; only matching rows were materialized into the scorer. No RESERVED qrel rows, answers, or content were exported to R3 artifacts.
