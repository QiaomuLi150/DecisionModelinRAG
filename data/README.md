# Released data

This directory contains no benchmark text or qrels. It stores only stable query/candidate IDs, hashes, selected IDs, model scores, and derived metrics. A subset of E2-N candidate IDs that contained path-like or private-looking strings was replaced with deterministic SHA-256 pseudonyms; identity is preserved inside the released projections, but the reverse mapping is intentionally not published. Use official benchmark downloads when re-scoring, and check each dataset license first.

| Path | Content/schema | Produced from |
|---|---|---|
| `candidates/bright_top20_ids.jsonl.gz` | One record per query: query ID, domain, ordered Top-20 candidate IDs, original slate/input hashes; no question or passage text | Frozen BRIGHT retrieval slate source |
| `candidates/cqadupstack_top20_ids.jsonl.gz` | Query ID/forum and ordered candidate IDs/hashes; no question or document text | Frozen CQADupStack Top-20 source |
| `scores/bright_perquery.csv` | `canonical_id,domain,policy,model,mode,budget_L,order_id,valid,error_class,ndcg5,ndcg5_bm25_fallback,n_calls` | BRIGHT per-query metrics |
| `scores/bright_metrics.jsonl.gz` | Query/model relevance summary, gold-presence flags, operational nDCG@5; no text | E9 frozen BRIGHT analysis projection |
| `scores/e2_n_model_dataset_summary.csv` | Model/dataset aggregate summary columns as in E2-N export | E2-N limited 12-policy study; not the 21×11 table |
| `scores/e2_n_top5_ids.jsonl.gz` | Cohort, dataset, query ID, policy, candidate IDs and selected IDs | E2-N Top-5 output; label-free ID view |
| `scores/transfer_perquery.csv` | Dataset, query ID, budget, model, validity, builder/model failures, gold count/reachability, nDCG@5/FS@5, token counts and input hashes | Corrected transfer confirmation |
| `scores/cqadupstack_perquery.csv` | Query ID/forum/length tercile, policy, validity, gold coverage and nDCG@5 | Frozen CQADupStack evaluation |
| `latency/bright_serving_perrequest.csv` | Per-policy/session request status, elapsed ms, memory, call counts and hashes; no query text or labels | Clean/recorded BRIGHT serving traces |
| `latency/cqadupstack_clean_perrequest.csv` | Clean serial session per-request timing, memory and validity | Clean CQADupStack Pass-B branch |
| `latency/bright_full_population_perrequest.csv` | BRIGHT full-population label-blind per-request timing; device-identifying fields removed | BRIGHT full-call serving session |

`.jsonl.gz` files are gzip-compressed parsed records or privacy/redistribution projections. `raw_traces/TRACE_MANIFEST.csv` records source and released-projection hashes; a projection is not byte-identical to its raw source. Removed fields are listed in `SANITIZATION_LOG.md`. `DATA_DICTIONARY.md` inventories every released CSV/JSONL schema and row count.

Metric caveats: BRIGHT uses nDCG@5 and failed requests score zero in all-request metrics. Transfer uses complete annotated support at Top-5 (FS@5) alongside nDCG in its separate per-query data; the paper's Table VIII uses FS@5 and cannot be pooled with BRIGHT nDCG@5. CQADupStack is frozen test data; the 20 dry-run records are not scored. The 333 absent MuSiQue slates are not synthesized.

The paper's transfer p95 figures are reported in the manuscript but cannot be reconstructed from the shipped timing exports. The archived 9.9×/15× branch is a superseded wrong-denominator diagnostic and must not be substituted.
