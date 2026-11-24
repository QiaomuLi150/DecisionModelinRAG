# E6-0R Endpoint Amendment V1 (Pre-Outcome)

## Owner decision requested

Accept the natural-relevance primary workload **BEIR/SciFact** and the separate conditional **GaRAGe K=15 supplied-pool grounding** secondary workload. Do not call either a natural complete-support CSSR benchmark.

## Primary N-REL

The future selector receives a frozen, gold-blind natural Top-20 produced by an independently frozen retrieval procedure over the 5,183-row SciFact corpus. The qrels are joined at the document unit. Primary metrics are binary nDCG@5 and Recall@5, with unconditional denominators over eligible qrel-linked queries, qrel coverage, and a retrieval-feasibility ceiling. No document is expanded into positive child passages without an authenticated mapping.

The cached release covers 1109 queries and 1109 qrel-linked queries. It has sparse binary judgments (1258 qrel rows and many unjudged corpus units). Unjudged is not explicit negative; judged-coverage and sensitivity are mandatory.

## Secondary F-GRD

GaRAGe exact K=15 rows (2329) can support a fixed supplied-pool `15 -> 5` grounding endpoint. The candidate ID is `sample_id:cite_i`; label mapping is `ANSWER-THE-QUESTION` positive, `RELATED-INFORMATION` related, `OUTDATED` invalid-at-date, `UNKNOWN`/empty unjudged. This is not natural retrieval and must not be compared as Top-20 retrieval.

## Frozen statistics and model scope

Use equal-source-group bootstrap only where source identity is defensible, 5,000 resamples, seed 731, two-sided percentile 95% CI; query-weighted sensitivity is mandatory. GaRAGe has no source-group identity, so source-independent CIs are unavailable. PPLX−Jina is the historical anchor; the conditional 4+4 roster remains unchanged, with StartLux and Amos pending native Top-5 validation. This amendment authorizes no inference.
