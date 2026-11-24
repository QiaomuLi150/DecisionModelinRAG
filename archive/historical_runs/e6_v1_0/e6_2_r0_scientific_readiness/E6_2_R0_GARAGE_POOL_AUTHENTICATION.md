# E6-2 R0 GaRAGe Pool Authentication

Local release: `data/external/e6_0r_datasets/garage/GaRAGe_benchmark.jsonl`
SHA-256: `419e3941f6e8eb4082a74ca2140c1f9337f8b467ff76656a6b8b0290ca3f3a72`
Rows: 2366
Unique sample IDs: 2366
Exact K=15 pools: 2329
Aligned grounding arrays: 2366

Candidate identity is positional (`sample_id:cite_i`). The release supplies `evidence_relevant`, `evidence_correct`, and cited-passage arrays, but no stable source-document IDs/URLs sufficient for source-group bootstrap. `ANSWER-THE-QUESTION` is a support/grounding label; `RELATED-INFORMATION`, `UNKNOWN`, `OUTDATED`, and empty values are not explicit negatives. This track is a fixed supplied-pool grounding task, not natural retrieval and not CSSR-U.

The eligible future endpoint is support-any@5 and graded grounding precision@5 with explicit unknown/unjudged accounting. No gold-aware candidate replacement or padding is permitted.
