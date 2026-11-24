# E5-0C Retrieval Task and Scope Audit

The audited implementation in `scripts/e5_0_data_protocol.py` builds BM25 and dense rankings over the full permitted corpus of 45,497 paragraph rows. No source-paper filter is applied. Query serialization passes `q['question']` only; source paper, support IDs, answers, qrels and selector outputs are not passed to either retrieval stage. Therefore the frozen task is **global retrieval over the permitted QASPER train/dev corpus for paper-originating questions**, not paper-conditioned retrieval.

The frozen contract is BM25 lowercase-regex `k1=1.2`, `b=0.75`, Top-100 plus Qwen3-VL-Embedding-2B dense Top-100, RRF `k=60`, stable candidate-ID tie-break, and Top-20 unique passage IDs. This is a global designated-support-retention task and may contain out-of-paper unlabeled answer-bearing passages; those candidates are not treated as negative gold.

The primary CSSR-U interpretation must therefore be limited to designated within-paper evidence retention under global retrieval. It is not general cross-paper evidence exhaustiveness and not answer correctness. The existing slate is retained unchanged; no paper-conditioned replacement is generated.
