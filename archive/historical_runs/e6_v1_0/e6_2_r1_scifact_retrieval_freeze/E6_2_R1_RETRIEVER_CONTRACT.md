# E6-2 R1 Retriever Contract

## Frozen historical method

BM25 uses lowercase Unicode-regex tokenization, `k1=1.2`, `b=0.75`, and Top-100. Dense retrieval uses `Qwen/Qwen3-VL-Embedding-2B`, Top-100 exact cosine. RRF uses `k=60`, unioned candidates, canonical-ID deduplication, and canonical-ID ascending tie-breaking. Final output is 20 unique document IDs.

The historical implementation is `scripts/m2_retrieve.py` (SHA-256 `0c71be200d9a2108446a4ff3663b01d89a38a9c48bf8dc639b45292e74b7d35a`). Its E5 execution used QASPER rows with `doc_id/title/full_text/question`; SciFact uses Arrow `_id/title/text`. R1 therefore freezes a transparent schema adapter: `scifact:<_id>` as the document ID, `(title + newline + text).strip()` for both document and query serialization. This is a pre-outcome data-schema compatibility amendment, not a retrieval-effectiveness optimization.

## Current compatibility result

The sparse implementation is mechanically compatible after the explicit adapter. The dense checkpoint reference observed locally is `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`, but E5 did not record a complete immutable embedding snapshot hash and the local cache contains only the `refs/main` pointer, not a verified snapshot. The embedding service and any GPU execution were not preflighted because fresh R1 device-specific authorization is absent.

No qrels, answers, support annotations, selector outputs or RESERVED content were used by the CPU check. The final RRF contract remains unexecuted and therefore no Top-20 slate is claimed.
