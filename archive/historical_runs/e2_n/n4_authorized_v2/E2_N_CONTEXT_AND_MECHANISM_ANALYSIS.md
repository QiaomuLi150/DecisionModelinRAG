# E2-N Context and Mechanism Analysis

The evaluator applied one deterministic Top-5 rule to all valid 20-candidate traces: descending score, then original candidate order, then candidate ID. Retrieval5 is the frozen first-five prefix.

Track A and Track B are reported separately. Complete-support CSSR means every identity-resolved gold document is in the selected Top-5; it is not a reader answer-accuracy measure.

- BGE and Laya support independent pointwise relevance only in this run.
- Clef supports both independent pointwise and native joint K=20 binary relevance; its full traces were reused from compatible E2-M1 execution with source hashes.
- Qwen service scores remain a separate limited branch; direct checkpoint Qwen traces are included only when E2_N_N4_INCLUDE_QWEN=1 and are never mixed with service scores.
- Laya’s checkpoint temperature warning is retained: score ordering was evaluated, but confidence calibration is not certified.
- No new reader answers were generated; these results assess selection quality proxies only.

Bootstrap protocol: 5000 query-level resamples, seed 731; intervals are exploratory and not confirmatory significance claims.
