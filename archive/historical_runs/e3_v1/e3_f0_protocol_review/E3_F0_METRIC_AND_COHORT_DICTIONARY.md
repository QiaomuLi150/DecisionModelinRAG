# E3-F0 Metric and Cohort Dictionary

`CSSR-U` is unconditional complete annotated support in Top-5. `CSSR-C` conditions on all annotated support being present in the frozen Top-20. `complete_gold_in_top20` is feasibility, not selector quality. `Track A pooled micro` weights questions; `Track A equal-dataset macro` weights HotpotQA, MuSiQue and 2Wiki equally; they must not be substituted. Track A, synthetic Track B, external MultiHop-RAG and FinanceBench are distinct. The historical counts are 4,267 questions overall, 1,490 Track A conditional rows and 330 Track B conditional rows.

`EXACT_RAW`, normalized EM/F1, numeric equivalence and legacy Finance match are reference-answer proxies. `MANUAL_VERIFIED` and independently verified factual correctness are unavailable. Finance evidence-page coverage is not exhaustive CSSR. Serialized p50/p95 is not sustainable offered-load throughput. A model-policy comparison is not an architecture-only causal comparison.

Reader denominators: 250 MultiHop per policy and 109 FinanceBench per policy. Selector denominator: 900 MultiHop and 109 FinanceBench per policy. Historical E3-A/B/C populations remain separate. Failures, overflow and missing labels must remain in unconditional future endpoints unless a protocol explicitly defines another estimand before outcome access.
