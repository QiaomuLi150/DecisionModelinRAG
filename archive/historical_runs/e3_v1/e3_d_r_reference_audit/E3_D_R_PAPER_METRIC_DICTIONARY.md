# E3-D-R metric dictionary

- `CSSR_U`: unconditional MultiHop complete annotated support in selected Top-5 over identity-valid query population.
- `CSSR_C`: conditional MultiHop complete support over queries whose frozen Top-20 contains all annotated support.
- `complete_gold_in_top20`: feasibility diagnostic; not a reader-quality result.
- FinanceBench annotated-page coverage: diagnostic only; annotations are not assumed exhaustive.
- `EXACT_RAW`, normalized EM/F1, and numeric/exact matches: deterministic reference-answer proxies.
- `NUMERIC_EQUIVALENT`: conservative parser endpoint, not human verification.
- `MANUAL_VERIFIED` and factual correctness: unavailable in this milestone.
- All R metrics are `POST_HOC_EXPLORATORY`. Track A/B/C domains are never pooled with the external MultiHop/Finance reader rows.
