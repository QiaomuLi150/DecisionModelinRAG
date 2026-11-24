# E6-3 Citation and Grounding Report

Structural citation fields are not claim-level factual verification.

`CLAIM_LEVEL_FACTUAL_SUPPORT_NOT_VERIFIED`

## representative_primary
- pplx_decider: n=200; citation-bearing=177; valid citation refs=611; invalid citation refs=0; cited-relevance precision=0.9046139359698682
- jina_v3_5: n=200; citation-bearing=182; valid citation refs=622; invalid citation refs=0; cited-relevance precision=0.9024725274725275
- startlux_decision_4b: n=200; citation-bearing=181; valid citation refs=671; invalid citation refs=0; cited-relevance precision=0.8960405156537753
- qwen3_reranker_0_6b: n=200; citation-bearing=175; valid citation refs=581; invalid citation refs=0; cited-relevance precision=0.8911428571428571
- original_fixed_pool_prefix5: n=200; citation-bearing=157; valid citation refs=457; invalid citation refs=0; cited-relevance precision=0.8792993630573248
## disagreement_diagnostic
- pplx_decider: n=60; citation-bearing=55; valid citation refs=193; invalid citation refs=0; cited-relevance precision=0.9163636363636364
- jina_v3_5: n=60; citation-bearing=53; valid citation refs=189; invalid citation refs=0; cited-relevance precision=0.8933962264150943
- startlux_decision_4b: n=60; citation-bearing=55; valid citation refs=210; invalid citation refs=0; cited-relevance precision=0.9003030303030303
- qwen3_reranker_0_6b: n=60; citation-bearing=53; valid citation refs=198; invalid citation refs=0; cited-relevance precision=0.9176100628930818
- original_fixed_pool_prefix5: n=60; citation-bearing=51; valid citation refs=182; invalid citation refs=0; cited-relevance precision=0.8741830065359477
## no_positive_diagnostic
- pplx_decider: n=64; citation-bearing=28; valid citation refs=80; invalid citation refs=0; cited-relevance precision=0.0
- jina_v3_5: n=64; citation-bearing=29; valid citation refs=76; invalid citation refs=0; cited-relevance precision=0.0
- startlux_decision_4b: n=64; citation-bearing=32; valid citation refs=90; invalid citation refs=0; cited-relevance precision=0.0
- qwen3_reranker_0_6b: n=64; citation-bearing=28; valid citation refs=74; invalid citation refs=0; cited-relevance precision=0.0
- original_fixed_pool_prefix5: n=64; citation-bearing=15; valid citation refs=30; invalid citation refs=0; cited-relevance precision=0.0

`evidence_correct` was retained as separate counts; RELATED-INFORMATION is not promoted to ANSWER-THE-QUESTION, and UNKNOWN is not treated as incorrect.
