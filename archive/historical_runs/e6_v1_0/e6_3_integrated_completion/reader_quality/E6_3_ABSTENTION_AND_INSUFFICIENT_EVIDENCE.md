# E6-3 Abstention and Insufficient-Evidence Analysis

No-positive means no annotated positive in the provided K15 pool; it does not establish global unanswerability.

## representative_primary
- pplx_decider: n=200; abstain=24; answer=177; no-positive-anywhere=9; abstain when selected evidence lacks positive=4/9; abstain when selected evidence has positive=20/191
- jina_v3_5: n=200; abstain=20; answer=178; no-positive-anywhere=9; abstain when selected evidence lacks positive=4/9; abstain when selected evidence has positive=16/191
- startlux_decision_4b: n=200; abstain=20; answer=180; no-positive-anywhere=9; abstain when selected evidence lacks positive=4/10; abstain when selected evidence has positive=16/190
- qwen3_reranker_0_6b: n=200; abstain=27; answer=172; no-positive-anywhere=9; abstain when selected evidence lacks positive=4/9; abstain when selected evidence has positive=23/191
- original_fixed_pool_prefix5: n=200; abstain=45; answer=154; no-positive-anywhere=9; abstain when selected evidence lacks positive=7/16; abstain when selected evidence has positive=38/184
## disagreement_diagnostic
- pplx_decider: n=60; abstain=7; answer=53; no-positive-anywhere=2; abstain when selected evidence lacks positive=2/3; abstain when selected evidence has positive=5/57
- jina_v3_5: n=60; abstain=9; answer=50; no-positive-anywhere=2; abstain when selected evidence lacks positive=2/4; abstain when selected evidence has positive=7/56
- startlux_decision_4b: n=60; abstain=7; answer=53; no-positive-anywhere=2; abstain when selected evidence lacks positive=2/4; abstain when selected evidence has positive=5/56
- qwen3_reranker_0_6b: n=60; abstain=13; answer=47; no-positive-anywhere=2; abstain when selected evidence lacks positive=3/3; abstain when selected evidence has positive=10/57
- original_fixed_pool_prefix5: n=60; abstain=13; answer=47; no-positive-anywhere=2; abstain when selected evidence lacks positive=3/5; abstain when selected evidence has positive=10/55
## no_positive_diagnostic
- pplx_decider: n=64; abstain=36; answer=28; no-positive-anywhere=64; abstain when selected evidence lacks positive=36/64; abstain when selected evidence has positive=0/0
- jina_v3_5: n=64; abstain=36; answer=27; no-positive-anywhere=64; abstain when selected evidence lacks positive=36/64; abstain when selected evidence has positive=0/0
- startlux_decision_4b: n=64; abstain=33; answer=30; no-positive-anywhere=64; abstain when selected evidence lacks positive=33/64; abstain when selected evidence has positive=0/0
- qwen3_reranker_0_6b: n=64; abstain=37; answer=26; no-positive-anywhere=64; abstain when selected evidence lacks positive=37/64; abstain when selected evidence has positive=0/0
- original_fixed_pool_prefix5: n=64; abstain=49; answer=15; no-positive-anywhere=64; abstain when selected evidence lacks positive=49/64; abstain when selected evidence has positive=0/0
