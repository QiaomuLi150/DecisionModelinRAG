# E3-F0 Paper-Ready Primary Tables

All rows are exploratory unless explicitly marked otherwise. Track A, synthetic enterprise Track B, external MultiHop-RAG and FinanceBench are separate populations. CSSR-U and CSSR-C are not pooled; reference answer metrics are proxies, not verified factual correctness.

### Historical E3-A paired evidence

| dataset | population | n | pplx_cssr | jina_cssr | difference_pplx_minus_jina | ci95_low | ci95_high | interpretation |
|---|---|---|---|---|---|---|---|---|
| hotpot | unconditional_eligible | 800 | 0.7125 | 0.7275 | -0.015 | -0.0275 | -0.00375 | paired exploratory evidence-retention association |
| hotpot | retrieval_conditional | 598 | 0.9531772575250836 | 0.9732441471571907 | -0.020066889632107024 | -0.03678929765886288 | -0.005016722408026756 | paired exploratory evidence-retention association |
| musique | unconditional_eligible | 2167 | 0.18874019381633594 | 0.19981541301338257 | -0.011075219197046609 | -0.02168897092754961 | -0.0009229349330872173 | paired exploratory evidence-retention association |
| musique | retrieval_conditional | 555 | 0.736936936936937 | 0.7801801801801802 | -0.043243243243243246 | -0.08468468468468468 | -0.0018018018018018018 | paired exploratory evidence-retention association |
| wiki2 | unconditional_eligible | 800 | 0.41125 | 0.385 | 0.02625 | 0.0125 | 0.04 | paired exploratory evidence-retention association |
| wiki2 | retrieval_conditional | 337 | 0.9762611275964391 | 0.913946587537092 | 0.06231454005934718 | 0.02967359050445104 | 0.09495548961424333 | paired exploratory evidence-retention association |
| __pooled__ | unconditional_eligible | 3767 | 0.3472259092115742 | 0.35120785771170693 | -0.003981948500132732 | -0.010883992567029467 | 0.002920095566764003 | paired exploratory evidence-retention association |
| __pooled__ | retrieval_conditional | 1490 | 0.8778523489932886 | 0.8879194630872483 | -0.010067114093959731 | -0.028187919463087248 | 0.008053691275167786 | paired exploratory evidence-retention association |
| enterprise | unconditional_eligible | 443 | 0.6862302483069977 | 0.5936794582392777 | 0.09255079006772009 | 0.06320541760722348 | 0.12415349887133183 | paired exploratory evidence-retention association |
| enterprise | retrieval_conditional | 330 | 0.9212121212121213 | 0.796969696969697 | 0.12424242424242424 | 0.08484848484848485 | 0.16666666666666666 | paired exploratory evidence-retention association |

### External E3-D selector scorecard

| policy | dataset | n_queries | retrieval_eligible_n | cssr_u | cssr_c | any_gold_top5 | complete_gold_top20 | finance_metric_note |
|---|---|---|---|---|---|---|---|---|
| retrieval5 | multihop | 900 | 802 | 0.3888888888888889 | 0.43640897755610975 | 0.9744444444444444 | 0.8911111111111111 |  |
| retrieval5 | financebench | 109 | 53 | NOT_MEASURED | NOT_MEASURED | 0.29357798165137616 | 0.48623853211009177 | annotated evidence-page diagnostic; not an exhaustive FinanceBench qrel |
| bge_pointwise | multihop | 900 | 802 | 0.3511111111111111 | 0.3940149625935162 | 0.9133333333333333 | 0.8911111111111111 |  |
| bge_pointwise | financebench | 109 | 53 | NOT_MEASURED | NOT_MEASURED | 0.5045871559633027 | 0.48623853211009177 | annotated evidence-page diagnostic; not an exhaustive FinanceBench qrel |
| jina35 | multihop | 900 | 802 | 0.46 | 0.516209476309227 | 0.9722222222222222 | 0.8911111111111111 |  |
| jina35 | financebench | 109 | 53 | NOT_MEASURED | NOT_MEASURED | 0.5963302752293578 | 0.48623853211009177 | annotated evidence-page diagnostic; not an exhaustive FinanceBench qrel |
| pplx_decider | multihop | 900 | 802 | 0.4488888888888889 | 0.5037406483790524 | 0.9688888888888889 | 0.8911111111111111 |  |
| pplx_decider | financebench | 109 | 53 | NOT_MEASURED | NOT_MEASURED | 0.48623853211009177 | 0.48623853211009177 | annotated evidence-page diagnostic; not an exhaustive FinanceBench qrel |
| liquid_joint20_binary | multihop | 900 | 802 | 0.3 | 0.33665835411471323 | 0.8988888888888888 | 0.8911111111111111 |  |
| liquid_joint20_binary | financebench | 109 | 53 | NOT_MEASURED | NOT_MEASURED | 0.5321100917431193 | 0.48623853211009177 | annotated evidence-page diagnostic; not an exhaustive FinanceBench qrel |
| zerank2_pointwise | multihop | 900 | 802 | 0.41888888888888887 | 0.470074812967581 | 0.9511111111111111 | 0.8911111111111111 |  |
| zerank2_pointwise | financebench | 109 | 53 | NOT_MEASURED | NOT_MEASURED | 0.5871559633027523 | 0.48623853211009177 | annotated evidence-page diagnostic; not an exhaustive FinanceBench qrel |

### E3-D-R FinanceBench deterministic proxies

| policy | n | legacy_success_n | legacy_success_rate | normalized_exact_n | numeric_equivalent_n | demonstrated_numeric_mismatch_n | manual_adjudication |
|---|---|---|---|---|---|---|---|
| retrieval5 | 109 | 8 | 0.07339449541284404 | 7 | 4 | 0 | MANUAL_ADJUDICATION_NOT_PERFORMED |
| bge_pointwise | 109 | 10 | 0.09174311926605505 | 9 | 4 | 2 | MANUAL_ADJUDICATION_NOT_PERFORMED |
| jina35 | 109 | 12 | 0.11009174311926606 | 8 | 5 | 1 | MANUAL_ADJUDICATION_NOT_PERFORMED |
| pplx_decider | 109 | 10 | 0.09174311926605505 | 7 | 5 | 2 | MANUAL_ADJUDICATION_NOT_PERFORMED |
| liquid_joint20_binary | 109 | 12 | 0.11009174311926606 | 8 | 5 | 1 | MANUAL_ADJUDICATION_NOT_PERFORMED |
| zerank2_pointwise | 109 | 12 | 0.11009174311926606 | 11 | 4 | 1 | MANUAL_ADJUDICATION_NOT_PERFORMED |

### E3-D-R matched evidence/answer cells

| evidence_cell | answer_transition | n | denominator | interpretation |
|---|---|---|---|---|
| BOTH_COMPLETE | PPLX_BETTER | 2 | 250 | reference-answer proxy; no causal or factual claim |
| BOTH_COMPLETE | EQUAL_PROXY | 82 | 250 | reference-answer proxy; no causal or factual claim |
| BOTH_COMPLETE | JINA_BETTER | 2 | 250 | reference-answer proxy; no causal or factual claim |
| BOTH_COMPLETE | ABSTAIN_OR_MISSING | 0 | 250 | reference-answer proxy; no causal or factual claim |
| PPLX_ONLY | PPLX_BETTER | 1 | 250 | reference-answer proxy; no causal or factual claim |
| PPLX_ONLY | EQUAL_PROXY | 26 | 250 | reference-answer proxy; no causal or factual claim |
| PPLX_ONLY | JINA_BETTER | 0 | 250 | reference-answer proxy; no causal or factual claim |
| PPLX_ONLY | ABSTAIN_OR_MISSING | 0 | 250 | reference-answer proxy; no causal or factual claim |
| JINA_ONLY | PPLX_BETTER | 1 | 250 | reference-answer proxy; no causal or factual claim |
| JINA_ONLY | EQUAL_PROXY | 27 | 250 | reference-answer proxy; no causal or factual claim |
| JINA_ONLY | JINA_BETTER | 4 | 250 | reference-answer proxy; no causal or factual claim |
| JINA_ONLY | ABSTAIN_OR_MISSING | 0 | 250 | reference-answer proxy; no causal or factual claim |
| NEITHER_COMPLETE | PPLX_BETTER | 3 | 250 | reference-answer proxy; no causal or factual claim |
| NEITHER_COMPLETE | EQUAL_PROXY | 98 | 250 | reference-answer proxy; no causal or factual claim |
| NEITHER_COMPLETE | JINA_BETTER | 4 | 250 | reference-answer proxy; no causal or factual claim |
| NEITHER_COMPLETE | ABSTAIN_OR_MISSING | 0 | 250 | reference-answer proxy; no causal or factual claim |

### E3-C native timing scope

| model | policy | n | p50_ms | p95_ms | serialized_throughput_rps | total_physical_forwards | status |
|---|---|---|---|---|---|---|---|
| liquid_d1_3b | liquid_joint20_binary | 300 | 172.15644650000002 | 649.70730395 | 3.911183974574965 | 300 | PASS |
| zerank_2 | zerank2_pointwise | 300 | 307.5840565 | 970.1943384000001 | 2.3831053704718927 | 900 | PASS |

### Reading rule

The tables support descriptive comparison of frozen model-policy implementations under specific input contracts. They do not support claims that a model family intrinsically reasons better, that PPLX is generally superior in enterprise settings, that reader proxy matches are factual correctness, or that serialized latency is sustainable production capacity.
