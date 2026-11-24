# DecisionRAG E3-D Final Scientific Report

## Status

`E3_D_OWNER_REVIEW_READY` — exploratory external replication artifacts are
complete with disclosed model- and corpus-specific limitations. This is not a
fresh confirmatory holdout evaluation.

## Frozen populations

The experiment used 900 nonreserved valid MultiHop-RAG questions and all 109
nonreserved public FinanceBench questions. One BM25+Qwen embedding RRF Top-20
slate was generated before labels were opened and was reused by every selector.
No reserved IDs, Jev calls, paid APIs, training, or multimodal inputs were used.

## Model and answer interpretation

BGE, Jina-v3.5, PPLX Decider, Liquid d1-3B, and zerank-2 were run through their
native local interfaces. Retrieval-only Top-5 is the fixed baseline. PPLX uses
the authenticated short-view contract because its native processor rejects
unbounded choice branches; this does not alter candidate identity or ordering.
Nimble is retained as an optional blocked control, not silently replaced.

MultiHop CSSR is reported with both unconditional and retrieval-conditional
denominators. FinanceBench results are annotated-page retrieval diagnostics;
they are not treated as exhaustive qrels. Fixed-reader EM/F1 and numeric
comparisons are reference-answer proxies, not independently verified factual
correctness.

See the machine-readable scorecard, paired bootstrap output, reader records,
and provenance manifest in this directory for exact values and hashes.

## Machine-readable scorecard snapshot

```csv
policy,dataset,n_queries,valid_selector_n,retrieval_eligible_n,cssr_u,cssr_c,any_gold_top5,complete_gold_top20,finance_metric_note
retrieval5,multihop,900,900,802,0.3888888888888889,0.43640897755610975,0.9744444444444444,0.8911111111111111,
retrieval5,financebench,109,109,53,NOT_MEASURED,NOT_MEASURED,0.29357798165137616,0.48623853211009177,annotated evidence-page diagnostic; not an exhaustive FinanceBench qrel
bge_pointwise,multihop,900,900,802,0.3511111111111111,0.3940149625935162,0.9133333333333333,0.8911111111111111,
bge_pointwise,financebench,109,109,53,NOT_MEASURED,NOT_MEASURED,0.5045871559633027,0.48623853211009177,annotated evidence-page diagnostic; not an exhaustive FinanceBench qrel
jina35,multihop,900,900,802,0.46,0.516209476309227,0.9722222222222222,0.8911111111111111,
jina35,financebench,109,109,53,NOT_MEASURED,NOT_MEASURED,0.5963302752293578,0.48623853211009177,annotated evidence-page diagnostic; not an exhaustive FinanceBench qrel
pplx_decider,multihop,900,900,802,0.4488888888888889,0.5037406483790524,0.9688888888888889,0.8911111111111111,
pplx_decider,financebench,109,109,53,NOT_MEASURED,NOT_MEASURED,0.48623853211009177,0.48623853211009177,annotated evidence-page diagnostic; not an exhaustive FinanceBench qrel
liquid_joint20_binary,multihop,900,900,802,0.3,0.33665835411471323,0.8988888888888888,0.8911111111111111,
liquid_joint20_binary,financebench,109,109,53,NOT_MEASURED,NOT_MEASURED,0.5321100917431193,0.48623853211009177,annotated evidence-page diagnostic; not an exhaustive FinanceBench qrel
zerank2_pointwise,multihop,900,900,802,0.41888888888888887,0.470074812967581,0.9511111111111111,0.8911111111111111,
zerank2_pointwise,financebench,109,109,53,NOT_MEASURED,NOT_MEASURED,0.5871559633027523,0.48623853211009177,annotated evidence-page diagnostic; not an exhaustive FinanceBench qrel
```

## Paired bootstrap snapshot

All paired differences use 5,000 bootstrap replicates with seed 731. The
complete CSV is authoritative; this embedded copy makes the owner-review
report self-contained.

```csv
left,right,dataset,metric,n,mean_delta,ci95_low,ci95_high,bootstrap_reps,seed,interpretation
pplx_decider,jina35,multihop,complete_gold_in_top5,900,-0.011111111111111112,-0.04,0.01888888888888889,5000,731,paired exploratory difference; no holdout claim
pplx_decider,jina35,multihop,any_gold_in_top5,900,-0.0033333333333333335,-0.016666666666666666,0.011111111111111112,5000,731,paired exploratory difference; no holdout claim
pplx_decider,jina35,financebench,complete_gold_in_top5,109,-0.08256880733944955,-0.13761467889908258,-0.03669724770642202,5000,731,paired exploratory difference; no holdout claim
pplx_decider,jina35,financebench,any_gold_in_top5,109,-0.11009174311926606,-0.1743119266055046,-0.05504587155963303,5000,731,paired exploratory difference; no holdout claim
liquid_joint20_binary,jina35,multihop,complete_gold_in_top5,900,-0.16,-0.19,-0.1288888888888889,5000,731,paired exploratory difference; no holdout claim
liquid_joint20_binary,jina35,multihop,any_gold_in_top5,900,-0.07333333333333333,-0.09333333333333334,-0.05333333333333334,5000,731,paired exploratory difference; no holdout claim
liquid_joint20_binary,jina35,financebench,complete_gold_in_top5,109,-0.03669724770642202,-0.08256880733944955,0.0,5000,731,paired exploratory difference; no holdout claim
liquid_joint20_binary,jina35,financebench,any_gold_in_top5,109,-0.06422018348623854,-0.11926605504587157,-0.009174311926605505,5000,731,paired exploratory difference; no holdout claim
zerank2_pointwise,jina35,multihop,complete_gold_in_top5,900,-0.04111111111111111,-0.06666666666666667,-0.015555555555555555,5000,731,paired exploratory difference; no holdout claim
zerank2_pointwise,jina35,multihop,any_gold_in_top5,900,-0.021111111111111112,-0.035555555555555556,-0.005555555555555556,5000,731,paired exploratory difference; no holdout claim
zerank2_pointwise,jina35,financebench,complete_gold_in_top5,109,-0.009174311926605505,-0.027522935779816515,0.0,5000,731,paired exploratory difference; no holdout claim
zerank2_pointwise,jina35,financebench,any_gold_in_top5,109,-0.009174311926605505,-0.027522935779816515,0.0,5000,731,paired exploratory difference; no holdout claim
liquid_joint20_binary,zerank2_pointwise,multihop,complete_gold_in_top5,900,-0.11888888888888889,-0.14777777777777779,-0.09111111111111111,5000,731,paired exploratory difference; no holdout claim
liquid_joint20_binary,zerank2_pointwise,multihop,any_gold_in_top5,900,-0.052222222222222225,-0.07333333333333333,-0.03111111111111111,5000,731,paired exploratory difference; no holdout claim
liquid_joint20_binary,zerank2_pointwise,financebench,complete_gold_in_top5,109,-0.027522935779816515,-0.07339449541284404,0.01834862385321101,5000,731,paired exploratory difference; no holdout claim
liquid_joint20_binary,zerank2_pointwise,financebench,any_gold_in_top5,109,-0.05504587155963303,-0.11009174311926606,0.0,5000,731,paired exploratory difference; no holdout claim
jina35,bge_pointwise,multihop,complete_gold_in_top5,900,0.10888888888888888,0.0811111111111111,0.13666666666666666,5000,731,paired exploratory difference; no holdout claim
jina35,bge_pointwise,multihop,any_gold_in_top5,900,0.058888888888888886,0.04111111111111111,0.07666666666666666,5000,731,paired exploratory difference; no holdout claim
jina35,bge_pointwise,financebench,complete_gold_in_top5,109,0.07339449541284404,0.01834862385321101,0.12844036697247707,5000,731,paired exploratory difference; no holdout claim
jina35,bge_pointwise,financebench,any_gold_in_top5,109,0.09174311926605505,0.03669724770642202,0.1559633027522936,5000,731,paired exploratory difference; no holdout claim
```

## Fixed-reader reference-proxy scorecard

```csv
policy,dataset,n,proxy_em,proxy_f1,numeric_or_exact_proxy,interpretation
retrieval5,multihop,250,0.572,0.5878,NOT_APPLICABLE,"reference string proxy, not independently adjudicated factual correctness"
retrieval5,financebench,109,NOT_APPLICABLE,NOT_APPLICABLE,0.07339449541284404,"deterministic FinanceBench answer proxy, not independently adjudicated factual correctness"
bge_pointwise,multihop,250,0.596,0.6050666666666668,NOT_APPLICABLE,"reference string proxy, not independently adjudicated factual correctness"
bge_pointwise,financebench,109,NOT_APPLICABLE,NOT_APPLICABLE,0.09174311926605505,"deterministic FinanceBench answer proxy, not independently adjudicated factual correctness"
jina35,multihop,250,0.584,0.5992761904761904,NOT_APPLICABLE,"reference string proxy, not independently adjudicated factual correctness"
jina35,financebench,109,NOT_APPLICABLE,NOT_APPLICABLE,0.11009174311926606,"deterministic FinanceBench answer proxy, not independently adjudicated factual correctness"
pplx_decider,multihop,250,0.576,0.5859333333333333,NOT_APPLICABLE,"reference string proxy, not independently adjudicated factual correctness"
pplx_decider,financebench,109,NOT_APPLICABLE,NOT_APPLICABLE,0.09174311926605505,"deterministic FinanceBench answer proxy, not independently adjudicated factual correctness"
liquid_joint20_binary,multihop,250,0.556,0.5788857142857142,NOT_APPLICABLE,"reference string proxy, not independently adjudicated factual correctness"
liquid_joint20_binary,financebench,109,NOT_APPLICABLE,NOT_APPLICABLE,0.11009174311926606,"deterministic FinanceBench answer proxy, not independently adjudicated factual correctness"
zerank2_pointwise,multihop,250,0.6,0.6217428571428572,NOT_APPLICABLE,"reference string proxy, not independently adjudicated factual correctness"
zerank2_pointwise,financebench,109,NOT_APPLICABLE,NOT_APPLICABLE,0.11009174311926606,"deterministic FinanceBench answer proxy, not independently adjudicated factual correctness"
```
