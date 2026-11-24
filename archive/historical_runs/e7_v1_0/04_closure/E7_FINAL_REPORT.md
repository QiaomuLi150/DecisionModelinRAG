# DecisionRAG E7 final integrity and robustness report

## Final gate

`E7_VALIDATED_EVIDENCE_READY`

## Native-contract audit

The E7 reference path independently constructed native payloads and restored score vectors to canonical candidate IDs for PPLX, StartLux, Jina and Qwen3-Reranker-0.6B. The primary audit is limited to the executed CAL workload and does not claim causal architecture comparisons. Secondary E6 policies were artifact-audited; no new broad secondary inference was authorized.

| Model | E7-B valid | E7-B attempts | Main contract verdict |
|---|---:|---:|---|
| PPLX Decider | 330 | 400 | PASS_WITH_LIMITATIONS: 70 native 8,192-token context failures |
| StartLux Decision | 400 | 400 | PASS |
| Jina v3.5 | 400 | 400 | PASS |
| Qwen3-Reranker-0.6B | 400 | 400 | PASS |

## Repeatability and order results

The frozen E7-B study contains 40 SciFact and 40 GaRAGe CAL questions, four policies and five physical conditions per model/query, for 1,600 scheduled native attempts. The active corrected freeze has 1,530 valid attempts and 70 explicit PPLX context-limit failures. Canonical A/B/C calls used byte-identical payload hashes; reversal and seeded permutation preserved candidate membership. The table reports exact-set agreement followed by mean Top-5 Jaccard in parentheses; relevance columns are canonical-minus-intervention binary nDCG@5 with 5,000 query-bootstrap 95% CIs.

| Model | Workload | A/B exact (Jaccard; n) | A/reverse exact (Jaccard; n) | A/seed exact (Jaccard; n) | nDCG A−reverse [95% CI] | nDCG A−seed [95% CI] |
|---|---|---|---|---|---|---|
| pplx_decider | scifact | 1.0 (1.0; n=26) | 0.07692307692307693 (0.4748168498168498; n=26) | 0.0 (0.37286324786324787; n=26) | 0.0 [-0.027680268482140686,0.027680268482140686] | 0.025 [-0.011907024642854248,0.06845351232142713] |
| pplx_decider | garage | 1.0 (1.0; n=40) | 0.0 (0.4558531746031746; n=40) | 0.025 (0.5083333333333333; n=40) | 0.01610511061604924 [-0.01490695042959847,0.04929184600103058] | 0.02456298460447879 [0.00015536979666568861,0.052305652941955216] |
| startlux_decision_4b | scifact | 1.0 (1.0; n=40) | 0.125 (0.5580357142857142; n=40) | 0.125 (0.5654761904761905; n=40) | -0.016443693042198154 [-0.06923941593561679,0.0375] | 0.007197517429394183 [-0.030495410827649727,0.03988411579898648] |
| startlux_decision_4b | garage | 1.0 (1.0; n=40) | 0.175 (0.650595238095238; n=40) | 0.175 (0.6773809523809524; n=40) | 0.017774821371450512 [-0.0027648726787015,0.03890482384260411] | 0.006525244258068277 [-0.008514014749389218,0.02314209095870444] |
| jina_v3_5 | scifact | 1.0 (1.0; n=40) | 0.225 (0.6880952380952381; n=40) | 0.275 (0.7300595238095238; n=40) | 0.017357918550455837 [-0.009375167497709336,0.05505084680749974] | 0.0037009417903692415 [-0.008058341506326575,0.018733468926351495] |
| jina_v3_5 | garage | 1.0 (1.0; n=40) | 0.275 (0.6928571428571428; n=40) | 0.3 (0.7086309523809524; n=40) | 0.004901905238182359 [-0.016772216497301838,0.02751112896888902] | 0.010623143664804663 [-0.016335201101780626,0.040835781016797204] |
| qwen3_reranker_0_6b | scifact | 1.0 (1.0; n=40) | 1.0 (1.0; n=40) | 1.0 (1.0; n=40) | 0.0 [0.0,0.0] | 0.0 [0.0,0.0] |
| qwen3_reranker_0_6b | garage | 1.0 (1.0; n=40) | 1.0 (1.0; n=40) | 1.0 (1.0; n=40) | 0.0 [0.0,0.0] | 0.0 [0.0,0.0] |

Same-input exact-set agreement is 100% for every model/workload among valid repeated calls. Order changes remain observable for PPLX, StartLux and Jina, while Qwen0.6B is stable on this CAL sample. The E6-4 order-instability finding therefore survives the E7 mapping/repeatability control as behavioral evidence, but not as an architecture-level causal claim. PPLX SciFact comparisons have n=26 valid canonical/order pairs because the remaining 14 questions hit the full-text context limit.

## Historical claim boundary

The corrected SciFact feasibility, E6-3A full-content selector boundary and reader lexical-proxy limitations are retained. The original E6-3 empty-passage serving estimate remains preserved and superseded. MultiHop-RAG and FinanceBench were previously exploratory exposures, not fresh confirmation. SciFact and GaRAGe are not pooled, and GaRAGe source independence remains unestablished.

## Restrictions

No Jev/paid APIs, new models, training, reader generation, broad E8/E9 work or protected holdout access occurred. Factual correctness is not independently verified.
