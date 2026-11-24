# Decision Models versus Specialized Rerankers for Top-20-to-Top-5 Evidence Selection

## Abstract

We study whether a general-purpose decision model can replace a specialized neural reranker for selecting five evidence passages from a frozen Top-20 text-only retrieval slate. The primary exploratory comparison uses QASPER train/dev exploratory questions, alternative annotated support sets, equal-source-group weighting, and the set-aware CSSR-U endpoint. PPLX Decider retained a valid support set in 306/1,502 questions (20.37%) and Jina v3.5 in 299/1,502 (19.91%); the equal-source-group difference was +0.5405 percentage points with a two-sided 95% bootstrap interval of [−0.3586,+1.4805] pp. The interval does not establish superiority, equivalence, or noninferiority. In a fixed-reader follow-up, the representative deterministic reference-agreement proxy was 18/200 versus 17/200, with a confidence interval crossing zero; this is not factual verification. On the same GPU and concurrency-one selector timing, PPLX p50/p95 were 1,320.085/1,616.623 ms versus Jina 62.245/88.790 ms. The record supports a scoped exploratory quality–cost comparison, not a universal replacement claim or production cost conclusion.

## 1. Introduction

Top-k retrieval systems often require a second-stage selector to reduce a broad candidate set to a context-sized evidence set. A general-purpose decision model may offer a richer native decision formulation, but its replacement value depends jointly on evidence retention, downstream reader utility, and inference cost. We therefore freeze candidate views and compare one general-purpose decision policy with one specialist reranker under their authenticated native contracts.

## 2. Research questions and claims boundary

The primary question is conditional and empirical: under the frozen QASPER text-only contract, does PPLX Decider provide competitive annotated support-set retention relative to Jina v3.5 at a usable selector cost? We do not test universal model-family behavior, verified factual answer accuracy, sustainable QPS, dollar cost, or prospective confirmation. E5-F remains sealed.

## 3. Data and governance

The E5-A exploratory population contains 1,502 valid QASPER questions from 587 eligible source groups; the full exploratory partition contains 643 groups. Each question uses one immutable natural Top-20 slate whose SHA-256 is `be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7`. E5-0C preserved alternative valid support sets rather than taking their union. The approved endpoint CSSR-U is one valid support set of size 1–5 completely contained in the selected Top-5. The retrieval-feasibility ceiling is 355/1,502 = 23.64%.

## 4. Model and inference contracts

PPLX revision `14c9d90137962d300f3de95633f7b6535e80c5ce` and Jina revision `e8a93f33f0b22108f8c2364f8484ce3422552fbc` selected exactly five passages from identical canonical Top-20 IDs and views. E5-D measured complete selector requests at concurrency 1 on GPU5 UUID `<REDACTED_GPU_UUID>`. PPLX was measured before Jina; order/thermal drift is therefore a limitation. The reader used Qwen3-VL-8B revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, identical inherited prompts, five passages and deterministic decoding.

## 5. Evaluation

CSSR-U is unconditional over 1,502 valid questions. CSSR-C conditions on the 355 questions for which complete support is present in Top-20. The primary PPLX−Jina effect is the mean of within-source-group policy differences, not the query-micro difference. Reader T1 is exact agreement between normalized answer output and prespecified QASPER answer-form alternatives. All answer-format sensitivity analyses are symmetric post-hoc diagnostics. No T2/T3 correctness endpoint was available.

## 6. Results

### 6.1 Evidence selection

PPLX: 306/1,502 CSSR-U and 306/355 CSSR-C. Jina: 299/1,502 and 299/355. Paired cells were both-success 275, PPLX-only 31, Jina-only 24, and neither 1,172. The equal-group effect was +0.5405 pp, 95% CI [−0.3586,+1.4805] pp. Thus the direction is numerically positive but inferentially inconclusive.

### 6.2 Reader utility

The representative population used 200 source-group-sampled questions. PPLX matched a reference form on 18/200 (9.0%); Jina matched on 17/200 (8.5%). The paired difference was +0.5 pp with 95% CI [−1.0,+2.0125] pp. The separate 55-question evidence-disagreement cohort is enriched and descriptive, not a population estimate. The 496 requests are fully explained by seven qids appearing in both cohorts: 510 nominal cohort slots collapse to 496 unique policy–qid requests.

### 6.3 Serving

PPLX p50/p95 latency was 1,320.085/1,616.623 ms; Jina was 62.245/88.790 ms. The measured p50 and p95 ratios are approximately 21.2× and 18.2×. PPLX fails the illustrative 1-second p95 threshold and meets 3 seconds; Jina meets both. Peak allocator observations were 53,188,925,952 bytes for PPLX and 1,551,129,600 bytes for Jina. These are not total physical memory or monetary cost.

### 6.4 Historical mechanisms

E2-M1 and E4-3 examined Clef pointwise/joint and related policy contracts on distinct cohorts and candidate-view lineages. E4-R found the named objectives comparable but heterogeneous; Clef joint contrasts were negative or inconclusive in E4-3 and matched isolated timing was blocked there. These results are preserved as scoped mechanism evidence and are not pooled with E5-A.

## 7. Why deterministic reference agreement is low

QASPER contains free-form, extractive, yes/no and unanswerable annotation forms, with alternative annotations and evidence strings. The reader requests one short answer, cap each passage at 256 tokenizer tokens, and may abstain. The frozen scorer compares normalized output against answer-form alternatives; it does not resolve all paraphrases, units, multi-claim entailment or factual grounding. A low match rate therefore identifies a validity limitation and motivates independent adjudication; it does not imply a 91% factual-error rate.

## 8. Limitations and threats

The quality endpoint is annotated evidence retention, not verified correctness. The reader endpoint is a deterministic reference proxy without T2/T3 adjudication. E5-D has fixed policy order and no randomized repeated-order measurement. A single specialist and one general-purpose decision model are tested on one scientific-document domain. Public pretraining exposure and source-group dependence limit independence claims. E5-F prospective confirmation remains blocked.

## 9. Reproducibility

All source artifacts, hashes, raw accepted reader rows, raw selector traces, contracts and new audit outputs are listed in `E5_QA_INPUT_PROVENANCE_INDEX.json` and `E5_QA_PROVENANCE_MANIFEST.json`. Historical artifacts were not modified. The full available regression suite and focused E5-QA tests are recorded in `E5_QA_TARGETED_AND_REGRESSION_TEST_REPORT.md`.

## 10. Conclusion

Under this frozen exploratory QASPER contract, PPLX is numerically competitive in support-set retention but not statistically established as better, and it is materially slower and more memory-intensive in the matched selector run. The defensible conclusion is a bounded exploratory comparison with a replacement boundary: deployment would require a latency budget that tolerates the observed PPLX tail and an independent answer-correctness and prospective validation study. No universal replacement, factual-accuracy, noninferiority, QPS or monetary-saving claim follows.

## Internal evidence references

- E5-A primary statistics: `results/non_jev/v2/e5_v1_1/e5_a_replacement_benchmark/E5_A_PRIMARY_GROUP_BALANCED_STATS.json`.
- E5-C raw accepted outputs: `results/non_jev/v2/e5_v1_1/e5_dc_serving_reader_validation/E5_C_READER_INPUTS_AND_OUTPUTS.jsonl`.
- E5-D raw traces: `results/non_jev/v2/e5_v1_1/e5_dc_serving_reader_validation/E5_D_NATIVE_SERVING_ATTEMPTS.jsonl`.
- E4-R reconciliation: `results/non_jev/v2/e4_v1_1/e4_r_reconciliation/E4_R_FINAL_SCIENTIFIC_RECONCILIATION_REPORT.md`.
