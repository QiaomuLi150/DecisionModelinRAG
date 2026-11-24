# When Do Decision Models Replace Specialized Rerankers in RAG? Evidence Quality, Reader Utility, and Serving Trade-offs

## Abstract

This exploratory study compares general-purpose decision models with specialized neural rerankers for selecting five passages from fixed candidate sets. Across SciFact and GaRAGe, relative evidence-selection quality is workload-dependent: PPLX is descriptively positive versus Jina on SciFact but negative on GaRAGe. The paired GaRAGe reader proxy does not show a demonstrated downstream advantage. A serving-integrity audit found that an earlier bounded E6-3 SciFact run accidentally supplied empty candidate passages; those rows are preserved but excluded from corrected serving claims. A versioned rerun on the identical serving sample restored full source views and retained context failures. The resulting evidence supports a qualified empirical paper, not a universal replacement or noninferiority claim.

## 1. Introduction and research questions

Reranking must balance evidence quality, context feasibility, latency and memory. We ask: (RQ1) when do decision-model selectors match specialized rerankers on fixed candidate slates; (RQ2) do native readout formulations change selection behavior; (RQ3) does selection difference translate to a downstream reader proxy; and (RQ4) what operational boundary is imposed by context and compute?

## 2. Models and native contracts

The eight-model roster comprises PPLX Decider, Liquid d1-3B, StartLux Decision, Amos 1 Small, Jina v3.5, Qwen3 Reranker 4B, BGE v2-m3 and Qwen3 Reranker 0.6B. Decision models use their native candidate-aligned categorical readouts where validated; specialists use native listwise or pairwise scores. Top-5 sorting is performed by stable score/order rules. Model revisions and contract notes are in the E6-3A manifest and manuscript tables.

## 3. Data and protocols

SciFact uses the frozen exploratory Top-20 slates and GaRAGe uses the frozen label-blind K15 inputs. E6-2 quality populations and E6-3 reader/serving populations are not pooled. The GaRAGe source-independence limitation remains. The E6-3A serving correction reuses the exact 100 SciFact and 100 GaRAGe sample IDs frozen before the original serving run; it changes only the SciFact source-view reconstruction.

## 4. Evidence-selection results

On SciFact, PPLX−Jina operational binary nDCG@5 was `0.017400` with 95% CI `[-0.010549, 0.045281]`; the interval includes zero. On GaRAGe, the corresponding exploratory PPLX−Jina binary evidence-relevant nDCG@5 effect was `-0.015802` with CI `[-0.022765, -0.008966]`, favoring Jina under that endpoint. These are distinct workload endpoints and do not establish cross-workload causality.

## 5. Reader utility

For the frozen representative 200-question reader cohort, normalized token-F1 was `0.116365` for PPLX and `0.119772` for Jina, with paired difference `-0.003407`. This is reference overlap against `answer_generate`, not independently adjudicated factual correctness. Abstentions, citation validity and annotated relevance are separate diagnostics. No reader inference was repeated in E6-3A.

## 6. Serving integrity and operational results

The original E6-3 SciFact serving measurements cannot support full-content latency or failure claims: the slate candidate objects lacked inline text and the implementation passed empty strings. The defect accounts for the absence of Amos failures in a sample containing `28` historical Amos-failure IDs and for the large latency reduction. Corrected E6-3A measurements are reported in machine-readable scorecards with explicit timing boundaries, physical GPU UUID, failure counts and memory. Sustainable QPS and monetary cost remain unmeasured.

## 7. Limitations

The quality studies are exploratory; GaRAGe source-document independence is not established. The reader endpoint is lexical and single-reference. No human factual adjudication, noninferiority margin, production offered-rate benchmark or monetary cost model is available. Historical E6-3 serving rows are a preserved invalid-input diagnostic. The corrected serving run is bounded and does not include retrieval.

## 8. Conclusion

Decision models can be viable selectors under some fixed workloads, but relative quality is workload-dependent and specialists remain competitive and often faster. Larger general-purpose models do not automatically produce better downstream utility. A replacement decision requires a workload-specific contract covering full-content feasibility, failure rate, quality endpoint, latency and memory. The current record supports a qualified exploratory manuscript; it does not support a universal replacement, factual-correctness, cost-savings or noninferiority claim.

## Reproducibility

All new audit outputs, corrected raw traces, hashes and test results are under `results/non_jev/v2/e6_v1_0/e6_3a_integrity_and_manuscript/`. Historical E2–E6-3 artifacts are preserved.
