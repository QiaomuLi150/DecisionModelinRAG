# When Do Decision Models Replace Specialized Rerankers in RAG?
## Evidence Quality, Robustness, and Serving Constraints

### Abstract

This exploratory study compares native decision-model policies with specialized rerankers for selecting five passages from fixed candidate pools. Evidence-selection quality is workload-dependent, and a controlled candidate-order experiment measures behavioral stability without changing candidate content. A prior serving defect was identified and corrected: the original E6-3 SciFact path supplied empty candidate passages. The corrected bounded serving diagnostics retain context failures, while downstream answer-reference agreement remains a lexical proxy rather than factual correctness. The record supports qualified external review, not universal replacement, noninferiority, or production-economics claims.

### Contributions and scope

1. Native inference-contract evaluation across decision and specialist policies.
2. Separate SciFact relevance and GaRAGe fixed-pool grounding endpoints.
3. Controlled candidate-order sensitivity using canonical, reversed, and seeded permutations.
4. Reader reference-agreement and abstention diagnostics, explicitly not factual adjudication.
5. Corrected full-content serving diagnostics and context-failure accounting.

No ranking algorithm was newly trained. Checkpoints have different training histories. Decision probability sorting is a derived Top-5 policy. SciFact and GaRAGe are different tasks; GaRAGe source-document independence is not established.

### Methods

E6-4 freezes 100 SciFact CAL and 100 GaRAGe CAL questions with seed 731. Four policies were selected before E6-4 inference: PPLX Decider, StartLux Decision, Jina v3.5, and Qwen3 Reranker 0.6B. Each receives the same question and candidate contents under three orderings. Relevance labels are loaded only after inference outputs are committed. Statistical intervals use 5,000 deterministic query-level paired bootstrap resamples, seed 731; secondary contrasts are exploratory.

### Results

The complete E6-4 structural scorecard is in `mechanism_inference/E6_4_ORDER_STABILITY_SCORECARD.csv`, and relevance changes are in `mechanism_scoring/E6_4_RELEVANCE_SENSITIVITY.csv`. The experiment reports changes and uncertainty rather than prespecifying a superiority direction. Historical SciFact PPLX−Jina nDCG@5 was +0.017400 (95% CI [-0.010549,+0.045281], n=277); GaRAGe was -0.015802 (95% CI [-0.022765,-0.008966], n=1397). These findings are not pooled.

Corrected E6-3A serving used full content. In the sampled SciFact workload, Amos had 56 context-limit failures out of 200 timed requests; the original empty-input serving rows are superseded for full-content claims. The corrected diagnostic boundary does not include retrieval or reader generation, and sustainable QPS and monetary cost were not measured.

The frozen 200-question representative reader proxy showed PPLX−Jina normalized token-F1 difference -0.003406734242 with CI [-0.009224723,+0.002650897]. This is reference agreement, not factual correctness.

### Limitations and claims boundary

No human factual adjudication was performed. No source-independent GaRAGe replication is claimed. The studies are exploratory and checkpoint comparisons, not architectural causal tests. The original E6-3 SciFact serving defect is preserved and explicitly superseded. E5-F and E6-F remain unopened. No sustainable production QPS, monetary cost, or noninferiority claim is supported.

### Conclusion

A decision model may be a practical replacement only under a workload-specific, full-content serving contract that verifies context feasibility, native input parity, relevance quality, and downstream utility. The current record supports external scientific review and a narrowly specified prospective confirmation protocol; it does not justify automatic deployment or universal specialist replacement.
