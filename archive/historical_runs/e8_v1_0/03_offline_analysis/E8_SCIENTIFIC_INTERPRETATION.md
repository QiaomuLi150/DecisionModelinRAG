# E8 scientific interpretation

E8 is a fixed original-rank input-format intervention, not an architecture-level causal test. A/R are the authenticated E7 unannotated controls; C/D are the new annotated diagnostic conditions.

## Interpretation rules

- C/D stability higher than A/R supports only an association under this exact serialization and cohort.
- Similar stability is inconclusive about internal rank use.
- Any additional context failures are part of the intervention cost.
- Relevance is conditional on valid selections; all-attempt failures are separately reported.

## Results

- `pplx_decider` / `scifact`: A/R exact=0.07692307692307693, C/D exact=0.08, A/R Jaccard=0.4748168498168498, C/D Jaccard=0.47349206349206346, common-complete n=25; Jaccard moderation=-0.010317460317460315 [-0.0680952380952381,0.04904761904761905].
- `pplx_decider` / `garage`: A/R exact=0.0, C/D exact=0.0, A/R Jaccard=0.4558531746031746, C/D Jaccard=0.4441468253968254, common-complete n=40; Jaccard moderation=-0.011706349206349205 [-0.05793650793650793,0.03888888888888888].
- `startlux_decision_4b` / `scifact`: A/R exact=0.125, C/D exact=0.2, A/R Jaccard=0.5580357142857142, C/D Jaccard=0.5904761904761904, common-complete n=40; Jaccard moderation=0.03244047619047619 [-0.012499999999999999,0.07827380952380952].
- `startlux_decision_4b` / `garage`: A/R exact=0.175, C/D exact=0.175, A/R Jaccard=0.650595238095238, C/D Jaccard=0.6267857142857143, common-complete n=40; Jaccard moderation=-0.023809523809523808 [-0.07738095238095237,0.030952380952380946].
- `jina_v3_5` / `scifact`: A/R exact=0.225, C/D exact=0.175, A/R Jaccard=0.6880952380952381, C/D Jaccard=0.6773809523809524, common-complete n=40; Jaccard moderation=-0.010714285714285716 [-0.04642857142857143,0.021428571428571425].
- `jina_v3_5` / `garage`: A/R exact=0.275, C/D exact=0.3, A/R Jaccard=0.6928571428571428, C/D Jaccard=0.713095238095238, common-complete n=40; Jaccard moderation=0.02023809523809524 [-0.022619047619047622,0.06547619047619048].
- `qwen3_reranker_0_6b` / `scifact`: A/R exact=1.0, C/D exact=1.0, A/R Jaccard=1.0, C/D Jaccard=1.0, common-complete n=40; Jaccard moderation=0.0 [0.0,0.0].
- `qwen3_reranker_0_6b` / `garage`: A/R exact=1.0, C/D exact=0.95, A/R Jaccard=1.0, C/D Jaccard=0.9833333333333334, common-complete n=40; Jaccard moderation=-0.01666666666666667 [-0.04166666666666667,0.0].

No E8 result is promoted to superiority, equivalence, internal-mechanism, factual-correctness, production-QPS or cost evidence. Small PPLX SciFact valid denominators and source dependence are retained.
