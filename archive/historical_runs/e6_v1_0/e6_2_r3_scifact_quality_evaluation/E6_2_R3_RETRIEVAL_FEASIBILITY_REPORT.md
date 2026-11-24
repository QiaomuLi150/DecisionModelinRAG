# SciFact retrieval feasibility

The frozen R1 Top-20 candidate universe was reused without regeneration. On the 277 exploratory questions, 277 have at least one judged positive somewhere in the Top-20 and 0 do not. This is a pre-reranker feasibility ceiling under sparse qrels, not factual correctness and not exhaustive negative evidence.

The R1 Top-5 baseline is scored with the identical binary nDCG@5 convention. Unjudged documents are not verified negatives; conventional nDCG assigns them zero gain under the explicit sparse-qrel scoring convention.
