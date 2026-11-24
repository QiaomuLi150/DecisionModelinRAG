# Limitations and validity threats

1. SciFact qrels are sparse. Unjudged passages are not independently verified negatives, so nDCG/Hit/Recall are judged-evidence metrics rather than exhaustive relevance or factual correctness.
2. The primary uncertainty is query-level paired bootstrap (5,000 resamples, seed 731). Qrel-connected source groups are reported descriptively only because the R0 source-independence contract does not justify group bootstrap.
3. The retrieval Top-20 ceiling is a property of the frozen hybrid retrieval slate; it bounds observable judged-positive recovery, not true evidence availability outside the slate.
4. Amos's 83 failures remain in the full denominator with zero operational utility. Its valid-only score is a feasibility-conditioned diagnostic. Failure token lengths are NOT_MEASURED in preserved R2 metadata.
5. Resource and latency values are inherited from historical R2/StartLux-repair traces; R3 performed zero new physical forwards and cannot establish a fresh serving comparison.
6. The models are separately trained checkpoints; these results do not identify architecture-level causality or factual answer correctness.
