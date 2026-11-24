# E9 Final Scientific Report

## Final gate

`E9_BENCHMARK_ADMISSION_BLOCKED`

The authorized benchmark admission sequence completed without model inference. TREC RAG 2025 failed its prespecified judged-coverage floors before outcome inspection, and the one authorized TREC RAG 2024 fallback also failed those floors. No native selector, reader, Jev, paid API or GPU inference calls were made for E9.

## Completed

- Official topic/qrel provenance for TREC 2025 and TREC 2024 was downloaded, hashed and audited.
- TREC 2025 candidate source was verified as a published RankQwen post-reranked pool with full text; its BM25 first-stage file was not available at the checked source path.
- TREC 2024 exact test-aligned published candidate source was checked; it has complete Top-20 text but insufficient judged coverage.
- The model protocol and outcome-blind reader subset were not frozen for execution because the admission gate failed.
- Historical E2-E8 artifacts were preserved; reserved data remained unopened.

## Not performed

- Native model comparisons: not started.
- Matched selector timings: not measured.
- nDCG or paired statistics: not computed.
- Reader diagnostic: not admitted and not run.

## Scientific interpretation

The external replacement question remains unresolved by E9. Running the models on these pools would not satisfy the prespecified external-benchmark validity contract, so no quality or deployment conclusion is drawn. The correct next decision is a separately governed data-admission repair or a new protocol, not post-outcome candidate substitution.
