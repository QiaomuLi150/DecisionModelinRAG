# DecisionRAG v1.3 priority continuation report

## Final gate

`PARTIAL_COMPLETE_R5_RESOURCE_BLOCKED`

R1 matched serving timing, the six-order R3 two-stage order robustness evaluation, and the corrected R4 reader-window experiment are complete. R5’s exact pinned runtime has now been built and authenticated, but the benchmark itself is not scientifically admissible under the current resource snapshot because no dedicated A100 is available.

## R1 — matched serving and marginal-cost timing

The frozen 120-query × 3-repeat cohort was run sequentially on GPU3. PPLX budgeted selection had p50/p95 1608.98/2378.94 ms; two-stage PPLX had 3461.20/6786.65 ms; Jina full had 285.66/1130.26 ms. The fixed marginal non-selection decision request interleaved with PPLX budget had p50 1485.03 ms and 1072.67 MiB marginal peak allocation. Three PPLX-related failures per policy remain in the denominator. These measurements exclude model load, which was approximately 26.04 s for PPLX and 1.47 s for Jina.

## R3 — shuffled candidate orders

The 1,384-query cohort was executed under reverse order and five seeded permutations. Every order had 1,381 valid and 3 failed requests. Candidate sets changed on 92.847%–94.436% of queries. All six order perturbations reduced nDCG relative to original order; reversed and seeds 3 and 5 had the strongest negative paired effects, but these are order-sensitivity diagnostics and not independent policy comparisons.

## R4 — reader windows

The corrected versioned run completed 2,400/2,400 outputs across four policies, two window sizes, and the 300-question CRAG cohort. Exact tokenizer-token offset scoring is the authoritative versioned result; prior failed attempts and word-offset scoring are retained as historical diagnostics. At 1024 tokens on the representative cohort, deterministic F1 was 0.2799 for retrieval_top5, 0.2762 for jina_full, 0.2872 for jina_budget, and 0.2926 for pplx_budget. These are answer/reference agreement proxies, not factual adjudication.

## R5 — exact Winnow runtime

The exact Winnow source and llama.cpp build are now available and unit-tested. Binary/model/tokenizer provenance is recorded in `r5_winnow_budget_v2/PROTOCOL_V2.json`. The R5 timed benchmark was not launched because every GPU had an active resident process at the resource check; sharing resident services would violate the frozen dedicated-A100 protocol. No R5 model calls occurred, and the previous runtime-unavailable blocker remains preserved unchanged.

## Integrity

No Jev or paid API calls were made. No protected holdout was accessed. Historical artifacts were not overwritten. The only versioned corrections are the R4 exact-token-offset scoring correction and the new R5 build/readiness package.
