# E9-BRIGHT final interpretation and evidence freeze

## Closure status

`E9_BRIGHT_INTERPRETATION_OWNER_REVIEW_READY`

The original E9-BRIGHT experiment was not rerun. Frozen outputs and the blocked TREC E9 attempt remain unchanged.

## Original operational endpoint

The recomputed PPLX−Jina macro-domain operational binary known-positive nDCG@5 effect is `-0.098372` with domain-resample 95% CI `[-0.15503154829573165, -0.0434499211615929]`. This reproduces the source E9 primary within floating-point tolerance. PPLX valid output rate was 611/1,384; Jina was 1,384/1,384.

## Common-valid diagnostics

PPLX/Jina common-valid n=611 across 11 domains. The paired result is in `E9_BRIGHT_COMPARISON_TABLE.csv`; it is a selected-success population and must not be interpreted as an all-query causal effect. Liquid/Qwen common-valid n=1382 across 12 domains. Fixed-domain query and hierarchical domain/query intervals are both reported.

## Retrieval feasibility

The frozen Top-20 contains at least one known positive for 607/1,384 queries across all 12 domains. Feasible-subset comparisons are reported separately. `gold_ids` are known-positive annotations, not exhaustive verified negatives or complete-support sets; no gold document was inserted and no slate was changed. Full-gold-set nDCG normalization and in-pool conditional normalization are kept separate.

## Failure-aware fallback

The post-hoc fallback policy uses the frozen retrieval Prefix-5 only after a native failure. PPLX fallback and Qwen fallback comparisons are reported in the comparison table. This is not the original endpoint and is not free: failed requests can consume native compute and latency.

## Reader and factual correctness

BRIGHT downstream reader evaluation was not admitted. No factual correctness, answer grounding, CSSR, or official BRIGHT leaderboard equivalence is claimed.

## Final interpretation

Existing decision models do not show a defensible universal replacement advantage over specialist rerankers in this record. PPLX is operationally disadvantaged by native context feasibility; Liquid versus Qwen is not a clear macro-domain superiority result; and reader utility remains unverified.

See the latency audit, final assessment, comparison table, evidence ledger, integrity report and provenance manifest for traceable details.
