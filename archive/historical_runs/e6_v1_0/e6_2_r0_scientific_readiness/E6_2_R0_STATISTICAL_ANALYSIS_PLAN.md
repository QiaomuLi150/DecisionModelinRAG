# E6-2 R0 Statistical Analysis Plan

Primary new contrast: SciFact PPLX binary nDCG@5 minus Jina binary nDCG@5 on identical frozen Top-20 slates in the 277-query EXPLORATORY cohort. Use per-query paired deltas and 5,000 deterministic paired bootstrap resamples, seed 731, percentile two-sided 95% interval. Report exact n, failures, no-positive qids and missingness.

Source-group resampling is reported only if the qrel-connected component mapping and partition provenance support it; otherwise query bootstrap is primary and source-group output is descriptive. The existing SciFact graph has 572 identity components, but that does not prove latent near-duplicate independence.

Secondary family: each decision model versus each prespecified specialist (Jina, Qwen3-4B, BGE-v2-m3, Qwen3-0.6B), retrieval Top-5 control, Recall@5, Hit@5 and judged-coverage sensitivity. Secondary multiplicity is controlled by reporting all planned comparisons with Holm-adjusted exploratory labels; no post-outcome endpoint selection or noninferiority margin is permitted.

GaRAGe uses query-level paired bootstrap only because stable source groups are unavailable. Track A historical effect is not pooled with SciFact or GaRAGe.
