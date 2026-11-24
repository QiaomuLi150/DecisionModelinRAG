# E2-N Scoring and Information-View Contract (N0)

Status: `N0_READY_AWAITING_GPU_AUTHORIZATION`; this document freezes the
model-neutral rules for owner review only. It authorizes no inference.

## Workload

- Reuse immutable Track A (3,767 questions) and Track B (500 questions) Top-20
  candidate slates. No retrieval, indexing, or holdout access is part of E2-N.
- Input is `(canonical nonempty question, ordered 20 candidate IDs, candidate
  views)` and output is exactly five unique selected IDs.
- Track A and Track B remain separate. Historical metrics are exploratory.

## View settings

Setting A is the common semantic view: canonical Unicode/whitespace-normalized
question, frozen candidate ID/order, title plus frozen passage text, with native
tokenization and truncation recorded. A model that cannot fit this view is
`INFEASIBLE`, not silently shortened.

Setting B is model-native and CAL-only configurable. Any model-specific prompt,
batching, chunking, or context limit must be versioned and hashed before the
historical run. It cannot replace Setting A in the common-information result.

The historical M1 text view is 64 question tokens, 128 passage tokens, and
Clef's recorded adapter ceiling of 16,384 encoded tokens. These are evidence,
not assumptions for newly proposed models.

## Score contracts

1. `INDEPENDENT_RELEVANCE`: one comparable score per candidate; descending
   score, then original frozen rank, then canonical ID.
2. `JOINT_BINARY_RELEVANCE`: one shared-slate request with candidate-aligned
   binary outputs. Scores are within-query ranking heuristics, not globally
   calibrated independent probabilities.
3. `LOCAL_CATEGORICAL`: group-specific denominator; never pool quartet scores.
4. `FULL_SLATE_CATEGORICAL`: not a Top-5 score without a validated native
   iterative/full-ranking readout.
5. `CONDITIONAL_GAIN_ESTIMATE`: exploratory only and gold-blind.

Every result must carry candidate alignment, score direction, revision/config
hash, raw response hash, actual model-input hash, physical-forward count,
truncation/overflow, cache key/provenance, timing/GPU telemetry, retry/error
status, and `gold_in_inference_input=false`.

## Failure behavior

Blank questions, duplicate/missing/out-of-slate IDs, score cardinality or
alignment errors, unknown score denominators, unsupported native interfaces,
unrecorded context overflow, cache reuse without exact-input provenance, or
missing physical-work evidence are explicit blocked/error records.
