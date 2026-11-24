# E2-P0v2 Scoring Contracts

## Contract types

1. **INDEPENDENT_RELEVANCE:** identical query/document prompt and readout for every candidate; one score per document; within-query sort is allowed.
2. **JOINT_BINARY_RELEVANCE:** one shared-slate request produces candidate-aligned binary logits/probabilities; candidate IDs and output positions must be checked. Scores are ranking heuristics, not independent calibrated probabilities.
3. **LOCAL_CATEGORICAL:** one winner among a recorded group. Its probabilities have a group-specific denominator and are not globally comparable. No historical Clef/Laya quartet probabilities may be merged for H2b.
4. **FULL_SLATE_CATEGORICAL:** a single mutually exclusive slate choice is not automatically a five-document evidence score; a validated iterative readout is required.
5. **CONDITIONAL_GAIN_ESTIMATE:** a model-estimated gain given selected set S; exploratory and gold-blind only.

## Required normalized record

Every output must carry `query_id`, ordered canonical `candidate_ids`, `selected_ids`, `score_type`, `score_contract_version`, `score_values` aligned by candidate ID, raw response hash, actual model-visible input hash, model-call ID, physical-forward counter when instrumentable, truncation flags, cache key/provenance, timestamps, retry/error status, model revision and configuration hash.

## Ranking and ties

Sort only scores explicitly declared comparable within the query. Use descending score, original frozen candidate rank, then canonical candidate ID as deterministic tie-break. Top-5 must be exactly five unique in-slate identities unless an explicit failure is recorded.

## Unsupported interfaces

Missing native context limits, undocumented token readouts, unknown score denominators, arbitrary generation strings, absent candidate alignment, and unmeasured physical passes are `UNVERIFIED`/`INFEASIBLE`, never silently promoted to PASS.
