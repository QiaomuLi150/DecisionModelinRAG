# E6-2 R0 SciFact Retrieval Readiness

## Status: `AUTHENTIC_SCIFACT_RETRIEVAL_FREEZE_REQUIRED`

Independent local Arrow audit found the authentic public release: 5,183 corpus documents, 1,109 queries, 919 train qrel rows, 339 test qrel rows, and 1,258 total qrel rows. Query/document joins are complete at the release-identity level; sparse qrels are not exhaustive negatives.

The historical E5 hash `be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7` is for the QASPER Top-20 manifest, not SciFact. E6-1’s `a1b006a79456aeba818fe3faf0059a56c1eb1c6adaf47483cac665845097f318` manifest is a deterministic technical candidate fixture. No authenticated natural SciFact Top-20 slate manifest was found in the repository search (`0 candidate file matches`).

## Required separate retrieval-only phase

Freeze before model outcomes: all 1,109 public SciFact corpus documents; query cohort (primary future quality cohort: the 277-query EXPLORATORY partition); the selected pretrained retriever(s), exact revision, tokenizer, index construction, BM25/dense/RRF parameters if hybrid, preprocessing, candidate deduplication, stable tie-breaking, Top-20 schema, and a qrel-free execution receipt. Join qrels only after slates are immutable. Preserve CAL (555) for adapter/development and do not open RESERVED (277).

This phase must not optimize against qrels or model outcomes and needs its own owner authorization. Until it is complete, E6-2 SciFact quality inference is not scientifically ready.
