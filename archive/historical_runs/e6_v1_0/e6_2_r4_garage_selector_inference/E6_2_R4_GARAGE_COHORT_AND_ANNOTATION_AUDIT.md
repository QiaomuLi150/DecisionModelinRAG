# GaRAGe cohort and annotation audit

- Authenticated rows: 2366
- Unique sample IDs: 2366
- Exact K=15 pools: 2329
- Aligned grounding/relevance/correct/cited arrays: 2366
- Dataset SHA-256: `419e3941f6e8eb4082a74ca2140c1f9337f8b467ff76656a6b8b0290ca3f3a72`
- License SHA-256: `41003d4a74749c0220e33dd415042164b5a1093ed401f36277234f772d22d3d0`
- Stable source-document IDs: NOT_PRESENT
- Existing approved GaRAGe exploratory partition: NOT_FOUND

The local R0/R0R partition manifests describe SciFact partitions, not GaRAGe. No GaRAGe CAL, EXPLORATORY, or RESERVED sample-ID manifest was found. Therefore the exact-K15 rows are authenticated as eligible fixed pools but are not authorized as an evaluation cohort. An outcome-blind exact-K15 cohort hash is recorded in the pre-inference manifest as a proposal only; it is not frozen.

Grounding labels remain separate from inference inputs. `ANSWER-THE-QUESTION` is positive support; `RELATED-INFORMATION`, `OUTDATED`, `UNKNOWN`, and empty values are not silently converted into negatives.
