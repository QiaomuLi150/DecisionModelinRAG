# E4-3 Input Provenance and Hash Audit

All listed inputs are read-only references. The E3-D exploratory protocol declares `reserved_ids_accessed=false`, `gold_accessed_during_generation=false`, `jev_calls=0`, and `paid_api_calls=0`. E4-2 artifacts are preserved and not overwritten.

## Input hashes

- `results/non_jev/v2/e3_v1/e3_d_external/E3_D_FROZEN_EXPLORATORY_PROTOCOL.json`: `ed325ad6fcefed15e80fedfc5398bca43ee28473c56a74e97b0c7ce336494f43` (26318 bytes)
- `results/non_jev/v2/e3_v1/e3_d_external/E3_D_EXPLORATORY_METADATA.jsonl`: `02e55ef3d91af0cc12f735169494c57ccf511901827b5da87ab0256091b8fe1f` (2586468 bytes)
- `results/non_jev/v2/e3_v1/e3_d_external/E3_D_FIRST_STAGE_CANDIDATES.jsonl`: `c532d37210e6381dd1bec8ae3533325c68bae7e16020677238c8a77ad77ffadc` (17179659 bytes)
- `results/non_jev/v2/e3_v1/e3_d_external/E3_D_CANDIDATE_VIEWS.jsonl`: `eef58ce32354ba0e13305ccd83d0295c48466be225525a972d3d54d77cc801d4` (236096536 bytes)
- `results/non_jev/v2/e3_v1/e3_d_external/E3_D_NATIVE_SELECTION_TRACES.jsonl`: `cd520976d181107c49edc5ef8098830bb861542cc9f914110a782e6ed53a2386` (13416146 bytes)
- `results/non_jev/v2/e4_v1_1/e4_2_facet_pilot/E4_2_INPUT_AND_COHORT_MANIFEST.json`: `333107471bea556f848bbfd64c0dac54d558128befddfe7e299ce051ae579fa0` (796314 bytes)
- `DecisionRAG_E4_3_Operating_Regimes_Execution_SDD_Codex.md`: `96563691fd29c814eb64e55872179358d75e1ee152b9faf59ff728e0c5b49c6c` (23032 bytes)

Prior E4-2 provenance status: `E4_2_PILOT_OWNER_REVIEW_READY`. New E4-3 cohort excludes `the prior E4-2 qid set` by deterministic ID comparison.
