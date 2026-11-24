# DecisionRAG v2.1 controlled replication — current closure

Generated: `2026-10-21T03:08:14.975893+00:00`

Final gate: **COMPLETE_STAGE0_AMENDED_STAGE3_COMPLETE**

The exact BM25-only input contract was unavailable. The owner-approved V2-HYBRID-RRF versioned amendment uses the preserved hybrid BM25-top100 + dense-top100 + RRF@60 Top-20 slates. The original blocked Stage 0 artifacts remain immutable.

## Native quality status

- `jina_full/original`: COMPLETE
- `jina_full/reversed`: COMPLETE
- `jina_full/perm_seed_20261020`: COMPLETE
- `jina_full/perm_seed_20261021`: COMPLETE
- `jina_budget/original`: COMPLETE
- `jina_budget/reversed`: COMPLETE
- `jina_budget/perm_seed_20261020`: COMPLETE
- `jina_budget/perm_seed_20261021`: COMPLETE
- `jina_two_stage_matched/original`: COMPLETE
- `jina_two_stage_matched/reversed`: COMPLETE
- `jina_two_stage_matched/perm_seed_20261020`: COMPLETE
- `jina_two_stage_matched/perm_seed_20261021`: COMPLETE
- `pplx_native/original`: COMPLETE
- `pplx_native/reversed`: MISSING_OR_INCOMPLETE
- `pplx_native/perm_seed_20261020`: MISSING_OR_INCOMPLETE
- `pplx_native/perm_seed_20261021`: MISSING_OR_INCOMPLETE
- `pplx_budget/original`: COMPLETE
- `pplx_budget/reversed`: COMPLETE
- `pplx_budget/perm_seed_20261020`: COMPLETE
- `pplx_budget/perm_seed_20261021`: COMPLETE
- `pplx_two_stage/original`: COMPLETE
- `pplx_two_stage/reversed`: COMPLETE
- `pplx_two_stage/perm_seed_20261020`: COMPLETE
- `pplx_two_stage/perm_seed_20261021`: COMPLETE

## Integrity and resource decisions

- Labels are joined only by the offline evaluator after raw traces are finalized.
- Smoke traces, partial traces and interrupted low-headroom workers are not scored as complete runs.
- The PPLX two-stage smoke on GPU 3 hit CUDA OOM during stage allocation; the PPLX budget permutation attempts on GPUs 6 and 1 were stopped/preserved when the scan showed near-zero headroom. No resident process was terminated.
- Stage 3 serving gate: `PASS_DEDICATED_GPU_SERVING_COMPLETE`. The timing session used one dedicated GPU, serial policy-major execution, 120 requests per dataset and three attempts per request.
- No Jev, paid API, training, new retrieval, or protected-holdout access occurred.

## Artifacts

- Protocol: `results/v2_1_replication/PROTOCOL_V2.json`
- Analysis exports: `results/v2_1_replication/stage4_analysis_v2_hybrid/export/`
- Hash manifest: `results/v2_1_replication/RAW_TRACE_SHA256_MANIFEST_V2.json`
- Gate: `results/v2_1_replication/FINAL_GATE_V2.json`
