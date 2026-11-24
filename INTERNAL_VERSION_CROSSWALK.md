# Internal experiment labels to paper sections

The mapping uses the arXiv v1 paper (2610.34816) and the source inventory. It does not imply identical protocols across rows.

| Internal path | Paper section/item | Use in release | Caveat |
|---|---|---|---|
| `results/non_jev/v2/e2_n/` | Table I historical screen contributors; supplementary model-policy evidence | E2-N score summaries are included as an explicitly different limited screen | Not the 21×11/104-run table; its own report documents a smaller roster and different question cohorts |
| `results/v1_3/r2_order_sensitivity/` | Section IV-B candidate-order controls | Per-query/order score exports | Keep policy/order populations distinct |
| `results/v1_3/r3_multipass_ordered/` | Section IV-B multi-stage and order controls | Ordered multi-pass exports | Not all values share input serialization with later matched-text control |
| `results/v1_5/c1_two_stage_specialist_gpu3/` | Section IV-B two-stage comparison | Per-query and contrast outputs | Preserve the Jina discrepancy; do not combine with matched-text run |
| `results/v1_6/d1_perwindow_jina/` | Section IV-B identical PPLX-window-text control | Per-query and domain outputs | Different two-stage Jina input serialization explains non-identical traces, but full cause was not identified |
| `results/v1_3/r1_serving_gate_v2/`, `results/v1_5/c2_latency_uncertainty/`, `results/v1_6/d2_matched_timing/` | Sections IV-C/D serving checks | Result summaries and per-request latency | Distinguish cohorts, sessions, and serial vs concurrent conditions |
| `results/v2_1_replication/` | Reproducibility and transfer-slate lineage | Frozen reports and derived transfer exports | Historical replication has multiple versions; use final v3 analysis where available |
| `results/v3_6_essential/stage1_quality_v2/` | Sections IV-A/B; BRIGHT all-request quality | Per-query quality table and summary | Do not replace frozen values |
| `results/v3_6_essential/followup_v3_repro_fixes/` | Section IV-E corrected transfer budget table and builder audit | Corrected per-query and summary exports | Shares frozen request builder; 2,167 MuSiQue requests only |
| `results/v3_6_essential/frozen_prospective_validation_v2/` | Section IV-F prospective CQADupStack validation | ID-only candidates, scored metrics, clean serving exports, final closure | 20 dry-run records excluded; clean and contended traces separated |
| `results/non_jev/v2/e9_bright_v1_0/` | BRIGHT robustness/benchmark extension; supports audits, not original Table I | Score-only trace and per-query metrics | E9 scope and policy roster differ from the main paper comparison |
| `results/non_jev/v2/e9_bright_interpretation_v1_0/` | Evidence reconciliation and discrepancy audit | Authoritative E2–E9 ledger and correction notes | Later interpretation only; does not overwrite earlier reports |
| `paper/tables/`, `paper/figures/` | Locally present paper-side candidate assets | Included only under `paper/local_assets/` | Their association with arXiv tables/figures is not authenticated |
