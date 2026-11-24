# E2-P0v2 CPU Validation Report

**Execution scope:** CPU-only static/mock validation; `CUDA_VISIBLE_DEVICES=''`; no model, GPU, network, Qdrant, service, Jev or paid API calls.

**Checks:** 18 PASS, 0 BLOCKED, 0 SKIPPED.

| Check | Status | Detail |
|---|---|---|
| `required_output_names` | `PASS` | all required files exist after generation |
| `m0_owner_gate` | `PASS` | E2_M0_COMPLETE_WITH_LIMITATIONS_READY_FOR_OWNER_REVIEW |
| `m0_pooled_denominators` | `PASS` | historical conditional denominators are retained |
| `jev_disabled` | `PASS` | local study config denies Jev |
| `no_empty_question_policy` | `PASS` | runtime contract contains fail-closed empty-query rule |
| `historical_categorical_is_local_only` | `PASS` | categorical score restriction present |
| `h2_separation` | `PASS` | three mechanisms separately specified |
| `holdout_not_falsely_frozen` | `PASS` | fresh holdout remains blocked |
| `ni_margin_owner_gate` | `PASS` | epsilon is not silently assumed |
| `power_owner_gate` | `PASS` | power/sample size remains unresolved |
| `slo_owner_gate` | `PASS` | numeric SLO remains unresolved |
| `model_capability_conservatism` | `PASS` | matrix does not promote unverified models |
| `cpu_only_imports` | `PASS` | validator imports standard library only |
| `no_network_literals` | `PASS` | validator has no network request path |
| `matrix_nonempty` | `PASS` | candidate roster represented |
| `mechanism_nonempty` | `PASS` | H2/H3/H4 matrix represented |
| `common_and_native_views` | `PASS` | both view settings specified |
| `telemetry_fail_closed` | `PASS` | telemetry requirements present |

A BLOCKED check here means an owner-gated or intentionally unresolved scientific decision, not a validator defect. The mandatory CPU implementation checks passed; the unresolved design decisions prevent scientific freeze.
