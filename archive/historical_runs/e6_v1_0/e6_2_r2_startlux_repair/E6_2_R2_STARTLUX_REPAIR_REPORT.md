# E6-2 R2 StartLux Runtime Repair Addendum

## Gate

`E6_2_R2_STARTLUX_REPAIRED_VALIDATED`

This is a versioned repair addendum. The original
`e6_2_r2_selector_inference` artifacts remain unchanged, including their
`E6_2_R2_SELECTIONS_FROZEN_WITH_EXCLUSIONS` gate.

## Root cause and repair

- The StartLux checkpoint was present locally at
  `data/models/e6_1b/startlux_decision_v1_4b`.
- The active shared environment lacked the CUDA dependencies required by the
  checkpoint: `causal-conv1d` and `flash-linear-attention`.
- Installed in the shared ML environment: `causal-conv1d==1.7.0` and
  `flash-linear-attention==0.5.2`.
- Transformers 5.16.1 no longer exposes the older
  `is_fast_path_available` marker. The bundled StartLux detector was updated
  to validate the actual `causal_conv1d` and FLA entry points instead.
- Direct dispatch audit passed for causal-conv1d and FLA. No slow-path
  fallback was used for the repaired run.

## Bounded native validation

- Population: the same 277 frozen, gold-blind R1 SciFact exploratory Top-20
  slates used by R2.
- Model revision: `9c0a32e9809d6a3f612434dd0bce150295dcc274`.
- GPU: physical GPU 5.
- Valid outputs: **277/277**.
- Failed outputs: **0**.
- Every output contained exactly five unique canonical candidate IDs.
- Observed physical forwards: **277** (one per query).
- Latency p50/p95: **775.807 / 1,373.746 ms**.
- Peak allocated/reserved memory: **12,094.73 / 17,230.00 MiB**.
- Input tokens: **4,141,025** total.
- Quality metrics: not run; qrels, reserved content, readers, Jev and paid
  APIs were not accessed.

## Smoke and regression validation

- Direct kernel-dispatch audit: PASS.
- Bundled kernel check after detector repair: `fast kernels: active`.
- CAL-only native Top-20 → Top-5 smoke: PASS; one physical forward and
  probability sum `1.0000000512`.
- Runtime and R2 integrity tests: **13/13 passed** (11 existing R2 tests plus
  2 StartLux repair tests).
- Full project regression suite: **227/227 passed**.
- The original R2 result bundle was not merged or overwritten.

## Provenance

| Artifact | SHA-256 |
|---|---|
| Repaired native trace | `eb118db268bcb3fc1cbb4b09bb5bdb286fac013dcc83486104d6060f338c5f57` |
| Patched StartLux runtime | `93fa62c4b447587cd28d4e8c22979674b7ede0153e5312da12887a1e7ec40bb3` |
| Repair regression test | `8c28508993ac22ff106c9a11f86106f811fa4f0a8e5a36a18fff54147f9f117a` |

R3 relevance scoring remains separately gated and was not started.
