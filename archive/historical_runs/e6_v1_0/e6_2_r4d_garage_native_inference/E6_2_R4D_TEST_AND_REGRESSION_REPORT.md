# DecisionRAG E6-2 R4D — Test and Regression Report

## Status

- R4D focused tests: **6/6 passed**.
- Full project regression suite: **251/251 passed**.
- Test command: `python -m pytest -q` using the shared MultiModalRAG Python 3.12 environment.
- Collection count: 251 tests.
- No model inference, quality scoring, reserved-data access, Jev call or paid API call occurred during testing.

## Coverage

The focused suite verifies:

1. frozen 1,397-question / 11,176-attempt population;
2. complete accounting for all eight models;
3. unique `(model, sample_id)` checkpoint keys;
4. exact 15-candidate inputs and valid five-item selections;
5. label-blind and reserved-data isolation;
6. readiness-gate values and explicit forward-count missingness.

The full suite includes all prior E2–E6-2 R4C tests plus the six R4D tests. Existing StartLux runtime warnings were non-failing dependency warnings; no test failed.

## Telemetry correction

The final derived reports distinguish measured hook counts, deterministic contract-derived counts, and `NOT_MEASURED`. A numeric zero from an unavailable internal hook is not treated as an observed zero. Raw append-only model chunks remain preserved.
