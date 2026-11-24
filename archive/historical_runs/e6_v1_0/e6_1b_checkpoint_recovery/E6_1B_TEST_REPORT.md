# DecisionRAG E6-1B Regression and Integrity Test Report

Generated: `2026-10-15T01:53:41.898213+00:00`

## Results

| suite | command/runtime | result |
|---|---|---|
| focused E6-1B | `pytest -q tests/test_e6_1b_checkpoint_recovery.py` | PASS, 5/5 |
| full relevant regression | shared `MultiModalRAG/venv`, `python -m pytest -q tests` | PASS, 203/203 |

The system-Python invocation was also attempted; it produced two environment-only failures in pre-existing E6-0R Arrow tests because `pyarrow` is absent from `/usr/bin/python3`. The designated shared environment contains `pyarrow==25.0.1`, and the complete 203-test suite passes there. No E6-1B assertion failed.

## Coverage added by this milestone

The focused suite covers frozen-input and attempt accounting, candidate-probability alignment and deterministic ties, gold-label exclusion from native requests, invalid-output gating, and preservation of the prior E6-1 gate/artifacts.

## Integrity checks

- E6-1 canonical input hash matched the frozen expected hash.
- Historical E5 Top-20 slate hash matched the frozen expected hash.
- All 12 files listed by the E6-1B provenance manifest matched their recorded SHA-256 values.
- E6-2 was not started; Jev and paid API calls remained zero.
