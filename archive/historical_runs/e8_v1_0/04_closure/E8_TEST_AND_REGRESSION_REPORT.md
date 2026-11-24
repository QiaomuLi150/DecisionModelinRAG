# E8 test and regression report

## Environment

- Python: `./Production_Project/MultiModalRAG/venv/bin/python`
- Focused command: `python -m pytest -q tests/test_e8_rank_metadata.py`
- Full command: `python -m pytest -q`
- Full collection command: `python -m pytest --collect-only -q`

## Results

- `focused_returncode: 0`
- Focused result: **6 passed, 0 failed, 0 skipped**.
- `full_returncode: 0`
- Full result: **301 passed, 0 failed, 0 skipped**.
- The previous E7 baseline of 295/295 is therefore preserved and extended by six E8 tests.
- Warnings: existing Triton platform fallback and torch JIT deprecation warnings only; no test failures.

## E8 integrity coverage

The focused tests validate the pre-outcome freeze, exact 640-attempt accounting, C/D rank identity under reversal, label-blind raw inputs, explicit failure rows, Top-5 uniqueness, separate workload denominators, and zero Jev/paid/protected access.

No model output was changed by testing. Historical E2–E7 directories were not modified.
