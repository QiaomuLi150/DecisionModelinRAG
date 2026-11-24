# E4-2 Test Report

Targeted E4-2 tests: **10 passed, 0 failed** (`python3 -m pytest -q tests/test_e4_2_facet_pilot.py`).

Combined E4-0/E4-1/E4-2 tests: **18 passed, 0 failed**.

Full repository regression: **135 passed, 0 failed** (`python3 -m pytest -q`).

Gate assertions covered exact frozen cohorts, question-only facets, no gold/reference/support leakage, five policies, exact Top-20/Top-5 IDs, explicit failures, separate representative/diagnostic strata, common candidate-view hashes, zero Jev/paid calls, no holdout outcomes, phase-separated work, and provenance input hashes.
