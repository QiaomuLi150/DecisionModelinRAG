# E6-2 R5 Test and Regression Report

- Focused R5 tests: **8/8 passed**.
- Full repository regression suite: **259/259 passed** (251 pre-existing tests plus 8 R5 tests).
- Toy metric cases covered: all-positive, zero-positive, rank-1, rank-5, >5 positives, ties, unresolved annotation, invalid selection.
- R4C partition/input and R4D prediction/provenance checks: PASS.
- No new model forwards, retrieval calls, reader calls, Jev calls or paid API calls.
- RESERVED rows exported to scorer: 0.
- Historical artifacts modified: false.

The full suite produced only previously observed StartLux dependency warnings; no test failure occurred.
