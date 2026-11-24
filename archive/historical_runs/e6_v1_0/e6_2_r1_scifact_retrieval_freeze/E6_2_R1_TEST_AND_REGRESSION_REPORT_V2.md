# E6-2 R1 Test and Regression Report

- Focused R1 tests: **6/6 passed**.
- Full authoritative suite: **214/214 passed** in the shared MultiModalRAG virtual environment, where the cached Arrow inputs are readable.
- System-Python comparison: 212/214 passed; 2 existing SciFact audit tests failed only because `/usr/bin/python3` lacks `pyarrow`.
- No inference was rerun during finalization. Selector, reader, Jev and paid API calls: 0.
