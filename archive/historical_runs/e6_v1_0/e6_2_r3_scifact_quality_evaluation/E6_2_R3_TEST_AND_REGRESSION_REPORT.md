# E6-2 R3 tests and regression

- Focused R3 tests: 6 passed, 0 failed.
- Full shared-environment suite: 233 passed, 0 failed.
- Default `/usr/bin/python3` suite: 230 passed, 3 failed because that interpreter lacks already-required `pyarrow` and `torch`; the same tests pass in the designated shared ML environment.
- No model inference, reader call, retrieval rebuild, Jev call, or paid API call was made by R3.
- Historical provenance checks: E6_2_R0_INPUT_OUTPUT_PROVENANCE_MANIFEST.json: 24/24, E6_2_R1_PROVENANCE_MANIFEST_V3.json: 27/27, E6_2_R2_PROVENANCE_MANIFEST.json: 30/30, E6_2_R2_STARTLUX_REPAIR_PROVENANCE.json: 5/5.
