# DecisionRAG v1.5 Regression and Integrity Report

Focused checks: 12/12 PASS.

| Check | Status | Detail |
|---|---|---|
| C2 request count | PASS | observed=3960 expected=3960 |
| C2 request-key uniqueness | PASS | rows=3960 unique=3960 |
| C2 fallback count | PASS | observed=360 expected=360 |
| C2 fallback candidate integrity | PASS | all fallback rows have five candidates |
| C2 policy summaries | PASS | rows=11 |
| C2 bootstrap rows | PASS | rows=16 |
| C2 R1 baseline ratios | PASS | three non-baseline policies × two bases |
| C2 E12 ratios | PASS | five non-baseline policies × two bases |
| blocked stages made no calls | PASS | C1/C3/C4 model_calls=0 |
| v1.5 artifact sanitization | PASS | no forbidden path/host/process tokens |
| historical source hashes | PASS | verified=11/11 |
| historical artifacts unchanged by v1.5 | PASS | source record states unchanged |

No GPU inference or external API calls were made by v1.5. Historical artifacts were not modified.

## Repository-wide pytest

The available repository suite completed with 3 dependency-gated failures and no v1.5-specific failures. The two SciFact closure failures require the already-cached pyarrow reader; the StartLux runtime test requires torch. These dependencies were not installed because v1.5 forbids model/runtime changes. The v1.5 focused integrity suite remains 12/12 PASS, and both new v1.5 scripts pass Python compilation.
