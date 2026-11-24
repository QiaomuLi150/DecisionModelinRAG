# E5-A Paper-Ready Result Tables

## Primary exploratory contrast

| Contrast | Estimate | 95% percentile CI | Questions | Eligible source groups | Bootstrap | Seed |
|---|---:|---:|---:|---:|---:|---:|
| PPLX Decider − Jina v3.5 equal-source-group CSSR-U | 0.005405478488953786 | [-0.003585963062112977, 0.01480540683053459] | 1502 | 587 | 5000 | 731 |

The estimate is approximately **0.541 percentage points**; the interval includes zero. The largest eligible group sensitivity is reported in `E5_A_PRIMARY_GROUP_BALANCED_STATS.json` and `E5_A_GROUP_EFFECTS_AND_SENSITIVITY.csv`.

## Scorecard and pair matrix

Use `E5_A_RETRIEVAL_CEILINGS_AND_SCORECARD.csv`, `E5_A_PAIRED_WIN_LOSS_AND_FAILURE_MATRIX.csv`, and `E5_A_SELECTED_IDS_AND_SET_AWARE_OUTCOMES.csv`. These tables retain unconditional denominators, conditional feasibility, exact selected canonical IDs, invalid outputs, and the exploratory label.

Optional BGE/Liquid/zerank-2 controls were not executed under the frozen primary completion scope; this is recorded explicitly in `E5_A_SECONDARY_POLICY_RESULTS.csv` and is not a zero score.
