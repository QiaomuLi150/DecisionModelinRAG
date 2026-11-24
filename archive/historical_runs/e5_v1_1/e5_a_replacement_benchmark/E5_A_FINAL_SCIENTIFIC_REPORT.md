# DecisionRAG E5-A Final Scientific Report

## Gate

`E5_A_OWNER_REVIEW_READY`

## Primary result

PPLX Decider minus Jina v3.5 equal-source-group set-aware CSSR-U: **0.005405478488953786** (about **0.541 percentage points**), 95% CI **[-0.003585963062112977, 0.01480540683053459]** (about **-0.359 to 1.481 percentage points**), over **1502** exploratory questions and **587** eligible source groups. The interval includes zero; this is exploratory evidence-selection retention, not factual correctness or a confirmation claim.

### Main scorecard

| Policy | CSSR-U successes / all | CSSR-U | Feasible Top-20 n | CSSR-C successes / feasible | CSSR-C |
|---|---:|---:|---:|---:|---:|
| Retrieval Top-5 | 220 / 1502 | 0.14647137150466044 | 355 | 220 / 355 | 0.6197183098591549 |
| PPLX Decider | 306 / 1502 | 0.20372836218375498 | 355 | 306 / 355 | 0.8619718309859155 |
| Jina v3.5 | 299 / 1502 | 0.19906790945406125 | 355 | 299 / 355 | 0.8422535211267606 |

The retrieval-feasibility ceiling was **355 / 1502** (0.23635153129161118); questions outside that ceiling are unconditional failures for CSSR-U. Paired outcomes were both-success **275**, PPLX-only **31**, Jina-only **24**, and both-failure **1172**. Exact machine-readable denominators are in `E5_A_RETRIEVAL_CEILINGS_AND_SCORECARD.csv` and `E5_A_PRIMARY_GROUP_BALANCED_STATS.json`.

Native failures, missingness, attempts and telemetry are preserved rather than filtered. A0 had 1502/1502 valid PPLX outcomes and 1502/1502 valid Jina outcomes; each had 1502 and 1502 recorded physical forward calls, respectively. H0 had 12/12 and 12/12 valid smoke outcomes. Regression tests: 5 targeted and 162 full-suite tests collected and passed.

No Jev or paid API calls were made. E5-B/C/D were not executed and E5-F remains locked.

## Claim limits

The result compares two pinned model-policy bundles on one frozen QASPER exploratory cohort. Source grouping does not prove no pretraining exposure; QASPER is a scientific-paper text corpus, not enterprise traffic; annotated support retention is not answer correctness or evidence exhaustiveness. No serving QPS/cost, reader utility or confirmation claim is made. E5-F remains locked.
