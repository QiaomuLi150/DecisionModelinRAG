# E6-2 R4C SciFact erratum closure

## Status: `SCIFACT_ERRATUM_VERIFIED`

The independent CPU audit evaluates frozen R1 Top-20 candidate membership against the original qrels. The corrected judged-positive feasibility is **254/277 (0.916967509)**, not the prior R3 report of 277/277. There are 23 frozen Top-20 slates with no judged-positive document.

The R3 helper defect is localized to feasibility reporting: it used existence of any qrel for a query instead of positive-qrel membership in that query's actual Top-20, and its `top20_judged_positive` field used selected Top-5 IDs. Affected fields are `top20_feasible`, `top20_feasible_n`, `top20_judged_positive`, and the feasibility report/scorecard. The frozen binary nDCG@5 primary does not depend on those fields.

Independent reproduction of the primary PPLX−Jina result: **Δ +0.017400; 95% CI [-0.010549, +0.045281]**, n=277, 5000 query bootstrap replicates, seed 731. Exact primary point and CI match the R3 recorded result. No R3/R4 file was edited.

The corrected feasibility is a versioned erratum and must replace, not silently mutate, the old feasibility fields in later reporting.
