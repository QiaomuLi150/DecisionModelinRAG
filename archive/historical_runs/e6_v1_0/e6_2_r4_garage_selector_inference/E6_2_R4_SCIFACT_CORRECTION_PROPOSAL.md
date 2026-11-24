# E6-2 R4 SciFact correction proposal

The R3 primary nDCG@5 result is reproducible and should be retained. The following R3 feasibility fields require versioned correction without modifying the frozen R3 artifacts:

- Reported Top-20 feasibility: 277/277.
- Correct Top-20 feasibility: 254/277 (0.916968).
- Missing judged-positive Top-20 slates: 23.
- Affected fields: per-query `top20_feasible`, `top20_feasible_n`, `top20_judged_positive`, and the R3 retrieval-feasibility report.
- Cause: the R3 helper populated `top20_feasible` from the existence of any qrel for the query, not from membership of a positive qrel document in the frozen Top-20 candidate IDs; its `top20_judged_positive` field likewise received selected Top-5 IDs rather than the Top-20 IDs.

This proposal is an erratum candidate, not an applied historical edit. Any corrected scorecard/report must be versioned and owner-reviewed before downstream interpretation.
