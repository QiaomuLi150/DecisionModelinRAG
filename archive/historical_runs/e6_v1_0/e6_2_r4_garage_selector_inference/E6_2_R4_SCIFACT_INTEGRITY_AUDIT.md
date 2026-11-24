# E6-2 R4 SciFact integrity audit

## Result: MATERIAL REPORTING DEFECT FOUND

The independent CPU implementation reproduced the frozen cohort, original qrel mapping, corpus IDs, retrieval-construction flags, binary nDCG arithmetic and PPLX–Jina bootstrap. It **did not reproduce the R3 Top-20 feasibility claim**: the correct count is 254/277 (0.916968), with 23 frozen slates containing no judged-positive document.

- Exploratory queries: 277
- qrel rows: 308
- Positive-label distribution: {1: 254, 2: 17, 3: 5, 5: 1}
- Correct Top-20 judged-positive feasibility: 254/277
- Candidate IDs absent from corpus: 0
- Positive-document ranks observed in Top-20: {1: 186, 2: 35, 4: 11, 18: 2, 15: 2, 7: 4, 6: 8, 13: 2, 12: 2, 11: 1, 17: 2, 8: 4, 20: 1, 3: 11, 5: 2, 19: 2, 16: 2, 10: 2}
- R1 rows with qrels-accessed flag: 0
- R1 rows with reserved-content flag: 0
- Reproduced PPLX–Jina Δ: 0.017400; CI [-0.010549, 0.045281]

The defect is in R3's feasibility field/reporting path: it tested whether the query had any qrel anywhere, rather than whether a judged-positive document occurred in that query's Top-20. The primary nDCG@5 arithmetic and PPLX–Jina bootstrap reproduce exactly. R3 artifacts remain unchanged; a separate correction proposal is provided. No retrieval or selector was rerun.
