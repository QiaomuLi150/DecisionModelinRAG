# E6-0 QASPER Answer-Type and Validity Report

The official QASPER evaluator contract was reproduced for **496** accepted reader rows. It scores answer F1 only, taking the maximum token F1 over annotation references. It does not score evidence F1 here, and it does not establish factual correctness.

| Cohort | Policy | QIDs | Accepted rows | Mean official Answer F1 | Group-bootstrap 95% CI |
|---|---|---:|---:|---:|---|
| representative | pplx_decider | 200 | 200 | 0.210600 | [0.16957719999316348, 0.25572076649950737] |
| representative | jina35 | 200 | 200 | 0.199394 | [0.15914806617313193, 0.24339191606109456] |
| diagnostic_enriched | pplx_decider | 55 | 55 | 0.251091 | [0.17934039913160854, 0.3448399701031046] |
| diagnostic_enriched | jina35 | 55 | 55 | 0.167772 | [0.10588985950473268, 0.24194656238085893] |

The scorer preserves QASPER's answer-type precedence: unanswerable, extractive span list joined by comma-space, free-form, then yes/no. Rows with multiple annotations are scored against each reference independently and retain the maximum; alternatives are not unioned. The source URL was inspected, but no local git commit or source-file hash was supplied by the archived release, so the scorer provenance is **limited rather than a fully pinned source snapshot**.

The frozen E5-C exact-match/reference-agreement results remain unchanged. These F1 values are post-hoc repair analyses only. `Unanswerable` is an answer string in the official scorer, not a factual-verification label.
