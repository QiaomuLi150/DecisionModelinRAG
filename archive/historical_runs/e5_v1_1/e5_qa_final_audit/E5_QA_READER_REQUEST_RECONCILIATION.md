# E5-QA Reader Request Reconciliation

Verdict: `ACCOUNTING_RECONCILED`.

The nominal count is 2 policies × (200 representative + 55 diagnostic) = **510** cohort slots. The representative and diagnostic qid sets actually overlap in **7** qids, listed in the slot ledger. Their union contains **248** qids, so the unique policy–qid request count is 2 × 248 = **496**, exactly matching the **496** raw reader rows.

- Physical fresh inference rows: 496.
- Accepted statuses: `{'OK': 496}`.
- Duplicate policy–qid rows: `[]`.
- Application-cache reuse rows: 0.
- Diagnostic analysis population remains 55 qids / 110 cohort slots, but only 96 diagnostic-only physical rows were needed because 14 shared slots were already represented by the same accepted outputs used by the representative cohort.
- No missing or failed accepted task was reconstructed; no row was invented.
- A separate attempt/retry event log was not present. The raw accepted file has one row per unique policy–qid and no non-OK rows, so retries cannot be independently enumerated beyond this bounded evidence.

This explains 496 versus 510 without treating cross-cohort shared physical rows as independent observations.
