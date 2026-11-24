# DecisionRAG v3.6 targeted audit status

| Stage | Status | Explanation |
|---|---|---|
| A1 | PARTIAL | The exact frozen v1.5/v1.6 pair was audited, but raw serialized text strings were not stored; exact serialized-input hashes and all list/set differences are exported. The claimed 197 differences are not present: 1,069 list and 789 set differences were observed. |
| A2 | PARTIAL | v1.5/v1.6 tokenizer/dtype/batch/attention settings are unrecoverable from their frozen receipts. Existing fp32 batch-1 C2 exports were reused without rerun; no historical setting cause is identifiable. |
| A3 | PARTIAL | Both historical values are retained with explicit protocol labels; no replacement value is scientifically justified. |
| B1 | PARTIAL | Clean timing completed for the available frozen 500 Enterprise + 2,167 MuSiQue cohort with two sessions/two passes and Jina anchors; the requested additional 333 MuSiQue slates do not exist. |
| B2 | PARTIAL | p95 and anchor-relative bootstrap ratios are exported for the available cohort; the full requested 2,500-question MuSiQue population remains blocked and sub-99% policies are NOT_COMPARABLE. |

No labels were included in raw timing traces. No paid API was called. No historical artifact was overwritten.
