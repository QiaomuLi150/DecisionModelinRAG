# E2-N2-R Forward Count Reconciliation

Status: `PASS_WITH_LIMITATIONS`

This is an offline reconciliation of normalized E2-N2 rows and their authenticated source manifests. No model or service was called.

## Row and forward definitions

- A normalized row is one policy/query evidence record in `E2_N2_MODEL_SCORES.jsonl`.
- `physical_forward_calls` is the adapter-reported physical forward count recorded in that row; it is not an HTTP/API request count.
- `physical_forward_evidence_rows` counts rows carrying the explicit evidence flag, not forward calls.
- API/service call counts and timestamps were not recorded in the frozen traces and are therefore `UNVERIFIABLE`, not inferred from row counts.

## Per-policy reconciliation

| Policy | Queries | Logical candidate decisions | API calls | Instrumented forwards | Source evidence | Blocked/error | Forward distribution | Cache reuse | Gold in input |
|---|---:|---:|---|---:|---|---:|---|---:|---:|
| retrieval5 | 4267 | 0 | NOT_APPLICABLE | 0 | results/non_jev/v11/track_a+track_b/*_RETRIEVAL_SLATES.jsonl | 0 |  | 0 | 0 |
| bge_pointwise | 4267 | 85340 | NOT_RECORDED | 4267 | results/non_jev/v2/e2_n/n3/E2_N_N3_BGE_FULL.jsonl | 0 | ok:4267 | 0 | 0 |
| laya_pointwise | 4267 | 85340 | NOT_RECORDED | 21335 | results/non_jev/v2/e2_n/n3/E2_N_N3_LAYA_FULL.jsonl | 0 | ok:4267 | 0 | 0 |
| clef_pointwise | 4267 | 85340 | NOT_RECORDED | 4267 | results/non_jev/v2/e2_n/n3/E2_N_N3_CLEF_FULL.jsonl | 0 | ok:4267 | 0 | 0 |
| clef_joint | 4267 | 85340 | NOT_RECORDED | 4267 | results/non_jev/v2/e2_n/n3/E2_N_N3_CLEF_FULL.jsonl | 0 | ok:4267 | 0 | 0 |
| gte | 4267 | 85340 | NOT_RECORDED | 4267 | results/non_jev/v2/e2_n/n3/E2_N_N3_GTE_FULL.jsonl | 0 | ok:4267 | 0 | 0 |
| minilm | 4267 | 85340 | NOT_RECORDED | 4267 | results/non_jev/v2/e2_n/n3/E2_N_N3_MINILM_FULL.jsonl | 0 | ok:4267 | 0 | 0 |
| jina35 | 4267 | 85340 | NOT_RECORDED | 4267 | results/non_jev/v2/e2_n/n3/E2_N_N3_JINA35_FULL.jsonl | 0 | ok:4267 | 0 | 0 |
| nimble | 4267 | 85340 | NOT_RECORDED | 4267 | results/non_jev/v2/e2_n/n3/E2_N_N3_NIMBLE_FULL.jsonl | 0 | ok:4267 | 0 | 0 |
| intern_decision | 4267 | 75340 | NOT_RECORDED | 3767 | results/non_jev/v2/e2_n/n3/E2_N_N3_INTERN_DECISION_FULL.jsonl | 500 | context_blocked_native_contract:500, ok:3767 | 0 | 0 |
| pplx_decider | 4267 | 85340 | NOT_RECORDED | 4267 | results/non_jev/v2/e2_n/n3/E2_N_N3_PPLX_DECIDER_FULL.jsonl | 0 | ok:4267 | 0 | 0 |
| qwen06 | 4267 | 85340 | NOT_RECORDED | 4267 | results/non_jev/v2/e2_n/n3/E2_N_N3_QWEN06_FULL.jsonl | 0 | ok:4267 | 0 | 0 |
| qwen4 | 4267 | 85340 | NOT_RECORDED | 4267 | results/non_jev/v2/e2_n/n3/E2_N_N3_QWEN4_FULL.jsonl | 0 | ok:4267 | 0 | 0 |
| qwen_direct | 4267 | 85340 | NOT_RECORDED | 4267 | results/non_jev/v2/e2_n/n3/E2_N_N3_QWEN_DIRECT_FULL.jsonl | 0 | ok:4267 | 0 | 0 |

Totals: **55,471 normalized rows**, **54,971 rows with physical-forward evidence**, and **72,039 recorded physical forwards**.
Both headline counts are valid under different definitions: 54,971 is the number of successful normalized evidence rows, whereas 72,039 is the sum of adapter-reported forward calls. They must not be added and neither is an API-call count.
The difference between recorded forwards and evidence rows is **17,068 = 4,267 × 4**, attributable to Laya's five internal forward batches per query while one normalized evidence row is retained per query. Laya's adapter source documents the five-batch pointwise path; this is not an API-request count.
Nimble and PPLX each record one adapter-observed forward per successful query; their native internal choice/selection steps are not separately instrumented. No additional Nimble/PPLX forwards are inferred. Clef pointwise and joint are separately retained one-forward-per-query contracts.

Intern-Decision has 3,767 successful Track-A/eligible rows and 500 explicit enterprise native-context-blocked rows. Those 500 rows are not relabeled as model failures and are excluded from scored quality populations.

## Source trace hash checks

| Source key | Rows in manifest | SHA-256 match |
|---|---:|---|
| bge_pointwise | 4267 | PASS |
| laya_pointwise | 4267 | PASS |
| clef | 8534 | PASS |
| gte | 4267 | PASS |
| minilm | 4267 | PASS |
| jina35 | 4267 | PASS |
| nimble | 4267 | PASS |
| intern_decision | 4267 | PASS |
| pplx_decider | 4267 | PASS |
| qwen06 | 4267 | PASS |
| qwen4 | 4267 | PASS |
| qwen_direct | 4267 | PASS |

Source/hash errors: `0`.

## Frozen selection artifact check

Normalized selection rows: `59738`; expected rows: `59738`; missing keys: `0`; selection mismatches: `0`.

## Limitations

Per-call timestamps, server queue intervals, HTTP/API request counts, and native internal selection steps are not reconstructible from these artifacts. They must be instrumented in a future E2-S/P run; no additive latency or cost claim is made here.
