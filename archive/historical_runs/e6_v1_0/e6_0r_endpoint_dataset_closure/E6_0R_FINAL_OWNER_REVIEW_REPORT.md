# DecisionRAG E6-0R Final Owner-Review Report

## Gate

**`E6_0R_OWNER_APPROVAL_READY`** — one natural relevance workload is admitted conditionally for owner signature; a separate fixed-pool grounding workload is also conditionally specified. E6-1 has not started. E6-2 and confirmation remain locked.

## Admitted workloads

1. **Primary N-REL: BEIR/SciFact.** Actual corpus: 5183 text documents; queries: 1109; qrels: 1258 binary rows covering 1109 queries. This supports natural document-relevance evaluation after a separately authorized, gold-blind retrieval preparation. It does not support CSSR-U because qrels are sparse document-level relevance judgments, not complete alternative support sets.
2. **Secondary F-GRD: GaRAGe exact K=15.** 2329 supplied passage pools are exactly 15 and can support an explicitly fixed `15 -> 5` grounding endpoint. The remaining variable-K rows are retained in the inventory, not silently converted. No natural Top-20 claim is permitted.

DocHop-QA is rejected for the current endpoint: its 11379 rows contain 3–5 supplied contexts and source PMC IDs but no packaged natural corpus, answer field, or evidence-support labels.

## Source independence and reservation

SciFact's qrel graph has 572 components; the largest contains 9 questions. A deterministic seed-731 component split was prepared. Reserved content was not exported, but no external sealed vault/OS deny boundary exists, so the reservation status is **metadata-only / CONFIRMATION_LOCKED**, not an independent confirmatory holdout. GaRAGe source-group inference is unavailable because stable source-document IDs are absent.

## Historical and integrity status

- E5 Top-20 SHA-256: `be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7`; expected `be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7`; match: **True**.
- Historical declared-manifest verification: **91/91** entries passed; failed: **0**.
- Exact text-overlap counts and adjacent-workspace SciFact exposure are in `E6_0R_HISTORICAL_EXCLUSION_INDEX.json`.
- E5 QASPER CSSR-U and E6-0 answer-F1 remain unchanged; no new endpoint overwrites them.
- New physical inference/model/GPU/API/Jev/paid operations: **zero**.

## Owner decisions required

1. Sign the N-REL SciFact amendment and accept sparse qrel/unjudged limitations.
2. If desired, separately accept GaRAGe K=15 fixed-pool grounding as a secondary exploratory endpoint.
3. Keep E6-1 limited to CAL adapter validation; do not start the full eight-model benchmark until native contracts and endpoint acceptance pass.
4. Keep E5-F/E6 reserved content locked and authorize any future retrieval/model workload separately.
