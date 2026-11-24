# DecisionRAG v3.6 Jina Contract-Repair Closure

## Gate

**JINA_SELECTION_GATE_PASS** for the repaired historical comparison. The overall v3.6 continuation remains **PARTIAL** because the independent N3 exact-BM25 artifact is unavailable and the historical N2-E Winnow branch lacks a pre-outcome protocol receipt.

## Defect and repair

The first v3.6 rerun changed two comparison-critical details: it hashed `documents` instead of the historical `candidate_texts` field, and it used serial `model.rerank` rather than the original v1.5 `score_many` batch-8 forward path. The corrected run freezes the E11 budgeted texts, uses the historical JSON serialization and replays the original batch-8 native scoring path.

The serial replay is retained as diagnostic evidence only; it is not used for the repaired gate.

## Exact reconciliation

| Check | Result |
|---|---:|
| Frozen population | 1384 |
| Historical successful rows | 1352 |
| Historical failures preserved without retry | 32 |
| New native successful rows | 1352 |
| Exact selected lists | 1352/1352 |
| Exact selected sets | 1352/1352 |
| Exact score vectors | 1352/1352 |
| Exact request hashes | 1352/1352 |
| Unexpected new failures | 0 |
| Physical batch forwards | 169 |

The repaired native trace is therefore an exact reproduction of the historical Jina result on every replayable request.

## Offline quality reference

The corrected Jina selections were joined offline to the frozen BRIGHT gold manifest after raw output freeze. Jina mean operational known-positive nDCG@5 is **0.173756**; PPLX historical L=8192 reuse is **0.192761**. The descriptive Jina-minus-PPLX query-weighted difference is **-0.019004**; equal-domain bootstrap CI is **[-0.028679, -0.009829]**, 40000 draws, seed 731. This is a repair/reconciliation statistic, not a new preregistered claim.

## Remaining issues

- **N3:** still blocked. Available transfer artifacts use frozen hybrid RRF first-stage slates; an exact BM25-only artifact was not found. No proxy retrieval result is substituted.
- **N2-E:** remains descriptive/partial. Its pre-outcome protocol receipt was not frozen before the anchor phase; this cannot be repaired retroactively. A new independently frozen rerun would be required for a primary matched claim.
- **Serving:** this repair validates the quality-input/scoring contract. Existing v3.6 serving records are not silently relabeled; any timing result must retain its own declared wrapper and timing boundary.

No historical file was overwritten, no protected holdout was accessed, and no paid/API call was made.
