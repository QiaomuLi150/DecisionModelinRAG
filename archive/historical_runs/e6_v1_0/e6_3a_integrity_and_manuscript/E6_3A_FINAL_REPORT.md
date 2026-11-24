# DecisionRAG E6-3A Final Report

## Overall gate: `E6_3A_READY_WITH_LIMITATIONS`

### Serving discrepancy root cause

The original E6-3 SciFact path read a missing `text` field from frozen slate candidates and passed empty passages. It therefore did not exercise the E6-2 full-content native contract. The sample itself was not filtered: it contained `28` of 83 historical Amos failure IDs. The apparent Amos recovery and latency reduction were caused by the invalid input view, not a model improvement.

### Correction

The exact 100 SciFact and 100 GaRAGe query sample, order, model revisions and native adapters were frozen again in `serving_audit/E6_3A_SERVING_CORRECTION_SPEC.json`. The SciFact view was restored via E6-2 corpus reconstruction. Original E6-3 rows remain unchanged; corrected rows are under `serving_audit/corrected_raw/`.

### Native parity and latency

Candidate ID/order hashes matched, but original E6-3 native input hashes did not. Historical E6-3 timing excluded candidate-view lookup and final Top-5 packaging, and used empty SciFact views. Corrected p50/p95 and failures are in `E6_3A_CORRECTED_SERVING_SCORECARD.csv`; no sustainable QPS or monetary cost is claimed.

### Reader validity

The frozen reader metric was reproduced with `0` mismatches. The representative PPLX−Jina lexical reference-overlap difference is `-0.003406734242` across `200` pairs. Factual correctness remains `NOT_VERIFIED`; no reader inference was rerun.

### Scientific conclusion

The record supports a qualified exploratory paper: decision-model utility is workload-dependent, specialists remain competitive, context limits and input fidelity are operationally decisive, and no downstream factual advantage or production-cost advantage is established. E6-4 and prospective confirmation remain unstarted.

### Required paths

- Manuscript: `manuscript/DECISIONRAG_MANUSCRIPT_DRAFT.md`
- Tables: `manuscript/DECISIONRAG_MANUSCRIPT_TABLES.md`
- Input audit: `serving_audit/E6_3A_SERVING_INPUT_PARITY_AUDIT.md`
- Corrected serving scorecard: `serving_audit/E6_3A_CORRECTED_SERVING_SCORECARD.csv`
- Reader audit: `reader_validity/E6_3A_READER_REFERENCE_METRIC_AUDIT.md`
- Provenance: `E6_3A_PROVENANCE_MANIFEST.json`
