# E6-2 R0 Historical Protocol Crosswalk

## Decision

The E6-1/E6-1B files are internally reproducible, but they do not implement the original natural-retrieval workload. They are retained as **technical native-adapter feasibility evidence**, not as SciFact ranking-quality evidence.

| Contract item | Original E6-1 authorization | Actual E6-1/E6-1B | Resolution |
|---|---|---|---|
| Dataset/input | Up to 120 QASPER CAL K20; up to 60 GaRAGe K15; up to 60 authentic SciFact CAL Top-20 if already available; 12 order diagnostics | 120 SciFact CAL questions; 120 technical candidate views; K15 and K20; 3 reversal questions/model | Deviation recorded; no rewrite or rerun |
| Candidate construction | Existing authentic retrieval slates only where available | SHA-256 deterministic ordering over the full public SciFact corpus; qrels not loaded | Gold-blind but synthetic/technical, not natural retrieval |
| K interpretation | Dataset-specific K15/K20 | Both K15 and K20 on the same technical views | Valid context/adapter feasibility only |
| Order diagnostics | 12 prespecified CAL questions | Three reversal queries per model | Smaller diagnostic; preliminary, not a population estimate |
| Owner amendment | E6-0R endpoint amendment approved; no evidence of a natural SciFact-slate amendment | E6-1 acceptance records explicitly say qrels absent and CAL feasibility only | Do not certify full original compliance |

The matching input hash (`a1b006...f318`) proves byte reproducibility of the executed fixture; it does not prove natural retrieval provenance. The historical QASPER Top-20 hash remains separate (`be9cf...`).
