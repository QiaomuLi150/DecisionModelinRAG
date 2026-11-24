# DecisionRAG E7 pre-execution protocol

- Seed: `731`.
- Primary native audit: pplx_decider, startlux_decision_4b, jina_v3_5, qwen3_reranker_0_6b.
- E7-B population: 40 SciFact CAL + 40 GaRAGe CAL, selected outcome-blind from the authenticated E6-4 CAL projection.
- Conditions per model/query: canonical A, canonical B, canonical C, reversed, seeded permutation.
- Planned physical attempts: 80 × 4 × 5 = 1,600.
- Native payloads are built and hashed by the independent E7 runner; full title+text is required and missing text is an error.
- No qrels, evidence labels, answers, reserved content, Jev or paid APIs are available to the inference worker.
- Raw outputs are frozen before any label join. Failed/invalid/context-limited attempts remain in denominators.
- E6-4 historical files are immutable; E7 is append-only.
- Protected access: E5-F/E6-F/RESERVED raw content, labels, references and predictions remain closed.

Manifest and hashes are in `E7_INPUT_INTEGRITY_BASELINE.json`; the B cohort is created by `freeze-b`.
