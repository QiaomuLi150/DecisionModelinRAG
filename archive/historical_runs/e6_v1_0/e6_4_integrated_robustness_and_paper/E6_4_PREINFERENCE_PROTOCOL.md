# DecisionRAG E6-4 Pre-inference protocol

- Seed: `731`.
- Models frozen before inference: pplx_decider, startlux_decision_4b, jina_v3_5, qwen3_reranker_0_6b.
- Revisions: `{"jina_v3_5": "e8a93f33f0b22108f8c2364f8484ce3422552fbc", "pplx_decider": "14c9d90137962d300f3de95633f7b6535e80c5ce", "qwen3_reranker_0_6b": "e61197ed45024b0ed8a2d74b80b4d909f1255473", "startlux_decision_4b": "9c0a32e9809d6a3f612434dd0bce150295dcc274"}`.
- SciFact CAL: 100; GaRAGe CAL: 100; planned attempts: 2,400.
- Orders: canonical frozen order; exact reversal; seeded permutation from `Random(SHA256(731|sample_id))`.
- Only candidate position changes. Query, candidate IDs, candidate content, checkpoint, native adapter, context limits, precision, readout and tie-breaking are unchanged.
- Inference reads label-blind CAL projections only. No qrels, evidence labels, answers, RESERVED content, Jev or paid API enters model input.
- Failure policy: retain every scheduled slot; no fallback, truncation, resampling, or outcome-based retry.
- Scoring is a separate post-freeze command.
- GPU policy: GPU 1 UUID `<REDACTED_GPU_UUID>`, sequential, no eviction.

Manifest: `E6_4_COHORT_AND_PERMUTATION_MANIFEST.json`.
