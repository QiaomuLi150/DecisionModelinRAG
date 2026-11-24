# E7 adapter defect and correction register

- The first E7 runner revision exposed two E7 implementation defects: Jina raw response serialization did not normalize NumPy integer indices, and StartLux tuple response unpacking was missing.
- Original raw records are preserved in `E7_B_NATIVE_OUTPUTS.jsonl`, `E7_B_NATIVE_OUTPUTS_jina_v3_5.jsonl`, and `E7_B_NATIVE_OUTPUTS_startlux_decision_4b.jsonl`.
- Versioned corrections are in `*_v2.jsonl`; corrected scoring uses `E7_B_NATIVE_OUTPUTS_CORRECTED_V2.jsonl`.
- PPLX 8,192-token failures are retained as native context-limit outcomes, not repaired by truncation.
- The E7 runner is independent of the E6 score-to-ID ranking helpers.
- Historical E6-3 empty-passage serving remains superseded by E6-3A and was not rewritten.
- No failure is imputed or silently dropped.
