# E8 Jina/StartLux correction lineage

E7 identified two implementation defects: Jina raw NumPy integer indices were not JSON-normalized, and StartLux tuple responses were not unpacked. The original raw traces remain at `E7_B_NATIVE_OUTPUTS.jsonl`, `E7_B_NATIVE_OUTPUTS_jina_v3_5.jsonl`, and `E7_B_NATIVE_OUTPUTS_startlux_decision_4b.jsonl`; corrected versioned traces are `E7_B_NATIVE_OUTPUTS_jina_v3_5_v2.jsonl`, `E7_B_NATIVE_OUTPUTS_startlux_decision_4b_v2.jsonl`, and `E7_B_NATIVE_OUTPUTS_CORRECTED_V2.jsonl`.

The correction affects E7-B native trace validity and its E7-B order/repeatability aggregates only. It does not silently replace unrelated E6 populations. E8 uses the corrected E7 path and records this diagnostic as a new input intervention. Historical E6-3A and E6-4 outputs remain immutable; no E6 aggregate is rewritten here.
