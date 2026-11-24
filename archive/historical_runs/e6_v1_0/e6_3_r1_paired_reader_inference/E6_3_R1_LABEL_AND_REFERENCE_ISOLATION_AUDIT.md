# E6-3 R1 Label and Reference Isolation Audit

- Reader prompts were built only from the frozen EXPLORATORY question and selected passage text projection.
- `evidence_relevant`, `evidence_correct`, `evidence_cited`, `answer_generate`, `answer_related_info` and `answer_validate` were not loaded into reader prompt construction.
- RESERVED question/content/labels were not loaded.
- The R5 no-positive IDs were used only as a pre-frozen cohort membership diagnostic; no labels were inserted into prompts.
- Raw model responses and parsed structural fields are preserved; no answer scoring ran in R1.
- Historical E5/E6-2 artifacts were not modified.
