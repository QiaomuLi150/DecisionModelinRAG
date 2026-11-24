# DecisionRAG v3.6 follow-up status

- Stage F CPU corrections: PASS.
- Stage 1 Jina repair: BLOCKED_UNVERIFIED_SELECTION_GATE; the corrected rerun is physically valid on budgetable requests but does not meet the prespecified 99% exact-list agreement gate.
- Stage 2 N1 full-population timing: 55360 records complete across the two sessions.
- Stage 2 N2: partial; session C contains Jina/PPLX anchors and local specialists; Drex-v2 has no same-session anchors; Winnow has same-session anchors but its pre-outcome protocol receipt was not frozen before the anchor phase.
- Stage 3 N3: BLOCKED because the available transfer protocol is hybrid RRF rather than the required exact BM25 artifact.
- Stage 4 CPU closure: PASS.
- Stage 5 packaging and hygiene: PASS.
- Overall: PARTIAL; no historical artifacts were overwritten and no protected holdouts or paid APIs were accessed.
