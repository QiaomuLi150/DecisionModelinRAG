# DecisionRAG v2.1 E2-M1 Final Report

Status: `E2_M1_PASS`

## Completed scope

- Track A frozen exploratory population: `3767` queries.
- Track B frozen enterprise population: `500` questions.
- Authentic Clef pointwise rows: `4267/4267`.
- Authentic native-joint K=20 rows: `4267/4267`.
- Physical native-joint forward calls: one per query; no quartet rescoring.
- Jev/paid calls: `0/0`.

## Enterprise identity correction

The enterprise frozen slates use path-qualified internal identities while the read-only gold sidecar preserves original `dsid_` values. The offline evaluator now resolves only one-to-one original-ID mappings from the frozen enterprise corpus. `499` questions resolve uniquely. `1` question remains excluded from identity-dependent metrics under `V11.1R1-GOLD-ID-AMBIGUITY-qst_0413`; all 500 questions remain in the manifest. Track B therefore reports 443 identity-resolved eligible questions and a separate 330-query retrieval-conditional denominator.

## Interpretation

The native joint mechanism is computationally feasible for K=20 on the pinned local Clef revision. Quality and paired comparisons are exploratory follow-up analyses, separated by Track A and Track B and reported with both unconditional eligible and retrieval-conditional denominators. Track A/B quality metrics are not reader factual-accuracy claims.

The structural oracle recheck is in `E2_M1_STRUCTURAL_ORACLE_RECHECK.csv`. It defines a quota oracle as a retrieval-conditional eligible query whose five quartet groups contain at most one gold document. It reports complete-gold collisions and verifies that observed quartet successes do not exceed this oracle.

No E2-N, E2-S/P, model-family expansion, model training, Jev, paid API, retrieval, or reader experiment was started.
