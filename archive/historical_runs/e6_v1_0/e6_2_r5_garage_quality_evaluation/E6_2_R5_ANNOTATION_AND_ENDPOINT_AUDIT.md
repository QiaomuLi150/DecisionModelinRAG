# E6-2 R5 Annotation and Endpoint Audit

## Authenticated endpoint

- Primary label: `evidence_relevant`; YES contributes binary gain 1 and NO contributes gain 0.
- `evidence_correct` remains categorical and is descriptive only.
- `evidence_cited` remains citation-related and is not used for primary relevance.
- K15 positions map canonically to `sample_id:cite_i`; original candidate order and text hashes are preserved.
- IDCG@5 uses all YES labels in the original K15 pool, not only selected passages.
- No-positive K15 rows remain in the full denominator with nDCG@5=0.

## Observed labels

- Exploratory rows: 1,397; annotation arrays: 15 per field.
- Binary YES/NO entries: 20954.
- Unresolved `evidence_relevant` entries: 1; one blank entry is retained as unresolved, not recoded.
- RESERVED rows exported to scorer: 0.

The frozen contract did not define a separate missing-label imputation rule. The operational metric keeps all 1,397 samples, treats the unresolved entry as non-positive for gain calculation only when computing the zero/IDCG convention, and flags the affected sample in every result row. This is a stated limitation, not a silent negative-label conversion.

No reader answer, answer-generation label, LLM adjudication, or factual-correctness label was accessed.
