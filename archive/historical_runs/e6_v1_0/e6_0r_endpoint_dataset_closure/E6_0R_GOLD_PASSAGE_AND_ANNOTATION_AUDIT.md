# E6-0R Gold Passage and Annotation Audit

## DocHop-QA

The audited schema is `id, question, context_list, task_type, hop_type, used_doc`. `context_list` has 11379 rows with 3–5 supplied contexts; no `answer`, `gold`, `evidence`, or `support` fields were present. A context's `pmc_id` is source metadata, not a proof that the listed contexts are exhaustive or that every listed context is necessary. Complete-support OR-set scoring is therefore not valid.

## GaRAGe

All 2366 rows have aligned `grounding`, `evidence_relevant`, `evidence_correct`, and `evidence_cited` list lengths. The `evidence_correct` values include `ANSWER-THE-QUESTION`, `RELATED-INFORMATION`, `UNKNOWN`, `OUTDATED`, and empty values. Empty/unknown remain unjudged, not explicit negatives. The rows have no alternative complete support-set field; per-passage labels do not justify CSSR-U. Candidate identity is only `sample_id:cite_i` within the supplied row.

The pre-outcome fixed-pool amendment maps `ANSWER-THE-QUESTION` to grounding-positive, `RELATED-INFORMATION` to related-but-not-sufficient, `OUTDATED` to answer-invalid at the question date, and `UNKNOWN`/empty to unjudged. This mapping must be signed before E6-1 outcomes and does not create a natural-retrieval claim.

## SciFact

SciFact has binary document-level qrels: all 1258 audited qrel rows have score 1, with sparse unjudged corpus units. A qrel-positive document must not be split into child passages and relabeled positive without a separate authenticated mapping. The admitted endpoint is qrel-unit natural relevance only.
