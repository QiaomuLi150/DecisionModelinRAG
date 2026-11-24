# E6-0R Historical Overlap and Duplicate Audit

{
  "adjacent_workspace_exposure": "P30C reports and p30n scripts contain SciFact references; DecisionRAG E2-E5 result files did not contain SciFact rows in bounded scan. Therefore SciFact is not claimed source-independent of the whole workspace.",
  "exact_overlap": {
    "qasper_vs_dochop_contexts": 0,
    "qasper_vs_dochop_questions": 0,
    "qasper_vs_garage_grounding": 0,
    "qasper_vs_garage_questions": 0,
    "qasper_vs_scifact_corpus": 0,
    "qasper_vs_scifact_queries": 0
  },
  "historical_reference": "E2-E5 permitted QASPER plus adjacent RAG_system source scan; sealed E5-F/E6 reservation untouched",
  "historical_source_notes": {
    "dochop": "new public release; source PMC IDs available but raw PMC corpus not bundled",
    "garage": "new public release; no source document identity in row schema",
    "qasper": "historically used and explicitly excluded from the new workload",
    "scifact": "actual SciFact cache present and adjacent RAG artifacts mention SciFact; use only as exploratory natural relevance unless owner accepts workspace exposure limitation"
  },
  "near_duplicate_method": "NOT_RUN; no semantic near-duplicate model or embedding operation authorized",
  "qasper_passage_hash_count": 59345,
  "qasper_question_hash_count": 3308
}
