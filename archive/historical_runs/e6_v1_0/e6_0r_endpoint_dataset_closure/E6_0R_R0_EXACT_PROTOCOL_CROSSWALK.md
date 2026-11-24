# E6-0R Exact Protocol Crosswalk

The historical E6-0 proposal was complete only for a future amendment choice: it named graded support/grounding as a possibility but did not select a single dataset, candidate K, qrel unit, unjudged policy or source-group contract. E6-0R freezes those missing fields before any new selector outcome:

- Primary: SciFact natural document relevance, binary qrel-unit nDCG@5/Recall@5.
- Secondary: GaRAGe exact K=15 supplied-pool grounding, support_any@5/grounding_precision@5.
- Historical QASPER CSSR-U is unchanged and not pooled.
- Failures are explicit status categories; not zero quality.
- Bootstrap seed 731 and 5,000 resamples; source groups only where justified.
