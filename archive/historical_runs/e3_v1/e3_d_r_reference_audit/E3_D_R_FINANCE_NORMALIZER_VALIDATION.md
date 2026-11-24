# FinanceBench normalizer validation

Status: `PASS_WITH_LIMITATIONS`; validation is deterministic and post-hoc exploratory.

The scorer preserves the legacy first-number/1% tolerance and exact yes/no endpoint. The added endpoints use Unicode NFKC, punctuation/spacing normalization, explicit thousand/million/billion/basis-point scales, and conservative one-to-one non-year numeric alignment. Percentage and percentage-point units are never conflated. Years are not treated as answer quantities.

Synthetic unit fixtures were evaluated offline: `$1.2 billion` equals `1200 million`; `12%` does not equal `12 percentage points`; wrong years, signs, mismatched quantities and multi-number alignment ambiguity remain negative/ambiguous. No model output was generated.

`MANUAL_ADJUDICATION_NOT_PERFORMED` and `factual_correctness=NOT_VERIFIABLE` apply to every real case.
