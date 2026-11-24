# E2-P0v2 Statistical Analysis Plan

## Quality estimands

For eligible identity-resolved nonempty gold sets with 1–5 documents, report unconditional complete annotated support retention over all eligible questions and retrieval-conditional retention over questions whose complete gold set is in Top-20. Track A, Track B, HERB citation recall and any new holdout remain separate. HERB citation recall is not complete necessary-support recall.

## Noninferiority

For a prespecified pair, `D_q = quality_decision_q - quality_specialist_q`. Use paired query-level resampling, with document/source cluster bootstrap when multiple questions share a source. Establish noninferiority only when the lower 95% CI bound is greater than `-epsilon`. `epsilon` is **NOT FROZEN**; owner justification in absolute percentage points is required before TEST.

## Primary and secondary contrasts

Recommendation: one H1 decision-model/specialist pair plus one H2b same-score policy contrast as the primary family. H2a, H2c, workload interactions, categories and historical A/B comparisons are secondary or exploratory unless the owner freezes otherwise. No TEST-driven winner selection.

## Resampling and multiplicity

Use paired bootstrap with query as unit, dataset-level estimates, equal-dataset macro as an explicit secondary estimand, seed and repetitions frozen before TEST. Recommended correction is Holm across the owner-defined primary family; this is a recommendation, not frozen. Missing/invalid outputs remain explicit and are not silently imputed.

## Power

A paired-power audit requires approved epsilon, target alpha, power, expected discordance/effect distribution and final holdout n. Historical outcomes may supply labeled planning estimates only. Because epsilon, power and holdout n are unresolved, the confirmation gate is **NOT FROZEN**.
