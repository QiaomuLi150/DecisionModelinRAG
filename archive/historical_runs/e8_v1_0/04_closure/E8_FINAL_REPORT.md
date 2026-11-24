# DecisionRAG E8 final scientific reconciliation

## Final gate

`E8_VALIDATED_EXPLANATION_AND_RECONCILIATION_READY`

## Scope and accounting

- New diagnostic: fixed identity-tied original retrieval-rank metadata, C/D conditions, exact E7 CAL cohort.
- Planned logical requests: 640 (4 models × 80 questions × 2 conditions).
- Attempted: 640; valid: 610; explicit failures: 30.
- Failure grouping: PPLX SciFact 30/80 attempts failed on 15 query IDs × {C,D}; StartLux, Jina and Qwen had 0 failures.
- Raw outputs were frozen before label loading. Jev calls: 0. Paid calls: 0. Reserved access: false.

## Stability and quality scorecard

| Model | Workload | Scheduled | A/R valid pairs | C/D valid pairs | Common complete | A/R exact | C/D exact | A/R Jaccard | C/D Jaccard |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pplx_decider | scifact | 40 | 26 | 25 | 25 | 0.07692307692307693 | 0.08 | 0.4748168498168498 | 0.47349206349206346 |
| pplx_decider | garage | 40 | 40 | 40 | 40 | 0.0 | 0.0 | 0.4558531746031746 | 0.4441468253968254 |
| startlux_decision_4b | scifact | 40 | 40 | 40 | 40 | 0.125 | 0.2 | 0.5580357142857142 | 0.5904761904761904 |
| startlux_decision_4b | garage | 40 | 40 | 40 | 40 | 0.175 | 0.175 | 0.650595238095238 | 0.6267857142857143 |
| jina_v3_5 | scifact | 40 | 40 | 40 | 40 | 0.225 | 0.175 | 0.6880952380952381 | 0.6773809523809524 |
| jina_v3_5 | garage | 40 | 40 | 40 | 40 | 0.275 | 0.3 | 0.6928571428571428 | 0.713095238095238 |
| qwen3_reranker_0_6b | scifact | 40 | 40 | 40 | 40 | 1.0 | 1.0 | 1.0 | 1.0 |
| qwen3_reranker_0_6b | garage | 40 | 40 | 40 | 40 | 1.0 | 0.95 | 1.0 | 0.9833333333333334 |

The machine-readable per-question results and 5,000-resample paired statistics are in `../03_offline_analysis/`. C/D is a modified-input diagnostic and cannot establish an internal model mechanism or universal replacement. Quality is conditional on valid selections; context failures remain operational failures.

## Common-complete paired moderation

The following are C/D minus A/R Jaccard moderations on the common-complete population; brackets are the prescribed 5,000-query-bootstrap 95% intervals. The analogous nDCG moderation is in `E8_PAIRED_STATISTICS.json`.

| Model | Workload | n | Jaccard moderation [95% CI] | nDCG moderation [95% CI] |
|---|---|---:|---:|---:|
| PPLX | SciFact | 25 | -0.0103 [-0.0681, 0.0490] | -0.0302 [-0.0759, 0.0000] |
| PPLX | GaRAGe | 40 | -0.0117 [-0.0579, 0.0389] | +0.0146 [-0.0151, 0.0452] |
| StartLux | SciFact | 40 | +0.0324 [-0.0125, 0.0783] | -0.0250 [-0.0802, 0.0323] |
| StartLux | GaRAGe | 40 | -0.0238 [-0.0774, 0.0310] | -0.0016 [-0.0164, 0.0126] |
| Jina | SciFact | 40 | -0.0107 [-0.0464, 0.0214] | +0.0103 [-0.0259, 0.0504] |
| Jina | GaRAGe | 40 | +0.0202 [-0.0226, 0.0655] | +0.0014 [-0.0189, 0.0224] |
| Qwen 0.6B | SciFact | 40 | +0.0000 [0.0000, 0.0000] | +0.0000 [0.0000, 0.0000] |
| Qwen 0.6B | GaRAGe | 40 | -0.0167 [-0.0417, 0.0000] | +0.0006 [0.0000, 0.0018] |

The C/D intervention therefore shows no consistent cross-model stability or relevance improvement. It is structurally descriptive and exploratory; intervals crossing zero are not evidence of equivalence.

## Lineage conclusions

E7's PPLX SciFact failures are valid for its 40-question full-content 8,192-token native contract (14 query IDs × five E7 conditions). E8 retains those 14 and adds `scifact:418` under the rank-tag prefix, giving 15 query IDs × two E8 conditions; no failure resolved. The added metadata overhead is therefore an observed feasibility cost for this intervention. Higher feasibility figures in earlier E6 artifacts are not treated as contradictory without exact cohort, payload, tokenizer and context-setting parity. E7's Jina/StartLux corrections are versioned, and original defects remain preserved with explicit supersession; E8 does not overwrite E6/E7 claims.

## Final recommendation

Freeze research for review only if the owner accepts the bounded exploratory status. No E9, protected holdout, new benchmark, reader inference or manuscript drafting was started. If a follow-up is required, the single essential integrity correction is to obtain exact earlier native payload/tokenizer/context metadata before making cross-stage PPLX feasibility claims.
