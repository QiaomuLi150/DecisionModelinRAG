# E4-2 Facet-Aware Decision Pilot

Status: `E4_2_PILOT_OWNER_REVIEW_READY` for bounded exploratory inference.

Five policies were evaluated on the frozen E4-1 representative (200) and diagnostic-enriched (50) cohorts; the strata are never pooled for population estimates.

## Frozen-input and native execution

Facet generation was deterministic and question-only; no support IDs, references, qrels, candidate texts or prior outcomes entered facet construction. The common optimizer, facet IDs, rank normalization, redundancy penalty and tie rule were frozen before native pilot outcomes.

Accepted native rows: 1400; physical forward calls: 7291. The common candidate view was `common_word_capped_256_v1` for all five policies.

## Policy scorecard

| Cohort | Policy | n | CSSR-U | CSSR-C | Any-gold-U | p50 ms | p95 ms | Calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| representative | jina_native_top5 | 200 | 0.3950 | 0.4293 | 0.9750 | 229.1172 | 243.0535 | 250 |
| diagnostic_enriched | jina_native_top5 | 50 | 0.2800 | 0.2800 | 0.9600 | 229.1172 | 243.0535 | 250 |
| representative | jina_facet_coverage | 200 | 0.4250 | 0.4620 | 0.9800 | 677.3427 | 915.0691 | 753 |
| diagnostic_enriched | jina_facet_coverage | 50 | 0.4000 | 0.4000 | 1.0000 | 677.3427 | 915.0691 | 753 |
| representative | liquid_native_top5 | 200 | 0.3050 | 0.3315 | 0.8450 | 338.7158 | 347.7040 | 250 |
| diagnostic_enriched | liquid_native_top5 | 50 | 0.2600 | 0.2600 | 0.8400 | 338.7158 | 347.7040 | 250 |
| representative | liquid_pointwise_facet_coverage | 200 | 0.3700 | 0.4022 | 0.9600 | 692.0933 | 732.2784 | 5000 |
| diagnostic_enriched | liquid_pointwise_facet_coverage | 50 | 0.3600 | 0.3600 | 0.9600 | 692.0933 | 732.2784 | 5000 |
| representative | liquid_joint_facet_coverage | 200 | 0.2700 | 0.2935 | 0.9150 | 492.5572 | 551.3242 | 250 |
| diagnostic_enriched | liquid_joint_facet_coverage | 50 | 0.2400 | 0.2400 | 0.9400 | 492.5572 | 551.3242 | 250 |

CSSR-U/C and Any-gold-U are annotated evidence-retention proxies. They do not establish factual answer correctness. Timing is shared-GPU and is not a production latency or cost estimate.

## Paired contrasts

- representative `joint_minus_pointwise`: n=200, delta=-0.1, wins/losses/ties=8/28/164, query-bootstrap 95% CI=[-0.16, -0.045]. Exploratory proxy only.
- representative `joint_minus_jina_facet`: n=200, delta=-0.155, wins/losses/ties=3/34/163, query-bootstrap 95% CI=[-0.21, -0.1]. Exploratory proxy only.
- representative `joint_minus_liquid_native`: n=200, delta=-0.035, wins/losses/ties=13/20/167, query-bootstrap 95% CI=[-0.09, 0.02]. Exploratory proxy only.
- representative `jina_facet_minus_jina_native`: n=200, delta=0.03, wins/losses/ties=15/9/176, query-bootstrap 95% CI=[-0.015, 0.08]. Exploratory proxy only.
- diagnostic_enriched `joint_minus_pointwise`: n=50, delta=-0.12, wins/losses/ties=1/7/42, query-bootstrap 95% CI=[-0.22, -0.02]. Exploratory proxy only.
- diagnostic_enriched `joint_minus_jina_facet`: n=50, delta=-0.16, wins/losses/ties=0/8/42, query-bootstrap 95% CI=[-0.26, -0.06]. Exploratory proxy only.
- diagnostic_enriched `joint_minus_liquid_native`: n=50, delta=-0.02, wins/losses/ties=6/7/37, query-bootstrap 95% CI=[-0.16, 0.12]. Exploratory proxy only.
- diagnostic_enriched `jina_facet_minus_jina_native`: n=50, delta=0.12, wins/losses/ties=13/7/30, query-bootstrap 95% CI=[-0.06, 0.3]. Exploratory proxy only.

## Interpretation and limitations

The Liquid pointwise/joint comparison uses the same d1-3B checkpoint and official `system_one` interface. Joint uses all candidates and candidate-facet questions in one shared state; pointwise uses one candidate state with the same facet questions. Any context overflow or parse failure remains explicit; no incompatible proxy was substituted.

Jina/Liquid raw score magnitudes are not treated as cross-model comparable; facet ranking is converted to a common within-facet rank score before the shared optimizer. Source-group sensitivity is reported separately and is limited by the dominant E3-D connected component. Entity-binding consistency and semantic placebo neutrality remain NOT_MEASURED/NOT_INDEPENDENTLY_VERIFIED.

No Jev/paid inference, training, new model family, multimodal work, or reserved holdout outcome access occurred. E4-3/E4-5 were not run. The facet method is not expanded automatically.
