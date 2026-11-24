# DecisionRAG E6-3 Integrated Scientific Report

Overall gate: E6_3_INTEGRATED_RESULTS_READY_WITH_LIMITATIONS

## Gates

- Reader: E6_3_READER_SCORING_COMPLETE.
- Serving: E6_3_MATCHED_SERVING_COMPLETE_WITH_HARDWARE_LIMITS.
- Scientific interpretation: integrated with limitations.

## Reader utility

The frozen R1 outputs were scored offline; no new reader inference was performed. The primary representative cohort is 200 questions with all 200 paired PPLX/Jina observations available.
Primary normalized-token-F1 contrast (PPLX minus Jina): {"cohort": "representative_primary", "jina_mean": 0.11977222812497733, "metric": "normalized_token_f1", "missing_ids": [], "missing_or_invalid_pairs": 0, "paired_n": 200, "pplx_mean": 0.1163654938834099, "pplx_minus_jina": {"ci95": [-0.009224723093481825, 0.0026508974716292887], "mean": -0.0034067342415674173, "n": 200, "resamples": 5000, "seed": 731}, "requested_questions": 200}.
This is a deterministic reference-overlap proxy against GaRAGe `answer_generate`, not independently verified factual correctness. Citation validity is structural; claim-level factual support is `CLAIM_LEVEL_FACTUAL_SUPPORT_NOT_VERIFIED`.

## Matched serving

The serving workload contained 3,200 timed attempts and 16 warmups: 100 frozen exploratory SciFact Top-20 queries and 100 frozen exploratory GaRAGe K15 queries, two repetitions, for each of eight models. All 3,200 timed requests and warmups returned a valid five-ID selection; failures=0. All measurements used GPU 1 UUID <REDACTED_GPU_UUID> sequentially.

Fastest descriptive p95 by workload (not a quality ranking):
- SciFact: jina_v3_5 p95=41.38536490499973 ms; PPLX p95=362.07984471693635 ms.
- GaRAGe: jina_v3_5 p95=84.43844527937463 ms; PPLX p95=720.8627523854371 ms.
All tested models satisfy the illustrative 1-second and 3-second successful-request p95 thresholds on these bounded samples. This does not establish sustainable QPS, because no offered-rate concurrency test was run; monetary cost is not modeled. Forward counts are adapter-observed only and are `NOT_MEASURED` where hooks did not expose internal work.

## Replacement boundary

The evidence supports a descriptive, workload-specific quality–resource comparison, not a formal noninferiority or replacement claim. Historical SciFact and GaRAGe quality metrics remain separate and are not pooled. A model can be operationally feasible while having lower historical evidence quality, and better evidence selection is not shown here to cause better factual answers.

## Scientific limitations and next decision

Source-group independence remains qualified by E6-2. The reader metric is lexical and no human/LLM adjudication was performed. E6-4 is deferred because the current evidence is sufficient for an explicitly exploratory replacement-boundary paper; any mechanism experiment would require a separate frozen protocol. E6-F, Jev, paid APIs, training and reserved holdout access remain locked.

Historical E2-E6-2/R1 artifacts were preserved.
