# E6-3 Manuscript-Ready Evidence Tables

## Table 1. Contracts

QASPER/E5 and SciFact/GaRAGe are separate exploratory contracts; GaRAGe annotations are incomplete/qualified and R1 reader reference agreement is lexical.

## Table 2. Historical evidence-selection quality

See authenticated E6-2 R3 and R5 model scorecards. Values are not recomputed here and are not combined across workloads.

## Table 3. Reader utility

Representative primary PPLX-Jina normalized token-F1: `{"cohort": "representative_primary", "jina_mean": 0.11977222812497733, "metric": "normalized_token_f1", "missing_ids": [], "missing_or_invalid_pairs": 0, "paired_n": 200, "pplx_mean": 0.1163654938834099, "pplx_minus_jina": {"ci95": [-0.009224723093481825, 0.0026508974716292887], "mean": -0.0034067342415674173, "n": 200, "resamples": 5000, "seed": 731}, "requested_questions": 200}`. This is reference overlap, not factual correctness.

## Table 4. Serving

See E6_3_SERVING_MODEL_SCORECARD.csv; successful-request percentiles and full-attempt failures are separate.

## Table 5. Replacement matrix

See E6_3_CROSS_WORKLOAD_REPLACEMENT_MATRIX.csv. No formal noninferiority or production-cost claim is made.
