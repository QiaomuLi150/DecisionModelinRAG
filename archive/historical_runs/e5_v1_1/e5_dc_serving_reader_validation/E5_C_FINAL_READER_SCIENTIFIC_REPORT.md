# E5-C Final Reader Scientific Report

Status: `C_ANALYSIS_PASS`.

The reader used the pinned local Qwen3-VL-8B-Instruct revision and identical deterministic E4-1 text-only prompts, five-passage budgets, generation settings and no-application-cache headers. Results are deterministic reference-agreement proxies, not independently verified factual correctness.

| cohort | n paired | PPLX matches | Jina matches | PPLX rate | Jina rate | delta | 95% CI | PPLX-only | Jina-only | both | neither |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| representative | 200 | 18 | 17 | 0.09 | 0.085 | 0.005 | [-0.01, 0.020124999999998183] | 2 | 1 | 16 | 181 |
| diagnostic_enriched | 55 | 4 | 2 | 0.07272727272727272 | 0.03636363636363636 | 0.03636363636363636 | [0.0, 0.08333333333333333] | 2 | 0 | 2 | 51 |

The representative cohort is the population-oriented exploratory estimate. The diagnostic cohort is the complete E5-A disagreement set and is not pooled into the representative estimate. T2 model adjudication and T3 independent factual correctness were not measured.
