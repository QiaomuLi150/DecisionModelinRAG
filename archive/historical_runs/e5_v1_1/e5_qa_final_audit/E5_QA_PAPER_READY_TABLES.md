# E5-QA Paper-Ready Tables

## Frozen E5-A primary

| Policy | CSSR-U | CSSR-C | Feasibility ceiling |
|---|---:|---:|---:|
| PPLX Decider | 306/1502 = 20.37% | 306/355 = 86.20% | 355/1502 = 23.64% |
| Jina v3.5 | 299/1502 = 19.91% | 299/355 = 84.23% | 355/1502 = 23.64% |

Equal-source-group PPLX−Jina difference: **0.5405 pp**, 95% CI [-0.3586, 1.4805] pp; 587 source groups; 5000 resamples; seed 731.

## E5-C representative reader proxy

| Policy | Matches / n | Rate |
|---|---:|---:|
| PPLX | 18/200 | 9.0% |
| Jina | 17/200 | 8.5% |

Difference +0.5 pp; 95% CI [−1.0,+2.0125] pp. This is deterministic reference agreement, not factual correctness.

## E5-D selector

| Policy | p50 | p95 |
|---|---:|---:|
| PPLX | 1320.085 ms | 1616.623 ms |
| Jina | 62.245 ms | 88.790 ms |

E5-D is sequential same-GPU concurrency-one timing; order was not randomized and no QPS/cost estimate is supported.

## Historical E2–E4 scope

E2-M1 measured heterogeneous Clef pointwise/joint exploratory comparisons. E4-R reconciled E2/E4 as comparable-but-heterogeneous, with negative/inconclusive Clef mechanism contrasts and blocked matched isolated timing. Those results are not pooled with E5-A or reinterpreted as a single causal model-family effect.
