# E4-R Paper-Ready Tables

## Table 1. E4-3 representative scorecard

| Policy | K | CSSR-U | CSSR-C | Complete-support ceiling | Any-gold Top-5 |
|---|---:|---:|---:|---:|---:|
| clef_pointwise_binary | K10 | 0.363333 | 0.529126 | 0.686667 | 0.923333 |
| clef_pointwise_binary | K20 | 0.306667 | 0.306667 | 1 | 0.866667 |
| clef_joint_binary | K10 | 0.33 | 0.480583 | 0.686667 | 0.896667 |
| clef_joint_binary | K20 | 0.263333 | 0.263333 | 1 | 0.85 |
| liquid_native_joint | K10 | 0.37 | 0.538835 | 0.686667 | 0.916667 |
| liquid_native_joint | K20 | 0.36 | 0.36 | 1 | 0.9 |
| jina35_native | K10 | 0.466667 | 0.679612 | 0.686667 | 0.966667 |
| jina35_native | K20 | 0.536667 | 0.536667 | 1 | 0.966667 |
| pplx_decider_native | K10 | 0.51 | 0.742718 | 0.686667 | 0.976667 |
| pplx_decider_native | K20 | 0.536667 | 0.536667 | 1 | 0.97 |
| bge_pointwise | K10 | 0.433333 | 0.631068 | 0.686667 | 0.926667 |
| bge_pointwise | K20 | 0.393333 | 0.393333 | 1 | 0.903333 |

## Table 2. Clef mechanism contrasts

| Contrast | Population | n | Difference | 95% CI |
|---|---|---:|---:|---|
| E2-M1 joint-pointwise | Track A conditional | 1490 | +14.295 pp | +11.678,+16.980 pp |
| E2-M1 joint-pointwise | Track B conditional | 330 | +20.000 pp | +14.848,+25.152 pp |
| E4-3 joint-pointwise | K10 feasible | 206 | -4.854 pp | -11.650,+1.942 pp |
| E4-3 joint-pointwise | K20 valid | 300 | -4.333 pp | -10.000,+1.000 pp |
| E4-3 K20-minus-K10 interaction | K10 feasible | 206 | 0.000 pp | -5.825,+5.825 pp |

## Table 3. Source sensitivity

| Policy | K | n | Groups | Largest n | Query bootstrap | Cluster bootstrap | Leave-largest-out |
|---|---:|---:|---:|---:|---|---|---|
| clef_pointwise_binary | K10 | 300 | 25 | 261 | 0.363333 [0.31, 0.42] | [0.334975, 0.722222] | 0.615385 (n=39) |
| clef_pointwise_binary | K20 | 300 | 25 | 261 | 0.306667 [0.256667, 0.36] | [0.275671, 0.6875] | 0.564103 (n=39) |
| clef_joint_binary | K10 | 300 | 25 | 261 | 0.33 [0.276667, 0.383333] | [0.304636, 0.692308] | 0.538462 (n=39) |
| clef_joint_binary | K20 | 300 | 25 | 261 | 0.263333 [0.213333, 0.31] | [0.24, 0.585366] | 0.435897 (n=39) |
| liquid_native_joint | K10 | 300 | 25 | 261 | 0.37 [0.316667, 0.423333] | [0.342351, 0.742857] | 0.615385 (n=39) |
| liquid_native_joint | K20 | 300 | 25 | 261 | 0.36 [0.306667, 0.416667] | [0.330948, 0.777778] | 0.615385 (n=39) |
| jina35_native | K10 | 300 | 25 | 261 | 0.466667 [0.41, 0.523333] | [0.431013, 0.888889] | 0.794872 (n=39) |
| jina35_native | K20 | 300 | 25 | 261 | 0.536667 [0.48, 0.593333] | [0.505521, 0.918919] | 0.820513 (n=39) |
| pplx_decider_native | K10 | 300 | 25 | 261 | 0.51 [0.453333, 0.566667] | [0.481149, 0.861111] | 0.769231 (n=39) |
| pplx_decider_native | K20 | 300 | 25 | 261 | 0.536667 [0.48, 0.593333] | [0.50309, 0.926829] | 0.846154 (n=39) |
| bge_pointwise | K10 | 300 | 25 | 261 | 0.433333 [0.376667, 0.486667] | [0.402187, 0.848485] | 0.717949 (n=39) |
| bge_pointwise | K20 | 300 | 25 | 261 | 0.393333 [0.336667, 0.446667] | [0.358025, 0.848485] | 0.717949 (n=39) |

All tables are exploratory. CSSR is evidence-retention, not factual answer correctness; source-group intervals are sensitivity diagnostics.
