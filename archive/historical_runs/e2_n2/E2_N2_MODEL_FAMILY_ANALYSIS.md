# E2-N2 Model-Family Analysis

Status: `EXPLORATORY_PASS_WITH_LIMITATIONS`

N2-4 was recomputed offline from the authenticated score traces with the frozen gold sidecars and 5,000 paired-bootstrap repetitions (seed 731). The outputs are exploratory complete-support selection proxies; they are not factual answer-accuracy results and must not be pooled across Track A and Track B.

## Highest conditional proxy by dataset

| Cohort | Dataset | Policy | CSSR-C |
|---|---|---|---:|
| `track_a` | `hotpot` | `jina35` | 0.9732 |
| `track_a` | `musique` | `jina35` | 0.7802 |
| `track_a` | `wiki2` | `pplx_decider` | 0.9763 |
| `track_b` | `enterprise` | `pplx_decider` | 0.9212 |

Intern-Decision is not measured for the 500 enterprise rows because of native context overflow. Small strata and model-family contrasts are descriptive only. Consult the machine-readable quality and paired-statistics tables for all denominators and intervals.
