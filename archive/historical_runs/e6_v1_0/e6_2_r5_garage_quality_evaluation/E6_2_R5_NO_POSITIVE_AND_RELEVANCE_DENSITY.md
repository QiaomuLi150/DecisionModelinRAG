# E6-2 R5 No-Positive and Relevance-Density Diagnostics

Operational denominator: **1397** EXPLORATORY samples.
No-positive samples: **64** (4.5812%). They remain in the primary denominator and receive nDCG@5=0 under the frozen convention.
Unresolved relevance entries: **1 sample**; the blank value is retained as unresolved and is not recoded to NO.

## Relevance-density strata

| stratum | n | PPLX nDCG@5 | Jina nDCG@5 | prefix nDCG@5 |
|---|---:|---:|---:|---:|
| zero_relevant | 64 | 0.000000 | 0.000000 | 0.000000 |
| one_relevant | 35 | 0.792570 | 0.788152 | 0.258159 |
| two_to_four_relevant | 115 | 0.707064 | 0.771023 | 0.319013 |
| five_or_more_relevant | 1183 | 0.936994 | 0.949568 | 0.833957 |

These strata are descriptive structural diagnostics; they were not used for model selection or a confirmatory claim. A no-positive K15 slate does not prove that useful evidence is absent outside the frozen pool.

Metadata such as question complexity/category was not carried into the label-blind R4D projection at sample level, so those analyses are NOT_MEASURED rather than reconstructed from aggregate partition counts.
