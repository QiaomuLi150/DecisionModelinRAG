# E3-F0 Power Methods and Validation

Power is planning sensitivity, not evidence. For each effect (2, 5, 10 percentage points), baseline success 0.46 and total paired discordance 0.20 were used. PPLX-only and Jina-only discordances were simulated as independent binomials; a one-sided McNemar-style z threshold of 1.959964 was applied. Five thousand deterministic replicates per n use seed 731 plus an effect/n offset. Results are in the CSV and minimum-n summary in the JSON.

This simple calculation does not model source-document clustering, missingness, multiplicity, or adaptive model selection. Future design must use cluster-aware simulation once the new source-group structure is known. The 2 pp and 5 pp planning values are illustrative and unapproved. No result is described as achieved power for an existing dataset.

Validation edge cases: zero discordance yields no positive superiority evidence; complete failure yields no usable success endpoint; high intra-cluster correlation reduces effective sample size; an effect whose simulated power does not reach 80% or 90% by n=5,000 is marked infeasible under these assumptions.
