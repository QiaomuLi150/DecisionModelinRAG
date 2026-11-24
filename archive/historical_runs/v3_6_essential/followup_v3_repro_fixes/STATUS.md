# DecisionRAG v3.6 Reproducibility Fixes

- **R1: PARTIAL.** The exact serialized builder was used and every requested key has an accounting record; 473 Enterprise L=2048 builder failures and two predeclared in-flight boundary cancellations remain explicit failures with no retry.
- **R2: DONE.** The exact available sweep/control artifacts contain 1,384 common queries, zero Top-5 list differences and zero Top-5 set differences; the requested 197-query mismatch is not reproduced by this pair.

Labels were joined only after raw outputs were frozen. No original artifacts were overwritten.
