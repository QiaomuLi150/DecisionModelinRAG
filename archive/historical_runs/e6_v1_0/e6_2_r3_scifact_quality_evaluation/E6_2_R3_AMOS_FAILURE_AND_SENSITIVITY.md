# Amos failure and sensitivity audit

Amos is scored on the full 277-query intent-to-evaluate population. The 83 preserved context-limit failures receive zero operational nDCG under the frozen conservative convention; no fallback, retry, truncation, or post-outcome selection was introduced.

- Full population: 277
- Valid selections: 194
- Preserved failures: 83
- Failure rate: 0.299639
- Success-conditioned nDCG@5: 0.761064 (diagnostic only)
- Full-population operational nDCG@5: 0.533020

The failure records do not carry a usable per-request input-token length in the frozen R2 schema; therefore a success/failure token-length distribution is **NOT_MEASURED**. R2 resource metadata reports the successful-run token total, but it cannot be used to infer failed-request lengths.
