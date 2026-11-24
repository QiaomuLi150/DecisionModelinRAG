# E5-0C Retrieval Feasibility

## CAL exact ceiling

`CAL_EXACT_CEILING_NOT_MEASURED`. The E5-0 artifact set contains no frozen CAL Top-20 slates and no matching CAL query-embedding cache. The permitted corpus embedding matrix exists, but recomputing the dense CAL query side would require a new embedding/model call, which is prohibited in CPU/offline E5-0C. No BM25-only proxy was substituted.

## Exploratory structural ceiling

Using only frozen 1,502 exploratory slates and the frozen designated support IDs, the pre-selector-outcome structural ceiling is:

- Any designated support present in Top-20: 428/1502 (0.2850)
- Complete designated support present in Top-20: 343/1502 (0.2284)
- Complete designated support already in first-stage Top-5: 212/1502 (0.1411)
- Source groups with at least one complete Top-20 query: 247/587

This is an exploratory retrieval-feasibility ceiling, not a selector result or a power calculation. Infeasible natural slates remain in the unconditional E5-A population and must be reported separately from CSSR-C.
