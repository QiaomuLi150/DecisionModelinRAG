# R0 CRAG Validity and Evaluation-Sensitivity Report

## Gate

`R0_VALID_BUT_LOW_INFORMATION`

### Contract conclusion

The corrected E11-V visible-text comparison is valid for a deterministic selector-to-reader reference-proxy audit. The four selector outputs contain 300 valid rows each; the reader contains 1,200 one-to-one policy/question records; all task/output request hashes and prompt hashes agree; the reader revision and generation settings are constant; application caching is disabled; and no gold answer appears in selector or reader inputs. The corrected namespace does not reuse the original 900 historical raw-HTML reader outputs.

### Why agreement is low

The 16–18% exact-reference rates should not be assigned to selector quality alone. CRAG does not provide passage-level support qrels or independent grounding labels. The reader applies a 256-token cap per selected page, and exact matching credits only normalized overlap with the stored answer or alternatives. The prompt also instructs the reader to abstain when supplied evidence is insufficient. These are measured evaluation/input constraints; their relative causal contribution is not identified by existing outputs.

### Selector-dependent sensitivity

PPLX and full-visible-text Jina selected different Top-5 IDs on all 300 queries. Their mean Top-5 ID Jaccard was `0.306190`. The reader answers were identical on `200` queries and different on `100`. Exact-reference proxy status changed on `15` queries. The primary contingency was 46 both-correct, 7 PPLX-only, 8 Jina-only, and 239 neither.

### Diagnostic evidence

A 40-question cohort was frozen by metadata-only domain/source-component hash ranking before diagnostic labels were inspected. The 160 resulting lexical checks are silver lower bounds only: phrase presence in Top-20 or selected Top-5 does not establish that the passage is sufficient, correct, current, or causally used by the reader. No gold-evidence injection, human adjudication, or new reader call was performed.

### Evaluation boundary

The exact and token-F1 metrics are deterministic answer/reference-overlap proxies. They are not factual correctness, grounding, support completeness, or citation validity. R1 therefore required an independent population before any new inference; that gate did not pass.
