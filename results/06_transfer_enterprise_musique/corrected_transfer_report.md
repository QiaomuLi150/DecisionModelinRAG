# DecisionRAG v3.6 Reproducibility Fixes — Final Output

## Gate and scope

This append-only follow-up executed the exact serialized request-builder correction (R1) and the offline Jina mismatch audit (R2). No new benchmark, model family, paid API, or reader inference was introduced. Raw model records were frozen before labels were joined. The final gate is **R1_PARTIAL_R2_DONE**: R1 is fully accounted for but retains explicit failures; R2 is complete for the exact available artifact pair.

## R1 — corrected transfer budget grid

The corrected builder measured the exact serialized request, including candidate IDs, template and separators, then decremented the shared per-candidate cap until the request was at most L−128. Each prepared manifest contains 2,667 rows (500 Enterprise and 2,167 MuSiQue).

| L | n | builder_failures | built_n | max_serialized_tokens | limit_L_minus_128 | all_built_within_limit |
|---|---|---|---|---|---|---|
| 2048 | 2667 | 473 | 2194 | 1920 | 1920 | True |
| 4096 | 2667 | 0 | 2667 | 3968 | 3968 | True |
| 6144 | 2667 | 0 | 2667 | 6014 | 6016 | True |
| 8192 | 2667 | 0 | 2667 | 8042 | 8064 | True |

The 473 L=2048 Enterprise failures are builder failures, not model failures. MuSiQue has no builder failures at any L. At L=4096, 6144 and 8192 all 2,667 requests build within the frozen limit.

### Quality and validity scorecard

The complete per-dataset/per-budget scorecard is in `exports/r1_summary.csv`; the condensed table below includes the two requested controls.

| dataset | L | policy | n | valid | builder_fail | model_fail | nDCG@5 | FS@5 | max_tok |
|---|---|---|---|---|---|---|---|---|---|
| enterprise | 2048 | hybrid_first_stage | 500 | 1.000 | 0 | 0 | 0.5099 | 0.5576 | 1917 |
| enterprise | 2048 | jina | 500 | 0.054 | 473 | 0 | 0.0114 | 0.0158 | 1917 |
| enterprise | 2048 | jina_full_text | 500 | 1.000 | 0 | 0 | 0.5384 | 0.5869 | - |
| enterprise | 2048 | pplx | 500 | 0.054 | 473 | 0 | 0.0137 | 0.0158 | 1917 |
| enterprise | 4096 | hybrid_first_stage | 500 | 1.000 | 0 | 0 | 0.5099 | 0.5576 | 3968 |
| enterprise | 4096 | jina | 500 | 1.000 | 0 | 0 | 0.3323 | 0.3770 | 3968 |
| enterprise | 4096 | jina_full_text | 500 | 1.000 | 0 | 0 | 0.5384 | 0.5869 | - |
| enterprise | 4096 | pplx | 500 | 1.000 | 0 | 0 | 0.5902 | 0.6230 | 3968 |
| enterprise | 6144 | hybrid_first_stage | 500 | 1.000 | 0 | 0 | 0.5099 | 0.5576 | 6014 |
| enterprise | 6144 | jina | 500 | 1.000 | 0 | 0 | 0.4344 | 0.4740 | 6014 |
| enterprise | 6144 | jina_full_text | 500 | 1.000 | 0 | 0 | 0.5384 | 0.5869 | - |
| enterprise | 6144 | pplx | 500 | 0.996 | 0 | 2 | 0.6134 | 0.6411 | 6014 |
| enterprise | 8192 | hybrid_first_stage | 500 | 1.000 | 0 | 0 | 0.5099 | 0.5576 | 8042 |
| enterprise | 8192 | jina | 500 | 1.000 | 0 | 0 | 0.4852 | 0.5327 | 8042 |
| enterprise | 8192 | jina_full_text | 500 | 1.000 | 0 | 0 | 0.5384 | 0.5869 | - |
| enterprise | 8192 | pplx | 500 | 1.000 | 0 | 0 | 0.6381 | 0.6569 | 8042 |
| musique | 2048 | hybrid_first_stage | 2167 | 1.000 | 0 | 0 | 0.3856 | 0.0992 | 1920 |
| musique | 2048 | jina | 2167 | 1.000 | 0 | 0 | 0.4949 | 0.1564 | 1920 |
| musique | 2048 | jina_full_text | 2167 | 1.000 | 0 | 0 | 0.5473 | 0.1970 | - |
| musique | 2048 | pplx | 2167 | 1.000 | 0 | 0 | 0.4363 | 0.1555 | 1920 |
| musique | 4096 | hybrid_first_stage | 2167 | 1.000 | 0 | 0 | 0.3856 | 0.0992 | 3935 |
| musique | 4096 | jina | 2167 | 1.000 | 0 | 0 | 0.5358 | 0.1878 | 3935 |
| musique | 4096 | jina_full_text | 2167 | 1.000 | 0 | 0 | 0.5473 | 0.1970 | - |
| musique | 4096 | pplx | 2167 | 1.000 | 0 | 0 | 0.4848 | 0.1763 | 3935 |
| musique | 6144 | hybrid_first_stage | 2167 | 1.000 | 0 | 0 | 0.3856 | 0.0992 | 5113 |
| musique | 6144 | jina | 2167 | 1.000 | 0 | 0 | 0.5442 | 0.1934 | 5113 |
| musique | 6144 | jina_full_text | 2167 | 1.000 | 0 | 0 | 0.5473 | 0.1970 | - |
| musique | 6144 | pplx | 2167 | 1.000 | 0 | 0 | 0.4820 | 0.1717 | 5113 |
| musique | 8192 | hybrid_first_stage | 2167 | 1.000 | 0 | 0 | 0.3856 | 0.0992 | 5744 |
| musique | 8192 | jina | 2167 | 1.000 | 0 | 0 | 0.5477 | 0.1980 | 5744 |
| musique | 8192 | jina_full_text | 2167 | 1.000 | 0 | 0 | 0.5473 | 0.1970 | - |
| musique | 8192 | pplx | 2167 | 1.000 | 0 | 0 | 0.4870 | 0.1754 | 5744 |

### Matched PPLX-minus-Jina contrasts

Bootstrap settings: 40,000 question-level resamples, seed 731, failed outputs scored 0, labels joined offline after raw freeze.

| dataset | L | n | delta | 95% CI |
|---|---|---|---|---|
| enterprise | 2048 | 443 | 0.0023 | [-0.0061, 0.0112] |
| enterprise | 4096 | 443 | 0.2580 | [0.2249, 0.2920] |
| enterprise | 6144 | 443 | 0.1789 | [0.1476, 0.2105] |
| enterprise | 8192 | 443 | 0.1529 | [0.1228, 0.1838] |
| musique | 2048 | 2167 | -0.0586 | [-0.0683, -0.0488] |
| musique | 4096 | 2167 | -0.0510 | [-0.0596, -0.0424] |
| musique | 6144 | 2167 | -0.0622 | [-0.0709, -0.0536] |
| musique | 8192 | 2167 | -0.0608 | [-0.0694, -0.0523] |

Enterprise is strongly positive for PPLX at L=4096–8192; its L=2048 contrast is near zero because only 27/500 requests build. MuSiQue is negative for PPLX at every budget, with CIs excluding zero. These are transfer-set observations, not a claim of general superiority.

### Raw accounting

- Jina: 10,668 total records; 10,195 valid native outputs and 473 L=2048 builder failures.
- PPLX: 10,668 total records; 10,193 valid native outputs, 473 L=2048 builder failures, and two explicit `CANCELLED_IN_FLIGHT_FOR_BOUNDARY_PARTITION` records. The two records were created when the first two workers were stopped at a predeclared disjoint boundary to avoid duplicate/partial retry; they were not retried.
- The completed suffix consisted of 5,292 valid native outputs across four workers (1,323 each). No suffix failures or OOMs were observed.
- Raw records retain `labels_joined=false`; labels were added only in the offline scorer.

## R2 — exact Jina mismatch audit

The audited pair is the available 1,384-query `jina_two_stage_8192` sweep and the earlier `jina_two_stage_pplxwindow` control. It contains 1,384 common queries, zero Top-5 list differences and zero Top-5 set differences. Stage score hashes agree for all 1,384 rows; request/final-stage hashes are identical for 1,380 rows. Four rows differ in recorded request-hash metadata but not in score hashes or selected lists. Tokenizer, dtype and batch size are not observable from these records.

Therefore the stated 197-query disagreement (0.1790 versus 0.1855) is **not present in this exact pair of frozen artifacts**. The trace audit does not justify a rerun, and it does not establish why a different historical comparison may have differed. `r2_differing_queries.csv` is retained and contains the empty differing-query result with its header.

## Integrity and reproducibility

- Protocol SHA-256: `34c4d9e7e3cef61fd4337f8aa0cc13224853087eaf22b2fda764cae7c8ecb677`.
- The raw-trace SHA-256 manifest is `R1_RAW_TRACE_SHA256_MANIFEST.json`.
- Prepared manifests satisfy the exact L−128 limit for every successfully built request.
- Duplicate/omitted raw-key accounting and raw label-blindness checks pass.
- Existing E2–v3.6 artifacts were preserved; no historical file was overwritten.
- The pinned project environment full regression suite passed: **301 passed, 15 warnings, 0 failures**. The system interpreter separately showed three dependency failures (two missing `pyarrow`, one missing `torch`); those are environment failures, not follow-up test failures.

## Export packages

- `decisionrag_v3_6_repro_fixes_exports.zip`: sanitized compact exports and reports.
- `decisionrag_v3_6_repro_fixes_raw_traces.zip`: sanitized raw JSONL traces and raw-trace manifest.

## Limitations

R1 includes explicit non-retried boundary cancellations and a deterministic L=2048 builder-infeasibility population; those are part of the reported missingness and not silently converted to valid model results. The R2 audit can only verify fields recorded in the frozen traces; tokenizer, dtype and batch size are not independently observable there. The score comparisons are benchmark-specific and do not establish production equivalence.
