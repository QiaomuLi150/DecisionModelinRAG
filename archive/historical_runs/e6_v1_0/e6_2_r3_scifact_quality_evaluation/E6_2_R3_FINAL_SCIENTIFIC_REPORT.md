# DecisionRAG E6-2 R3 — Independent SciFact Relevance Evaluation

## Gate

`E6_2_R3_SCIENTIFIC_RESULTS_READY_WITH_LIMITATIONS`

The frozen R1 retrieval slates and R2/StartLux selection outputs were authenticated and scored offline. R3 made zero new model, retrieval, reader, Jev, or paid API calls.

## Accounting

| population | planned | executed | valid | failed |
|---|---:|---:|---:|---:|
| combined eight-model registry | 2216 | 2216 | 2133 | 83 |

The 83 failures are Amos context-limit failures. StartLux contributes 277 repaired valid selections; its original R2 load-blocked artifact and gate remain preserved.

## Primary endpoint

The primary comparison is binary nDCG@5 on the same 277 exploratory queries, operationally scored with failed selections at zero. PPLX minus Jina = **0.017400**, query-bootstrap percentile 95% CI **[-0.010549, 0.045281]**, 5,000 resamples, seed 731. PPLX-only paired utility wins: 37; Jina-only: 34; equal: 206.

The primary CI is query-bootstrap because the authenticated R0 contract does not establish source-independent group resampling. Equal-source-group aggregation is descriptive only. An interval containing zero is not evidence of equivalence or non-inferiority.

## Model scorecard

See `E6_2_R3_MODEL_QUALITY_SCORECARD.csv`. It reports full-population operational means, valid-only diagnostics, failures, judged Recall@5/Hit@5, and Top-20 feasibility.

## Retrieval feasibility and sparse labels

The exact Top-20 feasibility count and qrel projection audit are in `E6_2_R3_RETRIEVAL_FEASIBILITY_REPORT.md` and `E6_2_R3_QREL_AND_ENDPOINT_AUDIT.md`. Unjudged documents are not verified negatives, and none of these metrics establish factual answer correctness.

## Resource boundary

R3 did not run inference. Latency, memory, and token fields are inherited from R2 and the separate StartLux repair. They are descriptive historical measurements, not a new matched serving benchmark.

## Limitations

Sparse qrels, descriptive-only source components, Amos context failures, separate checkpoint training, and inherited resource metadata limit causal and production claims. E6-3 remains a separate serving experiment and was not started.
