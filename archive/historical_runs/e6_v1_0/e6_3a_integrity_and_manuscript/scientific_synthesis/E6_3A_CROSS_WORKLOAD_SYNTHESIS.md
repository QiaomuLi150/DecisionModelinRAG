# E6-3A Cross-Workload Scientific Synthesis

## Evidence-selection quality

- SciFact E6-2 PPLX−Jina operational binary nDCG@5: `0.017399600216`, 95% CI `[-0.010548645189, 0.045281123175]`, n=`277`. Corrected Top-20 feasibility was 254/277. The interval includes zero.
- GaRAGe E6-2 PPLX−Jina binary evidence-relevant nDCG@5: `-0.015802136913`, 95% query-bootstrap CI `[-0.022765040204, -0.008965936101]`, n=`1397`. Source-document independence was not established, so this is exploratory and not a source-group confirmatory claim.

These endpoints are not pooled. They differ in candidate construction, annotation semantics and independence qualification.

## Downstream reader utility

The frozen 200-question reader proxy is PPLX mean `0.116365493883` versus Jina `0.119772228125`, paired difference `-0.003406734242`. This is lexical overlap against a stored reference; no independent factual adjudication exists.

## Serving

The original E6-3 SciFact scorecard is invalid for full-content comparison because 2,000 sampled candidate views were empty. The corrected E6-3A rows are separate and use the same 100-query sample with full authenticated E6-2 source views. Corrected failures, p50/p95 and memory are in `serving_audit/E6_3A_CORRECTED_SERVING_SCORECARD.csv`. They are bounded sequential diagnostics, not sustainable QPS or monetary cost.

## Boundary

The evidence supports workload-dependent descriptive feasibility. It does not establish universal superiority, noninferiority, causal downstream improvement, factual correctness or production economics. The dominant correction finding is operational: native context feasibility and input-view fidelity materially affect the apparent cost and failure profile.
