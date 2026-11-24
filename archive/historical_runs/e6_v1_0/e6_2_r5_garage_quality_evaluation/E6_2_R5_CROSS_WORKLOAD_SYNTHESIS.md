# E6-2 R5 Cross-Workload Synthesis

SciFact and GaRAGe are kept separate. SciFact uses natural Top-20 retrieval, document-level relevance judgments and 277 exploratory queries. GaRAGe uses supplied fixed K15 pools, passage-level `evidence_relevant` annotations and 1,397 exploratory samples.

| workload | PPLX−Jina | uncertainty | feasibility/limitations |
|---|---:|---|---|
| SciFact | +0.017400 nDCG@5 | query-bootstrap 95% CI [-0.010549, +0.045281] | corrected judged-positive Top-20 feasibility 254/277; Amos had 83 context failures in R2 |
| GaRAGe | -0.015802 nDCG@5 | query-bootstrap 95% CI [-0.022765, -0.008966] | fixed K15 passage pool; source identities and sealed reservation not established |

This is a cross-workload comparison, not an independent confirmation or pooled leaderboard. The candidate-generation process, pool size, annotation semantics, source independence and context workload differ. Selection quality does not establish factual answer correctness or production economics.
