# Paper tables and figures plan

No manuscript was drafted. The following tables/figures are recommended only if the owner approves publication preparation.

1. **Dataset and contract table:** BRIGHT domains, query counts, frozen BM25 Top-20 protocol, known-positive feasibility, model revisions and native readout contracts. Sources: `00_admission/*`, `01_retrieval/*`, `02_native/BRIGHT_NATIVE_RUN_*.json`.
2. **Primary ranking table:** operational macro-domain nDCG@5, PPLX/Jina and Liquid/Qwen effects, domain-bootstrap and query-weighted sensitivity. Source: `E9_BRIGHT_PAIRED_SENSITIVITY_STATISTICS.json`.
3. **Reliability table:** valid/failure counts, PPLX context failures, Qwen OOM, successful/failed/all-attempt latency. Source: `E9_BRIGHT_LATENCY_AND_FAILURE_AUDIT.md`.
4. **Domain forest plot:** per-domain PPLX−Jina and Liquid−Qwen effects with domain labels; exploratory and not pooled with incompatible E2–E8 metrics. Source: comparison statistics and scorecard.
5. **Feasibility plot:** per-domain known-positive Top-20 coverage and conditional nDCG; label as known-positive feasibility, not complete support. Source: `BRIGHT_RETRIEVAL_FEASIBILITY.json` and audit statistics.
6. **Evidence ledger table:** one row per claim with population, endpoint, status, source path and limitations. Source: `DECISIONRAG_E2_E9_AUTHORITATIVE_EVIDENCE_LEDGER.csv`.

Do not plot E6-3 original empty-passage timing as full-content serving evidence. Do not plot proxy answer agreement as verified factual accuracy.
