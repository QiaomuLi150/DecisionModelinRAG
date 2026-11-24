# DecisionRAG E3-D-R Final Scientific Report

Status: `E3_D_R_OWNER_REVIEW_READY`

This bounded audit used only existing E3-D artifacts. No model, GPU, HTTP, Jev, paid API, training, selector, reader, holdout outcome, or historical-artifact mutation occurred.

## R0 authenticated inputs
- Reader rows: **2154** = 250 MultiHop × 6 + 109 FinanceBench × 6.
- Full-stage selector selections reused: **6054**; native trace rows including CAL/smoke/full: **6348**.
- All reader rows real inference: **2154**; application-cache reuse: **0**; gold in reader input: **0**.
- Input hash checks matching recorded E3-D manifest: **47/47**.

## R1 FinanceBench answer audit
Legacy scoring was independently reproduced from each stored answer/reference pair; the historical E3-D rates are unchanged. The new endpoints are deterministic, post-hoc exploratory reference proxies only.

| policy | legacy matches / 109 | normalized-or-numeric matches / 109 | demonstrated numeric mismatches |
|---|---:|---:|---:|
| retrieval5 | 8 | 10 | 0 |
| bge_pointwise | 10 | 13 | 2 |
| jina35 | 12 | 13 | 1 |
| pplx_decider | 10 | 12 | 2 |
| liquid_joint20_binary | 12 | 13 | 1 |
| zerank2_pointwise | 12 | 15 | 1 |

No row is promoted to independently verified factual correctness. Manual adjudication was not performed. Blank/refusal, unit ambiguity, strict-format false negatives and deterministic numeric mismatches are reported separately; unresolved cases remain unknown.

## R2 matched MultiHop transitions
- Exact paired reader denominator: **250** questions, not the 900-query selector population.
- PPLX complete support: **113/250**; Jina complete support: **118/250**.
- PPLX-only complete-support cases: **27**; among these, PPLX/Jina proxy outcomes were better/equal/worse **1/26/0**.
- Four evidence cells: both=86, PPLX-only=27, Jina-only=32, neither=105.
- Four-cell table is in `E3_D_R_MULTIHOP_EVIDENCE_ANSWER_4X4.csv`; paired answer transitions and hashes are in `E3_D_R_MULTIHOP_PAIRED_ANSWER_TRANSITIONS.csv`.
- PPLX-vs-Jina answer outcomes are normalized reference proxies. No causal mediation or factual-correctness claim is made. Source-document-cluster bootstrap was not measured because the supplied E3-D reader artifact does not contain an auditable source-cluster key; query bootstrap is the only reported CI.

## R4 holdout
- Governance result: **E3_0_QUERY_HELD_OUT_ONLY**.
- Structural reserved/exploratory non-overlap and outcome-free metadata checks pass in the existing manifest. Reservation-before-analysis timestamp/commit is not independently verified, so E3-F remains locked.

## R5 tests and reproducibility
- New deterministic normalizer checks: **6/6**.
- Repository full regression executed with `PYTHONPATH=. pytest -q -rA`: **106/106 PASS**, including the six new E3-D-R tests. The historical E3-D report remains preserved as 100/100.
- Full new provenance manifest covers every R artifact and referenced input hash.

## Scientific conclusion
E3-D supports exploratory evidence-selection and reference-answer-proxy comparisons across the six policies. It does not establish factual answer accuracy, exhaustive FinanceBench support recall, causal evidence-to-answer mediation, or an independent confirmatory holdout. The appropriate next decision is owner review of the proposed E3-F protocol, not E3-F execution.
