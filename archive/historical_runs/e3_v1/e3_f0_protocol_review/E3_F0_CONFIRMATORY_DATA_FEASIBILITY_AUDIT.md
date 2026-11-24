# E3-F0 Confirmatory Data Feasibility Audit

## Verdict

`CONFIRMATORY_DATA_NOT_READY` with governance field `QUERY_ONLY_OR_PROSPECTIVITY_UNVERIFIED`.

The existing E3-0 manifest reports 195 reserved metadata IDs and 2,511 exploratory IDs, with zero recorded overlap and no answer/model-outcome use in partition assignment. The E3-D-R metadata audit also reports zero reserved IDs used by E3-D. However, the historical exposure audit remains pending and no immutable reservation-before-analysis timestamp, signed commit, or equivalent evidence is available. A manifest written after exploratory analysis cannot retroactively create a prospective confirmatory holdout.

This audit did not open reserved answers, qrels, evidence text, model outputs or evaluation outcomes. Public model-pretraining independence is unknowable. Query disjointness and structural source-group separation are therefore not promoted to independent confirmation.

## Feasible future data routes, not available datasets

1. A newly commissioned multi-hop release with source-document-group disjoint questions, locked before model selection and with complete support annotations.
2. New organizations/filings/reporting periods with independently controlled enterprise source groups and verified document-level gold.
3. A prospective answer-validation cohort with reference answers plus independent expert adjudication of numerical units, periods, entities and calculation chains.

FinanceBench's public 150-question release is already exposed to the exploratory work and its evidence annotations are not exhaustive; it is not a fresh enterprise confirmation population. Existing EnterpriseRAG-Bench is synthetic and cannot establish authentic enterprise transfer.

## Leakage and exposure risks retained

- exploratory model/policy selection and prompt-contract decisions;
- reused public source documents and possible document-level dependence;
- casebook and reference-proxy inspection;
- post-hoc ambiguity about reservation timing;
- public-model pretraining contamination, which cannot be audited here.

E3-F remains locked. A future confirmation requires owner approval of source rights, source-group separation, reservation timestamp, endpoint, missingness policy and analysis code before outcomes are opened.
