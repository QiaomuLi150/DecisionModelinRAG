# DecisionRAG R0–R3 Final Local-Model Scientific Report

## Final gate

`R3_FINAL_LOCAL_MODEL_EVIDENCE_FROZEN`

R0 and R1 were completed offline. R0 is valid but low-information; R1 admitted no independent population, so R2 was correctly skipped with zero new model or reader calls. This is a completed scientific closure, not a favorable-result gate.

## R0 comparability and sensitivity

- Corrected CRAG cohort: 300 questions, four policies, 1,200 reader records.
- One-to-one policy/question joins: `True`.
- Reader cache reuse: none; all corrected records are fresh, cache-disabled requests.
- Contract issues: `0`.
- All four selector arms were valid 300/300 and all reader records were successful 300/300 per arm.
- Selector outputs differed for every retrieval-vs-neural comparison; PPLX vs full-visible-text Jina differed on 300/300 queries. The PPLX/Jina mean Top-5 ID Jaccard was `0.306190`.
- PPLX vs Jina reader answers were identical on `200` queries and changed on `100`; exact-reference proxy status changed on `15`.

The corrected comparison is valid as a selector-dependent deterministic reference-proxy comparison. It is not a factual-grounding evaluation.

## PPLX versus Jina reference-proxy contingency

| Outcome | Count |
|---|---:|
| Both proxy-correct | 46 |
| PPLX-only proxy-correct | 7 |
| Jina-only proxy-correct | 8 |
| Neither proxy-correct | 239 |

PPLX exact-proxy total: 53/300. Full-visible-text Jina: 54/300. The exact-proxy difference is `-0.003333`, with query bootstrap CI `[-0.02674999999999997, 0.023333333333333334]` and source-group CI `[-0.047576832151300236, 0.016400709219858155]`. Token-F1 proxy difference is `0.011374`, with query CI `[-0.013333736700127566, 0.03669946708129382]` and source-group CI `[-0.02718648051364708, 0.032968001696545596]`.

## Why agreement is low

The audit supports several measured contributors: the CRAG corpus has no passage-level support/qrels; many selected views are truncated to the fixed 256-token reader page budget; the reader is instructed to abstain when evidence is not supported; and exact reference matching is a narrow deterministic proxy that does not credit every valid formulation unless it overlaps the stored answer/alternative. The data do not justify attributing the remaining error to selector quality alone. The silver lexical diagnostics are lower bounds only and are not gold judgments.

## R1 admission decision

`R1_NO_ADMISSIBLE_POPULATION`. CRAG was rejected for independent validation because the retained admission process exposed the release-wide metadata and the completed 300-query cohort is already development/exploratory evidence; the release also lacks passage-level support annotations and an attested source-independent partition. KILT-NQ was rejected because no authenticated local revision, retrieved-passage-ID corpus, or supported candidate contract exists without a new retrieval/index campaign. No other dataset was searched.

## R2

`R2_NOT_ADMISSIBLE`; no selector, reader, GPU, Jev, paid API, or holdout calls were made in this package.

## Historical evidence boundaries

E11's BRIGHT result remains post-E9 exploratory context-feasibility evidence. Its attributed operational decomposition remains contribution `+0.0172353803`, failure penalty `-0.1156074095`, total `-0.0983720293`, subject to its original hashes and bootstrap lineage. E11-V remains reference-proxy evidence. No result establishes universal replacement, equivalence, architecture causality, factual correctness, grounding, production QPS, or cost savings.

## Integrity

Full regression result: `PASS` over `301` collected tests. Historical hash verification checked `21` retained artifacts with `0` missing; status `PASS`. Historical artifacts were not modified; protected holdouts remained closed; Jev and paid calls were zero. The authoritative claims matrix, joined records, gates, diagnostics, and provenance manifest are in this namespace.
