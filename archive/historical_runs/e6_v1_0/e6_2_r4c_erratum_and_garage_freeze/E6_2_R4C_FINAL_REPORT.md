# DecisionRAG E6-2 R4C — SciFact Erratum Closure and GaRAGe Cohort Freeze

## Final gate

`E6_2_R4C_GARAGE_READY_WITH_INDEPENDENCE_LIMITATIONS`

The SciFact erratum is verified and closed as a versioned append-only correction. The primary PPLX−Jina nDCG@5 result reproduces exactly, while Top-20 judged-positive feasibility is corrected from 277/277 to 254/277. The GaRAGe exact-K15 exploratory population is frozen with all partitions and label-blind model inputs. The gate is qualified because the public release lacks stable source-document identities and sealed reservation attestation.

## Required closure facts

- SciFact corrected feasibility: **254/277 (0.916967509)**; 23 missing judged-positive Top-20 slates.
- SciFact primary: **Δ +0.017400; CI [-0.010549, +0.045281]**; exact point/CI reproduction: `True`.
- GaRAGe release: 2366 rows; exact-K15 aligned eligibility: 2329; excluded: 37.
- Frozen partition counts: `{'CAL': 466, 'EXPLORATORY': 1397, 'RESERVED': 466}`; EXPLORATORY model-input rows: 1397.
- Observed-content components: 1701; largest size 336; 768/2329 eligible rows (32.976%) are in non-singleton components. Exact repeated normalized passage hashes: 688/34,122 (2.016%); accepted approximate shingle pairs: 271, of which 45 crossed sample IDs. These are overlap diagnostics, not source-document identity.
- Endpoint: binary nDCG@5 over `evidence_relevant` YES/NO; no-positive rows retained with zero convention and separately reported.
- Eight native K15 contracts: prepared from R0; no new model validation or inference in R4C.
- Label-isolation projection: PASS; labels/answers absent from `results/non_jev/v2/e6_v1_0/e6_2_r4c_erratum_and_garage_freeze/E6_2_R4C_GARAGE_EXPLORATORY_INPUTS.jsonl`.
- Model calls/physical forwards/quality scores/GPU work: **0**.
- Jev calls/paid calls: **0/0**.
- Regression: focused checks **13/13**, committed R4C pytest checks **8/8**, full suite **245/245**.

## Scientific interpretation

The frozen EXPLORATORY population is suitable for an explicitly exploratory fixed-pool grounding evaluation after a separate owner authorization. It is not confirmation-grade: source-document IDs are absent, observed passage overlap is only a partial independence diagnostic, public model-pretraining exposure is unknown, and the reservation is a metadata-only partition rather than a sealed vault. No GaRAGe model quality result is claimed here.

R4C does not start R4 inference or E6-2 R5. A separate owner decision is required before any model loading.
