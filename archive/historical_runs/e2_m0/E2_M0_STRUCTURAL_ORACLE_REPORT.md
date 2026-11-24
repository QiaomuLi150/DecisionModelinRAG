# E2-M0 Structural Oracle Audit

**Status:** `E2_M0_COMPLETE_WITH_LIMITATIONS_READY_FOR_OWNER_REVIEW`

## Executive result

This is a CPU-only, read-only recomputation from frozen Track A/B slates, evaluation-only gold records, explicit saved group assignments, and frozen G5 selected IDs. No model, GPU, Qdrant, network, hosted API, or Jev request was made by this audit.

The oracle is annotation-conditioned: it measures whether the recorded Top-20 and one-winner-per-group policy could retain the supplied gold identity set. It is not a causal attribution of model error and does not prove that the annotated gold set is the only valid support set.

## Population waterfall

| Cohort | N_all | N_eligible (1–5, resolved) | N_conditional (all gold in Top-20) | Excluded |
|---|---:|---:|---:|---:|
| Track A prospective | 3767 | 3767 | 1490 | 0 |
| Track B enterprise | 500 | 443 | 330 | 57 |

Track A's gold sidecar contains resolved 2–4-document support for all 3,767 selected questions. Track B retains all 500 questions; 30 are empty-gold, 26 have more than five unique gold documents, and qst_0413 is excluded from identity-dependent evaluation under the accepted `V11.1R1-GOLD-ID-AMBIGUITY-qst_0413` exception. Therefore Track B has 443 identity-resolved 1–5-gold questions, not 469.

## Structural estimates

| Cohort | Policy | Population | Group-quota ceiling | Observed G5 | Quota loss | Residual gap |
|---|---|---|---:|---:|---:|---:|
| track_a/pooled | laya_g5 | unconditional eligible | 1196/3767 (0.3175) | 195/3767 (0.0518) | 294/3767 (0.0780) | 1001/3767 (0.2657) |
| track_a/pooled | laya_g5 | retrieval conditional | 1196/1490 (0.8027) | 195/1490 (0.1309) | 294/1490 (0.1973) | 1001/1490 (0.6718) |
| track_a/pooled | clef_g5 | unconditional eligible | 1196/3767 (0.3175) | 689/3767 (0.1829) | 294/3767 (0.0780) | 507/3767 (0.1346) |
| track_a/pooled | clef_g5 | retrieval conditional | 1196/1490 (0.8027) | 689/1490 (0.4624) | 294/1490 (0.1973) | 507/1490 (0.3403) |
| enterprise/pooled | laya_g5 | unconditional eligible | 318/443 (0.7178) | 124/443 (0.2799) | 12/443 (0.0271) | 194/443 (0.4379) |
| enterprise/pooled | laya_g5 | retrieval conditional | 318/330 (0.9636) | 124/330 (0.3758) | 12/330 (0.0364) | 194/330 (0.5879) |
| enterprise/pooled | clef_g5 | unconditional eligible | 318/443 (0.7178) | 206/443 (0.4650) | 12/443 (0.0271) | 112/443 (0.2528) |
| enterprise/pooled | clef_g5 | retrieval conditional | 318/330 (0.9636) | 206/330 (0.6242) | 12/330 (0.0364) | 112/330 (0.3394) |

## Dataset and category detail

| Scope | Policy | N_eligible | N_conditional | Collision queries | Quota loss | Residual gap | Observed |
|---|---|---:|---:|---:|---:|---:|---:|
| track_a/hotpot | laya_g5 | 800 | 598 | 114 | 114 | 391 | 93 |
| track_a/hotpot | clef_g5 | 800 | 598 | 114 | 114 | 149 | 335 |
| track_a/musique | laya_g5 | 2167 | 555 | 125 | 125 | 371 | 59 |
| track_a/musique | clef_g5 | 2167 | 555 | 125 | 125 | 248 | 182 |
| track_a/wiki2 | laya_g5 | 800 | 337 | 55 | 55 | 239 | 43 |
| track_a/wiki2 | clef_g5 | 800 | 337 | 55 | 55 | 110 | 172 |
| enterprise/basic | laya_g5 | 175 | 149 | 0 | 0 | 83 | 66 |
| enterprise/basic | clef_g5 | 175 | 149 | 0 | 0 | 51 | 98 |
| enterprise/completeness | laya_g5 | 4 | 3 | 1 | 1 | 1 | 1 |
| enterprise/completeness | clef_g5 | 4 | 3 | 1 | 1 | 1 | 1 |
| enterprise/conflicting_info | laya_g5 | 19 | 15 | 3 | 3 | 10 | 2 |
| enterprise/conflicting_info | clef_g5 | 19 | 15 | 3 | 3 | 8 | 4 |
| enterprise/constrained | laya_g5 | 30 | 26 | 1 | 1 | 14 | 11 |
| enterprise/constrained | clef_g5 | 30 | 26 | 1 | 1 | 10 | 15 |
| enterprise/high_level | laya_g5 | 0 | 0 | 0 | 0 | 0 | 0 |
| enterprise/high_level | clef_g5 | 0 | 0 | 0 | 0 | 0 | 0 |
| enterprise/info_not_found | laya_g5 | 0 | 0 | 0 | 0 | 0 | 0 |
| enterprise/info_not_found | clef_g5 | 0 | 0 | 0 | 0 | 0 | 0 |
| enterprise/intra_document_reasoning | laya_g5 | 40 | 37 | 0 | 0 | 22 | 15 |
| enterprise/intra_document_reasoning | clef_g5 | 40 | 37 | 0 | 0 | 13 | 24 |
| enterprise/miscellaneous | laya_g5 | 20 | 17 | 0 | 0 | 7 | 10 |
| enterprise/miscellaneous | clef_g5 | 20 | 17 | 0 | 0 | 1 | 16 |
| enterprise/project_related | laya_g5 | 30 | 18 | 7 | 7 | 9 | 2 |
| enterprise/project_related | clef_g5 | 30 | 18 | 7 | 7 | 8 | 3 |
| enterprise/semantic | laya_g5 | 125 | 65 | 0 | 0 | 48 | 17 |
| enterprise/semantic | clef_g5 | 125 | 65 | 0 | 0 | 20 | 45 |

The same invariant was checked per eligible row: `0 <= observed_G5 <= group_oracle <= retrieval_oracle <= 1`. The saved Laya and Clef partitions were explicit in their selector records and cross-checked against each full groupwise trace; no rank-contiguous groups were inferred.

## Collision summary

Collision means at least two annotated gold identities occur in the same recorded group, conditional on complete gold presence in Top-20. The per-dataset/category and multiplicity counts are in `E2_M0_AGGREGATE_ORACLES.csv` and per-query collision fields. Both policies are reported separately because their authenticated group records were not assumed identical.

## Historical reconciliation

Recomputed `top20_complete_gold=True` counts from the frozen metric rows: Track A laya_g5=1490, clef_g5=1490; Track B laya_g5=330, clef_g5=330. These equal the row-level oracle conditional denominators (1,490 and 330 respectively) and are used only as a cross-check, not as the oracle source.

## Optional analyses

Same-score policy ablation: `NOT_RUN — the core M0 audit does not promote saved heterogeneous pointwise traces to a verified globally comparable score contract.` No quartet softmax values were merged or globally sorted. HERB citation feasibility: `NOT_APPLICABLE_TO_CORE_M0 — no citation-set oracle was needed for the required A/B audit.`

## Integrity and limitations

Input records were resolved from 25 explicitly named files. Integrity rows: 1; case exclusions: 57. Existing artifacts were not written or modified; preservation hashes are recorded in the provenance manifest.

The result is limited by the benchmark annotations: a listed support set is treated as the supplied gold identity set, not as a proof of exhaustive necessary evidence. The residual gap combines within-group model scoring, prompt/context effects, and tie-breaking. It should not be described as intrinsic decision-model failure.

## P0v2 handoff

Future M1/P0v2 work must freeze a single globally comparable document scoring contract, explicit group partitions, a fresh untouched holdout, serving SLO/power/multiplicity decisions, and any license/GPU gates before inference. This audit does not authorize P0v2, M1, new retrieval, or paid evaluation.
