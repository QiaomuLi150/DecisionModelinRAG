# DecisionRAG E5-0C Final Readiness and Scientific Report

## Final gate

`E5_0C_PROTOCOL_AMENDMENT_REQUIRED`. The existing first-wrapper CSSR-U is not valid as the sole primary E5-A endpoint; a pre-outcome set-aware amendment and owner/reviewer approval are required before E5-A. E5-F remains locked.

## Authenticated facts

E5-0 output manifest: 36 files; missing=0; mismatches=0. Historical E4-R audit: checked=31, clean=True. Frozen Top-20 SHA-256 exact match: True. Existing E5-0 gate: `E5_0_CONDITIONAL_READY`.

## Evidence semantics

CAL has 519 mapping-valid questions, but answer wrappers and evidence spans are not universally exhaustive. Reconstructed first-designated support hashes match the E5-0 mapping for `True`. Multiple alternative/partial/non-text cases are reported separately. CSSR-U is defensible only as designated-support retention.

## Retrieval and feasibility

The frozen retriever is global over 45,497 permitted paragraphs, using question text only, BM25 Top-100 + Qwen dense Top-100 + RRF(60). Out-of-paper candidates are common and are not treated as negative gold. CAL exact hybrid feasibility is not measured because matching CAL query embeddings/slates are absent and new embedding calls are prohibited. Exploratory pre-outcome complete-support Top-20 feasibility is 343/1502 (0.2284).

## Source and governance limitations

Permitted exact source grouping is auditable. Cross-reservation near-duplicate disjointness is `RESERVED_NEAR_DUPLICATE_DISJOINTNESS_NOT_INDEPENDENTLY_VERIFIED`. External reservation attestation is unavailable and access isolation is partial. GPU5/6 historical smoke use is `GPU_5_6_AUTHORIZATION_UNVERIFIED` in the saved artifact record; E5-0C used no GPU.

## Decision

E5-A must not begin until the proposed set-aware endpoint amendment is approved and a separate GPU/model authorization is recorded. No E5-A result, factual correctness claim, parity claim, or confirmatory claim is made here. No historical artifacts were modified.
