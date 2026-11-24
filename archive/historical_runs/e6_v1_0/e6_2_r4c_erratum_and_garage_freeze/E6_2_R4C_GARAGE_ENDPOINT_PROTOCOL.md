# E6-2 R4C GaRAGe endpoint and statistical protocol

## Frozen population

The exact-K15 aligned population is 2329 rows, partitioned outcome-blind into CAL/EXPLORATORY/RESERVED using whole observed-content-overlap components and seed 731. The target is 20/60/20; actual counts are `{'CAL': 466, 'EXPLORATORY': 1397, 'RESERVED': 466}`. The EXPLORATORY input projection contains 1397 rows. RESERVED identifiers are retained as metadata only; no sealed-vault claim is made.

## Primary endpoint

For the later approved native selector run, binary nDCG@5 uses `evidence_relevant=YES` as gain 1 and NO as gain 0. All inference-population rows remain in the denominator; no-positive rows receive operational nDCG=0 and are reported separately. A relevance-evaluable subset with at least one YES is diagnostic only and cannot replace the primary denominator.

Secondary endpoints are Precision@5, Recall of annotated relevant passages, Hit@5, selected relevant count, category coverage, no-positive diagnostics and metadata-stratified descriptions. Selecting a relevant passage is not factual answer correctness.

The source-provided first five candidates are a fixed-pool prefix control only, not a natural retrieval baseline. The paired primary comparison is PPLX−Jina with 5000 bootstrap replicates, seed 731, two-sided percentile 95% intervals. A valid source-group bootstrap is not claimed: only observed-content-overlap clustered sensitivity and query bootstrap are allowed until defensible source identity and reservation attestation exist.

## Frozen model rules

The eight-model roster and previously validated native K15→Top5 policies are unchanged. No model was loaded or called in R4C; this document freezes the contract for any separately approved future inference.
