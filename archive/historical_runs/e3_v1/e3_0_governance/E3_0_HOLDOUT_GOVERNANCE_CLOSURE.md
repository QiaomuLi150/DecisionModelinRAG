# E3-0 Holdout Governance Closure

Status: `HOLDOUT_RESERVED_METADATA_ONLY`

## Dataset releases

- MultiHop-RAG: `71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82` from the official Hugging Face release and repository `c1c1287aa60a94acf9c4d20c891c9cd611a0f6e8`; 2,556 query records and 609 corpus records.
- FinanceBench: official repository `c1c1287aa60a94acf9c4d20c891c9cd611a0f6e8`; 150 public annotated questions and 360 document metadata records. The repository's larger described collection is not treated as locally available.

## Reservation rule

The partition is frozen from connected components over canonical source-document metadata using seed `731`. Stable hash bucket 0 of 4 is reserved; all other buckets are exploratory. The selection did not inspect answer text, evidence text, model outcomes, or reader outputs. Reserved question IDs are recorded for enforcement, but no reserved labels/results are included in exploratory manifests.

## Integrity

- Reserved IDs: **195**.
- Exploratory IDs: **2,511**.
- Valid identity-resolved reserved IDs: **107**.
- Valid identity-resolved exploratory IDs: **2,298**.
- Reserved/exploratory ID overlap: **0**.
- Partition mapping failures: **0**.
- Answer/model-outcome use during partition: **false**.

## Limitations

This is an experimenter-access reservation, not a guarantee of model-training novelty. FinanceBench's public sample is a reference-answer benchmark; its evidence annotations are not assumed exhaustive support sets. MultiHop-RAG records with no usable support-document identity remain in the manifest but are excluded from identity-dependent evaluation with an explicit reason.

The exploratory study may proceed only through the allowlisted nonreserved IDs. E3-F remains locked.
