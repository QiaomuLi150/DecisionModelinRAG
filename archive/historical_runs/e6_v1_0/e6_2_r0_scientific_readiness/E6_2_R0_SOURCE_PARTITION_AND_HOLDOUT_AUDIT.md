# E6-2 R0 Source Partition and Holdout Audit

SciFact source/qrel identity graph: 572 connected components. Frozen metadata split: CAL 555 queries / 287 components; EXPLORATORY 277 / 142; RESERVED 277 / 143. QID and component hashes are preserved in `E6_0R_SOURCE_SPLIT_MANIFEST.json`; split seed is 731.

The reserved receipt reports `METADATA_ONLY_RESERVED_LOCK`, raw question text exported = `False`, qrels exported = `False`. No reserved raw content, labels, candidate slates or outcomes were loaded by this R0. The repository convention is process-level only; no external sealed vault or OS deny boundary was established, so chronology and latent near-duplicate independence remain limitations.

Known E6-0R historical overlap audit reported zero exact identity overlap with QASPER, DocHop-QA and GaRAGe for the checked fields; near-duplicate analysis was not run. This supports a bounded identity statement, not a claim of complete semantic independence.

The full exploratory partition has 643 source groups in the broader historical accounting, while the SciFact admitted source graph reports 142 exploratory components. These are different scopes and must not be conflated.
