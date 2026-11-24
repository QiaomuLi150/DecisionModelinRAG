# E2-P0v2 Holdout and Leakage Plan

**Status:** `HOLDOUT_SELECTION_BLOCKED_PENDING_OWNER_REVIEW`

No existing Track A, Track B, HERB, Track C/R2, CAL or M0 population may be relabeled as a fresh confirmatory holdout. The candidate source must be selected from a locally verified release with source/revision hash, license, query/document mapping, evidence annotations, and a documented sampling frame. If no such source is approved, the new work remains exploratory.

Before sealing, perform query-ID, source-document, answer, normalized-text and candidate-corpus overlap checks against all prior evaluated populations. Keep the protected gold sidecar inaccessible to inference and CAL tuning. Freeze TRAIN/CAL/DEV/TEST manifests, random seed, sample size, source records and hashes in an immutable lock.

The owner must approve: dataset/revision, population and n, gold-set completeness contract, source-document independence, deduplication policy, retrieval/candidate construction, holdout access protocol, and confirmation status. Do not inspect proposed TEST labels or score models before this approval.

Recommended design: a separately released document-QA/multihop source with page/document-level evidence annotations and a new query/document-disjoint sample; exact source and n remain `TBD`.
