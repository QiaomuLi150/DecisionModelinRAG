# E2-N2 Authenticity Audit

Verdict: `PASS_WITH_LIMITATIONS`

Authenticated E2-N smoke/CAL/full traces were reused because their exact revisions, frozen candidate views, input hashes, raw-response hashes, and scoring contracts match the N2-0 contract. No raw inference was repeated in N2.

Verified source trace files: **12**; normalized score rows: **55471**; rows with physical-forward evidence: **54971**; rows with gold leakage: **0**.

The full traces include explicit Intern-Decision context-blocked records. The fresh confirmatory holdout was not accessed. No Jev, paid API, training, retrieval, indexing, or production-service change occurred.

All source trace SHA-256 values and corresponding summary hashes are recorded in the provenance manifest. Future newly executed traces must hash raw response bytes before normalization.
