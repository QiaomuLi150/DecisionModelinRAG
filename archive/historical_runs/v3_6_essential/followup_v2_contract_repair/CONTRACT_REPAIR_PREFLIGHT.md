# Jina Contract Repair Preflight

Status: **PASS**

The historical request hash is over `query + candidate_ids + candidate_texts`. The v3.6 rerun used `documents`, so its exact-input gate was invalid.

- Population: 1384
- Historical PASS: 1352
- Historical FAIL retained without retry: 32
- Candidate-ID exact: 1356
- Historical request-hash exact: 1352
- Native replay eligible: 1352
- Labels joined: NO
- Protected holdouts accessed: NO
- Native inference in preflight: NO

The manifest freezes exact E11 budgeted texts. Historical failures are preserved and not retried.

GPU snapshot status: `UNOBSERVABLE`.
