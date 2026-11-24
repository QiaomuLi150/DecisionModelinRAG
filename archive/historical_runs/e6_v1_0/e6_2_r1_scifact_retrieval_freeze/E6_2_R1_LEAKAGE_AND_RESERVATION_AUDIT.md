# E6-2 R1 Leakage and Reservation Audit

## Result

No dense retrieval or final slate construction occurred. The CPU validation used only the public SciFact corpus and the 555 CAL query texts. It did not load qrels, answers, support annotations, selector outputs, or RESERVED query text.

- CAL membership: 555, unique and present in the public query snapshot.
- EXPLORATORY membership: 277, unique and present in the public query snapshot; no content was read for this CPU check.
- RESERVED: 277 metadata count retained from the locked split manifest; content loaded = false.
- Gold injection: not possible in the executed CPU path; qrel files were not imported.
- Historical E5 slate hash preserved: `be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7`.

The final dense/RRF run must independently record that qrels are absent from the retrieval process. A qrel file being present elsewhere on disk is not treated as evidence of access.
