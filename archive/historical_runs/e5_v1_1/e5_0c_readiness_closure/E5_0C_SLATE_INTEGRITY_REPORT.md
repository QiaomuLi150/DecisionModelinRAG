# E5-0C Frozen Top-20 Slate Integrity Report

- Slate rows: 1502
- Invalid rows: 0
- Candidate rows: 30040
- Each row: exactly 20 unique candidate IDs, all in the permitted corpus: PASS
- Exact file SHA-256: `be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7`
- Required SHA-256: `be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7`
- Exact hash match against the original canonical JSONL byte stream: `True`
- Candidate source-paper fraction matching the query paper: 0.130226; out-of-paper candidates are retained and are not assumed negative.
- qrels/gold in generation: `False` in every frozen row; query text is not present in slate payloads, only its hash.

The canonical byte stream is the existing `E5_0_NATURAL_TOP20_MANIFEST.jsonl`; it was hashed directly, not reserialized.
