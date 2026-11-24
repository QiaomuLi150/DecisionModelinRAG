# MuSiQue blocker

Status: `BLOCKED` under decision rule A3-R1.

The repository contains the 2,167 selected MuSiQue records and prior V11 hybrid RRF Top-20 traces. The prior trace records `BM25 top-100 + dense top-100 + RRF@60` output and its RRF order; it does not preserve the exact BM25 top-100/top-20 list needed to reconstruct the required BM25-only Top-20 without running a new retrieval. Running BM25 over the corpus would violate the frozen-input-only rule.

All 2,167 selected source records and their annotated support cardinalities were audited offline; no relevance outcome was used to construct this blocker. No model call was made. Unblocking requires the exact hashable BM25 Top-20 slate and candidate text manifest, or an owner-approved versioned protocol amendment.
