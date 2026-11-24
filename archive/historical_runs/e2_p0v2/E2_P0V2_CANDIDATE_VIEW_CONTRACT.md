# E2-P0v2 Candidate View Contract

## Setting A — common information

Canonicalize the nonempty question with Unicode normalization and stable whitespace rules; reject blank/whitespace-only input. Preserve canonical document ID and frozen candidate order. Use the frozen source text/title view only after recording bytes, tokenizer tokens, truncation and overflow. The current historical view is 64 question tokens, 128 pointwise passage tokens and 64 groupwise passage tokens; this is historical evidence, not an automatic native contract for future models.

Every feasible model receives the same semantic document content. If its native context cannot contain the common view, mark the model infeasible for Setting A; do not silently shorten its evidence.

## Setting B — native optimized

Permit documented batching, chunking, context sizing, candidate ordering, prompt formulation and quantization after CAL-only tuning. Record all model-specific preprocessing and charge all preparation and repeated passes. Do not claim fairness from equal raw token counts when architectures tokenize/repeat differently; compare semantic evidence, physical work and cost transparently.

## Candidate identity

`candidate_ids` are canonical, unique and stable. A selected ID must be a member of the frozen Top-20. Duplicates, missing IDs, overflow, empty question, cache reuse without exact-input provenance and hidden filtering fail closed.
