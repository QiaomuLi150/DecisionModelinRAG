# E6-2 R0 E6-1 Deviation Reconciliation

## Finding

E6-1 and E6-1B used a deterministic SHA-256 candidate order over the public SciFact corpus. This is a valid label-blind adapter-feasibility fixture: qrels, answers and reserved data were not in the native input. It is not a frozen output of BM25, dense, hybrid or other natural retrieval, so it cannot support SciFact nDCG/Recall/Hit claims.

## Preserved evidence

- E6-1 input manifest SHA-256: `a1b006a79456aeba818fe3faf0059a56c1eb1c6adaf47483cac665845097f318`.
- E6-1 native results: 1,536 valid calls across six models; 2,697 observed/derived forward accounting entries.
- E6-1B: StartLux 256/256 valid; Amos 250/256 valid with six K20 context-limit failures and unmeasured physical forwards.
- No qrels, reader answers, Jev calls or paid calls were used.

## Claim boundary

The records support native interface validity, exact-ID output validation, context feasibility and preliminary order sensitivity. They do not support retrieval quality, model superiority, source-independent generalization or production economics.

## Prospective correction

Before E6-2 quality inference, execute a separately approved retrieval-only preparation that produces gold-blind, fixed SciFact Top-20 slates with documented retriever, corpus snapshot, index, candidate mapping, serialization, deduplication and tie-breaking. The current R0 does not run this preparation.
