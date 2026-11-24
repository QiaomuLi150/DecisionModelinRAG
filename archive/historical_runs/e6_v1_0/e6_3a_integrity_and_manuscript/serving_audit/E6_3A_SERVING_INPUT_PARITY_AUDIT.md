# E6-3A Serving Input Parity Audit

## Verdict

`SERVING_CONTRACT_DISCREPANCY_CONFIRMED`.

The original E6-3 SciFact serving path selected the frozen 100-query sample correctly, but its `canonical_item()` read `c.get("text", "")` from the R1 slate candidate objects. The authenticated R1 SciFact slate stores candidate IDs, titles and content hashes; it does not store a `text` field. The E6-3 path therefore submitted empty candidate passages.

Evidence: `2000/2000` sampled candidate records lacked `text`; the corresponding authenticated corpus reconstruction has non-empty source text for `2000/2000` records. Candidate ID/order hashes matched E6-2 for all audited overlaps, so the defect is content-view construction rather than sampling or candidate identity.

The sample was independently recomputed from all `277` exploratory questions using the recorded SHA-256 rule: match=`True`. It contains `28` of the historical `83` Amos context-limit failures, and `72` historical Amos successes. There is no evidence that the sample was filtered by Amos success, relevance, latency, token count or model output.

## Native-input and prediction evidence

See `E6_3A_NATIVE_CONTRACT_COMPARISON.json` and `E6_3A_PREDICTION_PARITY.csv`. For every model with an E6-2 row, candidate ID hashes match, while the E6-2 and E6-3 native input hashes do not match on the audited sample. Top-5 agreement is consequently diagnostic only. Amos has `28` historically failed sampled questions but zero E6-3 failures because it received empty passages.

The exact model revisions were retained. The discrepancy is not evidence of a model improvement and does not change the historical E6-2 failure count.
