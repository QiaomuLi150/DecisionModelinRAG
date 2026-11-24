# v3.6 Follow-up v2 Correction — Updated Available Results Snapshot

This status file records the post-C1-join state of the append-only correction namespace.

## Gate status

- **C1: DONE for the authorized transfer-set quality work.** The CPU length scan covered 2,667 questions at L=2048, 4096, 6144 and 8192. The completed native traces contain 21,336 rows across PPLX/Jina and all four budgets. The join produced 21,336 per-query rows, 16 summary rows and 8 paired contrasts using 40,000 bootstrap draws with seed 731. Labels were joined only after raw outputs were frozen.
- **C2: DONE for the diagnostic work.** The two fresh fp32, batch-1 Jina runs agree at the list and set levels on the jointly valid requests; the historical frozen-versus-corrected disagreement remains preserved as a diagnostic and does not replace the frozen result.
- **C3: DONE as a repair attempt, NOT COMPARABLE for latency.** Winnow has 483/1,080 valid requests with 597 HTTP 400 failures. Drex-v2 has 966/1,080 valid requests with 114 HTTP 400 failures. Both are below the 99% latency-comparison threshold.
- **C4: BLOCKED/NOT RUN.** Clean timing sessions were not started because the dedicated-GPU telemetry gate had not passed. No clean C4 timing claim is present.
- **C5: DONE offline.** The frontier was rebuilt from the existing v2 quality and N1 serving artifacts; it does not substitute for C4.

## Snapshot note

The refreshed ZIP is a point-in-time package. Existing historical and partial traces remain unchanged. The abandoned two-shard C1 partial traces remain preserved but are not included in the C1 join input.
