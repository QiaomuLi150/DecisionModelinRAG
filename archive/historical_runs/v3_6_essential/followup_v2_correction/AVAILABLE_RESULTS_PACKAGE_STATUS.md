# v3.6 Follow-up v2 Correction — Available Results Snapshot

This append-only snapshot packages the results currently available at packaging time. It does not claim completion of the whole correction milestone.

## Status at snapshot

- **C1:** PARTIAL. CPU length preparation is complete for 2,667 transfer questions at L=2048, 4096, 6144 and 8192. Jina matched-text traces and PPLX L=2048/4096 traces are complete. The three PPLX L=6144/8192 shards are still running/partial and are included as `.part` evidence. The earlier abandoned two-shard partials are preserved and are not valid join inputs.
- **C2:** COMPLETE for the available diagnostic work. Historical frozen-versus-corrected comparison and two fresh fp32 batch-1 runs are included. Fresh runs agreed at the list and set levels on their jointly valid requests; the historical frozen-versus-corrected disagreement remains documented.
- **C3:** COMPLETE as a repair attempt, but both policies remain invalid for latency comparison because their valid rates are below 99%. Winnow: 483/1,080 valid and 597 HTTP 400. Drex-v2: 966/1,080 valid and 114 HTTP 400. Five HTTP-400 bodies per policy are included.
- **C4:** NOT STARTED/BLOCKED at this snapshot because the clean dedicated-GPU telemetry gate had not passed. No clean C4 timing result is represented here.
- **C5:** COMPLETE as an offline frontier reconstruction from the existing v2 quality and N1 serving artifacts; it is not a substitute for missing clean C4 measurements.

## Included material

The ZIP contains the full current `followup_v2_correction` result namespace, including preparation manifests, raw traces, partial traces, diagnostic outputs, server-repair records, C5 frontier outputs, and this status file. Existing originals were copied; none were edited or deleted.

The archive is a point-in-time snapshot. Active C1 processes, if any, may produce later append-only updates that are not represented in this ZIP.
