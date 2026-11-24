# v1.5 C3 Bounded Concurrency Diagnostic

## Status

`C3_BOUNDED_EXPLORATORY_COMPLETE_D_SEPARATE_BLOCKED`

- Genuine label-blind native event records: `476`.
- GPU: `3` (`<REDACTED_GPU_UUID>`); post-run host check: released to 4 MiB used and 0% utilization.
- PPLX revision: `14787.462` ms load measurement; model revision is pinned in the protocol.
- D_separate was not run because no second dedicated A100 passed the isolation gate.

## Rmax diagnostic

- Best serial native-call capacity row: `8.029474` requests/s at tested concurrency `4`.
- This is a serial native-call capacity diagnostic, not sustainable production QPS.

## Observed bounded cells

| Config | Routing rate | Selection rate | Type | n | p50 ms | p95 ms | Throughput/s |
|---|---:|---:|---|---:|---:|---:|---:|
| A_routing_only | 0.3 | 0.0 | routing | 21 | 124.96578693389893 | 126.38780195266008 | 0.35 |
| A_routing_only | 0.6 | 0.0 | routing | 48 | 124.94488758966327 | 178.61741213943068 | 0.8 |
| B_consolidated | 0.3 | 0.25 | routing | 22 | 126.90607458353043 | 1133.0626597248497 | 0.36666666666666664 |
| B_consolidated | 0.3 | 0.25 | selection | 9 | 2190.869959598004 | 3173.566595936074 | 0.15 |
| B_consolidated | 0.6 | 0.25 | routing | 29 | 2443.380066045959 | 6609.611573253965 | 0.48333333333333334 |
| B_consolidated | 0.6 | 0.25 | selection | 18 | 3333.1302736291073 | 7730.532750018531 | 0.3 |
| B_consolidated | 0.3 | 0.5 | routing | 24 | 12331.129782084616 | 17461.995720169853 | 0.4 |
| B_consolidated | 0.3 | 0.5 | selection | 31 | 12503.506973661657 | 17650.650285663964 | 0.5166666666666667 |
| B_consolidated | 0.6 | 0.5 | routing | 37 | 14838.714784486576 | 24238.808675098484 | 0.6166666666666667 |
| B_consolidated | 0.6 | 0.5 | selection | 35 | 13053.286967059015 | 22542.714624231383 | 0.5833333333333334 |
| C_colocated | 0.3 | 0.25 | routing | 25 | 124.64279215782881 | 126.36100444942713 | 0.4166666666666667 |
| C_colocated | 0.3 | 0.25 | selection | 12 | 210.96963435411453 | 242.30448808521032 | 0.2 |
| C_colocated | 0.6 | 0.25 | routing | 42 | 124.39809320494533 | 126.5727047342807 | 0.7 |
| C_colocated | 0.6 | 0.25 | selection | 12 | 191.0944771952927 | 241.00135397166014 | 0.2 |
| C_colocated | 0.3 | 0.5 | routing | 12 | 125.26828749105334 | 329.72920242538027 | 0.2 |
| C_colocated | 0.3 | 0.5 | selection | 35 | 220.86147214813323 | 308.04105137395084 | 0.5833333333333334 |
| C_colocated | 0.6 | 0.5 | routing | 45 | 124.41891897469759 | 240.17048921716935 | 0.75 |
| C_colocated | 0.6 | 0.5 | selection | 19 | 222.0936417579651 | 355.4890010378265 | 0.31666666666666665 |

## Integrity limitations

- The full v2 C3 schedule was shortened to 10 s warm-up and 60 s measurement cells under versioned protocol `PROTOCOL_V3_BOUNDED.json`; these results are exploratory and not a replacement for the original 60/300-second design.
- The recovered R1 routing source contains one fixed request; the pool repeats that exact request with stable IDs. No prompt diversity is claimed.
- The queue is a deterministic FIFO virtual-arrival simulation using genuine main-thread native forwards. The failed thread-wrapper attempt is preserved by its traceback/session and is not included as a result.
- No labels or relevance outcomes entered any model input. No Jev/paid calls occurred.
