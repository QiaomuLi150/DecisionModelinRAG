# E4-3 Native Interface and Feasibility Audit

## Frozen population

- Representative: **300**; diagnostic: **80**.
- Source groups: **25**; largest group: **('multihop:component:cc11478c0a2ddecde4f0426a', 340)**.
- All candidates are the original E3-D Top-20 slates. K10 is the exact first-ten prefix; K20 is the exact original slate.

## Contracts

Clef pointwise uses `INDEPENDENT_RELEVANCE:v2`; Clef joint uses `JOINT_BINARY_RELEVANCE:v2`, the same Cloudflare/clef-flash revision and candidate view lineage. The current runner checks output-field count, candidate alignment and finite binary scores. Specialist controls use the existing E3-D native contract families and exact K20 reuse only when the input hash matches.

The PPLX checkpoint is a local 27B bfloat16 artifact. If its K10 load/inference failed, rows are retained as `BLOCKED` and no incompatible proxy is substituted. A successful historical K20 output is not treated as a new K10 call.

Valid new/reused rows: **4620**; blocked/error rows: **12**. Gold and support IDs were stored only as analysis-side fields and were not serialized into native request payloads.

Raw physical-call durations and GPU-memory peaks are retained as diagnostic telemetry. They were collected while the approved GPU was shared, so they are not isolated Top-K-to-Top-5 latency, sustainable QPS, or cost measurements.
