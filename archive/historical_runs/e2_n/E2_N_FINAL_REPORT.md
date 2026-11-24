# DecisionRAG v2.1 E2-N Expanded Model Benchmark — Final Owner-Review Package

Status: `PASS_WITH_LIMITATIONS`

## Completed scope

N0 offline verification, N1 authentic local smoke tests, N2 CAL configuration freezing, N3 frozen Track A/B historical exploratory scoring, and N4 independent offline analysis are complete for the verified, instrumented local branches. No training, retrieval/indexing, reader generation, Jev, paid API, or fresh confirmatory-holdout access occurred.

- N0: `4` verified-local roster entries; `11` exact proposed entries blocked by absent local checkpoint/dependency and no download authorization; `1` Jev entry not authorized.
- N0 CPU validation: 22 focused checks passed; existing repository suite: 65/65 passed.
- N1: BGE, Clef, and Laya strict gates passed; Qwen aligned smoke output passed with a physical-service telemetry limitation.
- N2: BGE 30 rows, Laya 30 rows, Clef 60 rows (30 queries × pointwise/native-joint) passed. Config hashes are in `n2/E2_N_CAL_FREEZE_MANIFEST.json`.
- N3: BGE 4,267 rows, Laya 4,267 rows, Clef 8,534 compatible reused rows; every row has candidate alignment, input/output hashes, and physical-forward evidence.
- N4: 21,335 frozen Top-5 selection rows, 30 dataset/policy quality rows, 32 paired bootstrap rows.

## Verified model roster

| Model | N0 | N1 | N2 | N3 | Final handling |
|---|---|---|---|---|---|
| Cloudflare/clef-flash | VERIFIED_LOCAL | PASS | PASS | PASS | included |
| convaiinnovations/laya | VERIFIED_LOCAL | PASS | PASS | PASS | included |
| BAAI/bge-reranker-v2-m3 | VERIFIED_LOCAL | PASS | PASS | PASS | included |
| Qwen/Qwen3-VL-Reranker-2B | VERIFIED_LOCAL | PASS_WITH_TELEMETRY_LIMITATION | BLOCKED_TELEMETRY | NOT_RUN | smoke-only; telemetry blocked |

New decision candidates Bespoke Nimble-9B-v2, Intern-Decision-4B, and PPLX Decider 27B, and new specialist candidates Qwen3-Reranker-0.6B/4B, Jina-v3.5, GTE-ModernBERT, and MiniLM-L6 remained blocked because exact local checkpoints/dependencies were absent and downloads were not authorized. No nearby substitute was silently used.

## Scorecard (complete-support CSSR proxy)

Track A and Track B are intentionally separate. `CSSR-U` is unconditional over identity-resolved eligible records; `CSSR-C` is conditional on all gold identities being present in the frozen Top-20. These are selector-support metrics, not factual answer accuracy.

| Cohort | Dataset | Policy | n eligible | n conditional | CSSR-U | CSSR-C |
|---|---|---|---:|---:|---:|---:|
| track_a | hotpot | retrieval5 | 800 | 598 | 0.4975 | 0.6656 |
| track_a | hotpot | bge_pointwise | 800 | 598 | 0.6875 | 0.9197 |
| track_a | hotpot | laya_pointwise | 800 | 598 | 0.1075 | 0.1438 |
| track_a | hotpot | clef_pointwise | 800 | 598 | 0.2762 | 0.3696 |
| track_a | hotpot | clef_joint | 800 | 598 | 0.4300 | 0.5753 |
| track_a | musique | retrieval5 | 2167 | 555 | 0.0992 | 0.3874 |
| track_a | musique | bge_pointwise | 2167 | 555 | 0.1680 | 0.6559 |
| track_a | musique | laya_pointwise | 2167 | 555 | 0.0125 | 0.0486 |
| track_a | musique | clef_pointwise | 2167 | 555 | 0.0355 | 0.1387 |
| track_a | musique | clef_joint | 2167 | 555 | 0.0697 | 0.2721 |
| track_a | wiki2 | retrieval5 | 800 | 337 | 0.2437 | 0.5786 |
| track_a | wiki2 | bge_pointwise | 800 | 337 | 0.3400 | 0.8071 |
| track_a | wiki2 | laya_pointwise | 800 | 337 | 0.0413 | 0.0979 |
| track_a | wiki2 | clef_pointwise | 800 | 337 | 0.2188 | 0.5193 |
| track_a | wiki2 | clef_joint | 800 | 337 | 0.2387 | 0.5668 |
| track_b | enterprise | retrieval5 | 443 | 330 | 0.5576 | 0.7485 |
| track_b | enterprise | bge_pointwise | 443 | 330 | 0.5372 | 0.7212 |
| track_b | enterprise | laya_pointwise | 443 | 330 | 0.2709 | 0.3636 |
| track_b | enterprise | clef_pointwise | 443 | 330 | 0.3070 | 0.4121 |
| track_b | enterprise | clef_joint | 443 | 330 | 0.4560 | 0.6121 |

## Runtime and integrity

- BGE selector p50/p95: 20.25/27.84 ms per 20-candidate query; Laya: 145.34/156.56 ms; Clef combined pointwise/native-joint trace rows: 1,195.80/2,385.23 ms. These are model-scoring scopes, not end-to-end serving latency.
- Laya physical forwards: 5 per query under its fixed candidate batch size; Clef preserves pointwise and native-joint semantics separately.
- GPUs used: GPU0 for Clef and Laya workloads; idle fallback GPU6 for BGE and Laya CAL/smoke workloads. GPUs1/2/4 hosted active services and were not disturbed.
- No checkpoint downloads, service restarts, model training, Jev calls, or paid API calls.
- All score traces contain frozen candidate IDs, actual input hashes, raw output hashes, physical-forward evidence, cache markers, and gold-blind inference markers.

## Remaining blockers and owner decision

1. Qwen requires durable service-level physical-forward telemetry before any full E2-N promotion.
2. The new official model roster requires separate download/network authorization and exact checkpoint/revision verification.
3. The current findings are exploratory on previously inspected Track A/B populations; they do not authorize Jev or paid inference.

The next owner decision is whether to resolve those blockers or accept the current three-model exploratory comparison as the E2-N endpoint. Jev remains untouched and requires separate authorization.

## Artifact locations

- N0: `results/non_jev/v2/e2_n/n0/`
- N1: `results/non_jev/v2/e2_n/n1/`
- N2: `results/non_jev/v2/e2_n/n2/`
- N3: `results/non_jev/v2/e2_n/n3/`
- N4: `results/non_jev/v2/e2_n/n4/`
- Machine-readable aggregate: `results/non_jev/v2/e2_n/E2_N_EXECUTION_MANIFEST.json`
