# DecisionRAG v3.6 Frozen Prospective Validation v2 — Final Closure

Gate: **`V2_OWNER_REVIEW_READY`**

## Scope and integrity

- Dataset: official BEIR CQADupStack test release, all 12 forums.
- Total query records: **13145**.
- Confirmatory quality population: **13125**; dry-run records excluded: **20**.
- Clean latency population: **2000** fixed IDs.
- Frozen protocol SHA-256: `f15cb0f6001327bfe47e0b1d605f73f7beefc3c6a5fc195689cd5ca1652ce86e`.
- Frozen latency-cohort SHA-256: `5d5f34e3b3a9cfd1b547744e1e4417c992e97b86a81edd94f8f175645652961c`.
- Pass-B labels/qrels: **not read**; all raw traces remained label-blind.
- Original split/contended Pass-B traces and the clean rerun are preserved separately.

## Pass-A quality scorecard

Primary scoring is all-request nDCG@5 with failures scored zero; confidence intervals are forum-bootstrap intervals from the existing frozen analysis.

| Policy | n | Valid rate | Mean nDCG@5 | Forum-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| BM25 Top-5 | 13125 | 1.000000 | 0.237047 | [0.202525, 0.271903] |
| PPLX 6144 | 13125 | 1.000000 | 0.328994 | [0.286773, 0.369579] |
| PPLX 4096 | 13125 | 1.000000 | 0.330125 | [0.287955, 0.370619] |
| Jina full | 13125 | 0.999924 | 0.319825 | [0.281166, 0.358066] |

Jina full had one recorded `CUDA_OOM`/missing quality record in the original Pass-A trace; it was retained and scored as zero. No request was silently dropped.

Primary PPLX 6144 contrasts:

- PPLX 6144 − BM25: **+0.091947**, CI [+0.078371, +0.102198].
- PPLX 6144 − Jina full: **+0.009169**, CI [+0.003114, +0.012811].

Therefore the frozen quality rule is met for PPLX 6144 against both BM25 and Jina. This is a ranking-quality result, not evidence of downstream answer correctness.

## Strict clean Pass-B serving result

The primary latency result uses the two new clean GPU3 sessions, each serial batch-1 and paired across all four admitted policies, with 60 Jina anchors at both the start and end.

| Policy | Valid rate | p50 ms (pooled) | p95 ms (pooled) | p95 ratio vs Jina | 95% ratio CI | P95 S verdict |
|---|---:|---:|---:|---:|---:|---|
| PPLX 4096 | 1.000000 | 940.362 | 1093.179 | 1.487 | [1.294, 1.693] | PASS |
| PPLX 6144 | 1.000000 | 1268.817 | 1612.484 | 2.193 | [1.924, 2.517] | INCONCLUSIVE |
| Jina trunc 8192 | 1.000000 | 138.895 | 222.192 | 0.302 | [0.265, 0.344] | PASS |

Ratios use 10,000 query-cluster bootstrap draws, seed 731, with both serial passes retained per query. All request latency is included; no fallback latency was substituted. PPLX 6144 is therefore **not statistically established within the 2× p95 SLO**, despite passing the quality endpoint. PPLX 4096 passes the measured latency S criterion.

Clean-session integrity checks: 16,000 data records, 240 anchors, zero request failures, zero missing/nonunique policy-query keys, zero cross-pass input-hash mismatches, zero reverse-order mismatches, and zero selected-ID mismatches among paired valid outputs. Allocator OOM warning lines appeared in stderr during the long sessions but did not create failed request records; they are a telemetry limitation, not silently discarded failures.

## Final scientific interpretation

The result supports a **conditional replacement boundary**: PPLX 4096 is competitive on this frozen CQADupStack ranking population and satisfies the measured latency S criterion; PPLX 6144 is the preregistered primary configuration and passes the quality endpoint, but its 2× p95 serving criterion is inconclusive. PPLX 4096's observed mean nDCG@5 is slightly higher in this scorecard, so the evidence does not justify claiming unconditional PPLX 6144 replacement, production cost savings, or answer-quality superiority.

## Tests and provenance

- Designated environment: full `pytest` suite passed in `Production_Project/MultiModalRAG/venv`; only Triton/deprecation warnings were emitted.
- System Python diagnostic: 3 tests could not import optional `pyarrow`/`torch`; this was an environment mismatch and was not used as the project regression result.
- The machine-readable hash and size inventory is in `PROVENANCE_MANIFEST.json`.
- Clean Pass-B analysis artifacts are in `../passB_analysis_clean_v1/`.
- No historical artifact was overwritten; the clean rerun and all earlier partial/contended traces are append-only.
