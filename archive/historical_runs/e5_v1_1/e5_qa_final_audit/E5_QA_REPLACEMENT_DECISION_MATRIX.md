# E5-QA Replacement Decision Matrix

| Dimension | PPLX Decider | Jina v3.5 | Interpretation |
|---|---|---|---|
| E5-A CSSR-U | 306/1502 = 20.37% | 299/1502 = 19.91% | +0.5405 pp equal-group effect; CI crosses zero |
| E5-A CSSR-C | 306/355 = 86.20% | 299/355 = 84.23% | Retrieval-conditional annotated support retention |
| E5-C representative T1 | 18/200 = 9.0% | 17/200 = 8.5% | +0.5 pp; reference proxy only |
| E5-D p50/p95 | 1320.085/1616.623 ms | 62.245/88.790 ms | Same GPU5, concurrency 1; fixed order |
| Factual correctness | NOT_MEASURED | NOT_MEASURED | No T2/T3 endpoint |
| Prospective confirmation | LOCKED | LOCKED | E5-F untouched |

Decision: `EXPLORATORY_COMPETITIVE_BUT_NOT_REPLACEMENT_READY`.

Quality is numerically competitive but not statistically superior/equivalent/non-inferior. PPLX has substantially higher measured selector latency and allocator peak. The result is conditional on QASPER, one PPLX policy, one Jina policy, frozen natural Top-20 slates and the declared contracts. No production QPS, dollar cost, universal family claim or factual-answer claim is supported.
