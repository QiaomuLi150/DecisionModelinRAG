# DecisionRAG E2-N Authorized Expansion — Owner Review Package v2

Status: `PASS_WITH_LIMITATIONS`

This is a versioned continuation of E2-N after explicit authorization for official checkpoint downloads and Qwen telemetry remediation. Earlier E2-N, M0, P0v2, M1, Track A/B/C, and HERB artifacts are preserved.

## Scientific and safety boundary

N3 uses only the previously inspected historical Track A/B populations and frozen candidate views. The fresh confirmatory holdout was not accessed. No training, retrieval/indexing, reader generation, production-service restart, Jev call, or paid API call occurred.

## N3 trace completeness

| Model | Revision | Rows | OK rows | Physical-forward rows | Status |
|---|---|---:|---:|---:|---|
| Alibaba-NLP/gte-reranker-modernbert-base | f7481e6055501a30fb19d090657df9ec1f79ab2c | 4267/4267 | 4267 | 4267 | PASS |
| Qwen/Qwen3-Reranker-0.6B | e61197ed45024b0ed8a2d74b80b4d909f1255473 | 4267/4267 | 4267 | 4267 | PASS |
| Qwen/Qwen3-Reranker-4B | 22e683669bc0f0bd69640a1354a6d0aebcfeede5 | 4267/4267 | 4267 | 4267 | PASS |
| Qwen/Qwen3-VL-Reranker-2B | 4bd860ac4f15ad1897a214615cccc700f8f71818 | 4267/4267 | 4267 | 4267 | PASS |
| bespokelabs/Bespoke-Nimble-9B-v2 | 4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe | 4267/4267 | 4267 | 4267 | PASS |
| bge | None | 4267/4267 | 4267 | 4267 | PASS |
| clef | None | 8534/8534 | 8534 | 8534 | PASS |
| cross-encoder/ms-marco-MiniLM-L6-v2 | 233902d25c440f23af6f7d6e94d2946bac0bee0a | 4267/4267 | 4267 | 4267 | PASS |
| internlm/Intern-Decision-4B | 0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd | 4267/4267 | 3767 | 3767 | BLOCKED |
| jinaai/jina-reranker-v3.5 | e8a93f33f0b22108f8c2364f8484ce3422552fbc | 4267/4267 | 4267 | 4267 | PASS |
| laya | None | 4267/4267 | 4267 | 4267 | PASS |
| perplexity-ai/pplx-decider-v1.1-27b | 14c9d90137962d300f3de95633f7b6535e80c5ce | 4267/4267 | 4267 | 4267 | PASS |

Direct Qwen-VL telemetry remediation is represented separately by the native checkpoint traces in `n1/E2_N_N1_QWEN_NATIVE_TELEMETRY_SUMMARY.json`, `n2/E2_N_N2_QWEN_DIRECT_CAL_SUMMARY.json`, and `n3/E2_N_N3_QWEN_DIRECT_FULL_SUMMARY.json`. The resident service remains a distinct, non-promoted telemetry path.

## Track A/B quality scorecard

These are complete-support selection proxies. Track A and Track B are not pooled, and these values are not reader factual accuracy.

| Cohort | Dataset | Policy | n eligible | n conditional | CSSR-U | CSSR-C |
|---|---|---|---:|---:|---:|---:|
| track_a | hotpot | retrieval5 | 800 | 598 | 0.4975 | 0.6656 |
| track_a | hotpot | bge_pointwise | 800 | 598 | 0.6875 | 0.9197 |
| track_a | hotpot | laya_pointwise | 800 | 598 | 0.1075 | 0.1438 |
| track_a | hotpot | clef_pointwise | 800 | 598 | 0.2762 | 0.3696 |
| track_a | hotpot | clef_joint | 800 | 598 | 0.4300 | 0.5753 |
| track_a | hotpot | qwen_direct | 800 | 598 | 0.6913 | 0.9247 |
| track_a | hotpot | qwen06 | 800 | 598 | 0.6637 | 0.8880 |
| track_a | hotpot | qwen4 | 800 | 598 | 0.7000 | 0.9365 |
| track_a | hotpot | gte | 800 | 598 | 0.6837 | 0.9147 |
| track_a | hotpot | minilm | 800 | 598 | 0.5950 | 0.7960 |
| track_a | hotpot | jina35 | 800 | 598 | 0.7275 | 0.9732 |
| track_a | hotpot | nimble | 800 | 598 | 0.6687 | 0.8946 |
| track_a | hotpot | intern_decision | 800 | 598 | 0.5750 | 0.7692 |
| track_a | hotpot | pplx_decider | 800 | 598 | 0.7125 | 0.9532 |
| track_a | musique | retrieval5 | 2167 | 555 | 0.0992 | 0.3874 |
| track_a | musique | bge_pointwise | 2167 | 555 | 0.1680 | 0.6559 |
| track_a | musique | laya_pointwise | 2167 | 555 | 0.0125 | 0.0486 |
| track_a | musique | clef_pointwise | 2167 | 555 | 0.0355 | 0.1387 |
| track_a | musique | clef_joint | 2167 | 555 | 0.0697 | 0.2721 |
| track_a | musique | qwen_direct | 2167 | 555 | 0.1809 | 0.7063 |
| track_a | musique | qwen06 | 2167 | 555 | 0.1601 | 0.6252 |
| track_a | musique | qwen4 | 2167 | 555 | 0.1740 | 0.6793 |
| track_a | musique | gte | 2167 | 555 | 0.1509 | 0.5892 |
| track_a | musique | minilm | 2167 | 555 | 0.1431 | 0.5586 |
| track_a | musique | jina35 | 2167 | 555 | 0.1998 | 0.7802 |
| track_a | musique | nimble | 2167 | 555 | 0.1601 | 0.6252 |
| track_a | musique | intern_decision | 2167 | 555 | 0.1366 | 0.5333 |
| track_a | musique | pplx_decider | 2167 | 555 | 0.1887 | 0.7369 |
| track_a | wiki2 | retrieval5 | 800 | 337 | 0.2437 | 0.5786 |
| track_a | wiki2 | bge_pointwise | 800 | 337 | 0.3400 | 0.8071 |
| track_a | wiki2 | laya_pointwise | 800 | 337 | 0.0413 | 0.0979 |
| track_a | wiki2 | clef_pointwise | 800 | 337 | 0.2188 | 0.5193 |
| track_a | wiki2 | clef_joint | 800 | 337 | 0.2387 | 0.5668 |
| track_a | wiki2 | qwen_direct | 800 | 337 | 0.3475 | 0.8249 |
| track_a | wiki2 | qwen06 | 800 | 337 | 0.3312 | 0.7864 |
| track_a | wiki2 | qwen4 | 800 | 337 | 0.3525 | 0.8368 |
| track_a | wiki2 | gte | 800 | 337 | 0.3187 | 0.7567 |
| track_a | wiki2 | minilm | 800 | 337 | 0.3262 | 0.7745 |
| track_a | wiki2 | jina35 | 800 | 337 | 0.3850 | 0.9139 |
| track_a | wiki2 | nimble | 800 | 337 | 0.3700 | 0.8783 |
| track_a | wiki2 | intern_decision | 800 | 337 | 0.3287 | 0.7804 |
| track_a | wiki2 | pplx_decider | 800 | 337 | 0.4113 | 0.9763 |
| track_b | enterprise | retrieval5 | 443 | 330 | 0.5576 | 0.7485 |
| track_b | enterprise | bge_pointwise | 443 | 330 | 0.5372 | 0.7212 |
| track_b | enterprise | laya_pointwise | 443 | 330 | 0.2709 | 0.3636 |
| track_b | enterprise | clef_pointwise | 443 | 330 | 0.3070 | 0.4121 |
| track_b | enterprise | clef_joint | 443 | 330 | 0.4560 | 0.6121 |
| track_b | enterprise | qwen_direct | 443 | 330 | 0.5508 | 0.7394 |
| track_b | enterprise | qwen06 | 443 | 330 | 0.5260 | 0.7061 |
| track_b | enterprise | qwen4 | 443 | 330 | 0.4628 | 0.6212 |
| track_b | enterprise | gte | 443 | 330 | 0.5666 | 0.7606 |
| track_b | enterprise | minilm | 443 | 330 | 0.5440 | 0.7303 |
| track_b | enterprise | jina35 | 443 | 330 | 0.5937 | 0.7970 |
| track_b | enterprise | nimble | 443 | 330 | 0.6117 | 0.8212 |
| track_b | enterprise | intern_decision | 0 | 0 | NOT_MEASURED | NOT_MEASURED |
| track_b | enterprise | pplx_decider | 443 | 330 | 0.6862 | 0.9212 |

## Integrity and resource accounting

- Every promoted N3 trace must have the expected 4,267 rows, candidate identity alignment, input/output hashes, physical-forward evidence, and zero gold-in-input rows; the machine-readable manifest records these checks.
- GPU0 and GPU6 were used by the new runs. GPUs hosting resident services or unrelated workloads were not stopped, restarted, or reconfigured.
- No monetary GPU cost is inferred from utilization; cost remains `NOT_MEASURED`.
- Historical results remain exploratory follow-up analyses, not prospective confirmation.

## Remaining limitations / owner decision

- incomplete or weak physical evidence: internlm/Intern-Decision-4B
- internlm/Intern-Decision-4B: 500 native context-overflow rows under the frozen 8192-token contract
- CSSR is a complete-support selection proxy, not independently verified reader factual accuracy.
- Qwen service-side telemetry remains separate from direct checkpoint telemetry.
- GPU dollar cost was not measured; no monetary claim is made.

Owner review is required before any fresh holdout, E2-S/P, training, Jev smoke test, or paid inference. Jev remains at zero calls.
