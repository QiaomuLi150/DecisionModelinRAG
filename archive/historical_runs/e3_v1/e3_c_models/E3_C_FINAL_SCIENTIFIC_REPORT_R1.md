# DecisionRAG E3-C Final Scientific Report — C5 Completion Amendment

This versioned amendment supersedes the pre-C5 blocked timing conclusion while preserving that report as `E3_C_FINAL_SCIENTIFIC_REPORT_PRE_C5_BLOCKED.md`.

# DecisionRAG E3-C — C5 Owner-Review Closure

Verdict: `E3_C_OWNER_REVIEW_READY`

Scientific status: `E3_C_PASS_WITH_LIMITATIONS`. C0-C4 and C6 remain valid; C5 is now `C5_PASS_GPU0_ISOLATED`. GPU0 passed the isolation guard before each model, no unrelated process was interrupted, and it was empty after cleanup. The complete 300-query Top-20→Top-5 task was physically measured for both validated native policies.

## Measured C5 results

| Model | n | valid | load ms | warmup ms | p50 ms | p95 ms | p99 ms | 1s hit | 3s hit | peak allocated MiB | forwards |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| liquid_d1_3b | 300 | 300 | 12029.11 | 2182.99 | 172.16 | 649.71 | 700.25 | 1.0000 | 1.0000 | 7780.0 | 300 |
| zerank_2 | 300 | 300 | 3893.28 | 814.55 | 307.58 | 970.19 | 1066.20 | 0.9700 | 1.0000 | 10556.4 | 900 |

Full machine-readable request traces are in `E3_C_SELECTOR_LATENCY_RAW.jsonl`; quality results remain in `E3_C_DATASET_QUALITY_SUMMARY.csv` and `E3_C_PRIMARY_PAIRED_STATISTICS.csv`. The timing study is serialized single-client only. It does not establish sustainable QPS, kernel-seconds, or monetary cost. E3-0 remains `E3_0_HOLDOUT_RESERVATION_PENDING_METADATA`.

No fresh holdout, Jev, paid API, training, multimodal experiment, retrieval rebuild or production-service change occurred.


## Preserved C0-C4 report

# DecisionRAG E3-C Final Scientific Report

Verdict: `E3_C_PASS_WITH_LIMITATIONS`; owner gate: `E3_C_OWNER_REVIEW_READY`.

## Scope and frozen inputs

C0-C4 completed on the frozen 4,267-query historical population: Track A has 3,767 rows (HotpotQA 800, MuSiQue 2,167, 2Wiki 800) and Track B has 500 enterprise rows. Track B identity eligibility is 443 and its retrieval-conditional denominator is 330. Native inputs contain the query and the frozen 20 candidate text views only; gold/qrels/reference answers were loaded after inference. Historical retrieval-only, BGE, Jina and Perplexity baselines were reused from SHA-verified artifacts.

Liquid d1-3B uses the official `system_one` API with twenty native `noul` yes probabilities in one shared state. This is a valid exploratory selection interface, not proof of cross-head calibration or complementarity. zerank-2 uses the official Sentence-Transformers `CrossEncoder.predict()` contract and raw Yes logits; no sigmoid was applied. Official sources: [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B), [Liquid prompt.py](https://huggingface.co/LiquidAI/d1-3B/blob/main/prompt.py), and [zeroentropy/zerank-2-reranker](https://huggingface.co/zeroentropy/zerank-2-reranker).

## Native execution

| model | smoke | CAL | full valid | physical forwards | batch contract |
|---|---:|---:|---:|---:|---|
| liquid_d1_3b | 12/12 | 30/30 | 4267/4267 | 4267 | one native shared-state call/query |
| zerank_2 | 12/12 | 30/30 | 4267/4267 | 12801 | 20 pairs, batch size 8, three decoder forwards/query |

Every final accepted native row has twenty unique canonical candidates, finite scores, a stable Top-5, input and score/output hashes, `gold_in_inference_input=false`, and status `VALID`. The append-only raw trace retains prior Liquid adapter failures and earlier zerank attempts; they are not used as model outputs. The final accepted rows are deterministic across repeated attempts: no score, selected-ID, or input-hash mismatches were found.

Historical raw-trace attempt counts retained:
- `liquid_d1_3b/cal/INFERENCE_ERROR`: 30
- `liquid_d1_3b/cal/VALID`: 30
- `liquid_d1_3b/full/INFERENCE_ERROR`: 4267
- `liquid_d1_3b/full/VALID`: 4267
- `liquid_d1_3b/smoke/INFERENCE_ERROR`: 12
- `liquid_d1_3b/smoke/VALID`: 12
- `zerank_2/cal/VALID`: 90
- `zerank_2/full/VALID`: 12801
- `zerank_2/smoke/VALID`: 36

## Quality scorecard

CSSR-U is unconditional over identity-eligible queries; CSSR-C is conditional on every gold document being present in the frozen candidate set. Any-gold coverage is a retrieval diagnostic. These are evidence-selection metrics, not factual answer accuracy.

| policy | dataset | eligible n | conditional n | valid n | CSSR-U | CSSR-C | any-gold coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| retrieval5 | hotpot | 800 | 598 | 800 | 0.4975 | 0.6656 | 0.9888 |
| retrieval5 | musique | 2167 | 555 | 2167 | 0.0992 | 0.3874 | 0.9229 |
| retrieval5 | wiki2 | 800 | 337 | 800 | 0.2437 | 0.5786 | 0.9962 |
| retrieval5 | enterprise | 443 | 330 | 443 | 0.5576 | 0.7485 | 0.7833 |
| retrieval5 | __pooled__ | 3767 | 1490 | 3767 | 0.2145 | 0.5423 | 0.9525 |
| bge_pointwise | hotpot | 800 | 598 | 800 | 0.6875 | 0.9197 | 0.9888 |
| bge_pointwise | musique | 2167 | 555 | 2167 | 0.1680 | 0.6559 | 0.9229 |
| bge_pointwise | wiki2 | 800 | 337 | 800 | 0.3400 | 0.8071 | 0.9962 |
| bge_pointwise | enterprise | 443 | 330 | 443 | 0.5372 | 0.7212 | 0.7833 |
| bge_pointwise | __pooled__ | 3767 | 1490 | 3767 | 0.3148 | 0.7960 | 0.9525 |
| jina35 | hotpot | 800 | 598 | 800 | 0.7275 | 0.9732 | 0.9888 |
| jina35 | musique | 2167 | 555 | 2167 | 0.1998 | 0.7802 | 0.9229 |
| jina35 | wiki2 | 800 | 337 | 800 | 0.3850 | 0.9139 | 0.9962 |
| jina35 | enterprise | 443 | 330 | 443 | 0.5937 | 0.7970 | 0.7833 |
| jina35 | __pooled__ | 3767 | 1490 | 3767 | 0.3512 | 0.8879 | 0.9525 |
| pplx_decider | hotpot | 800 | 598 | 800 | 0.7125 | 0.9532 | 0.9888 |
| pplx_decider | musique | 2167 | 555 | 2167 | 0.1887 | 0.7369 | 0.9229 |
| pplx_decider | wiki2 | 800 | 337 | 800 | 0.4113 | 0.9763 | 0.9962 |
| pplx_decider | enterprise | 443 | 330 | 443 | 0.6862 | 0.9212 | 0.7833 |
| pplx_decider | __pooled__ | 3767 | 1490 | 3767 | 0.3472 | 0.8779 | 0.9525 |
| liquid_joint20_binary | hotpot | 800 | 598 | 800 | 0.6512 | 0.8712 | 0.9888 |
| liquid_joint20_binary | musique | 2167 | 555 | 2167 | 0.1541 | 0.6018 | 0.9229 |
| liquid_joint20_binary | wiki2 | 800 | 337 | 800 | 0.3925 | 0.9318 | 0.9962 |
| liquid_joint20_binary | enterprise | 443 | 330 | 443 | 0.5124 | 0.6879 | 0.7833 |
| liquid_joint20_binary | __pooled__ | 3767 | 1490 | 3767 | 0.3103 | 0.7846 | 0.9525 |
| zerank2_pointwise | hotpot | 800 | 598 | 800 | 0.6350 | 0.8495 | 0.9888 |
| zerank2_pointwise | musique | 2167 | 555 | 2167 | 0.1449 | 0.5658 | 0.9229 |
| zerank2_pointwise | wiki2 | 800 | 337 | 800 | 0.3350 | 0.7953 | 0.9962 |
| zerank2_pointwise | enterprise | 443 | 330 | 443 | 0.6433 | 0.8636 | 0.7833 |
| zerank2_pointwise | __pooled__ | 3767 | 1490 | 3767 | 0.2894 | 0.7315 | 0.9525 |
| retrieval5 | track_a_equal_dataset_macro | 3767 | 1490 | 3767 | 0.2802 | 0.5439 | 0.9693 |
| bge_pointwise | track_a_equal_dataset_macro | 3767 | 1490 | 3767 | 0.3985 | 0.7942 | 0.9693 |
| jina35 | track_a_equal_dataset_macro | 3767 | 1490 | 3767 | 0.4374 | 0.8891 | 0.9693 |
| pplx_decider | track_a_equal_dataset_macro | 3767 | 1490 | 3767 | 0.4375 | 0.8888 | 0.9693 |
| liquid_joint20_binary | track_a_equal_dataset_macro | 3767 | 1490 | 3767 | 0.3993 | 0.8016 | 0.9693 |
| zerank2_pointwise | track_a_equal_dataset_macro | 3767 | 1490 | 3767 | 0.3716 | 0.7368 | 0.9693 |

The `__pooled__` rows are Track-A-only pooled rows; enterprise remains separate. `track_a_equal_dataset_macro` is the prespecified unweighted macro across HotpotQA, MuSiQue and 2Wiki and is distinct from pooled micro aggregation.

## Paired statistics

Paired bootstrap used seed `731` and `5000` resamples. Track A and Track B are never pooled. The four model-versus-authenticated-baseline comparisons and the practical Liquid-versus-zerank comparison are shown below; intervals are exploratory and no causal architecture claim is made.

| comparison | cohort | population | n | difference | 95% CI | wins/losses/ties |
|---|---|---:|---:|---:|---:|---:|
| liquid_joint20_binary_minus_jina35 | track_a | cssr_c | 1490 | -0.1034 | [-0.1255, -0.0819] | 67/221/1202 |
| liquid_joint20_binary_minus_jina35 | track_a | cssr_u | 3767 | -0.0409 | [-0.0494, -0.0321] | 67/221/3479 |
| liquid_joint20_binary_minus_jina35 | track_b | cssr_c | 330 | -0.1091 | [-0.1636, -0.0545] | 27/63/240 |
| liquid_joint20_binary_minus_jina35 | track_b | cssr_u | 443 | -0.0813 | [-0.1219, -0.0384] | 27/63/353 |
| liquid_joint20_binary_minus_pplx_decider | track_a | cssr_c | 1490 | -0.0933 | [-0.1148, -0.0732] | 61/200/1229 |
| liquid_joint20_binary_minus_pplx_decider | track_a | cssr_u | 3767 | -0.0369 | [-0.0454, -0.0284] | 61/200/3506 |
| liquid_joint20_binary_minus_pplx_decider | track_b | cssr_c | 330 | -0.2333 | [-0.2818, -0.1848] | 5/82/243 |
| liquid_joint20_binary_minus_pplx_decider | track_b | cssr_u | 443 | -0.1738 | [-0.2122, -0.1377] | 5/82/356 |
| zerank2_pointwise_minus_jina35 | track_a | cssr_c | 1490 | -0.1564 | [-0.1799, -0.1329] | 54/287/1149 |
| zerank2_pointwise_minus_jina35 | track_a | cssr_u | 3767 | -0.0619 | [-0.0711, -0.0528] | 54/287/3426 |
| zerank2_pointwise_minus_jina35 | track_b | cssr_c | 330 | 0.0667 | [0.0152, 0.1182] | 50/28/252 |
| zerank2_pointwise_minus_jina35 | track_b | cssr_u | 443 | 0.0497 | [0.0113, 0.0880] | 50/28/365 |
| zerank2_pointwise_minus_pplx_decider | track_a | cssr_c | 1490 | -0.1463 | [-0.1691, -0.1242] | 55/273/1162 |
| zerank2_pointwise_minus_pplx_decider | track_a | cssr_u | 3767 | -0.0579 | [-0.0672, -0.0486] | 55/273/3439 |
| zerank2_pointwise_minus_pplx_decider | track_b | cssr_c | 330 | -0.0576 | [-0.0970, -0.0212] | 11/30/289 |
| zerank2_pointwise_minus_pplx_decider | track_b | cssr_u | 443 | -0.0429 | [-0.0722, -0.0158] | 11/30/402 |
| liquid_joint20_binary_minus_zerank2_pointwise | track_a | cssr_c | 1490 | 0.0530 | [0.0282, 0.0772] | 207/128/1155 |
| liquid_joint20_binary_minus_zerank2_pointwise | track_a | cssr_u | 3767 | 0.0210 | [0.0114, 0.0305] | 207/128/3432 |
| liquid_joint20_binary_minus_zerank2_pointwise | track_b | cssr_c | 330 | -0.1758 | [-0.2273, -0.1242] | 17/75/238 |
| liquid_joint20_binary_minus_zerank2_pointwise | track_b | cssr_u | 443 | -0.1309 | [-0.1716, -0.0903] | 17/75/351 |

## Timing and resource limitations

C5 is `TIMING_BLOCKED_SHARED_GPU`. GPU1 was shared by unrelated resident processes at preflight; GPUs0/2/4 were also occupied. The SDD requires exclusive or genuinely contention-free A100 timing for complete Top-20-to-Top-5 selection. Therefore isolated p50/p95/p99, SLO compliance, model-load cost and matched recurring serving cost are `NOT_MEASURED_ISOLATION_BLOCKED`. `E3_C_SELECTOR_LATENCY_RAW.jsonl` is intentionally empty, not a zero-latency result. No shared-GPU timing is promoted as a hardware-equal comparison.

## Integrity and owner gate

Model revisions are pinned in `E3_C_FROZEN_EXPLORATORY_PROTOCOL.json`; source, baseline and snapshot hashes are recorded there. No fresh confirmatory holdout, Jev, paid API, training, E3-D, E3-F, multimodal input or production-service modification occurred. E3-0 remains `E3_0_HOLDOUT_RESERVATION_PENDING_METADATA`. The direct Liquid-versus-zerank comparison is practical and confounded by model family, training objective and input formulation; it is not a causal architecture ablation.

`E3_C_OWNER_REVIEW_READY`
