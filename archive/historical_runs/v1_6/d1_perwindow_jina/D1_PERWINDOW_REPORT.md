# DecisionRAG v1.6 D1 — Exact PPLX-Window Jina Quality Run

Status: `D1_COMPLETE_OWNER_REVIEW_READY`

The D1 run evaluated Jina v3.5 on the exact recoverable per-window strings
that PPLX received in the frozen R3 records. The R3 traces retained hashes and
caps rather than raw strings. Reconstruction was authenticated against every
stored R3 stage hash: 9,688/9,688 request-order records matched exactly.
Stage 2 used the frozen R3 stage-1-survivor carry-forward rule.

## Raw execution

- Frozen population: 1,384 queries × 7 orders = 9,688 records.
- Jina valid records: 9,667; failures: 21.
- The 21 failures are exactly the three pre-existing PPLX source failures in
  each order. No additional Jina-native failure class occurred.
- Labels were absent from all native inputs and were joined only after raw
  output files were closed and hashed.
- Post-freeze determinism check: 20/20 records, zero selected-ID or score-trace
  mismatches.
- Survivor-set matches between Jina stage-1 survivors and PPLX stage-2 set:
  original 8/1,381; reversed 6/1,381; permutation seeds 1–5: 7, 4, 7, 5,
  10 respectively.

## Primary contrasts

The metric is known-positive nDCG@5, normalized by `min(gold_count, 5)`;
failures score zero. Confidence intervals are 40,000 whole-domain bootstrap
draws, seed 731. Sign-flip p-values enumerate all 2^12 domain-sign patterns.
Holm adjustment is over the two declared contrasts.

| contrast | query-weighted delta | 95% CI | 90% CI | exact sign-flip p | Holm p | positive domains |
|---|---:|---|---|---:|---:|---:|
| PPLX − Jina, original BM25 order | 0.025226 | [0.012333, 0.041851] | [0.014300, 0.039088] | 0.003906 | 0.007812 | 10/12 |
| PPLX − Jina, mean six shuffled orders | 0.021345 | [0.007313, 0.037994] | [0.009301, 0.035399] | 0.019531 | 0.019531 | 10/12 |

These are exploratory ranking-quality results on the frozen BRIGHT known-positive
endpoint, not factual-answer or production-latency evidence.

## Answer to the D1 question

Yes. On this frozen known-positive nDCG@5 endpoint, two-stage PPLX still
outperforms two-stage Jina when Jina receives the exact recoverable PPLX
per-window strings. The result remains positive for the original order and for
the six-order mean, with the intervals and multiplicity-adjusted p-values above.
This does not establish superiority on other endpoints or equal cost.

## Artifacts

- `PROTOCOL.json`
- `D1_INPUT_MANIFEST.jsonl` and `D1_INPUT_MANIFEST_LOCK.json`
- `D1_RAW_TRACE_SHA256_MANIFEST.json`
- `D1_DETERMINISM_RECHECK.json`
- `D1_OFFLINE_SUMMARY.json`
- `export/d1_perquery.csv`
- `export/d1_domain_summary.csv`
- `export/d1_contrasts.csv`
- `raw/` contains append-only native traces; the package contains hashes and
  exports, not unsanitized raw traces.
