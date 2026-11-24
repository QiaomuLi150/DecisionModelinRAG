# R3 Ordered PPLX Two-Stage Order-Robustness Report

Status: COMPLETE — six frozen non-original candidate orders were executed and analyzed offline after raw-output freeze.

- Population: 1,384 BRIGHT queries × 6 orders = 8,304 selector attempts.
- Native model revision: `14c9d90137962d300f3de95633f7b6535e80c5ce`.
- Candidate IDs were permuted only; the frozen Top-20 slate identity and source hash were preserved.
- Gold labels were not read by inference workers; all scoring below was joined offline after all shards were sealed.

## Order-level summary

| order | valid | failed | all-attempt nDCG@5 | valid nDCG@5 | set changed vs original | paired Δ query | paired Δ domain |
|---|---:|---:|---:|---:|---:|---:|---:|
| reversed | 1381 | 3 | 0.202500 | 0.202940 | 0.944 | -0.008272 | -0.009389 |
| perm_seed_1 | 1381 | 3 | 0.203606 | 0.204048 | 0.944 | -0.007166 | -0.008078 |
| perm_seed_2 | 1381 | 3 | 0.209310 | 0.209764 | 0.940 | -0.001463 | -0.001832 |
| perm_seed_3 | 1381 | 3 | 0.206848 | 0.207297 | 0.942 | -0.003924 | -0.004273 |
| perm_seed_4 | 1381 | 3 | 0.202771 | 0.203211 | 0.934 | -0.008001 | -0.008213 |
| perm_seed_5 | 1381 | 3 | 0.202338 | 0.202777 | 0.928 | -0.008434 | -0.008980 |

## Interpretation

The order rows measure sensitivity of the same two-stage policy to candidate ordering. They are not independent model comparisons. The original-order R3 result is retained as the authenticated historical anchor; all six ordered runs use identical query slates, model revision, budget rule and native three-call contract.

See `multipass_ordered_bootstrap.json` for paired query/domain uncertainty, `multipass_ordered_stability.csv` for cross-order dispersion, and `multipass_ordered_provenance.json` for raw hashes.
