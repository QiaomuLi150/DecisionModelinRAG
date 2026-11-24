# E2-M1 Authenticity Audit

| Check | Result |
|---|---|
| Pointwise rows | 4267 total; 4267 authentic `ok` |
| Native joint rows | 4267 total; 4267 authentic `ok` |
| Candidate count/alignment | Each successful row records 20 canonical IDs and score order |
| Gold in inference inputs | `0` by construction; gold is loaded only after inference for offline evaluation |
| Synthetic/fallback scores | `0` |
| Application cache reuse | `false`; only authenticated resume records can be reused |
| Checkpoint revision | `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` |
| GPU | `[0]` |
| Jev/paid calls | `0/0` |

The runner fails closed on missing candidate text, duplicate IDs, non-finite scores, invalid binary option fields, context overflow, or incomplete native output. Original historical artifacts are read-only and unchanged; all M1 outputs are under `results/non_jev/v2/e2_m1/`.
