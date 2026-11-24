# Stage 3 Serving Analysis

Raw rows: 3600; failures: 0; raw SHA-256: `1c7cc9a30fc3a78f782837d27b8daddd0045f2865762de1c36508518d767e78c`.

All 3,600 rows were measured in one serial session on the dedicated GPU. Model loading was excluded from request latency. No labels or holdout outcomes were accessed.

| Dataset | Policy | p50 ms | p95 ms | p95 ratio vs Jina full | 95% ratio interval | P(ratio <= 2) | alloc p95 MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| enterprise | jina_full | 268.720 | 350.151 | 1.0000 | [1.0000, 1.0000] | 1.0000 | 2232.105 |
| enterprise | jina_budget | 167.908 | 182.432 | 0.5210 | [0.4922, 0.5722] | 1.0000 | 1762.536 |
| enterprise | jina_two_stage_matched | 339.870 | 417.420 | 1.1921 | [1.1403, 1.2700] | 1.0000 | 1662.342 |
| enterprise | pplx_budget | 2137.164 | 2267.096 | 6.4746 | [6.1314, 7.1470] | 0.0000 | 51287.165 |
| enterprise | pplx_two_stage | 4209.500 | 4842.669 | 13.8302 | [13.3276, 14.9801] | 0.0000 | 50984.873 |
| musique | jina_full | 72.417 | 113.409 | 1.0000 | [1.0000, 1.0000] | 1.0000 | 1514.465 |
| musique | jina_budget | 72.413 | 113.734 | 1.0029 | [0.9499, 1.0096] | 1.0000 | 1512.724 |
| musique | jina_two_stage_matched | 132.577 | 168.959 | 1.4898 | [1.4025, 1.5508] | 1.0000 | 1342.960 |
| musique | pplx_budget | 924.649 | 1283.115 | 11.3140 | [10.6757, 11.7332] | 0.0000 | 50670.493 |
| musique | pplx_two_stage | 1509.348 | 1959.232 | 17.2757 | [16.7179, 18.2651] | 0.0000 | 50280.352 |

Failures were scored on the all-request basis; no failure occurred, so no BM25 fallback packaging time was added. These are serial matched-session measurements, not sustainable QPS or monetary cost estimates.
