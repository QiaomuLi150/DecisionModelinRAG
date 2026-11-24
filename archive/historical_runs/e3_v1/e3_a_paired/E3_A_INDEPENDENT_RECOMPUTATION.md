# E3-A Independent Re-computation

The second path recomputed conditional cells directly from canonical query rows and `(jina_complete, pplx_complete)` booleans, without reading the primary cell CSV.

```json
[
  {
    "cohort": "track_a",
    "dataset": "hotpot",
    "n": 598,
    "both": 564,
    "pplx_only": 6,
    "jina_only": 18,
    "neither": 10
  },
  {
    "cohort": "track_a",
    "dataset": "musique",
    "n": 555,
    "both": 355,
    "pplx_only": 54,
    "jina_only": 78,
    "neither": 68
  },
  {
    "cohort": "track_a",
    "dataset": "wiki2",
    "n": 337,
    "both": 303,
    "pplx_only": 26,
    "jina_only": 5,
    "neither": 3
  },
  {
    "cohort": "track_b",
    "dataset": "enterprise",
    "n": 330,
    "both": 257,
    "pplx_only": 47,
    "jina_only": 6,
    "neither": 20
  }
]
```
