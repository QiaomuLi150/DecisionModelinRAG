# v1.3 R6 — LeetCode context-trim separation

Status: `PARTIAL_NATIVE_CONTROLS_COMPLETE_BUDGETED_JINA_BLOCKED`

The 142-question LeetCode cohort and its Kev/Drex longest-prefix sub-slates
were frozen before labels were used. Native Kev and Drex scoring, their
deterministic E11-A budget views, and matched Jina native controls were run
label-blind. The exact budgeted strings needed for Jina-on-budget controls were
not persisted in the original R6 manifest, so those controls are blocked rather
than reconstructed from token counts.

| Policy | Rows | Valid | Failed | Valid-row mean known-positive nDCG@5 |
|---|---:|---:|---:|---:|
| Kev native sub | 142 | 132 | 10 | 0.108184 |
| Kev budget sub | 142 | 139 | 3 | 0.084906 |
| Jina on Kev native sub | 142 | 139 | 3 | 0.265348 |
| Jina on Kev budget sub | 142 | 0 | 142 | NOT_MEASURED |
| Drex native sub | 142 | 126 | 16 | 0.159618 |
| Drex budget sub | 142 | 140 | 2 | 0.135589 |
| Jina on Drex native sub | 142 | 140 | 2 | 0.266711 |
| Jina on Drex budget sub | 142 | 0 | 142 | NOT_MEASURED |

On valid paired native subsets, Kev minus matched Jina is `-0.169962` over
132 queries (6 positive, 44 negative, 82 unchanged); Drex minus matched Jina is
`-0.116349` over 126 queries (8 positive, 40 negative, 78 unchanged). These are
diagnostic workload results, not broad replacement claims. Budgeted model
quality is not compared to Jina because the corresponding Jina inputs are
missing, and the invalid comparison is not filled with a proxy.

Raw hashes:

- Kev: `e92c2e5573250bf52e7943aeb0bd04a0026edf06db5027b9d61b6b80c5ab8469`
- Drex: `22588f8f49bbb6a57bdc8fd18f9ec7f637242e18b09041d1e735e4f50ccf7ff4`
- Jina native controls: `3e09ad4daf30187438c1548fc3b3f6b2bb082c774e63aff9797d3eb9f1ce9d83`

Machine-readable outputs:

- `export/leetcode_sub_perquery.csv`
- `export/leetcode_sub_summary.csv`
- `export/leetcode_sub_paired.json`

