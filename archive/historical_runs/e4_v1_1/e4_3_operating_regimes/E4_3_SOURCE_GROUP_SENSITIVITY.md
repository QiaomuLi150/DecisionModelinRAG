# E4-3 Source-Group Sensitivity

The exploratory cohort is not source-group balanced. It contains **25** groups; the largest has **340** of **380** selected records. Representative-query CSSR-U is the primary exploratory estimand. Query bootstrap treats questions as independent; cluster bootstrap resamples whole source groups and is reported as a sensitivity interval. Leave-largest-group-out is descriptive because the dominant group is unusually large. No representative/diagnostic pooling is used.

| Policy | K | n | Groups | Largest group | Query CSSR-U [95%] | Cluster bootstrap [95%] | Leave-largest-out CSSR-U |
|---|---:|---:|---:|---|---|---|---|
| clef_pointwise_binary | K10 | 300 | 25 | multihop:component:cc11478c0a2ddecde4f0426a (261) | 0.36333333333333334 [0.31, 0.42] | [0.3355018587360595, 0.71875] | 0.6153846153846154 (n=39) |
| clef_pointwise_binary | K20 | 300 | 25 | multihop:component:cc11478c0a2ddecde4f0426a (261) | 0.30666666666666664 [0.25666666666666665, 0.36] | [0.27598566308243727, 0.6857142857142857] | 0.5641025641025641 (n=39) |
| clef_joint_binary | K10 | 300 | 25 | multihop:component:cc11478c0a2ddecde4f0426a (261) | 0.33 [0.28, 0.38333333333333336] | [0.3044982698961938, 0.6923076923076923] | 0.5384615384615384 (n=39) |
| clef_joint_binary | K20 | 300 | 25 | multihop:component:cc11478c0a2ddecde4f0426a (261) | 0.2633333333333333 [0.21333333333333335, 0.31333333333333335] | [0.23985239852398524, 0.5833333333333334] | 0.4358974358974359 (n=39) |
| liquid_native_joint | K10 | 300 | 25 | multihop:component:cc11478c0a2ddecde4f0426a (261) | 0.37 [0.31666666666666665, 0.42333333333333334] | [0.34290843806104127, 0.7352941176470589] | 0.6153846153846154 (n=39) |
| liquid_native_joint | K20 | 300 | 25 | multihop:component:cc11478c0a2ddecde4f0426a (261) | 0.36 [0.30666666666666664, 0.41333333333333333] | [0.3309608540925267, 0.78125] | 0.6153846153846154 (n=39) |
| jina35_native | K10 | 300 | 25 | multihop:component:cc11478c0a2ddecde4f0426a (261) | 0.4666666666666667 [0.41333333333333333, 0.5233333333333333] | [0.43148148148148147, 0.8857142857142857] | 0.7948717948717948 (n=39) |
| jina35_native | K20 | 300 | 25 | multihop:component:cc11478c0a2ddecde4f0426a (261) | 0.5366666666666666 [0.48, 0.59] | [0.5055658627087198, 0.918918918918919] | 0.8205128205128205 (n=39) |
| pplx_decider_native | K10 | 300 | 25 | multihop:component:cc11478c0a2ddecde4f0426a (261) | 0.51 [0.4533333333333333, 0.5666666666666667] | [0.48201438848920863, 0.8620689655172413] | 0.7692307692307693 (n=39) |
| pplx_decider_native | K20 | 300 | 25 | multihop:component:cc11478c0a2ddecde4f0426a (261) | 0.5366666666666666 [0.48, 0.5933333333333334] | [0.5030750307503075, 0.9285714285714286] | 0.8461538461538461 (n=39) |
| bge_pointwise | K10 | 300 | 25 | multihop:component:cc11478c0a2ddecde4f0426a (261) | 0.43333333333333335 [0.38, 0.4866666666666667] | [0.40269277845777235, 0.8484848484848485] | 0.717948717948718 (n=39) |
| bge_pointwise | K20 | 300 | 25 | multihop:component:cc11478c0a2ddecde4f0426a (261) | 0.3933333333333333 [0.33666666666666667, 0.45] | [0.3585434173669468, 0.8529411764705882] | 0.717948717948718 (n=39) |
