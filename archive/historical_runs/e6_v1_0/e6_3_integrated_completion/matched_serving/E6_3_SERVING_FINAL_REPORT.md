# E6-3 Matched Serving Final Report

Gate: E6_3_MATCHED_SERVING_COMPLETE_WITH_HARDWARE_LIMITS

All eight models were scheduled sequentially on GPU 1 where safe, with two timed repetitions for 100 SciFact and 100 GaRAGe queries. Warmups are excluded. See the scorecard for complete accounting.

Sustainable QPS and monetary cost were not measured/modelled. Observed forward counts are reported as NOT_MEASURED when adapter hooks did not expose them.

- pplx_decider / scifact: 200/200 valid; p50=322.6558822207153; p95=362.07984471693635; failure=0.0; 1s=1; 3s=1
- pplx_decider / garage: 200/200 valid; p50=655.858242418617; p95=720.8627523854371; failure=0.0; 1s=1; 3s=1
- liquid_d1_3b / scifact: 200/200 valid; p50=50.29613524675369; p95=55.68707254715264; failure=0.0; 1s=1; 3s=1
- liquid_d1_3b / garage: 200/200 valid; p50=139.93514562025666; p95=193.9601795282212; failure=0.0; 1s=1; 3s=1
- startlux_decision_4b / scifact: 200/200 valid; p50=113.18892985582352; p95=113.87569620274007; failure=0.0; 1s=1; 3s=1
- startlux_decision_4b / garage: 200/200 valid; p50=267.33667124062777; p95=381.39071241021156; failure=0.0; 1s=1; 3s=1
- amos_1_small_2b / scifact: 200/200 valid; p50=54.1147543117404; p95=57.004027627408504; failure=0.0; 1s=1; 3s=1
- amos_1_small_2b / garage: 200/200 valid; p50=129.759076051414; p95=188.12109297141427; failure=0.0; 1s=1; 3s=1
- jina_v3_5 / scifact: 200/200 valid; p50=39.6644938737154; p95=41.38536490499973; failure=0.0; 1s=1; 3s=1
- jina_v3_5 / garage: 200/200 valid; p50=58.02957294508815; p95=84.43844527937463; failure=0.0; 1s=1; 3s=1
- qwen3_reranker_4b / scifact: 200/200 valid; p50=177.01944289729; p95=193.77970406785607; failure=0.0; 1s=1; 3s=1
- qwen3_reranker_4b / garage: 200/200 valid; p50=237.73502837866545; p95=322.87260727025546; failure=0.0; 1s=1; 3s=1
- bge_reranker_v2_m3 / scifact: 200/200 valid; p50=78.15387984737754; p95=93.25797338970004; failure=0.0; 1s=1; 3s=1
- bge_reranker_v2_m3 / garage: 200/200 valid; p50=154.40051537007093; p95=232.31142130680374; failure=0.0; 1s=1; 3s=1
- qwen3_reranker_0_6b / scifact: 200/200 valid; p50=113.58962347730994; p95=120.57305020280182; failure=0.0; 1s=1; 3s=1
- qwen3_reranker_0_6b / garage: 200/200 valid; p50=88.2295579649508; p95=118.10890776105224; failure=0.0; 1s=1; 3s=1
