# DecisionRAG E9-BRIGHT final scientific report

## Final gate

`E9_BRIGHT_VALIDATED_WITH_LIMITATIONS`

## Scope and admission

The official `xlangai/BRIGHT` short-query and short-document configurations were admitted at pinned dataset revision `3066d29c9651a576c8aba4832d249807b181ecae`; the official repository commit is `d99e8391d967d4c2b3a74732530d2309e2fc92b6`. The run contains 1384 frozen Top-20 slates over 12 domains. The official BM25 analyzer and parameters (`k1=0.9`, `b=0.4`) were used. The original TREC E9 blocked tree was not modified.

Post-freeze feasibility found 607/1384 queries with at least one known-positive `gold_id` in Top-20 across 12/12 domains. This clears the preregistered low-information threshold, but complete-gold support is not implied.

## Primary endpoint and comparisons

The endpoint is binary known-positive operational nDCG@5: failed native attempts score zero in the operational denominator; valid-only values are reported separately in the scorecard. The primary effect is unweighted macro-domain mean with 5,000 bootstrap resamples, seed 731. Query-weighted sensitivity is also reported. The exact values are in `03_analysis/BRIGHT_PRIMARY_PAIRED_STATISTICS.json`.

`{
  "Liquid_minus_Qwen0.6B": {
    "domain_count": 12,
    "domain_effects": [
      {
        "a_mean": 0.2481729687532525,
        "b_mean": 0.16854087071806262,
        "domain": "biology"
      },
      {
        "a_mean": 0.25670725378052806,
        "b_mean": 0.25692397579096327,
        "domain": "earth_science"
      },
      {
        "a_mean": 0.1390271748315582,
        "b_mean": 0.14603838425869947,
        "domain": "economics"
      },
      {
        "a_mean": 0.15268395785287361,
        "b_mean": 0.17692581889968428,
        "domain": "psychology"
      },
      {
        "a_mean": 0.14708836887219942,
        "b_mean": 0.1566138726335308,
        "domain": "robotics"
      },
      {
        "a_mean": 0.14011696095944837,
        "b_mean": 0.1839064709192476,
        "domain": "stackoverflow"
      },
      {
        "a_mean": 0.18252299417936826,
        "b_mean": 0.12801796981683938,
        "domain": "sustainable_living"
      },
      {
        "a_mean": 0.09962616832829611,
        "b_mean": 0.14054467534337575,
        "domain": "pony"
      },
      {
        "a_mean": 0.21121616301008111,
        "b_mean": 0.28597938816259266,
        "domain": "leetcode"
      },
      {
        "a_mean": 0.04016706594474071,
        "b_mean": 0.08321191820650863,
        "domain": "aops"
      },
      {
        "a_mean": 0.08219292853505653,
        "b_mean": 0.14057159939811703,
        "domain": "theoremqa_questions"
      },
      {
        "a_mean": 0.08825748742943591,
        "b_mean": 0.11846367518133523,
        "domain": "theoremqa_theorems"
      }
    ],
    "macro_domain_primary": {
      "ci95": [
        -0.03884249994547151,
        0.00944420916659197
      ],
      "effect": -0.016496593904343163,
      "n": 12,
      "reps": 5000,
      "seed": 731
    },
    "query_weighted_sensitivity": {
      "ci95": [
        -0.032279883290989395,
        -0.00973714361950452
      ],
      "effect": -0.02080266176396467,
      "n": 1384,
      "reps": 5000,
      "seed": 731
    }
  },
  "PPLX_minus_Jina": {
    "domain_count": 12,
    "domain_effects": [
      {
        "a_mean": 0.20604808618770568,
        "b_mean": 0.19437600232381735,
        "domain": "biology"
      },
      {
        "a_mean": 0.09397818688466465,
        "b_mean": 0.308761577871133,
        "domain": "earth_science"
      },
      {
        "a_mean": 0.04523306719639678,
        "b_mean": 0.18818671328280173,
        "domain": "economics"
      },
      {
        "a_mean": 0.06671875998999739,
        "b_mean": 0.22354915889655289,
        "domain": "psychology"
      },
      {
        "a_mean": 0.08589433612615824,
        "b_mean": 0.19040521211232592,
        "domain": "robotics"
      },
      {
        "a_mean": 0.0,
        "b_mean": 0.18980956295786516,
        "domain": "stackoverflow"
      },
      {
        "a_mean": 0.184793344749221,
        "b_mean": 0.1944272315888792,
        "domain": "sustainable_living"
      },
      {
        "a_mean": 0.20145710529583694,
        "b_mean": 0.13060618036469157,
        "domain": "pony"
      },
      {
        "a_mean": 0.0,
        "b_mean": 0.28061706321923746,
        "domain": "leetcode"
      },
      {
        "a_mean": 0.025040813652078126,
        "b_mean": 0.07429177496092536,
        "domain": "aops"
      },
      {
        "a_mean": 0.05534483192242948,
        "b_mean": 0.1276896560937095,
        "domain": "theoremqa_questions"
      },
      {
        "a_mean": 0.05560913665172259,
        "b_mean": 0.09786188618360772,
        "domain": "theoremqa_theorems"
      }
    ],
    "macro_domain_primary": {
      "ci95": [
        -0.15381727491417288,
        -0.0442565978717294
      ],
      "effect": -0.09837202926661133,
      "n": 12,
      "reps": 5000,
      "seed": 731
    },
    "query_weighted_sensitivity": {
      "ci95": [
        -0.11866908935010775,
        -0.08793649606006218
      ],
      "effect": -0.10311112905783595,
      "n": 1384,
      "reps": 5000,
      "seed": 731
    }
  }
}`

## Per-domain scorecard

The complete machine-readable scorecard is `03_analysis/BRIGHT_SCORECARD_BY_DOMAIN.csv`.

```text
model,domain,n,valid_n,failure_n,operational_ndcg5,valid_ndcg5,hit5,p50_latency_ms,p95_latency_ms,p95_memory_mib
jina_v3_5,aops,111,111,0,0.07429177496092536,0.07429177496092536,0.1981981981981982,370.807544,890.3948375,3619.867919921875
jina_v3_5,biology,103,103,0,0.19437600232381735,0.19437600232381735,0.39805825242718446,89.571348,451.9320415999999,2384.5598144531236
jina_v3_5,earth_science,116,116,0,0.308761577871133,0.308761577871133,0.6120689655172413,486.898798,3202.8476945,9452.0478515625
jina_v3_5,economics,103,103,0,0.18818671328280173,0.18818671328280173,0.3300970873786408,449.886834,1053.7529284999998,4023.359033203125
jina_v3_5,leetcode,142,142,0,0.28061706321923746,0.28061706321923746,0.43661971830985913,703.260441,873.6795668999999,3655.653247070312
jina_v3_5,pony,112,112,0,0.13060618036469157,0.13060618036469157,0.49107142857142855,139.6620085,174.8974936,1667.988427734375
jina_v3_5,psychology,101,101,0,0.22354915889655289,0.22354915889655289,0.3465346534653465,623.857112,1523.026184,4238.533203125
jina_v3_5,robotics,101,101,0,0.19040521211232592,0.19040521211232592,0.33663366336633666,475.668833,1712.362947,3709.76904296875
jina_v3_5,stackoverflow,117,117,0,0.18980956295786516,0.18980956295786516,0.36752136752136755,1116.477484,1976.5882486,5941.79765625
jina_v3_5,sustainable_living,108,108,0,0.1944272315888792,0.1944272315888792,0.3888888888888889,207.4921315,530.4175015499991,2373.536035156249
jina_v3_5,theoremqa_questions,194,194,0,0.1276896560937095,0.1276896560937095,0.16494845360824742,296.00472449999995,755.0964947,2984.9521484375
jina_v3_5,theoremqa_theorems,76,76,0,0.09786188618360772,0.09786188618360772,0.14473684210526316,365.5171755,718.2229434999999,2936.5540771484375
liquid_d1_3b,aops,111,111,0,0.04016706594474071,0.04016706594474071,0.11711711711711711,587.342677,1116.2683685000002,8283.74560546875
liquid_d1_3b,biology,103,103,0,0.2481729687532525,0.2481729687532525,0.44660194174757284,206.150675,713.8270756999999,7554.9958984375
liquid_d1_3b,earth_science,116,116,0,0.25670725378052806,0.25670725378052806,0.5,755.361997,3870.3039315,12372.95849609375
liquid_d1_3b,economics,103,103,0,0.1390271748315582,0.1390271748315582,0.24271844660194175,739.512932,1345.5566890999994,8724.116113281249
liquid_d1_3b,leetcode,142,142,0,0.21121616301008111,0.21121616301008111,0.38028169014084506,1349.7036170000001,1845.2796195,8362.383642578125
liquid_d1_3b,pony,112,112,0,0.09962616832829611,0.09962616832829611,0.36607142857142855,390.997301,657.88729555,6801.469677734375
liquid_d1_3b,psychology,101,101,0,0.15268395785287361,0.15268395785287361,0.27722772277227725,784.874073,1421.943932,8899.78173828125
liquid_d1_3b,robotics,101,101,0,0.14708836887219942,0.14708836887219942,0.2871287128712871,507.447588,1242.838483,8620.14111328125
liquid_d1_3b,stackoverflow,117,117,0,0.14011696095944837,0.14011696095944837,0.3162393162393162,1316.780698,1964.5205835999993,9838.41669921875
liquid_d1_3b,sustainable_living,108,108,0,0.18252299417936826,0.18252299417936826,0.3333333333333333,407.04087,854.1125743499994,7604.099658203124
liquid_d1_3b,theoremqa_questions,194,194,0,0.08219292853505653,0.08219292853505653,0.13917525773195877,528.3024379999999,1049.7529673,7948.46396484375
liquid_d1_3b,theoremqa_theorems,76,76,0,0.08825748742943591,0.08825748742943591,0.11842105263157894,633.9930995,1017.918962,7887.00830078125
pplx_decider,aops,111,32,79,0.025040813652078126,0.086860322355646,0.06306306306306306,52.498796,2349.589484,51374.80124511719
pplx_decider,biology,103,91,12,0.20604808618770568,0.23321926238828225,0.39805825242718446,922.106793,1658.4492775,50933.401611328125
pplx_decider,earth_science,116,47,69,0.09397818688466465,0.231946163374917,0.20689655172413793,135.101544,2079.4126435000003,51281.25815429688
pplx_decider,economics,103,23,80,0.04523306719639678,0.20256547483603776,0.05825242718446602,59.098209,1937.180373599999,51308.03994140625
pplx_decider,leetcode,142,0,142,0.0,0.0,0.0,60.7776465,82.58596539999998,
pplx_decider,pony,112,112,0,0.20145710529583694,0.20145710529583694,0.5803571428571429,2237.837712,2861.4622987499997,50911.46276855469
pplx_decider,psychology,101,25,76,0.06671875998999739,0.2695437903595894,0.1188118811881188,64.180364,2303.08883,51359.11591796875
pplx_decider,robotics,101,53,48,0.08589433612615824,0.16368543299513175,0.1485148514851485,1050.502656,2335.42238,51350.091796875
pplx_decider,stackoverflow,117,5,112,0.0,0.0,0.0,74.775144,108.3932797999999,51229.20869140625
pplx_decider,sustainable_living,108,87,21,0.184793344749221,0.22939863486110193,0.3333333333333333,1332.8280855,2240.2362644,51285.57817382812
pplx_decider,theoremqa_questions,194,108,86,0.05534483192242948,0.0994157166014011,0.08247422680412371,1535.809702,2763.4082488,51356.807397460936
pplx_decider,theoremqa_theorems,76,28,48,0.05560913665172259,0.15093908519753274,0.06578947368421052,45.3051965,2750.28719875,51380.28076171875
qwen3_reranker_0_6b,aops,111,111,0,0.08321191820650863,0.08321191820650863,0.21621621621621623,293.211474,506.3863165,3508.972412109375
qwen3_reranker_0_6b,biology,103,103,0,0.16854087071806262,0.16854087071806262,0.36893203883495146,181.69046,956.6331721999999,6133.778759765623
qwen3_reranker_0_6b,earth_science,116,116,0,0.25692397579096327,0.25692397579096327,0.5172413793103449,889.9097535000001,7924.729609,27435.557495117188
qwen3_reranker_0_6b,economics,103,103,0,0.14603838425869947,0.14603838425869947,0.27184466019417475,356.985833,847.9356238999987,5561.133398437494
qwen3_reranker_0_6b,leetcode,142,142,0,0.28597938816259266,0.28597938816259266,0.44366197183098594,623.0412105,1025.9498033999998,3760.3679199218745
qwen3_reranker_0_6b,pony,112,112,0,0.14054467534337575,0.14054467534337575,0.4642857142857143,257.29007850000005,312.80622495,1832.41748046875
qwen3_reranker_0_6b,psychology,101,100,1,0.17692581889968428,0.17869507708868113,0.297029702970297,726.827383,2333.430194,35634.866796874994
qwen3_reranker_0_6b,robotics,101,101,0,0.1566138726335308,0.1566138726335308,0.2871287128712871,317.306766,2201.453925,38685.54736328125
qwen3_reranker_0_6b,stackoverflow,117,117,0,0.1839064709192476,0.1839064709192476,0.3418803418803419,474.110039,930.8356991999997,33993.7650390625
qwen3_reranker_0_6b,sustainable_living,108,107,1,0.12801796981683938,0.12921439944129584,0.28703703703703703,249.6165975,1232.2329861999958,32790.92739257812
qwen3_reranker_0_6b,theoremqa_questions,194,194,0,0.14057159939811703,0.14057159939811703,0.18556701030927836,234.8521915,417.0535855,17905.939038085937
qwen3_reranker_0_6b,theoremqa_theorems,76,76,0,0.11846367518133523,0.11846367518133523,0.14473684210526316,275.16873699999996,725.380067,19530.875366210938
retrieval_prefix5,aops,111,111,0,0.060026893039621325,0.060026893039621325,0.17117117117117117,,,
retrieval_prefix5,biology,103,103,0,0.16699066156515135,0.16699066156515135,0.36893203883495146,,,
retrieval_prefix5,earth_science,116,116,0,0.24287632547555063,0.24287632547555063,0.49137931034482757,,,
retrieval_prefix5,economics,103,103,0,0.13246593247452698,0.13246593247452698,0.24271844660194175,,,
retrieval_prefix5,leetcode,142,142,0,0.22054939117234537,0.22054939117234537,0.3380281690140845,,,
retrieval_prefix5,pony,112,112,0,0.083553939337468,0.083553939337468,0.3392857142857143,,,
retrieval_prefix5,psychology,101,101,0,0.11648103147370903,0.11648103147370903,0.24752475247524752,,,
retrieval_prefix5,robotics,101,101,0,0.12049912445261529,0.12049912445261529,0.27722772277227725,,,
retrieval_prefix5,stackoverflow,117,117,0,0.16920106329056148,0.16920106329056148,0.3333333333333333,,,
retrieval_prefix5,sustainable_living,108,108,0,0.11910786097624068,0.11910786097624068,0.24074074074074073,,,
retrieval_prefix5,theoremqa_questions,194,194,0,0.09473743782978895,0.09473743782978895,0.14948453608247422,,,
retrieval_prefix5,theoremqa_theorems,76,76,0,0.03545716685060505,0.03545716685060505,0.07894736842105263,,,
```

## Native reliability and failure casebook

There were 775 native failure rows across the four model files. The full casebook is `03_analysis/BRIGHT_NATIVE_FAILURE_CASEBOOK.jsonl`. PPLX had explicit native context rejection (`Question branch exceeds the 8192-token limit; no input was truncated`) on long inputs; Qwen0.6B had two CUDA allocator OOM failures. These are operational failures, not silently repaired inputs. Liquid and Jina completed their attempted rows. No gold labels or answers were included in model inputs.

## Resource measurements

The complete resource table is `04_resources/BRIGHT_SELECTOR_RESOURCE_SUMMARY.csv`; per-run GPU snapshots and raw-output hashes are in `02_native`. These are observed selector-call timings and memory, not matched isolated hardware claims, sustainable production QPS, or dollar savings.

```text
model,n_attempts,valid_n,failure_n,p50_selector_ms,p95_selector_ms,p95_peak_memory_mib,matched_hardware,note
pplx_decider,1384,611,773,82.879796,2419.5117656499997,51334.747314453125,False,per-model GPU allocation and model-loading order are recorded in native run manifests; no unmeasured QPS/cost inference
liquid_d1_3b,1384,1384,0,602.7278779999999,1734.2709429499998,9210.712890625,False,per-model GPU allocation and model-loading order are recorded in native run manifests; no unmeasured QPS/cost inference
jina_v3_5,1384,1384,0,382.141571,1422.2969771999979,4497.824902343747,False,per-model GPU allocation and model-loading order are recorded in native run manifests; no unmeasured QPS/cost inference
qwen3_reranker_0_6b,1384,1382,2,323.2234465,1524.2218250999988,33618.5220703125,False,per-model GPU allocation and model-loading order are recorded in native run manifests; no unmeasured QPS/cost inference
```

## Downstream reader and interpretation limits

`BRIGHT_READER_NOT_EVALUABLE` is the final reader status. No reader inference was run: the release `gold_answer` and reasoning fields do not provide an independently verified answer-scoring contract. Accordingly this milestone establishes ranking-quality and native-reliability evidence only, not factual correctness, complete-support CSSR, downstream utility, architectural causality, or production economics. Public benchmark pretraining exposure cannot be proved absent from metadata.

## Provenance and tests

Raw native outputs were frozen before gold-label joins. The provenance manifest covers the append-only E9-BRIGHT tree. The full project regression suite passed; focused checks cover Top-20 uniqueness, label-blind inputs, raw-output freeze ordering, reader non-admission and preservation of the TREC E9 tree. Jev and paid API calls are zero.
