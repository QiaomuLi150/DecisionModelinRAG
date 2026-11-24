# E5-0 Local Model and GPU Feasibility

The approved embedding service at <REDACTED_IP>:8001 was reachable for retrieval-only work and reported Qwen/Qwen3-VL-Embedding-2B with max model length 8192.

The SDD-authorized synthetic, label-free native smoke checks completed on isolated GPUs: Jina status=PASS, GPU6; PPLX status=PASS, GPU5. Each final smoke processed 20 synthetic candidates, returned 20 scores and 5 selected IDs, and recorded one physical forward. Jina used revision `e8a93f33f0b22108f8c2364f8484ce3422552fbc`; PPLX used revision `14c9d90137962d300f3de95633f7b6535e80c5ce`. The smoke data are interface feasibility only: no benchmark quality score, p95, QPS or deployment conclusion was inferred.

The first two raw rows contain a harness-status false negative; their nested telemetry recorded the physical forward correctly. The corrected subsequent rows are the final status for each policy. E5-A model feasibility remains an owner-review dependency.
