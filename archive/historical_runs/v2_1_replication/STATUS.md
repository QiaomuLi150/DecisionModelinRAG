# DecisionRAG v2.1 controlled replication status

Overall status: `BLOCKED` (`R2_BOTH_BENCHMARKS_BLOCKED`).

Stage 0 completed CPU-only inventory and froze a pre-outcome protocol receipt. Both benchmark branches lack the exact existing BM25-only Top-20 artifact required by the contract. No inference or GPU workload was started. See `stage0_preflight/BLOCKER_ENTERPRISE.md`, `stage0_preflight/BLOCKER_MUSIQUE.md`, `stage0_preflight/PROTOCOL.json`, and `stage0_preflight/DEVIATIONS.md`.

Protocol SHA-256: `b9c417a5932474941d0e1cba42687e8cae578ec2b1e8b49bb9393425b07813ab`

Resume after an approved input-contract resolution: `python3 scripts/v2_1_replication_stage0.py`
