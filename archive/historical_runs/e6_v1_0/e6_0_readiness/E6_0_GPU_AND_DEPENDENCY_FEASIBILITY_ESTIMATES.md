# E6-0 GPU and Dependency Feasibility Estimates

This milestone performed no GPU work and downloaded no model weights. Existing historical observations show PPLX Decider is a 27B model with materially higher memory/latency than Jina; future runs require a memory-reservation check before loading it. Liquid, Jina, BGE and Qwen snapshots exist locally; StartLux and Amos exact checkpoint revisions are not locally pinned. Estimates are therefore qualitative and are not execution-time measurements or cost claims.

Required before E6-1: authorized GPU window, per-device occupancy snapshot, model-specific peak-memory dry-run/telemetry, serial or matched-device policy, timeout/retry rules, and no unrelated workload eviction.
