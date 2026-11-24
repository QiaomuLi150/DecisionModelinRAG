#!/usr/bin/env python3
"""Add non-inference provenance fields to completed M5 traces.

This does not alter measurements or rerun a model.  It only records the frozen
topology hash and makes unavailable per-request GPU active time explicit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M5 = ROOT / "data/results/m5"

TOPOLOGY = {
    "study_id": "decisionrag-frozen-A4-B3-C4-D3-E3-F1-G2",
    "local_gpu": "CUDA_VISIBLE_DEVICES=0",
    "services": {
        "embedding": "http://service.example.invalid/v1/embeddings",
        "qdrant": "http://service.example.invalid",
        "qwen_reranker": "http://service.example.invalid/v1/rerank",
        "reader": "http://service.example.invalid/v1/chat/completions",
    },
    "model_revisions": {
        "bge": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "qwen_reranker": "4bd860ac4f15ad1897a214615cccc700f8f71818",
        "laya": "7b928d828b7b0e022f929d9bd2e44165aa270148",
        "clef": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
        "reader": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    },
    "reader_model": "Qwen/Qwen3-VL-8B-Instruct",
    "embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
    "reranker_model": "Qwen/Qwen3-VL-Reranker-2B",
}
TOPOLOGY_HASH = hashlib.sha256(json.dumps(TOPOLOGY, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    changed = 0
    for path in sorted(M5.glob("*_serving_traces.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            row.setdefault("model_topology_hash", TOPOLOGY_HASH)
            row.setdefault("gpu_compute_seconds", None)
            row.setdefault(
                "gpu_compute_measurement_method",
                "NOT_MEASURED: per-request CUDA active-time events were not instrumented in the completed serving run; see selector_wall_seconds_per_1k_proxy",
            )
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        changed += len(rows)
    manifest = {
        "topology": TOPOLOGY,
        "model_topology_hash": TOPOLOGY_HASH,
        "trace_rows_repaired": changed,
        "inference_reruns": 0,
        "jev_calls": 0,
        "hosted_cloudflare_calls": 0,
    }
    (M5 / "m5_topology_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
