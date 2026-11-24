#!/usr/bin/env python3
"""E5-0 label-free native interface smoke check.

The harness intentionally uses synthetic query/candidate content.  It is not
an evaluation runner and must never be pointed at an E5 or historical query.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e5_v1_1/e5_0_data_protocol/E5_0_PREFLIGHT_TELEMETRY.jsonl"
sys.path.insert(0, str(ROOT / "scripts"))


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["jina35", "pplx_decider"], required=True)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()
    started = time.time()
    row = {
        "stage": "E5_0_SYNTHETIC_INTERFACE_PREFLIGHT",
        "policy": args.policy,
        "gpu_requested": args.gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "synthetic_only": True,
        "quality_scored": False,
        "query_id": "e5_0_synthetic_smoke_q0",
        "question": "Synthetic interface smoke query: select the candidate containing the stated synthetic fact.",
        "candidate_ids": [f"e5_0_synthetic_candidate_{i:02d}" for i in range(20)],
        "status": "ERROR",
    }
    try:
        import torch
        from e3_d_external import ExternalNativeAdapter

        row["torch_version"] = torch.__version__
        row["gpu_name"] = torch.cuda.get_device_name(0)
        row["gpu_memory_before_bytes"] = int(torch.cuda.memory_allocated(0))
        question = row["question"]
        docs = [{
            "doc_id": cid,
            "title": f"Synthetic candidate {i:02d}",
            "full_text": f"This synthetic candidate contains only label-free smoke content {i:02d}; it is not a benchmark document.",
        } for i, cid in enumerate(row["candidate_ids"])]
        row["synthetic_input_sha256"] = sha({"question": question, "candidates": docs})
        adapter = ExternalNativeAdapter(args.policy)
        spec = adapter.spec
        row["model_id"] = spec.get("id")
        row["model_revision"] = spec.get("revision")
        row["native_kind"] = spec.get("kind")
        scores, telemetry = adapter.score(question, docs)
        ordered = sorted(zip(scores, row["candidate_ids"]), key=lambda item: (-float(item[0]), item[1]))
        physical = bool(telemetry.get("physical_inference_evidence")) or int(telemetry.get("physical_forward_calls") or 0) > 0 or int(telemetry.get("native_api_calls") or 0) > 0
        row.update({
            "status": "PASS" if len(scores) == 20 and len({cid for _, cid in ordered[:5]}) == 5 and physical else "BLOCKED",
            "score_count": len(scores),
            "score_sha256": sha([float(x) for x in scores]),
            "selected_ids": [cid for _, cid in ordered[:5]],
            "actual_model_input_sha256": telemetry.get("actual_model_input_sha256"),
            "duration_ms": telemetry.get("duration_ms"),
            "physical_forward_calls": telemetry.get("physical_forward_calls"),
            "native_api_calls": telemetry.get("native_api_calls"),
            "physical_inference_evidence": physical,
            "telemetry": telemetry,
        })
        adapter.close()
        row["gpu_memory_after_bytes"] = int(torch.cuda.memory_allocated(0))
        row["gpu_peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(0))
    except Exception as exc:
        row["error_type"] = type(exc).__name__
        row["error"] = str(exc)[:2000]
    row["elapsed_wall_s"] = time.time() - started
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({k: row[k] for k in ("policy", "status", "model_id", "model_revision", "score_count", "selected_ids", "duration_ms", "physical_forward_calls", "native_api_calls", "error_type") if k in row}, indent=2))
    return 0 if row["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
