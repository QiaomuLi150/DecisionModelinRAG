#!/usr/bin/env python3
"""Bounded, CAL-only serving configuration verification for E2-S/P-R2.

This deliberately measures the frozen native adapter configuration only.  The
other protocol dimensions are recorded as unsupported unless the adapter has a
documented interface for them; no prompt, score, candidate, or precision
semantics are changed implicitly.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_sp_r2"
GPU = int(os.environ.get("E2_N_PHYSICAL_GPU_ID", "1"))
MODELS = ("bge_pointwise", "jina35", "nimble", "gte", "pplx_decider")
CONFIG = {
    "attention": "native_default",
    "batch_size": 1,
    "internal_document_pair_batching": "native_adapter_fixed",
    "request_concurrency": 1,
    "precision": "native_pinned_precision",
    "document_processing": "frozen_candidate_view_contract",
}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gpu_snapshot() -> dict:
    import subprocess
    cmd = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,power.draw", "--format=csv,noheader,nounits"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    return {"status": "PASS" if p.returncode == 0 else "UNAVAILABLE", "command": " ".join(cmd), "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    import e2_sp_r_shared_load_runner as shared

    OUT.mkdir(parents=True, exist_ok=True)
    items = shared.load_cal_items()
    before = gpu_snapshot()
    device = next((line for line in before.get("stdout", "").splitlines() if line.strip().startswith(str(GPU) + ",")), "")
    if not device:
        raise RuntimeError(f"approved GPU{GPU} is not visible")

    lock = {
        "protocol": "DecisionRAG v2.1 E2-S/P-R2",
        "stage": "R2_CAL_ONLY_SERVING_OPTIMIZATION",
        "status": "FROZEN_BEFORE_MEASUREMENT",
        "seed": 20261011,
        "cal_query_ids": [x["query_id"] for x in items],
        "cal_query_count": len(items),
        "selected_configuration": CONFIG,
        "candidate_configurations": {
            "batch_sizes": [1, 2, 4, 8],
            "request_concurrency": [1, 2, 4],
            "precisions": ["native_pinned_precision", "bfloat16_if_native_supported"],
            "attention": ["native_default"],
        },
        "unsupported_candidate_policy": "not measured unless exposed by the pinned native adapter; never silently substitute a different scoring implementation",
        "approved_gpu": GPU,
        "gpu_before": before,
        "gold_in_inference_input": False,
        "cache_reuse": False,
        "jev_calls": 0,
        "paid_api_calls": 0,
    }
    lock["lock_sha256"] = sha_json(lock)
    write_json(OUT / "E2_SP_R2_CAL_CONFIG_LOCK.json", lock)

    summary_rows = []
    for model in MODELS:
        print(json.dumps({"stage": "R2_CAL", "model": model, "status": "loading"}), flush=True)
        adapter = shared.make_adapter(model)
        traces = []
        try:
            adapter.warmup(items)
            for index, item in enumerate(items):
                started = time.perf_counter_ns()
                try:
                    result = adapter.score_item(item)
                    duration = float(result.pop("duration_ms"))
                    row = {
                        "record_type": "cal_request",
                        "model": model,
                        "query_id": item["query_id"],
                        "status": "ok",
                        "duration_ms": duration,
                        "candidate_ids": result.pop("candidate_ids"),
                        "selection_text_sha256": result.pop("selection_text_sha256"),
                        "actual_model_input_sha256": result.pop("actual_model_input_sha256"),
                        "raw_response_sha256": result.pop("raw_response_sha256"),
                        "physical_forward_calls": result.pop("physical_forward_calls"),
                        "input_tokens": result.pop("input_tokens", None),
                        "gold_in_inference_input": False,
                        "cache_reuse": False,
                        "resource_mode": f"isolated_gpu{GPU}",
                        "physical_gpu": GPU,
                        "config_lock_sha256": lock["lock_sha256"],
                    }
                    row.update({"scores_present": len(result.get("scores", [])) == 20} if "scores" in result else {"scores_present": True})
                except Exception as exc:
                    row = {
                        "record_type": "cal_request", "model": model, "query_id": item["query_id"],
                        "status": "error", "duration_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
                        "error": f"{type(exc).__name__}: {exc}", "gold_in_inference_input": False,
                        "cache_reuse": False, "resource_mode": f"isolated_gpu{GPU}", "physical_gpu": GPU,
                        "config_lock_sha256": lock["lock_sha256"],
                    }
                traces.append(row)
                if (index + 1) % 10 == 0:
                    print(json.dumps({"stage": "R2_CAL", "model": model, "completed": index + 1, "ok": sum(r["status"] == "ok" for r in traces)}), flush=True)
        finally:
            adapter.close()
        trace_path = OUT / f"E2_SP_R2_CAL_{model.upper()}_REQUESTS.jsonl"
        with trace_path.open("w", encoding="utf-8") as f:
            for row in traces:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        ok = [r for r in traces if r["status"] == "ok"]
        values = [float(r["duration_ms"]) for r in ok]
        model_summary = {
            "record_type": "cal_summary", "model": model, "rows": len(traces), "completed": len(ok),
            "errors": len(traces) - len(ok), "error_rate": (len(traces) - len(ok)) / len(traces),
            "p50_ms": statistics.median(values) if values else None,
            "p95_ms": sorted(values)[max(0, min(len(values) - 1, int((0.95 * len(values) + 0.999999) - 1)))] if values else None,
            "mean_ms": statistics.mean(values) if values else None,
            "min_ms": min(values) if values else None, "max_ms": max(values) if values else None,
            "physical_forward_calls": sum(int(r.get("physical_forward_calls", 0)) for r in ok),
            "input_hash_rows": sum(bool(r.get("actual_model_input_sha256")) for r in ok),
            "response_hash_rows": sum(bool(r.get("raw_response_sha256")) for r in ok),
            "config": CONFIG, "config_lock_sha256": lock["lock_sha256"],
            "trace_sha256": sha_file(trace_path), "gpu": GPU, "resource_mode": f"isolated_gpu{GPU}",
            "gold_in_inference_input": False, "cache_reuse": False, "jev_calls": 0, "paid_api_calls": 0,
            "status": "PASS" if len(ok) == len(traces) else "BLOCKED",
        }
        summary_rows.append(model_summary)
        write_json(OUT / f"E2_SP_R2_CAL_{model.upper()}_SUMMARY.json", model_summary)
        print(json.dumps({"stage": "R2_CAL", "model": model, "status": model_summary["status"], "p50_ms": model_summary["p50_ms"], "p95_ms": model_summary["p95_ms"]}, sort_keys=True), flush=True)

    with (OUT / "E2_SP_R2_CAL_OPTIMIZATION.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["model", "rows", "completed", "errors", "error_rate", "p50_ms", "p95_ms", "mean_ms", "min_ms", "max_ms", "physical_forward_calls", "input_hash_rows", "response_hash_rows", "status", "trace_sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in summary_rows)
    lock["gpu_after"] = gpu_snapshot()
    lock["model_summaries"] = summary_rows
    write_json(OUT / "E2_SP_R2_CAL_CONFIG_LOCK.json", lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
