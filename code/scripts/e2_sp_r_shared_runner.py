#!/usr/bin/env python3
"""Run the frozen E2-S/P-R native smoke/CAL workloads on shared GPU0.

This runner is intentionally separate from the isolated-GPU runner.  It does
not evict or restart the process already attached to GPU0.  Every output row
and summary is marked ``shared_gpu_contention_limited``; these traces can
support native adapter and CAL correctness checks, but they cannot establish
isolated open-loop capacity or a fair production latency comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_sp_r/shared_gpu0"
GPU = 0
MIN_FREE_MIB = 25_000

MODELS = ("bge_pointwise", "jina35", "nimble", "gte", "pplx_decider")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def shared_guard(base, model: str) -> dict:
    before = base.gpu_snapshot()
    row = None
    for line in before.get("stdout", "").splitlines():
        fields = [x.strip() for x in line.split(",")]
        if fields and fields[0] == str(GPU):
            row = {
                "index": GPU,
                "name": fields[1],
                "memory_total_mib": int(fields[2]),
                "memory_used_mib": int(fields[3]),
                "memory_free_mib": int(fields[4]),
                "utilization_gpu_percent": int(fields[5]),
            }
            break
    if row is None:
        raise RuntimeError("GPU0 not present in nvidia-smi output")
    apps = base.compute_apps()
    if row["memory_free_mib"] < MIN_FREE_MIB:
        raise RuntimeError(f"GPU0 free memory {row['memory_free_mib']} MiB below safety floor {MIN_FREE_MIB} MiB")
    return {
        "resource_mode": "shared_gpu0_contention_limited",
        "isolation": False,
        "integrity_gate": "SHARED_CONTENTION_LIMITED",
        "physical_gpu_id": GPU,
        "visible_cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model": model,
        "minimum_free_mib": MIN_FREE_MIB,
        "gpu_before": row,
        "compute_apps_before": apps,
        "no_process_eviction": True,
        "no_service_restart": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=("S0_SMOKE", "S1_CAL"), required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
        raise RuntimeError(f"shared GPU0 runner requires CUDA_VISIBLE_DEVICES=0, got {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    sys.path.insert(0, str(ROOT / "scripts"))
    import e2_sp_gpu5_runner as base

    # Reuse the already reviewed native adapters while redirecting all output
    # into the R package's shared-device namespace.
    base.GPU = GPU
    base.OUT = OUT
    original_stamp = base.stamp_rows
    original_summary = base.summary

    def stamp(rows, model_key, stage, guard):
        rows = original_stamp(rows, model_key, stage, guard)
        for row in rows:
            row["e2_sp_resource_mode"] = "shared_gpu0_contention_limited"
            row["e2_sp_physical_gpu_id"] = GPU
            row["e2_sp_integrity_gate"] = "SHARED_CONTENTION_LIMITED"
            row["e2_sp_isolated"] = False
            row["e2_sp_no_process_eviction"] = True
            row["e2_sp_no_service_restart"] = True
        return rows

    def summary(rows, model_key, stage, spec, guard_before, path, extra=None):
        result = original_summary(rows, model_key, stage, spec, guard_before, path, extra)
        result.update({
            "resource_mode": "shared_gpu0_contention_limited",
            "physical_gpu_id": GPU,
            "isolated": False,
            "integrity_gate": "SHARED_CONTENTION_LIMITED",
            "capacity_validity": "NOT_VALID_FOR_ISOLATED_CAPACITY",
        })
        return result

    base.stamp_rows = stamp
    base.summary = summary
    items = base.load_cohort_items()
    selected = base.smoke_items(items) if args.stage == "S0_SMOKE" else base.cal_items()
    guard = shared_guard(base, args.model)
    print(json.dumps({"model": args.model, "stage": args.stage, "population": len(selected), "guard": guard}, sort_keys=True), flush=True)
    result = base.run_model(args.model, selected, args.stage, guard)
    result["resource_mode"] = "shared_gpu0_contention_limited"
    result["physical_gpu_id"] = GPU
    result["isolated"] = False
    result["integrity_gate"] = "SHARED_CONTENTION_LIMITED"
    stem = f"E2_SP_R_{args.stage}_{args.model.upper()}_SHARED_GPU0"
    write_json(OUT / f"{stem}_SUMMARY.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result.get("status") == "PASS" or all(v.get("status") == "PASS" for v in result.values() if isinstance(v, dict) and "status" in v) else 2


if __name__ == "__main__":
    raise SystemExit(main())
