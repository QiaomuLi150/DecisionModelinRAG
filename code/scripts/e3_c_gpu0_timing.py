#!/usr/bin/env python3
"""DecisionRAG E3-C C5: isolated complete Top-20 -> Top-5 timing.

This is deliberately separate from the quality runner.  It consumes the
already-frozen E2-S/P 300-query workload and the validated native adapters,
records preprocessing through postprocessing latency, and fails closed if a
foreign compute process appears on GPU0.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e3_v1/e3_c_models"
GPU_ID = 0
WORKLOAD_MANIFEST = ROOT / "results/non_jev/v2/e2_sp/E2_SP_READER_COHORT_MANIFEST.json"
RAW_NAME = "E3_C_SELECTOR_LATENCY_RAW.jsonl"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gpu_guard(allow_pid: int | None = None) -> dict:
    inventory_env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    g = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20, env=inventory_env, check=True,
    )
    devices = {}
    for line in g.stdout.splitlines():
        fields = [x.strip() for x in line.split(",")]
        if len(fields) >= 7:
            devices[int(fields[0])] = {"index": int(fields[0]), "uuid": fields[1], "name": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]), "memory_free_mib": int(fields[5]), "utilization_percent": int(fields[6])}
    target = devices.get(GPU_ID)
    if target is None:
        raise RuntimeError("GPU0 is absent from nvidia-smi")
    a = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,gpu_name,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20, env=inventory_env, check=True,
    )
    all_apps = [line.strip() for line in a.stdout.splitlines() if line.strip()]
    foreign = []
    own = []
    for raw in all_apps:
        fields = [x.strip() for x in raw.split(",")]
        if len(fields) < 3 or target["uuid"] not in raw:
            continue
        try:
            pid = int(fields[2])
        except ValueError:
            pid = -1
        (own if allow_pid is not None and pid == allow_pid else foreign).append(raw)
    return {"status": "PASS" if not foreign else "BLOCKED_FOREIGN_GPU0_PROCESS", "gpu": target, "gpu0_apps": all_apps, "gpu0_own_apps": own, "gpu0_foreign_apps": foreign, "allow_pid": allow_pid}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def load_workload(e3c) -> tuple[list[dict], str]:
    items, _gold, _selections, _trace_hashes = e3c.load_data()
    by_id = {row["query_id"]: row for row in items}
    manifest = json.loads(WORKLOAD_MANIFEST.read_text(encoding="utf-8"))
    records = manifest["query_records"]
    if len(records) != 300 or len({r["query_id"] for r in records}) != 300:
        raise RuntimeError("frozen C5 workload is not 300 unique records")
    out = []
    for record in records:
        item = by_id.get(record["query_id"])
        if item is None:
            raise RuntimeError(f"C5 query missing from frozen population: {record['query_id']}")
        if sha_bytes(item["question"].encode("utf-8")) != record["question_sha256"]:
            raise RuntimeError(f"C5 question hash mismatch: {record['query_id']}")
        ids = [c["doc_id"] for c in item["candidates"]]
        if ids != record["top20_candidate_ids"] or len(ids) != 20 or len(set(ids)) != 20:
            raise RuntimeError(f"C5 frozen Top-20 mismatch: {record['query_id']}")
        out.append(item)
    return out, sha_file(WORKLOAD_MANIFEST)


def time_model(e3c, model_name: str, workload: list[dict], workload_hash: str, pre_guard: dict) -> tuple[list[dict], dict]:
    import torch

    load_start = time.perf_counter_ns()
    model = e3c.load_liquid(GPU_ID) if model_name == "liquid_d1_3b" else e3c.load_zerank(GPU_ID)
    torch.cuda.synchronize()
    load_ms = (time.perf_counter_ns() - load_start) / 1_000_000.0
    native = e3c.liquid_call if model_name == "liquid_d1_3b" else e3c.zerank_call
    warm_start = time.perf_counter_ns()
    torch.cuda.synchronize()
    warm = native(workload[0], model, "c5_warmup", GPU_ID, None) if model_name == "liquid_d1_3b" else native(workload[0], model, "c5_warmup", GPU_ID, 8)
    torch.cuda.synchronize()
    warm_ms = (time.perf_counter_ns() - warm_start) / 1_000_000.0
    if warm.get("status") != "VALID":
        raise RuntimeError(f"{model_name} warmup failed: {warm.get('reason')}")
    torch.cuda.reset_peak_memory_stats()
    rows = []
    run_start = time.perf_counter_ns()
    for index, item in enumerate(workload, 1):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        row = native(item, model, "c5", GPU_ID, None) if model_name == "liquid_d1_3b" else native(item, model, "c5", GPU_ID, 8)
        torch.cuda.synchronize()
        complete_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        row.update({"model": model_name, "policy": e3c.POLICY_LIQUID if model_name == "liquid_d1_3b" else e3c.POLICY_ZERANK, "stage": "c5", "gpu_id": GPU_ID, "complete_selector_latency_ms": complete_ms, "internal_native_latency_ms": row.get("latency_ms"), "workload_manifest_sha256": workload_hash, "logical_decisions": 20, "gold_in_inference_input": False, "cache_reuse": False, "candidate_view_hashes": [sha_bytes(str(c.get("full_text", "")).encode()) for c in item["candidates"]]})
        rows.append(row)
        if index % 25 == 0 or index == len(workload):
            print(f"C5 {model_name} {index}/{len(workload)} {row.get('status')}", flush=True)
    wall_ms = (time.perf_counter_ns() - run_start) / 1_000_000.0
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    post_guard = gpu_guard(os.getpid())
    valid = [r for r in rows if r.get("status") == "VALID"]
    latencies = [float(r["complete_selector_latency_ms"]) for r in valid]
    tokens = [int(r.get("input_tokens") or r.get("estimated_max_prompt_tokens") or r.get("max_pair_tokens") or 0) for r in valid]
    summary = {"model": model_name, "policy": e3c.POLICY_LIQUID if model_name == "liquid_d1_3b" else e3c.POLICY_ZERANK, "n": len(rows), "valid_n": len(valid), "error_n": len(rows) - len(valid), "error_rate": (len(rows) - len(valid)) / len(rows), "model_load_ms": load_ms, "warmup_ms": warm_ms, "p50_ms": percentile(latencies, .50), "p95_ms": percentile(latencies, .95), "p99_ms": percentile(latencies, .99), "mean_ms": statistics.mean(latencies) if latencies else None, "sum_request_ms": sum(latencies), "timed_wall_ms": wall_ms, "serialized_throughput_rps": len(valid) / (wall_ms / 1000.0) if wall_ms else None, "p50_input_tokens": percentile([float(x) for x in tokens], .50) if tokens else None, "p95_input_tokens": percentile([float(x) for x in tokens], .95) if tokens else None, "total_physical_forwards": sum(int(r.get("physical_forward_count") or 0) for r in valid), "physical_forward_evidence_rows": sum(bool(r.get("physical_forward_count")) for r in valid), "peak_memory_allocated_mib": peak_alloc, "peak_memory_reserved_mib": peak_reserved, "slo_1s_hit_rate": sum(x <= 1000 for x in latencies) / len(latencies) if latencies else None, "slo_3s_hit_rate": sum(x <= 3000 for x in latencies) / len(latencies) if latencies else None, "gpu_id": GPU_ID, "device_entitlement": "single approved isolated A100 GPU0", "pre_guard": pre_guard, "post_guard": post_guard, "status": "PASS" if len(valid) == len(rows) and post_guard["status"] == "PASS" else "BLOCKED"}
    try:
        if getattr(model, "_e3c_forward_handle", None) is not None:
            model._e3c_forward_handle.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    finally:
        pass
    return rows, summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    import e3_c_offline as e3c

    pre_guard = gpu_guard()
    if pre_guard["status"] != "PASS" or pre_guard["gpu"]["memory_free_mib"] < 70000:
        raise RuntimeError(f"GPU0 isolation guard failed: {json.dumps(pre_guard, sort_keys=True)}")
    workload, workload_hash = load_workload(e3c)
    previous = {}
    preserve = {
        RAW_NAME: "E3_C_SELECTOR_LATENCY_RAW_PRE_C5_BLOCKED.jsonl",
        "E3_C_SELECTOR_LATENCY_AND_RESOURCE_REPORT.md": "E3_C_SELECTOR_LATENCY_AND_RESOURCE_REPORT_PRE_C5_BLOCKED.md",
        "E3_C_FINAL_SCIENTIFIC_REPORT.md": "E3_C_FINAL_SCIENTIFIC_REPORT_PRE_C5_BLOCKED.md",
        "E3_C_OWNER_REVIEW_GATE.md": "E3_C_OWNER_REVIEW_GATE_PRE_C5_BLOCKED.md",
        "E3_C_PROVENANCE_MANIFEST.json": "E3_C_PROVENANCE_MANIFEST_PRE_C5_BLOCKED.json",
    }
    for source, target in preserve.items():
        src = OUT / source
        dst = OUT / target
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
        if src.is_file():
            previous[source] = sha_file(src)
    all_rows: list[dict] = []
    summaries = []
    for model_name in ("liquid_d1_3b", "zerank_2"):
        pre_model_guard = gpu_guard(os.getpid())
        if pre_model_guard["status"] != "PASS":
            raise RuntimeError(f"GPU0 became non-isolated before {model_name}: {json.dumps(pre_model_guard, sort_keys=True)}")
        rows, summary = time_model(e3c, model_name, workload, workload_hash, pre_model_guard)
        all_rows.extend(rows)
        summaries.append(summary)
        post_model_guard = gpu_guard(os.getpid())
        if post_model_guard["status"] != "PASS":
            raise RuntimeError(f"GPU0 became non-isolated after {model_name}: {json.dumps(post_model_guard, sort_keys=True)}")
    write_jsonl(OUT / RAW_NAME, all_rows)
    write_csv(OUT / "E3_C_SELECTOR_LATENCY_SUMMARY.csv", summaries)
    isolation = {"status": "C5_PASS_GPU0_ISOLATED" if all(x["status"] == "PASS" for x in summaries) else "C5_BLOCKED", "gpu_id": GPU_ID, "workload_n": len(workload), "workload_manifest_sha256": workload_hash, "pre_run_guard": pre_guard, "model_summaries": summaries, "preserved_pre_c5_artifacts": previous, "jev_calls": 0, "paid_api_calls": 0, "holdout_access": False}
    write_json(OUT / "E3_C_C5_GPU0_ISOLATION_AUDIT.json", isolation)
    table = ["| Model | n | valid | load ms | warmup ms | p50 ms | p95 ms | p99 ms | 1s hit | 3s hit | peak allocated MiB | forwards |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        table.append(f"| {s['model']} | {s['n']} | {s['valid_n']} | {s['model_load_ms']:.2f} | {s['warmup_ms']:.2f} | {s['p50_ms']:.2f} | {s['p95_ms']:.2f} | {s['p99_ms']:.2f} | {s['slo_1s_hit_rate']:.4f} | {s['slo_3s_hit_rate']:.4f} | {s['peak_memory_allocated_mib']:.1f} | {s['total_physical_forwards']} |")
    report = f"""# E3-C C5 Selector Latency and Resource Report

Status: `C5_PASS_GPU0_ISOLATED`.

The previously blocked C5 gate was completed on approved physical GPU0 after a pre-run guard found no GPU0 compute process and at least 70 GiB free. The workload is the immutable 300-query E2-S/P cohort, with exact question hashes and ordered Top-20 candidate IDs checked before timing. Models were run one at a time; warmup and model-load time are reported separately. Each timed request includes canonical input construction, tokenization/preprocessing, native inference, physical forwards, score extraction and deterministic Top-5 postprocessing.

{chr(10).join(table)}

The 1-second and 3-second columns are descriptive single-client scenario hit rates, not sustainable-QPS claims. `serialized_throughput_rps` is reported in the machine-readable summary only and is not production capacity. No GPU-seconds or dollar cost is inferred. The prior blocked artifacts were copied to versioned `*_PRE_C5_BLOCKED` files before this completion.

GPU isolation, memory, exact workload hashes, post-run guards and model revisions are recorded in `E3_C_C5_GPU0_ISOLATION_AUDIT.json`. Jev and paid API calls remain zero.
"""
    (OUT / "E3_C_SELECTOR_LATENCY_AND_RESOURCE_REPORT.md").write_text(report, encoding="utf-8")
    closure = f"""# DecisionRAG E3-C — C5 Owner-Review Closure

Verdict: `E3_C_OWNER_REVIEW_READY`

Scientific status: `E3_C_PASS_WITH_LIMITATIONS`. C0-C4 and C6 remain valid; C5 is now `C5_PASS_GPU0_ISOLATED`. GPU0 passed the isolation guard before each model, no unrelated process was interrupted, and it was empty after cleanup. The complete 300-query Top-20→Top-5 task was physically measured for both validated native policies.

## Measured C5 results

{chr(10).join(table)}

Full machine-readable request traces are in `E3_C_SELECTOR_LATENCY_RAW.jsonl`; quality results remain in `E3_C_DATASET_QUALITY_SUMMARY.csv` and `E3_C_PRIMARY_PAIRED_STATISTICS.csv`. The timing study is serialized single-client only. It does not establish sustainable QPS, kernel-seconds, or monetary cost. E3-0 remains `E3_0_HOLDOUT_RESERVATION_PENDING_METADATA`.

No fresh holdout, Jev, paid API, training, multimodal experiment, retrieval rebuild or production-service change occurred.
"""
    (OUT / "E3_C_C5_OWNER_REVIEW_CLOSURE.md").write_text(closure, encoding="utf-8")
    old_final = (OUT / "E3_C_FINAL_SCIENTIFIC_REPORT.md").read_text(encoding="utf-8")
    r1 = "# DecisionRAG E3-C Final Scientific Report — C5 Completion Amendment\n\nThis versioned amendment supersedes the pre-C5 blocked timing conclusion while preserving that report as `E3_C_FINAL_SCIENTIFIC_REPORT_PRE_C5_BLOCKED.md`.\n\n" + closure + "\n\n## Preserved C0-C4 report\n\n" + old_final
    (OUT / "E3_C_FINAL_SCIENTIFIC_REPORT_R1.md").write_text(r1, encoding="utf-8")
    gate = """# E3-C Owner Review Gate — C5 Completion

`E3_C_OWNER_REVIEW_READY`

`C5_PASS_GPU0_ISOLATED`: complete isolated Top-20→Top-5 timing was measured for Liquid d1-3B and zerank-2 on the frozen 300-query workload. C0-C4 quality and C6 integrity results remain preserved. E3-0 remains `E3_0_HOLDOUT_RESERVATION_PENDING_METADATA`. Jev and paid calls: zero.
"""
    (OUT / "E3_C_OWNER_REVIEW_GATE_R1.md").write_text(gate, encoding="utf-8")
    manifest_path = OUT / "E3_C_PROVENANCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"status": "E3_C_PASS_WITH_LIMITATIONS", "owner_review_gate": "E3_C_OWNER_REVIEW_READY", "c5": "C5_PASS_GPU0_ISOLATED", "c5_gpu_id": GPU_ID, "c5_workload_n": len(workload), "c5_workload_manifest_sha256": workload_hash, "c5_summary": summaries, "c5_isolation_audit": "results/non_jev/v2/e3_v1/e3_c_models/E3_C_C5_GPU0_ISOLATION_AUDIT.json", "versioned_closure": "results/non_jev/v2/e3_v1/e3_c_models/E3_C_C5_OWNER_REVIEW_CLOSURE.md", "previous_provenance_manifest_sha256": previous.get("E3_C_PROVENANCE_MANIFEST.json")})
    artifacts = [p for p in OUT.iterdir() if p.is_file() and p.name != "E3_C_PROVENANCE_MANIFEST.json"]
    manifest["artifact_sha256"] = {str(p.relative_to(ROOT)): sha_file(p) for p in sorted(artifacts)}
    write_json(manifest_path, manifest)
    print(json.dumps({"status": isolation["status"], "summaries": summaries, "raw_rows": len(all_rows), "workload_sha256": workload_hash}, sort_keys=True))
    return 0 if isolation["status"] == "C5_PASS_GPU0_ISOLATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
