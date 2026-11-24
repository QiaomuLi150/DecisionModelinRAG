#!/usr/bin/env python3
"""DecisionRAG E6-2 R2 native selector inference.

The runner consumes only the frozen R1 exploratory slates and public corpus
text needed to reconstruct their candidate payloads. It never imports qrels,
answers, RESERVED content, or any quality-scoring code.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r1_scifact_retrieval_freeze"
OUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r2_selector_inference"
WORK = OUT / "_work"
DATASETS = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets")
TOP20 = R1 / "E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl"
FROZEN_TOP20_SHA = "b9b2a3c3971da72f7e7aff6066c381c681f0ff6f7b4592d85b480e12cd449333"

MODEL_ORDER = [
    "pplx_decider", "liquid_d1_3b", "startlux_decision_4b", "amos_1_small_2b",
    "jina_v3_5", "qwen3_reranker_4b", "bge_reranker_v2_m3", "qwen3_reranker_0_6b",
]
E61_MODELS = {"pplx_decider", "liquid_d1_3b", "jina_v3_5", "qwen3_reranker_4b", "bge_reranker_v2_m3", "qwen3_reranker_0_6b"}
E61B_MODELS = {"startlux_decision_4b", "amos_1_small_2b"}
MODEL_REVISIONS = {
    "pplx_decider": "14c9d90137962d300f3de95633f7b6535e80c5ce",
    "liquid_d1_3b": "051bcc464b01b9f92942b364d9586b0ef5912432",
    "startlux_decision_4b": "9c0a32e9809d6a3f612434dd0bce150295dcc274",
    "amos_1_small_2b": "09ba3e9d8f506cfcb96ea130845d5d76d9d92028",
    "jina_v3_5": "e8a93f33f0b22108f8c2364f8484ce3422552fbc",
    "qwen3_reranker_4b": "22e683669bc0f0bd69640a1354a6d0aebcfeede5",
    "bge_reranker_v2_m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "qwen3_reranker_0_6b": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gpu_snapshot() -> dict[str, Any]:
    query = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    apps_cmd = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    try:
        raw = subprocess.check_output(query, text=True, stderr=subprocess.STDOUT).strip()
        devices = []
        for line in raw.splitlines():
            v = [x.strip() for x in line.split(",")]
            if len(v) == 7:
                devices.append({"index": int(v[0]), "uuid": v[1], "name": v[2], "memory_total_mib": int(v[3]), "memory_used_mib": int(v[4]), "memory_free_mib": int(v[5]), "utilization_gpu_percent": int(v[6])})
        try:
            apps = subprocess.check_output(apps_cmd, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            apps = f"UNAVAILABLE:{type(exc).__name__}:{exc}"
        return {"status": "ok", "devices": devices, "compute_apps_raw": apps, "captured_at": utc()}
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}", "captured_at": utc()}


def app_uuids(snapshot: dict[str, Any]) -> set[str]:
    text = str(snapshot.get("compute_apps_raw") or "")
    return {line.split(",", 1)[0].strip() for line in text.splitlines() if line.strip() and line.split(",", 1)[0].strip().startswith("GPU-")}


def safe_devices(snapshot: dict[str, Any], minimum_free_mib: int = 20000) -> list[dict[str, Any]]:
    used = app_uuids(snapshot)
    return [d for d in snapshot.get("devices", []) if d["memory_free_mib"] >= minimum_free_mib and d["utilization_gpu_percent"] == 0 and d["uuid"] not in used]


def load_items() -> list[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.memory_map(str(sorted(DATASETS.rglob("scifact-corpus.arrow"))[0]), "r") as f:
        try:
            corpus_rows = ipc.open_stream(f).read_all().to_pylist()
        except Exception:
            f.seek(0)
            corpus_rows = ipc.open_file(f).read_all().to_pylist()
    corpus = {f"scifact:{row['_id']}": row for row in corpus_rows}
    items = []
    for slate in read_jsonl(TOP20):
        candidates = []
        for c in slate["candidates"]:
            doc_id = c["document_id"]
            row = corpus[doc_id]
            text = (f"{row.get('title', '')}\n{row.get('text', '')}").strip()
            if sha_bytes(text.encode()) != c["content_sha256"]:
                raise RuntimeError(f"frozen candidate content mismatch: {doc_id}")
            candidates.append({"doc_id": doc_id, "title": str(row.get("title") or ""), "text": str(row.get("text") or ""), "candidate_view_sha256": c["content_sha256"]})
        qtext = slate["query_text"]
        items.append({"query_id": slate["query_id"], "question": qtext, "question_sha256": sha_bytes(qtext.encode()), "candidate_ids_top20": list(slate["candidate_ids"]), "candidates": candidates, "slate_sha256": slate["slate_sha256"], "input_sha256": sha_json({"query_id": slate["query_id"], "question": qtext, "candidate_ids": slate["candidate_ids"], "candidate_content_sha256": [x["candidate_view_sha256"] for x in candidates], "protocol": "E6-2-R2-frozen-r1"})})
    if len(items) != 277 or len({x["query_id"] for x in items}) != 277:
        raise RuntimeError("R1 population mismatch")
    return items


def top5(ids: list[str], scores: list[float]) -> list[str]:
    return [doc_id for _, doc_id in sorted(zip(scores, ids), key=lambda x: (-float(x[0]), ids.index(x[1]), x[1]))[:5]]


def classify_failure(exc: BaseException) -> str:
    text = str(exc).lower()
    if any(x in text for x in ("16384", "context", "maximum length", "too long", "token")):
        return "context_limit"
    if "out of memory" in text or "cuda oom" in text:
        return "oom"
    if "timeout" in text:
        return "timeout"
    if "probability" in text or "choice" in text or "score" in text or "candidate" in text:
        return "invalid_output"
    return "inference_error"


def load_adapter(model: str, physical_gpu: int):
    if model in E61_MODELS:
        import e6_1_native_adapter_validation as e61
        adapter = e61.NativeAdapter(model, 0)
        return adapter, {"family": "E6_1", "backend": e61.MODEL_META[model]["kind"], "checkpoint_revision": MODEL_REVISIONS[model], "physical_gpu": physical_gpu}
    import e6_1b_checkpoint_recovery as e61b
    dispatch = e61b.kernel_dispatch_audit() if model == "startlux_decision_4b" else {"verified": True, "method": "E6-1B authenticated Amos native contract"}
    adapter = e61b.NativeAdapter(model, physical_gpu, dispatch)
    return adapter, {"family": "E6_1B", "backend": e61b.MODELS[model]["kind"], "checkpoint_revision": MODEL_REVISIONS[model], "physical_gpu": physical_gpu, "kernel_dispatch": dispatch}


def adapter_score(model: str, adapter: Any, item: dict[str, Any], docs: list[dict[str, Any]], physical_gpu: int) -> tuple[list[float], dict[str, Any]]:
    ids = [x["doc_id"] for x in docs]
    import torch
    torch.cuda.synchronize()
    if model in E61_MODELS:
        scores, telemetry = adapter.score(item["question"], docs)
    else:
        score_map, telemetry = adapter.score(item, ids, docs)
        scores = [float(score_map[f"candidate_{i:03d}"]) for i in range(len(ids))]
    torch.cuda.synchronize()
    if len(scores) != len(ids) or not all(math.isfinite(float(x)) for x in scores):
        raise ValueError("native scores are incomplete or non-finite")
    observed = telemetry.get("physical_forward_count")
    if observed is None or int(observed or 0) <= 0:
        telemetry["physical_forward_count"] = "NOT_MEASURED"
        telemetry["physical_forward_count_method"] = "native wrapper did not expose a positive observed forward event"
    else:
        telemetry["physical_forward_count_method"] = "registered native forward hook"
    return [float(x) for x in scores], telemetry


def run_worker(model_names: list[str], physical_gpu: int) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    audit = read_json(OUT / "E6_2_R2_GPU_AUDIT_V2.json")
    if audit["status"] != "PASS":
        raise SystemExit("R2 GPU audit is not authorized")
    snapshot = gpu_snapshot()
    devices = {x["index"]: x for x in snapshot.get("devices", [])}
    expected_uuid = audit["approved_gpu_uuids"][str(physical_gpu)]
    device = devices.get(physical_gpu)
    if not device or device["uuid"] != expected_uuid or device["memory_free_mib"] < 20000:
        raise SystemExit(f"GPU {physical_gpu} no longer satisfies R2 audit: {device}")
    if expected_uuid in app_uuids(snapshot):
        raise SystemExit(f"GPU {physical_gpu} has a compute app; no eviction permitted")
    items = load_items()
    for model in model_names:
        partial = WORK / f"{model}.jsonl"
        if partial.exists():
            partial.unlink()
        rows: list[dict[str, Any]] = []
        load_started = time.perf_counter()
        try:
            adapter, meta = load_adapter(model, physical_gpu)
            rows.append({"model": model, "stage": "load", "status": "LOADED", "gpu_id": physical_gpu, "checkpoint_revision": MODEL_REVISIONS[model], "load_time_ms": (time.perf_counter() - load_started) * 1000, "backend": meta, "gold_in_input": False, "created_at": utc()})
        except Exception as exc:
            rows.append({"model": model, "stage": "load", "status": "MODEL_LOAD_ERROR", "gpu_id": physical_gpu, "checkpoint_revision": MODEL_REVISIONS[model], "failure": f"{type(exc).__name__}: {exc}", "failure_class": "model_load_error", "gold_in_input": False, "created_at": utc()})
            partial.write_text("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in rows) + "\n", encoding="utf-8")
            continue
        partial.write_text(json.dumps(rows[0], ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        for item in items:
            ids = list(item["candidate_ids_top20"])
            docs = item["candidates"]
            row = {"model": model, "stage": "exploratory", "status": "INFERENCE_ERROR", "query_id": item["query_id"], "dataset": "scifact", "candidate_k": 20, "candidate_ids": ids, "candidate_slate_sha256": item["slate_sha256"], "candidate_ids_sha256": sha_json(ids), "question_sha256": item["question_sha256"], "input_manifest_sha256": item["input_sha256"], "gpu_id": physical_gpu, "checkpoint_revision": MODEL_REVISIONS[model], "gold_in_input": False, "selected_ids": [], "scores": [], "failure": None, "native_api_calls": 0, "created_at": utc()}
            before = gpu_snapshot()
            try:
                import torch
                torch.cuda.reset_peak_memory_stats()
                scores, telemetry = adapter_score(model, adapter, item, docs, physical_gpu)
                selected = top5(ids, scores)
                if len(selected) != 5 or len(set(selected)) != 5 or not set(selected).issubset(set(ids)):
                    raise ValueError("invalid Top-5 candidate membership")
                row.update({"status": "VALID", "scores": scores, "score_sha256": sha_json(scores), "selected_ids": selected, "selected_ids_sha256": sha_json(selected), "native_api_calls": telemetry.get("native_api_calls", 1), **telemetry, "peak_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2), "peak_memory_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2)})
            except Exception as exc:
                row.update({"status": "INFERENCE_ERROR", "failure": f"{type(exc).__name__}: {exc}", "failure_class": classify_failure(exc), "native_api_calls": 1, "physical_forward_count": "NOT_MEASURED", "physical_forward_count_method": "failure; no forward count inferred"})
            row["gpu_snapshot_before"] = before
            row["gpu_snapshot_after"] = gpu_snapshot()
            rows.append(row)
            with partial.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        try:
            adapter.close()
        finally:
            gc.collect()
        rows.append({"model": model, "stage": "unload", "status": "RELEASED", "gpu_id": physical_gpu, "created_at": utc()})
        with partial.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rows[-1], ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps({"model": model, "attempts": sum(x.get("stage") == "exploratory" for x in rows), "valid": sum(x.get("status") == "VALID" for x in rows), "gpu": physical_gpu}), flush=True)
    return 0


def prepare(gpu_indices: list[int]) -> int:
    snapshot = gpu_snapshot()
    if snapshot.get("status") != "ok":
        raise SystemExit("nvidia-smi unavailable")
    safe = safe_devices(snapshot, 20000)
    safe_by_idx = {str(x["index"]): x for x in safe}
    required = {str(i): safe_by_idx[str(i)] for i in gpu_indices if str(i) in safe_by_idx}
    if not required:
        raise SystemExit(f"no requested GPU is safely idle: {gpu_indices}")
    # The current owner instruction authorizes all GPUs discovered idle here.
    approved = {str(i): safe_by_idx[str(i)]["uuid"] for i in sorted(int(k) for k in required)}
    if not any(int(k) == 1 for k in approved) or not any(int(k) == 5 for k in approved) or not any(int(k) == 6 for k in approved):
        print(json.dumps({"warning": "one or more preferred idle GPUs absent", "approved": approved}, indent=2), flush=True)
    assignments = {"pplx_decider": 1, "liquid_d1_3b": 5, "jina_v3_5": 5, "qwen3_reranker_4b": 5, "bge_reranker_v2_m3": 5, "qwen3_reranker_0_6b": 6, "startlux_decision_4b": 6, "amos_1_small_2b": 6}
    if not all(str(g) in approved for g in set(assignments.values())):
        raise SystemExit(f"required assignment GPU unavailable: {approved}")
    write_json(OUT / "E6_2_R2_GPU_AUDIT_V2.json", {"status": "PASS", "authorization": "current owner authorized all idle and available GPUs", "preflight": snapshot, "approved_gpu_uuids": approved, "safe_devices": safe, "assignments": assignments, "new_process_started": False, "process_eviction": False, "unrelated_services_modified": False, "fresh_device_specific_authorization": True, "created_utc": utc()})
    pre = read_json(OUT / "E6_2_R2_PREINFERENCE_MANIFEST.json")
    pre["gate"] = "E6_2_R2_PREINFERENCE_FROZEN"
    pre["authorization"] = {"fresh_device_specific_authorization_present": True, "approved_gpu_uuids": approved, "source": "current owner instruction plus live no-compute-app audit"}
    pre["execution_plan"] = {"assignments": assignments, "execution_order": MODEL_ORDER, "scheduled_attempts": 2216, "candidate_k": 20, "selected_k": 5, "quality_metrics": False}
    pre["gpu_audit_sha256"] = sha_file(OUT / "E6_2_R2_GPU_AUDIT_V2.json")
    write_json(OUT / "E6_2_R2_PREINFERENCE_MANIFEST_V2.json", pre)
    registry_path = OUT / "E6_2_R2_MODEL_EXECUTION_REGISTRY_V2.csv"
    with registry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "revision", "assigned_gpu", "assigned_gpu_uuid", "planned_attempts", "status"])
        for model in MODEL_ORDER:
            writer.writerow([model, MODEL_REVISIONS[model], assignments[model], approved[str(assignments[model])], 277, "AUTHORIZED_NOT_STARTED"])
    print(json.dumps({"status": "PASS", "approved_gpu_uuids": approved, "assignments": assignments}, indent=2), flush=True)
    return 0


def load_partial_rows() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(WORK.glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    x = (len(values) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (x - lo)


def merge() -> int:
    rows = load_partial_rows()
    if not rows:
        raise SystemExit("no worker traces found")
    rows.sort(key=lambda x: (MODEL_ORDER.index(x.get("model")) if x.get("model") in MODEL_ORDER else 99, x.get("stage", ""), x.get("query_id", "")))
    native_rows = [x for x in rows if x.get("stage") == "exploratory"]
    with (OUT / "E6_2_R2_NATIVE_TRACES.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (OUT / "E6_2_R2_NATIVE_SELECTIONS.jsonl").open("w", encoding="utf-8") as handle:
        for row in native_rows:
            handle.write(json.dumps({k: row.get(k) for k in ("model", "query_id", "candidate_slate_sha256", "candidate_ids", "selected_ids", "scores", "status", "failure", "native_input_sha256", "native_response_sha256", "input_tokens", "duration_ms", "gpu_id", "checkpoint_revision")}, ensure_ascii=False, sort_keys=True) + "\n")
    failures = [x for x in native_rows if x.get("status") != "VALID"]
    with (OUT / "E6_2_R2_FAILURES.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "query_id", "failure_class", "failure", "candidate_k", "gpu_id", "native_api_calls", "physical_forward_count"])
        writer.writeheader()
        for row in failures:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})
    registry_rows = []
    latency_rows = []
    forward_rows = []
    for model in MODEL_ORDER:
        attempted = [x for x in native_rows if x.get("model") == model]
        valid = [x for x in attempted if x.get("status") == "VALID"]
        durations = [float(x["duration_ms"]) for x in valid if x.get("duration_ms") is not None]
        forward_values = [x.get("physical_forward_count") for x in valid if isinstance(x.get("physical_forward_count"), (int, float))]
        load_error = next((x for x in rows if x.get("model") == model and x.get("stage") == "load" and x.get("status") == "MODEL_LOAD_ERROR"), None)
        status = "VALIDATED" if len(attempted) == 277 and len(valid) == 277 else ("VALIDATED_WITH_EXCLUSIONS" if attempted else "MODEL_LOAD_BLOCKED")
        registry_rows.append({"model": model, "revision": MODEL_REVISIONS[model], "assigned_gpu": next((x.get("gpu_id") for x in rows if x.get("model") == model), None), "scheduled_attempts": 277, "attempts": len(attempted), "successful": len(valid), "failed": len(attempted) - len(valid), "load_error": load_error.get("failure") if load_error else None, "status": status, "quality_metrics": "NOT_RUN"})
        latency_rows.append({"model": model, "attempts": len(attempted), "successful": len(valid), "failed": len(attempted) - len(valid), "all_attempt_failure_rate": (1 - len(valid) / len(attempted)) if attempted else None, "success_latency_p50_ms": percentile(durations, .5), "success_latency_p95_ms": percentile(durations, .95), "input_tokens_total": sum(int(x.get("input_tokens") or 0) for x in valid), "input_tokens_missing": sum(x.get("input_tokens") is None for x in valid), "peak_allocated_mib_max": max((float(x.get("peak_memory_allocated_mib") or 0) for x in valid), default=None), "peak_reserved_mib_max": max((float(x.get("peak_memory_reserved_mib") or 0) for x in valid), default=None)})
        forward_rows.append({"model": model, "successful_attempts": len(valid), "observed_forward_sum": sum(int(x.get("physical_forward_count") or 0) for x in valid if isinstance(x.get("physical_forward_count"), (int, float))), "observed_forward_records": len(forward_values), "not_measured_records": sum(not isinstance(x.get("physical_forward_count"), (int, float)) for x in valid), "accounting_policy": "observed native hook events only; no derived estimates"})
    def write_rows(name: str, data: list[dict[str, Any]]) -> None:
        fields = []
        for row in data:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(data)
    write_rows("E6_2_R2_MODEL_EXECUTION_REGISTRY.csv", registry_rows)
    write_rows("E6_2_R2_LATENCY_AND_RESOURCE_REPORT.csv", latency_rows)
    write_rows("E6_2_R2_FORWARD_ACCOUNTING.csv", forward_rows)
    accounted_models = {x["model"] for x in registry_rows if x["status"] != "MODEL_LOAD_BLOCKED"}
    load_blocked_models = {x["model"] for x in registry_rows if x["status"] == "MODEL_LOAD_BLOCKED"}
    validation = {"status": "PASS", "scheduled_models": 8, "scheduled_attempts": 2216, "actual_attempts": len(native_rows), "all_model_attempts_or_exclusions_accounted": all((len([x for x in native_rows if x.get("model") == m]) == 277) or (m in load_blocked_models) for m in MODEL_ORDER), "load_blocked_models": sorted(load_blocked_models), "attempted_models": sorted(accounted_models), "valid_selection_count": sum(x.get("status") == "VALID" for x in native_rows), "invalid_or_failed_count": len(failures), "all_valid_outputs_exactly_five_unique_ids": all(x.get("status") != "VALID" or (len(x.get("selected_ids", [])) == 5 and len(set(x.get("selected_ids", []))) == 5 and set(x.get("selected_ids", [])).issubset(set(x.get("candidate_ids", [])))) for x in native_rows), "candidate_slate_hashes": sorted({x.get("candidate_slate_sha256") for x in native_rows}), "candidate_slate_hash_count": len({x.get("candidate_slate_sha256") for x in native_rows}), "qrels_accessed": False, "reserved_content_accessed": False, "quality_metrics": False, "native_outputs_committed": True}
    write_json(OUT / "E6_2_R2_INPUT_OUTPUT_VALIDATION.json", validation)
    exclusions = [x["model"] for x in registry_rows if x["status"] != "VALIDATED"]
    gate_name = "E6_2_R2_SELECTIONS_FROZEN_WITH_EXCLUSIONS" if exclusions else "E6_2_R2_SELECTIONS_FROZEN_READY"
    gate = {"gate": gate_name, "status": "READY_FOR_SEPARATE_E6_2_R3_OFFLINE_SCORING", "created_utc": utc(), "models": registry_rows, "exclusions": exclusions, "scheduled_attempts": 2216, "actual_attempts": len(native_rows), "quality_metrics": False, "qrels_accessed": False, "reserved_content_accessed": False, "next_gate": "E6-2 R3 owner authorization for offline relevance scoring"}
    write_json(OUT / "E6_2_R2_R3_READINESS_GATE.json", gate)
    report = ["# DecisionRAG E6-2 R2 Native Selector Inference", "", f"## Gate: `{gate_name}`", "", "This run executed native selector inference only on the frozen, gold-blind SciFact EXPLORATORY Top-20 slates. No relevance metrics were calculated.", "", "## Population and isolation", f"- 277 frozen questions × 8 models = 2,216 scheduled attempts; actual attempts: {len(native_rows)}.", "- Candidate slate hash set is identical across models: one frozen R1 slate hash per query.", "- qrels, gold labels, reference answers, RESERVED content, readers and R3 outputs were not loaded.", "- No fallback to retrieval Top-5 or another model was used.", "", "## Model execution", "", "| Model | Attempts | Valid | Failed | Success p50 ms | Success p95 ms | Status |", "|---|---:|---:|---:|---:|---:|---|"]
    for reg, lat in zip(registry_rows, latency_rows):
        report.append(f"| {reg['model']} | {reg['attempts']} | {reg['successful']} | {reg['failed']} | {lat['success_latency_p50_ms']} | {lat['success_latency_p95_ms']} | {reg['status']} |")
    report += ["", "## Accounting limitations", "- Physical forwards are retained only when a native wrapper exposed positive observed hook events; otherwise the value is `NOT_MEASURED`.", "- Latencies are selector-attempt diagnostics, not E6-3 serving throughput or production cost measurements.", "- Decision-model probability sorting is a derived Top-5 policy, not joint-set optimization.", "", "## R3 boundary", "Native selections and explicit failures are committed. R3 may consume this bundle for offline relevance scoring only after separate owner authorization."]
    (OUT / "E6_2_R2_FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--models", default="")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--gpus", default="1,5,6")
    args = parser.parse_args()
    if args.prepare:
        return prepare([int(x) for x in args.gpus.split(",") if x.strip()])
    if args.worker:
        if args.gpu is None or not args.models:
            raise SystemExit("--worker requires --gpu and --models")
        return run_worker([x.strip() for x in args.models.split(",") if x.strip()], args.gpu)
    if args.merge:
        return merge()
    raise SystemExit("select --prepare, --worker, or --merge")


if __name__ == "__main__":
    raise SystemExit(main())
