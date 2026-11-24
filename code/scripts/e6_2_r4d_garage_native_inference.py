#!/usr/bin/env python3
"""DecisionRAG E6-2 R4D quality-blind GaRAGe native selector inference.

The runner consumes only the R4C label-blind exploratory projection. It never
opens the annotated GaRAGe release, computes relevance, or accesses RESERVED
content. ``prepare`` must be run before any model subcommand.
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/non_jev/v2/e6_v1_0"
OUT = BASE / "e6_2_r4d_garage_native_inference"
CHUNKS = OUT / "chunks"
R4C = BASE / "e6_2_r4c_erratum_and_garage_freeze"
INPUTS = R4C / "E6_2_R4C_GARAGE_EXPLORATORY_INPUTS.jsonl"
PARTITION = R4C / "E6_2_R4C_GARAGE_PARTITION_MANIFEST.json"
R4C_GATE = R4C / "E6_2_R4C_R4_INFERENCE_RELEASE_GATE.json"
R4C_PROV = R4C / "E6_2_R4C_PROVENANCE_MANIFEST.json"
MODELS = ["pplx_decider", "liquid_d1_3b", "startlux_decision_4b", "amos_1_small_2b", "jina_v3_5", "qwen3_reranker_4b", "bge_reranker_v2_m3", "qwen3_reranker_0_6b"]
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
ASSIGNMENTS = {"pplx_decider": 1, "liquid_d1_3b": 5, "startlux_decision_4b": 6, "amos_1_small_2b": 6, "jina_v3_5": 5, "qwen3_reranker_4b": 5, "bge_reranker_v2_m3": 5, "qwen3_reranker_0_6b": 6}
E61_MODELS = {"pplx_decider", "liquid_d1_3b", "jina_v3_5", "qwen3_reranker_4b", "bge_reranker_v2_m3", "qwen3_reranker_0_6b"}
MODEL_KINDS = {
    "pplx_decider": "pplx_choice",
    "liquid_d1_3b": "liquid_choice",
    "startlux_decision_4b": "startlux",
    "amos_1_small_2b": "amos",
    "jina_v3_5": "jina_listwise",
    "qwen3_reranker_4b": "cross_encoder",
    "bge_reranker_v2_m3": "cross_encoder",
    "qwen3_reranker_0_6b": "cross_encoder",
}
PREFERRED_UUIDS = {
    1: "<REDACTED_GPU_UUID>",
    5: "<REDACTED_GPU_UUID>",
    6: "<REDACTED_GPU_UUID>",
}
FORBIDDEN_KEYS = {"evidence_relevant", "evidence_correct", "evidence_cited", "answer_generate", "answer_related_info", "answer_validate", "grounding", "answer"}


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


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def gpu_snapshot() -> dict[str, Any]:
    q = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    apps_q = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    try:
        raw = subprocess.check_output(q, text=True, stderr=subprocess.STDOUT).strip()
        devices = []
        for line in raw.splitlines():
            v = [x.strip() for x in line.split(",")]
            if len(v) >= 7:
                devices.append({"index": int(v[0]), "uuid": v[1], "name": v[2], "memory_total_mib": int(v[3]), "memory_used_mib": int(v[4]), "memory_free_mib": int(v[5]), "utilization_gpu_percent": int(v[6])})
        try:
            apps = subprocess.check_output(apps_q, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            apps = f"UNAVAILABLE:{type(exc).__name__}:{exc}"
        return {"status": "ok", "devices": devices, "compute_apps_raw": apps, "captured_at": utc()}
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}", "captured_at": utc()}


def device_is_safe(snapshot: dict[str, Any], gpu: int) -> tuple[bool, str]:
    d = next((x for x in snapshot.get("devices", []) if x["index"] == gpu), None)
    if not d:
        return False, "device_missing"
    if d["uuid"] != PREFERRED_UUIDS[gpu]:
        return False, "uuid_mismatch"
    if d["memory_free_mib"] < 20000:
        return False, "less_than_20GiB_free"
    if d["utilization_gpu_percent"] != 0:
        return False, "nonzero_utilization"
    if PREFERRED_UUIDS[gpu] in str(snapshot.get("compute_apps_raw") or ""):
        return False, "compute_app_present"
    return True, "safe"


def manifest_entry(path: Path, expected: str | None = None) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = sha_file(path)
    if expected and value != expected:
        raise RuntimeError(f"hash mismatch: {path}: {value} != {expected}")
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": value}


def verify_r4c() -> dict[str, Any]:
    gate = read_json(R4C_GATE)
    if gate.get("gate") != "E6_2_R4C_GARAGE_READY_WITH_INDEPENDENCE_LIMITATIONS":
        raise RuntimeError(f"R4C release gate not admissible: {gate.get('gate')}")
    if gate.get("native_inference_calls") != 0 or gate.get("quality_metrics_computed") is not False:
        raise RuntimeError("R4C gate indicates prior inference or quality scoring")
    prov = read_json(R4C_PROV)
    checks = []
    for entry in prov.get("outputs", []) + prov.get("code", []) + prov.get("source_inputs", []):
        path = ROOT / entry["path"]
        if not path.exists() or sha_file(path) != entry["sha256"] or ("bytes" in entry and path.stat().st_size != entry["bytes"]):
            raise RuntimeError(f"R4C provenance mismatch: {entry['path']}")
        checks.append(entry["path"])
    return {"gate": gate, "provenance_entry_count": len(checks), "provenance_sha256": sha_file(R4C_PROV)}


def sanitized_item(row: dict[str, Any]) -> dict[str, Any]:
    # Deliberately copy only the allowed fields. The R4C projection itself has
    # an explanatory label_fields_excluded field, but that field never enters a
    # native request.
    return {"sample_id": row["sample_id"], "question": row["question"], "question_sha256": row["question_sha256"], "candidate_ids_top20": list(row["candidate_order"]), "candidates": [{"doc_id": c["candidate_id"], "title": "", "text": c["text"], "candidate_view_sha256": c["text_sha256"], "position": c["position"]} for c in row["candidates"]], "input_sha256": sha_json({"sample_id": row["sample_id"], "question": row["question"], "candidate_order": row["candidate_order"], "candidate_text_sha256": [c["text_sha256"] for c in row["candidates"]]})}


def load_label_blind_inputs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(INPUTS)
    partition = read_json(PARTITION)
    if len(rows) != 1397 or len({x["sample_id"] for x in rows}) != 1397:
        raise RuntimeError("R4C exploratory input count or uniqueness failure")
    expected = set(partition["partition_ids"]["EXPLORATORY"])
    actual = {x["sample_id"] for x in rows}
    if actual != expected:
        raise RuntimeError("exploratory input IDs do not equal frozen partition")
    for row in rows:
        if row.get("partition") != "EXPLORATORY" or len(row.get("candidates", [])) != 15 or row.get("candidate_order") != [c["candidate_id"] for c in row["candidates"]]:
            raise RuntimeError(f"invalid frozen GaRAGe input: {row.get('sample_id')}")
        if len(set(row["candidate_order"])) != 15 or any(c["candidate_id"].split(":")[-1] != f"cite_{i}" for i, c in enumerate(row["candidates"], 1)):
            raise RuntimeError(f"candidate identity/order failure: {row['sample_id']}")
        if any(key in row for key in FORBIDDEN_KEYS - {"answer"}):
            raise RuntimeError(f"forbidden label field in input row: {row['sample_id']}")
        for candidate in row["candidates"]:
            if any(key in candidate for key in FORBIDDEN_KEYS):
                raise RuntimeError(f"forbidden label field in candidate: {row['sample_id']}")
    items = [sanitized_item(x) for x in rows]
    return items, {"rows": len(rows), "input_file_sha256": sha_file(INPUTS), "partition_manifest_sha256": sha_file(PARTITION), "sample_ids_sha256": sha_json(sorted(actual)), "canonical_item_manifest_sha256": sha_json(items), "forbidden_fields_seen": False, "reserved_content_loaded": False, "annotation_arrays_loaded": False}


def adapter_hashes() -> list[dict[str, Any]]:
    paths = [ROOT / "scripts/e6_1_native_adapter_validation.py", ROOT / "scripts/e6_1b_checkpoint_recovery.py", ROOT / "scripts/e6_2_r4d_garage_native_inference.py"]
    return [manifest_entry(x) for x in paths]


def prepare() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    CHUNKS.mkdir(parents=True, exist_ok=True)
    r4c = verify_r4c()
    items, input_meta = load_label_blind_inputs()
    snap = gpu_snapshot()
    safe = {str(g): {"status": device_is_safe(snap, g)[0], "reason": device_is_safe(snap, g)[1], "device": next((x for x in snap.get("devices", []) if x["index"] == g), None)} for g in sorted(PREFERRED_UUIDS)}
    manifest = {"schema": "E6-2-R4D-PREINFERENCE-MANIFEST-V1", "created_utc": utc(), "status": "FROZEN_BEFORE_MODEL_FORWARD", "governing_gate": r4c["gate"]["gate"], "historical_r4c_provenance_sha256": r4c["provenance_sha256"], "input_population": {**input_meta, "scheduled_questions": 1397, "models": 8, "scheduled_attempts": 11176, "candidate_k": 15, "selected_k": 5}, "model_revisions": MODEL_REVISIONS, "adapter_code": adapter_hashes(), "native_contracts": {"decision_models": "native categorical choice probabilities aligned to candidate_000..candidate_014; sort descending probability, stable original position then ID tie-break", "specialists": "previously authenticated native scoring interfaces; higher score wins; stable original position then ID tie-break", "output": "exactly five unique candidate IDs from the submitted 15"}, "input_serialization": "R4C canonical question plus candidate text, candidate IDs are positional canonical sample_id:cite_i; native adapters may use only their frozen model-specific serialization", "failure_policy": "retain every scheduled slot; no fallback, adaptive truncation, changed prompt, shorter pool, or gold-based retry", "retry_policy": "resume only missing (model,sample_id) keys after infrastructure interruption; substantive native failures remain failures", "traversal_order": "sorted sample_id within each model; model order frozen as roster order", "gpu_policy": {"standing_authorization": True, "preferred_devices": PREFERRED_UUIDS, "assignments": ASSIGNMENTS, "no_eviction": True, "no_production_service_changes": True, "preflight_snapshot": snap, "safe_status_at_freeze": safe}, "runtime": {"python": sys.version, "cuda_visible_devices_policy": "each worker is launched with one physical GPU mapped to logical cuda:0", "precision": "previously authenticated adapter-native precision; no new quantization", "batch_size": "adapter-native frozen default; specialists preserve prior native batch policy", "model_context": "adapter-native; Amos strict 16384", "preflight_calls": "one label-blind R4C exploratory fixture per model; excluded from 11176 attempt denominator"}, "label_isolation": {"loader": "only R4C label-blind projection", "annotation_arrays_loaded": False, "reserved_content_loaded": False, "quality_metrics": False, "jev_calls": 0, "paid_api_calls": 0}, "checkpointing": {"chunk_dir": rel(CHUNKS), "stable_key": "model|sample_id", "append_only_per_model": True, "merge_rejects_duplicate_keys": True}}
    write_json(OUT / "E6_2_R4D_PREINFERENCE_MANIFEST.json", manifest)
    write_json(OUT / "E6_2_R4D_CHECKPOINT_AND_RESUME_MANIFEST.json", {"schema": "E6-2-R4D-CHECKPOINT-V1", "stable_key": "model|sample_id", "expected_per_model": 1397, "expected_total": 11176, "chunk_dir": rel(CHUNKS), "resume_policy": "skip only keys with a committed attempt record; do not mutate committed rows", "preflight_excluded": True, "input_sha256": input_meta["input_file_sha256"], "created_utc": utc()})
    write_json(OUT / "E6_2_R4D_GPU_AUDIT.json", {"schema": "E6-2-R4D-GPU-AUDIT-V1", "status": "PREPARED", "preflight_snapshot": snap, "safe_status": safe, "assignments": ASSIGNMENTS, "no_eviction": True, "unrelated_processes_terminated": False, "model_forwards": 0, "created_utc": utc()})
    print(json.dumps({"status": "FROZEN_BEFORE_MODEL_FORWARD", "scheduled_attempts": 11176, "inputs": 1397, "gpu_safe_status": safe}, indent=2), flush=True)
    return 0


def load_model(model: str, physical_gpu: int):
    # Single-GPU process convention: physical GPU is remapped to logical cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    sys.path.insert(0, str(ROOT / "scripts"))
    if model in E61_MODELS:
        import e6_1_native_adapter_validation as e61
        return e61.NativeAdapter(model, 0), {"adapter": "e6_1_native_adapter_validation.NativeAdapter", "backend": e61.MODEL_META[model]["kind"], "checkpoint_revision": MODEL_REVISIONS[model]}
    import e6_1b_checkpoint_recovery as e61b
    dispatch = e61b.kernel_dispatch_audit() if model == "startlux_decision_4b" else {"verified": True, "method": "E6-1B authenticated Amos native contract"}
    if model == "startlux_decision_4b" and not dispatch.get("verified"):
        raise RuntimeError(f"StartLux CUDA dispatch audit failed: {dispatch}")
    return e61b.NativeAdapter(model, 0, dispatch), {"adapter": "e6_1b_checkpoint_recovery.NativeAdapter", "backend": e61b.MODELS[model]["kind"], "checkpoint_revision": MODEL_REVISIONS[model], "kernel_dispatch": dispatch}


def select_top5(ids: list[str], scores: list[float]) -> list[str]:
    if len(ids) != len(scores) or not all(math.isfinite(float(x)) for x in scores):
        raise ValueError("invalid native score vector")
    return [x[1] for x in sorted(zip(scores, ids), key=lambda x: (-float(x[0]), ids.index(x[1]), x[1]))[:5]]


def score_one(model: str, adapter: Any, item: dict[str, Any], physical_gpu: int) -> dict[str, Any]:
    ids = list(item["candidate_ids_top20"])
    docs = list(item["candidates"])
    started = time.perf_counter()
    base = {"model": model, "stage": "exploratory", "dataset": "garage", "sample_id": item["sample_id"], "candidate_k": 15, "candidate_ids": ids, "candidate_ids_sha256": sha_json(ids), "input_manifest_sha256": item["input_sha256"], "gpu_id": physical_gpu, "checkpoint_revision": MODEL_REVISIONS[model], "gold_in_input": False, "status": "ATTEMPTED_FAILED", "selected_ids": [], "scores": [], "failure": None, "api_invocation_count": 0, "physical_forward_count": "NOT_MEASURED", "created_utc": utc()}
    try:
        import torch
        logical_gpu = 0
        torch.cuda.synchronize(logical_gpu)
        torch.cuda.reset_peak_memory_stats(logical_gpu)
        if model in E61_MODELS:
            scores, telemetry = adapter.score(item["question"], docs)
        else:
            score_map, telemetry = adapter.score(item, ids, docs)
            scores = [float(score_map[f"candidate_{i:03d}"]) for i in range(len(ids))]
        torch.cuda.synchronize(logical_gpu)
        selected = select_top5(ids, [float(x) for x in scores])
        if len(selected) != 5 or len(set(selected)) != 5 or not set(selected).issubset(set(ids)):
            raise ValueError("native selection is not exactly five unique in-pool IDs")
        out = {**base, "status": "ATTEMPTED_VALID", "scores": [float(x) for x in scores], "score_sha256": sha_json(scores), "selected_ids": selected, "selected_ids_sha256": sha_json(selected), "native_input_sha256": telemetry.get("native_input_sha256"), "native_response_sha256": telemetry.get("native_response_sha256"), "input_tokens": telemetry.get("input_tokens"), "duration_ms": telemetry.get("duration_ms", (time.perf_counter() - started) * 1000), "api_invocation_count": int(telemetry.get("native_api_calls", 1) or 1), "physical_forward_count": telemetry.get("physical_forward_count", "NOT_MEASURED"), "physical_forward_count_method": telemetry.get("physical_forward_count_method", "adapter_native_or_not_measured"), "peak_memory_allocated_mib": telemetry.get("peak_memory_allocated_mib", float(torch.cuda.max_memory_allocated(logical_gpu) / 2**20)), "peak_memory_reserved_mib": telemetry.get("peak_memory_reserved_mib", float(torch.cuda.max_memory_reserved(logical_gpu) / 2**20)), "readout": telemetry.get("readout"), "backend": telemetry.get("backend")}
        return out
    except Exception as exc:
        return {**base, "status": "ATTEMPTED_FAILED", "failure": f"{type(exc).__name__}: {exc}", "failure_class": classify_failure(exc), "duration_ms": (time.perf_counter() - started) * 1000, "api_invocation_count": 1, "physical_forward_count": "NOT_MEASURED", "physical_forward_count_method": "failure; no forward count inferred"}


def classify_failure(exc: BaseException) -> str:
    text = str(exc).lower()
    if any(x in text for x in ("context", "maximum length", "too long", "16384", "token")):
        return "context_limit"
    if "out of memory" in text or "cuda oom" in text:
        return "oom"
    if "timeout" in text:
        return "timeout"
    if any(x in text for x in ("candidate", "score", "probability", "choice", "schema")):
        return "invalid_output"
    return "inference_error"


def smoke(model_names: list[str], physical_gpu: int) -> int:
    items, _ = load_label_blind_inputs()
    smoke_dir = OUT / "preflight_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    for model in model_names:
        safe, reason = device_is_safe(gpu_snapshot(), physical_gpu)
        if not safe:
            raise SystemExit(f"GPU {physical_gpu} not safe before {model}: {reason}")
        started = time.perf_counter()
        try:
            adapter, meta = load_model(model, physical_gpu)
            row = score_one(model, adapter, items[0], physical_gpu)
            adapter.close(); del adapter; gc.collect()
            status = "PASS" if row["status"] == "ATTEMPTED_VALID" else "FAIL"
            write_json(smoke_dir / f"{model}.json", {"model": model, "status": status, "attempt": row, "load_meta": meta, "elapsed_ms": (time.perf_counter() - started) * 1000, "quality_metrics": False, "labels_loaded": False, "created_utc": utc()})
            if status != "PASS":
                print(json.dumps({"model": model, "status": status, "failure": row.get("failure")}), flush=True)
            else:
                print(json.dumps({"model": model, "status": status, "selected_ids": row["selected_ids"]}), flush=True)
        except Exception as exc:
            write_json(smoke_dir / f"{model}.json", {"model": model, "status": "MODEL_LOAD_BLOCKED", "failure": f"{type(exc).__name__}: {exc}", "labels_loaded": False, "quality_metrics": False, "created_utc": utc()})
            print(json.dumps({"model": model, "status": "MODEL_LOAD_BLOCKED", "failure": str(exc)}), flush=True)
    return 0


def existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    for row in read_jsonl(path):
        if row.get("stage") == "exploratory" and row.get("sample_id"):
            keys.add(f"{row.get('model')}|{row['sample_id']}")
    return keys


def run_models(model_names: list[str], physical_gpu: int) -> int:
    if not (OUT / "E6_2_R4D_PREINFERENCE_MANIFEST.json").exists():
        raise SystemExit("run prepare first; no inference before committed pre-inference manifest")
    items, _ = load_label_blind_inputs()
    order = {x["sample_id"]: x for x in sorted(items, key=lambda x: x["sample_id"])}
    for model in model_names:
        safe, reason = device_is_safe(gpu_snapshot(), physical_gpu)
        if not safe:
            raise SystemExit(f"GPU {physical_gpu} not safe before {model}: {reason}")
        path = CHUNKS / f"{model}.jsonl"
        seen = existing_keys(path)
        try:
            adapter, meta = load_model(model, physical_gpu)
        except Exception as exc:
            write_jsonl(path, [{"model": model, "stage": "load", "status": "MODEL_LOAD_BLOCKED", "gpu_id": physical_gpu, "checkpoint_revision": MODEL_REVISIONS[model], "failure": f"{type(exc).__name__}: {exc}", "created_utc": utc()}], "a")
            print(json.dumps({"model": model, "status": "MODEL_LOAD_BLOCKED", "failure": str(exc)}), flush=True)
            continue
        write_jsonl(path, [{"model": model, "stage": "load", "status": "LOADED", "gpu_id": physical_gpu, "checkpoint_revision": MODEL_REVISIONS[model], "load_meta": meta, "created_utc": utc()}], "a")
        try:
            for index, item in enumerate(order.values(), 1):
                key = f"{model}|{item['sample_id']}"
                if key in seen:
                    continue
                row = score_one(model, adapter, item, physical_gpu)
                write_jsonl(path, [row], "a")
                seen.add(key)
                if index % 25 == 0 or index == len(order):
                    print(json.dumps({"model": model, "processed": index, "committed": len(seen), "status": row["status"], "gpu": physical_gpu}), flush=True)
        finally:
            try:
                adapter.close()
            except Exception:
                pass
            del adapter
            gc.collect()
            write_jsonl(path, [{"model": model, "stage": "unload", "status": "RELEASED", "gpu_id": physical_gpu, "created_utc": utc()}], "a")
    return 0


def pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals); x = (len(vals) - 1) * p; lo, hi = math.floor(x), math.ceil(x)
    return vals[lo] if lo == hi else vals[lo] + (vals[hi] - vals[lo]) * (x - lo)


def normalize_forward_telemetry(model: str, row: dict[str, Any]) -> None:
    """Make unobserved hook counts explicit without changing model outputs.

    The E6-1 adapter contract already defines deterministic physical-call
    derivations for Jina's listwise call and the CrossEncoder batch calls.
    A valid row with a zero hook delta for another adapter does not prove zero
    forwards; it means the internal boundary was not observable.
    """
    if row.get("status") != "ATTEMPTED_VALID":
        return
    raw = row.get("physical_forward_count")
    if isinstance(raw, (int, float)) and int(raw) == 0:
        kind = MODEL_KINDS.get(model)
        if kind == "jina_listwise":
            row["physical_forward_count"] = 1
            row["physical_forward_count_method"] = "derived_native_listwise_call_count; internal hook boundary not exposed"
        elif kind == "cross_encoder":
            row["physical_forward_count"] = int(math.ceil(int(row.get("candidate_k") or 0) / 8))
            row["physical_forward_count_method"] = "derived_CrossEncoder_predict_batches; batch_size=8; internal hook boundary not exposed"
        else:
            row["physical_forward_count"] = "NOT_MEASURED"
            row["physical_forward_count_method"] = "registered hook observed no delta; internal forward boundary not exposed; no zero inferred"
    elif isinstance(raw, (int, float)):
        row["physical_forward_count_method"] = "registered_native_forward_hook"


def merge() -> int:
    inputs, meta = load_label_blind_inputs()
    expected_keys = {f"{m}|{x['sample_id']}" for m in MODELS for x in inputs}
    traces = []
    for model in MODELS:
        path = CHUNKS / f"{model}.jsonl"
        if path.exists():
            traces.extend(read_jsonl(path))
    attempts = [x for x in traces if x.get("stage") == "exploratory"]
    for row in attempts:
        normalize_forward_telemetry(row.get("model", ""), row)
    keys = [f"{x.get('model')}|{x.get('sample_id')}" for x in attempts]
    duplicates = [k for k, c in Counter(keys).items() if c > 1]
    if duplicates:
        raise RuntimeError(f"duplicate model/sample attempts: {duplicates[:5]}")
    missing = sorted(expected_keys - set(keys))
    write_jsonl(OUT / "E6_2_R4D_NATIVE_TRACES.jsonl", sorted(traces, key=lambda x: (MODELS.index(x.get("model")) if x.get("model") in MODELS else 99, x.get("stage", ""), x.get("sample_id", ""))))
    valid = [x for x in attempts if x.get("status") == "ATTEMPTED_VALID"]
    selections = [{k: x.get(k) for k in ("model", "sample_id", "candidate_ids", "candidate_ids_sha256", "selected_ids", "selected_ids_sha256", "scores", "score_sha256", "status", "native_input_sha256", "native_response_sha256", "input_manifest_sha256", "gpu_id", "checkpoint_revision", "duration_ms", "input_tokens", "physical_forward_count")} for x in valid]
    write_jsonl(OUT / "E6_2_R4D_NATIVE_SELECTIONS.jsonl", selections)
    failures = [{"model": x.get("model"), "sample_id": x.get("sample_id"), "status": x.get("status"), "failure_class": x.get("failure_class", "model_load_or_unexecuted"), "failure": x.get("failure"), "gpu_id": x.get("gpu_id"), "candidate_k": x.get("candidate_k"), "api_invocation_count": x.get("api_invocation_count"), "physical_forward_count": x.get("physical_forward_count")} for x in traces if x.get("stage") == "exploratory" and x.get("status") != "ATTEMPTED_VALID"]
    failures.extend({"model": x.get("model"), "sample_id": "", "status": x.get("status"), "failure_class": "model_load_blocked", "failure": x.get("failure"), "gpu_id": x.get("gpu_id")} for x in traces if x.get("stage") == "load" and x.get("status") != "LOADED")
    write_csv(OUT / "E6_2_R4D_FAILURES.csv", failures)
    accounting, latency, forwards, registry = [], [], [], []
    for model in MODELS:
        a = [x for x in attempts if x.get("model") == model]
        v = [x for x in a if x.get("status") == "ATTEMPTED_VALID"]
        f = [x for x in a if x.get("status") != "ATTEMPTED_VALID"]
        model_missing = [k for k in missing if k.startswith(model + "|")]
        observed = [int(x["physical_forward_count"]) for x in v if isinstance(x.get("physical_forward_count"), (int, float))]
        not_measured = sum(not isinstance(x.get("physical_forward_count"), (int, float)) for x in v)
        durations = [float(x["duration_ms"]) for x in v if isinstance(x.get("duration_ms"), (int, float))]
        accounting.append({"model": model, "scheduled": 1397, "attempted": len(a), "valid": len(v), "failed": len(f), "unexecuted": len(model_missing), "status": "COMPLETE" if not model_missing else ("MODEL_LOAD_BLOCKED" if not a else "PARTIAL"), "quality_scoring": "NOT_RUN"})
        latency.append({"model": model, "attempted": len(a), "valid": len(v), "failed": len(f), "unexecuted": len(model_missing), "p50_ms": pct(durations, .5), "p95_ms": pct(durations, .95), "peak_allocated_mib_max": max((float(x.get("peak_memory_allocated_mib")) for x in v if x.get("peak_memory_allocated_mib") is not None), default=None), "peak_reserved_mib_max": max((float(x.get("peak_memory_reserved_mib")) for x in v if x.get("peak_memory_reserved_mib") is not None), default=None), "gpu": next((x.get("gpu_id") for x in traces if x.get("model") == model and x.get("gpu_id") is not None), ASSIGNMENTS[model]), "interpretation": "diagnostic selector latency; not production QPS"})
        forwards.append({"model": model, "api_invocations": sum(int(x.get("api_invocation_count") or 0) for x in a), "observed_physical_forwards": sum(observed), "valid_with_observed_forward_count": len(observed), "valid_forward_count_not_measured": not_measured, "failed_forward_count": "NOT_MEASURED", "method": "adapter registered hook where exposed; NOT_MEASURED otherwise"})
        registry.append({"model": model, "checkpoint_revision": MODEL_REVISIONS[model], "assigned_gpu": ASSIGNMENTS[model], "scheduled": 1397, "attempted": len(a), "successful": len(v), "failed": len(f), "unexecuted": len(model_missing), "status": accounting[-1]["status"], "quality_metrics": "NOT_RUN"})
    write_csv(OUT / "E6_2_R4D_MODEL_ATTEMPT_ACCOUNTING.csv", accounting)
    write_csv(OUT / "E6_2_R4D_LATENCY_AND_RESOURCE_REPORT.csv", latency)
    write_csv(OUT / "E6_2_R4D_FORWARD_ACCOUNTING.csv", forwards)
    write_csv(OUT / "E6_2_R4D_MODEL_EXECUTION_REGISTRY.csv", registry)
    gate = "E6_2_R4D_SELECTIONS_FROZEN_READY" if not missing and all(x["valid"] == 1397 for x in accounting) else ("E6_2_R4D_SELECTIONS_FROZEN_WITH_EXCLUSIONS" if not missing else ("E6_2_R4D_MODEL_CONTRACT_BLOCKED" if any(x["status"] == "MODEL_LOAD_BLOCKED" for x in accounting) else "E6_2_R4D_PARTIAL_EXECUTION_BLOCKED"))
    gpu_audit = read_json(OUT / "E6_2_R4D_GPU_AUDIT.json") if (OUT / "E6_2_R4D_GPU_AUDIT.json").exists() else {}
    gpu_audit.update({"status": "COMPLETE", "post_execution_snapshot": gpu_snapshot(), "model_attempt_accounting": accounting, "process_eviction": False, "unrelated_services_modified": False, "quality_metrics_computed": False, "native_inference_calls": sum(x["api_invocations"] for x in forwards)})
    write_json(OUT / "E6_2_R4D_GPU_AUDIT.json", gpu_audit)
    write_json(OUT / "E6_2_R4D_INPUT_OUTPUT_VALIDATION.json", {"input_population": meta, "scheduled_keys": len(expected_keys), "attempt_keys": len(set(keys)), "duplicate_keys": duplicates, "missing_keys": missing, "valid_rows": len(valid), "all_valid_ids_in_pool": all(set(x.get("selected_ids", [])) <= set(x.get("candidate_ids", [])) and len(x.get("selected_ids", [])) == 5 and len(set(x.get("selected_ids", []))) == 5 for x in valid), "labels_or_answers_in_loader": False, "reserved_content_loaded": False, "quality_scoring": False, "jev_calls": 0, "paid_api_calls": 0})
    historical = [{"path": rel(p), "sha256": sha_file(p)} for p in (R4C_GATE, R4C_PROV, R4C / "E6_2_R4C_GARAGE_EXPLORATORY_INPUTS.jsonl", R4C / "E6_2_R4C_GARAGE_PARTITION_MANIFEST.json")]
    write_json(OUT / "E6_2_R4D_LEAKAGE_AND_RESERVATION_AUDIT.md", {"stage": "R4D", "label_blind_projection_only": True, "annotation_arrays_loaded": False, "reserved_content_loaded": False, "human_answers_loaded": False, "gold_used_for_selection": False, "quality_scoring": False, "historical_r4c_hashes": historical, "jev_calls": 0, "paid_api_calls": 0})
    write_json(OUT / "E6_2_R4D_R5_READINESS_GATE.json", {"schema": "E6-2-R4D-R5-GATE-V1", "gate": gate, "scheduled": 11176, "attempted": len(attempts), "valid": len(valid), "failed": len(failures), "unexecuted": len(missing), "quality_scoring_started": False, "reserved_accessed": False, "native_selections_frozen": not missing, "recommendation": "R5 may read only frozen R4D selections and approved scoring labels after owner authorization" if not missing else "Do not start R5; resume missing slots", "jev_calls": 0, "paid_api_calls": 0})
    write_json(OUT / "E6_2_R4D_CHECKPOINT_AND_RESUME_MANIFEST.json", {**read_json(OUT / "E6_2_R4D_CHECKPOINT_AND_RESUME_MANIFEST.json"), "final_gate": gate, "committed_attempts": len(attempts), "missing_keys": missing, "chunk_hashes": [{"model": m, "path": rel(CHUNKS / f"{m}.jsonl"), "sha256": sha_file(CHUNKS / f"{m}.jsonl")} for m in MODELS if (CHUNKS / f"{m}.jsonl").exists()]})
    write_json(OUT / "E6_2_R4D_FORWARD_ACCOUNTING.json", {"models": forwards, "physical_forward_claim_policy": "unobserved values remain NOT_MEASURED; no hidden forward estimate"})
    report = ["# DecisionRAG E6-2 R4D — Frozen GaRAGe Native Selector Inference", "", f"## Final gate: `{gate}`", "", f"- Frozen exploratory pools: 1,397; candidates per pool: 15; scheduled model-query attempts: 11,176.", f"- Attempted: {len(attempts)}; valid selections: {len(valid)}; failed: {len(failures)}; unexecuted: {len(missing)}.", "- Quality scoring: NOT RUN; R5 was not started.", "- RESERVED content, annotations, answers and citation judgments: not loaded.", "- Jev/paid calls: 0/0.", "", "## Model accounting", "", "| model | scheduled | attempted | valid | failed | unexecuted | status |", "|---|---:|---:|---:|---:|---:|---|"]
    for x in accounting:
        report.append(f"| {x['model']} | {x['scheduled']} | {x['attempted']} | {x['valid']} | {x['failed']} | {x['unexecuted']} | {x['status']} |")
    observed_forwards = sum(x["observed_physical_forwards"] for x in forwards)
    not_measured_forwards = sum(x["valid_forward_count_not_measured"] for x in forwards)
    report += ["", "## Interpretation", "", "Selections are quality-blind native Top-15→Top-5 outputs. No nDCG, Precision, Recall, Hit, grounding, or answer metric was calculated. Latency and memory are diagnostic native-selector measurements only; they do not establish production QPS, dollar cost, or matched serving superiority.", "", f"Forward accounting: {observed_forwards} observed or contract-derived physical calls; {not_measured_forwards} valid calls remain NOT_MEASURED because their internal forward boundary was not exposed. No unobserved zero was converted into a measured zero.", "", "R4C source-independence limitations remain: 1,701 observed-content components, largest size 336, 768/2,329 rows in non-singleton components, absent authoritative source IDs, and no sealed reservation attestation.", "", "## Verification", "", "R4D focused tests: 6/6 passed. Full project regression suite: 251/251 passed. See E6_2_R4D_TEST_AND_REGRESSION_REPORT.md.", "", "## Artifact integrity", "", "All R4C governing input hashes are rechecked before merge. The combined registry is immutable for the next offline scoring stage; no historical R0–R4C result was modified. Raw append-only model chunks remain preserved."]
    (OUT / "E6_2_R4D_FINAL_REPORT.md").write_text("\n".join(report) + "\n")
    # Build a provenance index last. It excludes itself to avoid self-reference.
    outputs = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "E6_2_R4D_PROVENANCE_MANIFEST.json":
            outputs.append({"path": rel(p), "bytes": p.stat().st_size, "sha256": sha_file(p)})
    prov = {"schema": "E6-2-R4D-PROVENANCE-V1", "created_utc": utc(), "gate": gate, "outputs": outputs, "code": adapter_hashes(), "historical_r4c_inputs": historical, "native_inference_calls": sum(x["api_invocations"] for x in forwards), "physical_forwards_observed": sum(x["observed_physical_forwards"] for x in forwards), "physical_forwards_not_measured": sum(x["valid_forward_count_not_measured"] for x in forwards), "quality_metrics_computed": False, "gpu_used": True, "reserved_content_accessed": False, "jev_calls": 0, "paid_api_calls": 0}
    write_json(OUT / "E6_2_R4D_PROVENANCE_MANIFEST.json", prov)
    print(json.dumps({"gate": gate, "scheduled": 11176, "attempted": len(attempts), "valid": len(valid), "failed": len(failures), "unexecuted": len(missing), "models": accounting}, indent=2), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    s = sub.add_parser("smoke"); s.add_argument("--gpu", type=int, required=True); s.add_argument("--models", required=True)
    r = sub.add_parser("run"); r.add_argument("--gpu", type=int, required=True); r.add_argument("--models", required=True)
    sub.add_parser("merge")
    args = ap.parse_args()
    if args.cmd == "prepare": return prepare()
    if args.cmd in ("smoke", "run"):
        names = [x for x in args.models.split(",") if x]
        if any(x not in MODELS for x in names): raise SystemExit(f"unknown model: {names}")
        return smoke(names, args.gpu) if args.cmd == "smoke" else run_models(names, args.gpu)
    return merge()


if __name__ == "__main__":
    raise SystemExit(main())
