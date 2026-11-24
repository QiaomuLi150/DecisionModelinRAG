#!/usr/bin/env python3
"""E2-S/P GPU5 continuation runner.

GPU5 is used only after an explicit resource check confirms that it has no
compute owner and essentially all memory is free.  The runner consumes the
already-frozen E2-S/P reader cohort and frozen Top-20 slates; it never creates
or tunes a retrieval pool and never reads gold labels while scoring.

The primary measurements are serialized on one isolated A100.  The script
does not touch the earlier blocked package or any resident model service.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_sp/gpu5"
COHORT = ROOT / "results/non_jev/v2/e2_sp/E2_SP_READER_COHORT_MANIFEST.json"
GPU = 5
COHORT_SEED = "E2-SP-READER-COHORT-20261013"

SPECIALISTS = {
    "qwen06": {"runner": "specialist", "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473"},
    "qwen4": {"runner": "specialist", "revision": "22e683669bc0f0bd69640a1354a6d0aebcfeede5"},
    "jina35": {"runner": "specialist", "revision": "e8a93f33f0b22108f8c2364f8484ce3422552fbc"},
    "gte": {"runner": "specialist", "revision": "f7481e6055501a30fb19d090657df9ec1f79ab2c"},
    "minilm": {"runner": "specialist", "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a"},
}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def gpu_snapshot() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    inventory_env = {key: value for key, value in os.environ.items() if key != "CUDA_VISIBLE_DEVICES"}
    proc = subprocess.run(command, capture_output=True, text=True, timeout=20, env=inventory_env)
    return {"returncode": proc.returncode, "command": " ".join(command), "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def gpu5_row(snapshot: dict) -> dict:
    for line in snapshot.get("stdout", "").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0] == str(GPU):
            return {
                "index": GPU,
                "name": fields[1],
                "memory_total_mib": int(fields[2]),
                "memory_used_mib": int(fields[3]),
                "memory_free_mib": int(fields[4]),
                "utilization_gpu_percent": int(fields[5]),
            }
    raise RuntimeError("GPU5 not present in nvidia-smi output")


def compute_apps() -> list[dict]:
    command = ["nvidia-smi", "--query-compute-apps=gpu_uuid,gpu_name,pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    inventory_env = {key: value for key, value in os.environ.items() if key != "CUDA_VISIBLE_DEVICES"}
    proc = subprocess.run(command, capture_output=True, text=True, timeout=20, env=inventory_env)
    return [{"raw": line.strip()} for line in proc.stdout.splitlines() if line.strip()] if proc.returncode == 0 else [{"error": proc.stderr.strip()}]


def gpu_uuids() -> dict[int, str]:
    command = ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    inventory_env = {key: value for key, value in os.environ.items() if key != "CUDA_VISIBLE_DEVICES"}
    proc = subprocess.run(command, capture_output=True, text=True, timeout=20, env=inventory_env)
    if proc.returncode != 0:
        raise RuntimeError(f"cannot resolve GPU UUIDs: {proc.stderr.strip()}")
    return {int(parts[0].strip()): parts[1].strip() for parts in (line.split(",", 1) for line in proc.stdout.splitlines() if line.strip())}


def guard_gpu5(min_free_mib: int) -> dict:
    before = gpu_snapshot()
    row = gpu5_row(before)
    apps = compute_apps()
    uuid = gpu_uuids().get(GPU)
    gpu5_apps = [app for app in apps if uuid and uuid in app.get("raw", "")]
    if row["memory_free_mib"] < min_free_mib:
        raise RuntimeError(f"GPU5 free memory {row['memory_free_mib']} MiB below safety floor {min_free_mib} MiB")
    if gpu5_apps:
        raise RuntimeError(f"GPU5 has compute applications: {gpu5_apps}")
    return {"gpu": row, "compute_apps": apps, "guard": "PASS", "minimum_free_mib": min_free_mib}


def load_cohort_items() -> list[dict]:
    manifest = json.loads(COHORT.read_text(encoding="utf-8"))
    records = manifest["query_records"]
    if len(records) != 300 or len({row["query_id"] for row in records}) != 300:
        raise RuntimeError("frozen E2-S/P cohort is not 300 unique records")
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_m1_experiment import load_full_items

    all_items = {row["query_id"]: row for row in load_full_items()}
    selected = []
    for record in records:
        item = all_items.get(record["query_id"])
        if item is None:
            raise RuntimeError(f"frozen cohort query is absent from frozen source rows: {record['query_id']}")
        if sha_bytes(item["question"].encode("utf-8")) != record["question_sha256"]:
            raise RuntimeError(f"question hash mismatch: {record['query_id']}")
        ids = [candidate["doc_id"] for candidate in item["candidates"]]
        if ids != record["top20_candidate_ids"]:
            raise RuntimeError(f"candidate slate mismatch: {record['query_id']}")
        item = dict(item)
        item["e2_sp_cohort"] = "frozen_reader_300"
        selected.append(item)
    return selected


def smoke_items(items: list[dict]) -> list[dict]:
    # Two deterministic records per declared dataset, retaining the frozen
    # candidate slates and including enterprise in the smoke set.
    out = []
    for dataset in ("hotpot", "musique", "wiki2", "enterprise"):
        group = [item for item in items if item["dataset"] == dataset]
        if not group:
            raise RuntimeError(f"no frozen smoke record for {dataset}")
        out.extend(group[:2])
    return out


def cal_items() -> list[dict]:
    """Load the predeclared CAL-only validation cohort without gold fields."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import e2_m1_experiment

    rows = e2_m1_experiment.load_local_cal_items()
    if len(rows) != 30 or len({row["query_id"] for row in rows}) != 30:
        raise RuntimeError(f"CAL validation cohort is not exactly 30 unique rows: {len(rows)}")
    return rows


def stamp_rows(rows: list[dict], model_key: str, stage: str, guard: dict) -> list[dict]:
    for row in rows:
        row["e2_sp_stage"] = stage
        row["e2_sp_model_key"] = model_key
        row["e2_sp_resource_mode"] = "isolated_single_a100_gpu5"
        row["e2_sp_physical_gpu_id"] = GPU
        row["e2_sp_gpu_guard"] = guard
        row["e2_sp_gold_in_inference_input"] = bool(row.get("gold_in_inference_input", False))
        row["e2_sp_cache_reuse"] = bool(row.get("cache_reuse", False))
    return rows


def summary(rows: list[dict], model_key: str, stage: str, spec: dict, guard_before: dict, path: Path, extra: dict | None = None) -> dict:
    ok = [row for row in rows if row.get("status") == "ok"]
    strict = [row for row in ok if row.get("physical_forward_evidence") is True and int(row.get("physical_forward_calls", 0)) > 0]
    result = {
        "e2_sp_stage": stage,
        "model_key": model_key,
        "model_id": spec.get("model_id"),
        "revision": spec.get("revision"),
        "resource_mode": "isolated_single_a100_gpu5",
        "physical_gpu_id": GPU,
        "rows": len(rows),
        "expected_rows": len(rows),
        "unique_queries": len({row.get("query_id") for row in rows}),
        "ok_rows": len(ok),
        "physical_forward_evidence_rows": len(strict),
        "physical_forward_calls": sum(int(row.get("physical_forward_calls", 0)) for row in strict),
        "input_and_response_hash_rows": sum(bool(row.get("actual_model_input_sha256") and row.get("raw_response_sha256")) for row in ok),
        "valid_unique_20_candidate_rows": sum(len(row.get("candidate_ids", [])) == 20 and len(set(row.get("candidate_ids", []))) == 20 for row in ok),
        "gold_in_inference_input_rows": sum(bool(row.get("gold_in_inference_input", False)) for row in rows),
        "cache_reuse_rows": sum(bool(row.get("cache_reuse", False)) for row in rows),
        "gpu_guard_before": guard_before,
        "gpu_snapshot_after": gpu_snapshot(),
        "trace_sha256": sha_file(path),
        "jev_calls": 0,
        "paid_api_calls": 0,
        "status": "PASS" if len(ok) == len(rows) and (not strict or len(strict) == len(rows)) else "PARTIAL_BLOCKED",
    }
    if extra:
        result.update(extra)
    return result


def run_specialist(model_key: str, items: list[dict], stage: str, path: Path, guard: dict) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    import e2_n_specialist_runner as runner

    # Its native adapter is already pinned and instrumented.  We use its
    # resumable implementation only in this new output namespace.
    native_stage = "n3"
    rows = runner.run(model_key, native_stage, items, path)
    loaded = load_jsonl(path)
    spec = runner.MODELS[model_key]
    spec = {"model_id": spec["id"], "revision": spec["revision"], "adapter": spec["kind"]}
    loaded = stamp_rows(loaded, model_key, stage, guard)
    write_jsonl(path, loaded)
    return summary(loaded, model_key, stage, spec, guard, path, {"native_runner_summary": rows})


def run_nimble(items: list[dict], stage: str, path: Path, guard: dict) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    import e2_n_nimble as runner

    runner.run_items = None
    # The implementation is called directly, so the old e2_n output path is
    # never used.
    result = runner.run("n3", items, path)
    loaded = stamp_rows(load_jsonl(path), "nimble", stage, guard)
    write_jsonl(path, loaded)
    spec = {"model_id": runner.MODEL_ID, "revision": runner.REV, "adapter": "native ParallelScorer"}
    return summary(loaded, "nimble", stage, spec, guard, path, {"native_runner_summary": result})


def run_pplx(items: list[dict], stage: str, path: Path, guard: dict) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    import e2_n_pplx as runner

    result = runner.run("n3", items, path)
    loaded = stamp_rows(load_jsonl(path), "pplx_decider", stage, guard)
    write_jsonl(path, loaded)
    spec = {"model_id": runner.MODEL_ID, "revision": runner.REV, "adapter": "native autojev DecisionModel"}
    return summary(loaded, "pplx_decider", stage, spec, guard, path, {"native_runner_summary": result, "native_formulation": "native_slate_categorical"})


def run_qwen_direct(items: list[dict], stage: str, path: Path, guard: dict) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    import e2_n_qwen_direct as runner

    result_items, result = runner.run_direct(items, "n3", path, guard)
    loaded = stamp_rows(result_items, "qwen_direct", stage, guard)
    write_jsonl(path, loaded)
    spec = {"model_id": runner.QWEN_ID, "revision": runner.QWEN_REV, "adapter": "native checkpoint-side Qwen3-VL reranker"}
    return summary(loaded, "qwen_direct", stage, spec, guard, path, {"native_runner_summary": result})


def run_bge(items: list[dict], stage: str, path: Path, guard: dict) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_n_specialist_runner import snapshot, tensor_fingerprint, views, MODELS
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_m3_selectors import common_tokenizer
    sys.path.insert(0, str(ROOT / "src"))
    from decisionrag.selection import pointwise_pair

    spec = {"id": "BAAI/bge-reranker-v2-m3", "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "owner": "BAAI", "repo": "bge-reranker-v2-m3"}
    model_path = snapshot(spec)
    model_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    view_tokenizer = common_tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
    original_forward = model.forward
    forward_count = {"value": 0}
    captured: list[dict] = []

    def counted_forward(*args, **kwargs):
        forward_count["value"] += 1
        captured.append({"args": [tensor_fingerprint(value) for value in args], "kwargs": {key: tensor_fingerprint(value) for key, value in kwargs.items()}})
        return original_forward(*args, **kwargs)

    model.forward = counted_forward
    rows = []
    existing = {row["query_id"]: row for row in load_jsonl(path) if row.get("status") == "ok"}
    for index, item in enumerate(items, 1):
        if item["query_id"] in existing:
            rows.append(existing[item["query_id"]])
            continue
        item_views = views(item, view_tokenizer)
        pairs = [pointwise_pair(view) for view in item_views]
        row = {"e2_sp_stage": stage, "query_id": item["query_id"], "cohort": item.get("cohort"), "dataset": item["dataset"], "model_id": spec["id"], "revision": spec["revision"], "adapter": "native BGE sequence classifier", "candidate_ids": [view.doc_id for view in item_views], "selection_text_sha256": [view.selection_text_sha256 for view in item_views], "score_direction": "higher_is_more_relevant", "cache_reuse": False, "cache_provenance": "none", "gold_in_inference_input": False, "status": "error", "real_inference": False, "physical_gpu_id": GPU, "visible_cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES")}
        row["request_hash"] = sha_json({"query_id": item["query_id"], "candidate_ids": row["candidate_ids"], "selection_text_sha256": row["selection_text_sha256"], "contract": "INDEPENDENT_RELEVANCE:v2"})
        before = forward_count["value"]
        captured_before = len(captured)
        try:
            encoded = model_tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            input_hash = sha_json({key: tensor_fingerprint(value) for key, value in encoded.items()})
            started = time.perf_counter_ns()
            with torch.inference_mode():
                scores = model(**encoded).logits.reshape(-1).float().detach().cpu().tolist()
            actual = captured[captured_before:]
            if len(scores) != 20 or any(not math.isfinite(float(value)) for value in scores):
                raise RuntimeError("BGE returned invalid 20-score vector")
            row.update({"status": "ok", "real_inference": True, "scores": [float(value) for value in scores], "raw_response_sha256": sha_json(scores), "actual_model_input_sha256": sha_json(actual), "tokenized_input_sha256": input_hash, "physical_forward_calls": forward_count["value"] - before, "physical_forward_evidence": bool(actual) and forward_count["value"] > before, "input_forward_batches": len(actual), "input_tokens": int(encoded["attention_mask"].sum().item()), "duration_ms": (time.perf_counter_ns() - started) / 1_000_000.0})
        except Exception as exc:
            row.update({"error_type": f"{type(exc).__name__}: {exc}", "physical_forward_calls": forward_count["value"] - before, "physical_forward_evidence": bool(captured[captured_before:])})
        rows.append(row)
        if index % 10 == 0 or index == len(items):
            write_jsonl(path, rows)
            print(f"bge {stage} {index}/{len(items)} ok={sum(row.get('status') == 'ok' for row in rows)}", flush=True)
    write_jsonl(path, rows)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    rows = stamp_rows(rows, "bge_pointwise", stage, guard)
    write_jsonl(path, rows)
    return summary(rows, "bge_pointwise", stage, {"model_id": spec["id"], "revision": spec["revision"], "adapter": "native BGE sequence classifier"}, guard, path)


def run_clef(items: list[dict], stage: str, point_path: Path, joint_path: Path, guard: dict) -> dict:
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    import e2_m1_experiment as runner
    from e2_n_specialist_runner import tensor_fingerprint
    from run_m3_selectors import common_tokenizer

    guard_clef = guard
    module, model, processor, torch_module = runner.import_clef()
    tokenizer = common_tokenizer()
    original_forward = model.forward
    forward_count = {"value": 0}
    captured: list[dict] = []

    def counted_forward(*args, **kwargs):
        forward_count["value"] += 1
        captured.append({"args": [tensor_fingerprint(value) for value in args], "kwargs": {key: tensor_fingerprint(value) for key, value in kwargs.items()}})
        return original_forward(*args, **kwargs)

    model.forward = counted_forward
    point_rows, joint_rows = [], []
    for index, item in enumerate(items, 1):
        try:
            point_before = forward_count["value"]
            point_captured_before = len(captured)
            point_row = runner.pointwise_one(item, module, model, processor, tokenizer, torch_module, batch_size=4)
            point_actual = captured[point_captured_before:]
            point_row.update({"actual_model_input_sha256": sha_json(point_actual), "raw_response_sha256": point_row.get("response_hash"), "physical_forward_calls": forward_count["value"] - point_before, "physical_forward_evidence": bool(point_actual) and forward_count["value"] > point_before, "input_forward_batches": len(point_actual)})
            point_rows.append(point_row)

            joint_before = forward_count["value"]
            joint_captured_before = len(captured)
            joint_row = runner.joint_one(item, module, model, processor, tokenizer, torch_module)
            joint_actual = captured[joint_captured_before:]
            joint_row.update({"actual_model_input_sha256": sha_json(joint_actual), "raw_response_sha256": joint_row.get("response_hash"), "physical_forward_calls": forward_count["value"] - joint_before, "physical_forward_evidence": bool(joint_actual) and forward_count["value"] > joint_before, "input_forward_batches": len(joint_actual)})
            joint_rows.append(joint_row)
        except Exception as exc:
            error = {"query_id": item["query_id"], "cohort": item.get("cohort"), "dataset": item["dataset"], "status": "error", "real_inference": False, "error_type": f"{type(exc).__name__}: {exc}", "gold_in_inference_input": False, "cache_reuse": False}
            point_rows.append(dict(error, formulation="pointwise_binary"))
            joint_rows.append(dict(error, formulation="native_joint_binary"))
        if index % 10 == 0 or index == len(items):
            print(f"clef {stage} {index}/{len(items)} point_ok={sum(r.get('status') == 'ok' for r in point_rows)} joint_ok={sum(r.get('status') == 'ok' for r in joint_rows)}", flush=True)
    point_rows = stamp_rows(point_rows, "clef_pointwise", stage, guard_clef)
    joint_rows = stamp_rows(joint_rows, "clef_joint", stage, guard_clef)
    write_jsonl(point_path, point_rows)
    write_jsonl(joint_path, joint_rows)
    del model, processor
    torch.cuda.empty_cache()
    gc.collect()
    spec = {"model_id": runner.CLEF_ID, "revision": runner.CLEF_REV, "adapter": "native Clef pointwise/joint"}
    point_summary = summary(point_rows, "clef_pointwise", stage, spec, guard, point_path)
    joint_summary = summary(joint_rows, "clef_joint", stage, spec, guard, joint_path, {"native_formulation": "native_joint_binary"})
    return {"pointwise": point_summary, "joint": joint_summary}


def run_model(model_key: str, items: list[dict], stage: str, guard: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    if model_key == "bge_pointwise":
        path = OUT / f"E2_SP_{stage}_BGE_POINTWISE.jsonl"
        return run_bge(items, stage, path, guard)
    if model_key == "clef":
        return run_clef(items, stage, OUT / f"E2_SP_{stage}_CLEF_POINTWISE.jsonl", OUT / f"E2_SP_{stage}_CLEF_JOINT.jsonl", guard)
    if model_key in SPECIALISTS:
        path = OUT / f"E2_SP_{stage}_{model_key.upper()}.jsonl"
        return run_specialist(model_key, items, stage, path, guard)
    if model_key == "nimble":
        return run_nimble(items, stage, OUT / f"E2_SP_{stage}_NIMBLE.jsonl", guard)
    if model_key == "pplx_decider":
        return run_pplx(items, stage, OUT / f"E2_SP_{stage}_PPLX_DECIDER.jsonl", guard)
    if model_key == "qwen_direct":
        return run_qwen_direct(items, stage, OUT / f"E2_SP_{stage}_QWEN_DIRECT.jsonl", guard)
    raise ValueError(f"unsupported model key {model_key}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("bge_pointwise", "qwen06", "qwen4", "jina35", "gte", "minilm", "clef", "nimble", "pplx_decider", "qwen_direct"))
    parser.add_argument("--stage", choices=("S0_SMOKE", "S1_CAL", "S2_FULL"), default="S0_SMOKE")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "5"):
        raise RuntimeError(f"E2-S/P GPU5 runner requires CUDA_VISIBLE_DEVICES=5, got {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    items = load_cohort_items()
    selected = smoke_items(items) if args.stage == "S0_SMOKE" else (cal_items() if args.stage == "S1_CAL" else items)
    min_free = 70000 if args.model == "pplx_decider" else 25000
    guard = guard_gpu5(min_free)
    print(json.dumps({"model": args.model, "stage": args.stage, "population": len(selected), "gpu_guard": guard}, sort_keys=True), flush=True)
    result = run_model(args.model, selected, args.stage, guard)
    summary_path = OUT / f"E2_SP_{args.stage}_{args.model.upper()}_SUMMARY.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if (result.get("status") == "PASS" or all(value.get("status") == "PASS" for value in result.values() if isinstance(value, dict) and "status" in value)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
