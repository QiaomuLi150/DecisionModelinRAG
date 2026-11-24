#!/usr/bin/env python3
"""E2-N N1 bounded authentic CAL smoke tests.

Only locally present checkpoints/services are used.  No download, retrieval,
index build, holdout access, Jev, paid API, or training path exists here.
Run in the separately authorized host runtime with CUDA_VISIBLE_DEVICES set to
one approved/idle device.  GPU 6 is also accepted when explicitly selected by
the owner as an idle fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_n/n1"
HF_ROOT = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub")
SEED = 20261011
QWEN_ID = "Qwen/Qwen3-VL-Reranker-2B"
QWEN_REV = "4bd860ac4f15ad1897a214615cccc700f8f71818"
QWEN_URL = "http://service.example.invalid/v1/rerank"
QWEN_MODELS_URL = "http://service.example.invalid/v1/models"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def snapshot(owner: str, repo: str, revision: str) -> Path:
    return HF_ROOT / f"models--{owner}--{repo}" / "snapshots" / revision


def gpu_snapshot() -> dict:
    cmd = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=20)
        return {"status": "PASS" if p.returncode == 0 else "UNAVAILABLE", "command": " ".join(cmd), "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "command": " ".join(cmd), "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def load_items() -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_m1_experiment import load_local_cal_items

    all_items = load_local_cal_items()
    chosen = []
    for dataset in ("hotpot", "musique", "wiki2"):
        group = sorted([x for x in all_items if x["dataset"] == dataset], key=lambda x: (len(x["question"]), x["query_id"]))
        # Deterministic short/long coverage: two endpoints per dataset, plus
        # two interior cases in the first two datasets for schema diversity.
        chosen.extend([group[0], group[-1]])
        if dataset in {"hotpot", "musique"}:
            chosen.extend([group[1], group[-2]])
    unique = {x["query_id"]: x for x in chosen}
    output = [unique[qid] for qid in sorted(unique)]
    if len(output) != 10:
        raise RuntimeError(f"expected 10 deterministic CAL smoke items, got {len(output)}")
    return output


def base_trace(item: dict, model_id: str, revision: str, formulation: str, candidate_ids: list[str], view_hashes: list[str]) -> dict:
    return {
        "stage": "N1",
        "seed": SEED,
        "query_id": item["query_id"],
        "dataset": item["dataset"],
        "model_id": model_id,
        "revision": revision,
        "formulation": formulation,
        "candidate_ids": candidate_ids,
        "candidate_count": len(candidate_ids),
        "selection_text_sha256": view_hashes,
        "score_direction": "higher_is_more_relevant",
        "cache_reuse": False,
        "cache_provenance": "not_applicable_new_smoke_request",
        "gold_in_inference_input": False,
        "retry_count": 0,
        "status": "error",
        "real_inference": False,
        "physical_gpu_id": os.environ.get("E2_N_PHYSICAL_GPU_ID"),
        "visible_cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def selected_views(item: dict, tokenizer):
    sys.path.insert(0, str(ROOT / "src"))
    from decisionrag.selection import make_views
    return [make_views(item["question"], c, tokenizer) for c in item["candidates"]]


def run_bge(items: list[dict]) -> list[dict]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_m3_selectors import common_tokenizer
    from decisionrag.selection import pointwise_pair

    revision = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    model_id = "BAAI/bge-reranker-v2-m3"
    path = snapshot("BAAI", "bge-reranker-v2-m3", revision)
    view_tokenizer = common_tokenizer()
    model_tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.bfloat16).to("cuda").eval()
    rows = []
    for item in items:
        views = selected_views(item, view_tokenizer)
        ids = [x.doc_id for x in views]
        row = base_trace(item, model_id, revision, "INDEPENDENT_RELEVANCE", ids, [x.selection_text_sha256 for x in views])
        try:
            pairs = [pointwise_pair(x) for x in views]
            encoded = model_tokenizer([x[0] for x in pairs], [x[1] for x in pairs], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            input_payload = {k: v.detach().cpu().tolist() for k, v in encoded.items()}
            row["actual_model_input_sha256"] = sha_json(input_payload)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            with torch.inference_mode():
                logits = model(**encoded).logits.reshape(-1).float().detach().cpu().tolist()
            torch.cuda.synchronize()
            row.update({
                "status": "ok",
                "real_inference": True,
                "scores": [float(x) for x in logits],
                "raw_response_sha256": sha_json(logits),
                "request_hash": sha_json({"pairs": pairs}),
                "physical_forward_calls": 1,
                "physical_forward_evidence": True,
                "input_tokens": int(encoded["attention_mask"].sum().item()),
                "duration_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            })
        except Exception as exc:
            row["error_type"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    del model
    torch.cuda.empty_cache()
    return rows


def run_laya(items: list[dict]) -> list[dict]:
    from laya import Router
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT / "src"))
    from decisionrag.selection import pointwise_question, pointwise_state

    revision = "7b928d828b7b0e022f929d9bd2e44165aa270148"
    model_id = "convaiinnovations/laya"
    path = snapshot("convaiinnovations", "laya", revision)
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
    router = Router(models={"english": str(path)}, device="cuda", default="english", max_loaded=1)
    agent = router.load("english")
    forward_counter = {"count": 0}
    captured_batches = []
    original_forward = agent._forward

    def counted_forward(batch):
        forward_counter["count"] += 1
        captured = {}
        for key, value in batch.items():
            if hasattr(value, "detach"):
                captured[key] = value.detach().cpu().tolist()
        captured_batches.append(captured)
        return original_forward(batch)

    agent._forward = counted_forward
    rows = []
    for item in items:
        views = selected_views(item, tokenizer)
        ids = [x.doc_id for x in views]
        payload = [{"state": pointwise_state(x), "questions": pointwise_question(), "model": "english", "max_len": 512} for x in views]
        row = base_trace(item, model_id, revision, "INDEPENDENT_RELEVANCE", ids, [x.selection_text_sha256 for x in views])
        row["actual_model_input_sha256"] = sha_json(payload)
        before_forwards = forward_counter["count"]
        before_batches = len(captured_batches)
        try:
            started = time.perf_counter_ns()
            results = router.predict_batch(payload, batch_size=4, sort_by_length=True)
            scores = [float(x["answers"]["relevance"]["noul"]) for x in results]
            actual_batches = captured_batches[before_batches:]
            row.update({"status": "ok", "real_inference": True, "scores": scores, "raw_response_sha256": sha_json(results), "request_hash": sha_json(payload), "actual_model_input_sha256": sha_json(actual_batches), "physical_forward_calls": forward_counter["count"] - before_forwards, "physical_forward_evidence": True, "duration_ms": (time.perf_counter_ns() - started) / 1_000_000.0, "responses": results})
        except Exception as exc:
            row["error_type"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def run_qwen(items: list[dict]) -> list[dict]:
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_m3_selectors import common_tokenizer
    from decisionrag.selection import pointwise_pair

    identity = requests.get(QWEN_MODELS_URL, timeout=30).json()
    served = identity["data"][0]
    if served.get("id") != QWEN_ID or QWEN_REV not in str(served.get("root", "")):
        raise RuntimeError(f"unexpected Qwen service identity: {served}")
    tokenizer = common_tokenizer()
    session = requests.Session()
    rows = []
    for item in items:
        views = selected_views(item, tokenizer)
        ids = [x.doc_id for x in views]
        documents = [pointwise_pair(x)[1] for x in views]
        payload = {"model": QWEN_ID, "query": views[0].question_view, "documents": documents, "chat_template_kwargs": {"add_generation_prompt": True}}
        row = base_trace(item, QWEN_ID, QWEN_REV, "INDEPENDENT_RELEVANCE", ids, [x.selection_text_sha256 for x in views])
        row.update({"actual_model_input_sha256": sha_json(payload), "physical_forward_calls": "SERVICE_NOT_INSTRUMENTED", "physical_forward_evidence": False, "physical_gpu_id": "SERVICE_GPU_NOT_INSTRUMENTED"})
        try:
            started = time.perf_counter_ns()
            response = session.post(QWEN_URL, json=payload, timeout=300, headers={"Cache-Control": "no-cache", "X-DecisionRAG-Cache": "disabled"})
            response.raise_for_status()
            body = response.json()
            ordered = sorted(body["results"], key=lambda x: int(x["index"]))
            if len(ordered) != 20 or [int(x["index"]) for x in ordered] != list(range(20)):
                raise RuntimeError("Qwen response index alignment invalid")
            scores = [float(x["relevance_score"]) for x in ordered]
            row.update({"status": "ok_with_telemetry_limit", "real_inference": True, "scores": scores, "raw_response_sha256": sha_bytes(response.content), "request_hash": sha_json(payload), "duration_ms": (time.perf_counter_ns() - started) / 1_000_000.0, "service_identity": served, "usage": body.get("usage")})
        except Exception as exc:
            row["error_type"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def run_clef(items: list[dict]) -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    import e2_m1_experiment as m1

    m1.GPU_ID = int(os.environ.get("E2_N_PHYSICAL_GPU_ID", "0"))
    module, model, processor, torch = m1.import_clef()
    tokenizer = __import__("transformers", fromlist=["AutoTokenizer"]).AutoTokenizer.from_pretrained(str(snapshot("convaiinnovations", "laya", "7b928d828b7b0e022f929d9bd2e44165aa270148") / "tokenizer"))
    rows = []
    for item in items:
        views = selected_views(item, tokenizer)
        ids = [x.doc_id for x in views]
        view_hashes = [x.selection_text_sha256 for x in views]
        point = m1.pointwise_one(item, module, model, processor, tokenizer, torch, batch_size=4)
        point["stage"] = "N1"
        point["physical_gpu_id"] = os.environ.get("E2_N_PHYSICAL_GPU_ID")
        point["cache_provenance"] = "not_applicable_new_smoke_request"
        point["raw_response_sha256"] = point.get("response_hash")
        point["actual_model_input_sha256"] = point.get("actual_input_sha256")
        point["physical_forward_evidence"] = True
        point["cache_reuse"] = False
        point["gold_in_inference_input"] = False
        rows.append(point)
        joint = m1.joint_one(item, module, model, processor, tokenizer, torch)
        joint["stage"] = "N1"
        joint["physical_gpu_id"] = os.environ.get("E2_N_PHYSICAL_GPU_ID")
        joint["cache_provenance"] = "not_applicable_new_smoke_request"
        joint["raw_response_sha256"] = joint.get("response_hash")
        joint["actual_model_input_sha256"] = joint.get("actual_input_sha256")
        joint["physical_forward_evidence"] = True
        joint["cache_reuse"] = False
        joint["gold_in_inference_input"] = False
        rows.append(joint)
    del model
    torch.cuda.empty_cache()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("bge", "laya", "qwen", "clef"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_items()
    gpu_before = gpu_snapshot()
    if args.model == "bge": rows = run_bge(items)
    elif args.model == "laya": rows = run_laya(items)
    elif args.model == "qwen": rows = run_qwen(items)
    else: rows = run_clef(items)
    path = OUT / f"E2_N_N1_{args.model.upper()}_SMOKE.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    ok = sum(row.get("status") in {"ok", "ok_with_telemetry_limit"} for row in rows)
    strict = sum(row.get("status") == "ok" and row.get("physical_forward_evidence") is True for row in rows)
    summary = {"stage": "N1", "model": args.model, "rows": len(rows), "ok_or_limited": ok, "strict_telemetry_ok": strict, "unique_queries": len(items), "candidate_count_per_query": 20, "gpu_snapshot_before": gpu_before, "gpu_snapshot_after": gpu_snapshot(), "jev_calls": 0, "paid_api_calls": 0, "downloads": 0, "network": args.model == "qwen", "status": "PASS" if ok == len(rows) and strict == len(rows) else "PASS_WITH_LIMITATIONS" if ok == len(rows) else "BLOCKED"}
    (OUT / f"E2_N_N1_{args.model.upper()}_SMOKE_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if ok == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
