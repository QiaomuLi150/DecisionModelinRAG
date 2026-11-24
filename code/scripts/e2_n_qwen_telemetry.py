#!/usr/bin/env python3
"""Qwen3-VL reranker telemetry remediation using the pinned local checkpoint.

The resident vLLM service on :8002 is not restarted or modified.  This bounded
smoke uses the official checkpoint-side scorer on an idle GPU and instruments
its actual model forward boundary, while retaining the existing service smoke
scores for alignment diagnostics.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_n/n1"
HF_ROOT = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub")
QWEN_REV = "4bd860ac4f15ad1897a214615cccc700f8f71818"
QWEN_ID = "Qwen/Qwen3-VL-Reranker-2B"


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gpu_snapshot() -> dict:
    import subprocess
    cmd = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=20)
    return {"status": "PASS" if p.returncode == 0 else "UNAVAILABLE", "stdout": p.stdout.strip(), "stderr": p.stderr.strip(), "command": " ".join(cmd)}


def main() -> int:
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_n_n1_smoke import load_items, selected_views
    from qwen_semantic_final_audit import NativeQwenReference, snapshot
    from run_m3_selectors import common_tokenizer
    from decisionrag.selection import pointwise_pair

    items = load_items()
    model_path = snapshot("Qwen", "Qwen3-VL-Reranker-2B", QWEN_REV)
    template_path = model_path / "additional_chat_templates" / "reranker.jinja"
    if not model_path.is_dir() or not template_path.is_file():
        raise RuntimeError(f"missing local Qwen checkpoint/template: {model_path}")
    before = gpu_snapshot()
    reference = NativeQwenReference(model_path, template_path)
    tokenizer = common_tokenizer()
    forward_count = {"count": 0}
    captured = []
    # NativeQwenReference wraps the official scorer in ``reference``; the
    # actual multimodal backbone forward boundary is nested there.
    original_forward = reference.reference.model.forward

    def counted_forward(*args, **kwargs):
        forward_count["count"] += 1
        payload = {}
        for key, value in kwargs.items():
            if hasattr(value, "detach"):
                payload[key] = value.detach().cpu().tolist()
        if args:
            payload["args"] = [x.detach().cpu().tolist() if hasattr(x, "detach") else repr(x) for x in args]
        captured.append(payload)
        return original_forward(*args, **kwargs)

    reference.reference.model.forward = counted_forward
    service_rows = {r["query_id"]: r for r in (json.loads(x) for x in (OUT / "E2_N_N1_QWEN_SMOKE.jsonl").read_text().splitlines() if x.strip())}
    rows = []
    for item in items:
        views = selected_views(item, tokenizer)
        ids = [v.doc_id for v in views]
        row = {
            "stage": "N1_QWEN_TELEMETRY_REMEDIATION",
            "query_id": item["query_id"],
            "dataset": item["dataset"],
            "model_id": QWEN_ID,
            "revision": QWEN_REV,
            "adapter": "official checkpoint-side Qwen3VLReranker",
            "candidate_ids": ids,
            "score_direction": "higher_is_more_relevant",
            "cache_reuse": False,
            "cache_provenance": "not_applicable_new_direct_checkpoint_smoke",
            "gold_in_inference_input": False,
            "physical_gpu_id": os.environ.get("E2_N_PHYSICAL_GPU_ID"),
            "visible_cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "status": "error",
        }
        before_forwards = forward_count["count"]
        before_batches = len(captured)
        scores = []
        started = time.perf_counter_ns()
        try:
            for view in views:
                document = pointwise_pair(view)[1]
                score, _meta = reference.score(views[0].question_view, document)
                scores.append(float(score))
            actual = captured[before_batches:]
            if len(scores) != 20 or not all(math.isfinite(x) for x in scores):
                raise RuntimeError("invalid direct Qwen score vector")
            service = service_rows.get(item["query_id"], {})
            service_scores = service.get("scores", [])
            direct_order = sorted(range(20), key=lambda i: (-scores[i], i, ids[i]))
            service_order = sorted(range(20), key=lambda i: (-float(service_scores[i]), i, ids[i])) if len(service_scores) == 20 else []
            row.update({
                "status": "ok",
                "real_inference": True,
                "scores": scores,
                "raw_response_sha256": sha_json(scores),
                "actual_model_input_sha256": sha_json(actual),
                "physical_forward_calls": forward_count["count"] - before_forwards,
                "physical_forward_evidence": True,
                "duration_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
                "input_forward_batches": len(actual),
                "service_smoke_available": len(service_scores) == 20,
                "service_smoke_top5_overlap": len(set(ids[i] for i in direct_order[:5]).intersection(ids[i] for i in service_order[:5])) / 5 if service_order else None,
                "service_smoke_score_vector_sha256": sha_json(service_scores) if service_scores else None,
            })
        except Exception as exc:
            row["error_type"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    after = gpu_snapshot()
    path = OUT / "E2_N_N1_QWEN_NATIVE_TELEMETRY.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    ok = [r for r in rows if r.get("status") == "ok"]
    strict = [r for r in ok if r.get("physical_forward_evidence") is True and r.get("physical_forward_calls", 0) > 0]
    summary = {
        "stage": "N1_QWEN_TELEMETRY_REMEDIATION",
        "model": QWEN_ID,
        "revision": QWEN_REV,
        "rows": len(rows),
        "ok_rows": len(ok),
        "strict_physical_telemetry_rows": len(strict),
        "total_physical_forward_calls": sum(r.get("physical_forward_calls", 0) for r in strict),
        "service_smoke_comparison_rows": sum(r.get("service_smoke_available") is True for r in rows),
        "mean_service_smoke_top5_overlap": sum(r["service_smoke_top5_overlap"] for r in rows if r.get("service_smoke_top5_overlap") is not None) / max(1, sum(r.get("service_smoke_top5_overlap") is not None for r in rows)),
        "trace_sha256": sha_file(path),
        "gpu_snapshot_before": before,
        "gpu_snapshot_after": after,
        "resident_service_restarted": False,
        "resident_service_modified": False,
        "downloads": 0,
        "jev_calls": 0,
        "paid_api_calls": 0,
        "status": "PASS" if len(strict) == len(rows) else "BLOCKED",
    }
    (OUT / "E2_N_N1_QWEN_NATIVE_TELEMETRY_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
