#!/usr/bin/env python3
"""Direct, instrumented E2-N runs for the pinned Qwen3-VL reranker.

This runner deliberately does not contact or restart the resident vLLM
service.  It uses the official checkpoint-side scorer already used by the
semantic audit, batches the twenty frozen candidates per query, and records a
hash of the tensors observed at the model-forward boundary.  The historical
stage is resumable by query ID and consumes only the frozen CAL/Track A/B
slates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HF_ROOT = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub")
OUT_N1 = ROOT / "results/non_jev/v2/e2_n/n1"
OUT_N2 = ROOT / "results/non_jev/v2/e2_n/n2"
OUT_N3 = ROOT / "results/non_jev/v2/e2_n/n3"
QWEN_ID = "Qwen/Qwen3-VL-Reranker-2B"
QWEN_REV = "4bd860ac4f15ad1897a214615cccc700f8f71818"
DEFAULT_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."
SEED = 20261011


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot() -> Path:
    return HF_ROOT / "models--Qwen--Qwen3-VL-Reranker-2B" / "snapshots" / QWEN_REV


def gpu_snapshot() -> dict:
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return {"status": "PASS" if p.returncode == 0 else "UNAVAILABLE", "command": " ".join(command), "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "command": " ".join(command), "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_items(stage: str) -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_m1_experiment import load_full_items, load_local_cal_items

    if stage == "n2":
        rows = load_local_cal_items()
        if len(rows) != 30:
            raise RuntimeError(f"expected 30 CAL items, got {len(rows)}")
    else:
        rows = load_full_items()
        if len(rows) != 4267:
            raise RuntimeError(f"expected 4267 historical items, got {len(rows)}")
    if len({row["query_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate query IDs in frozen input population")
    return rows


def selected_views(item: dict, tokenizer):
    sys.path.insert(0, str(ROOT / "src"))
    from decisionrag.selection import make_views

    return [make_views(item["question"], candidate, tokenizer) for candidate in item["candidates"]]


def base_row(item: dict, views: list, stage: str) -> dict:
    return {
        "stage": "N2_CAL" if stage == "n2" else "N3_FULL_HISTORICAL",
        "seed": SEED,
        "query_id": item["query_id"],
        "cohort": item.get("cohort", "cal" if stage == "n2" else "historical"),
        "dataset": item["dataset"],
        "model_id": QWEN_ID,
        "revision": QWEN_REV,
        "adapter": "official checkpoint-side Qwen3VLReranker",
        "formulation": "INDEPENDENT_RELEVANCE",
        "score_contract": "INDEPENDENT_RELEVANCE:v2",
        "candidate_ids": [view.doc_id for view in views],
        "selection_text_sha256": [view.selection_text_sha256 for view in views],
        "score_direction": "higher_is_more_relevant",
        "cache_reuse": False,
        "cache_provenance": "not_applicable_new_direct_checkpoint_request",
        "gold_in_inference_input": False,
        "physical_gpu_id": os.environ.get("E2_N_PHYSICAL_GPU_ID"),
        "visible_cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "status": "error",
        "real_inference": False,
    }


def tensor_fingerprint(value):
    """Hash tensor bytes without copying full token arrays into every trace."""
    if not hasattr(value, "detach"):
        return repr(value)
    tensor = value.detach().to("cpu").contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": sha_bytes(tensor.numpy().tobytes()),
    }


def run_direct(items: list[dict], stage: str, output_path: Path, gpu_before: dict) -> tuple[list[dict], dict]:
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    from qwen_semantic_final_audit import NativeQwenReference
    from run_m3_selectors import common_tokenizer
    from decisionrag.selection import pointwise_pair

    model_path = snapshot()
    template_path = model_path / "additional_chat_templates" / "reranker.jinja"
    if not model_path.is_dir() or not template_path.is_file():
        raise RuntimeError(f"missing pinned Qwen snapshot/template: {model_path}")
    reference = NativeQwenReference(model_path, template_path)
    view_tokenizer = common_tokenizer()
    forward_count = {"value": 0}
    captured = []
    original_forward = reference.reference.model.forward

    def counted_forward(*args, **kwargs):
        forward_count["value"] += 1
        captured.append({
            "args": [tensor_fingerprint(value) for value in args],
            "kwargs": {key: tensor_fingerprint(value) for key, value in kwargs.items()},
        })
        return original_forward(*args, **kwargs)

    reference.reference.model.forward = counted_forward
    existing = {row["query_id"]: row for row in load_jsonl(output_path) if row.get("status") == "ok"}
    rows = []
    for index, item in enumerate(items, 1):
        if item["query_id"] in existing:
            rows.append(existing[item["query_id"]])
            continue
        views = selected_views(item, view_tokenizer)
        if len(views) != 20 or len({view.doc_id for view in views}) != 20:
            raise RuntimeError(f"invalid frozen candidate view count for {item['query_id']}")
        row = base_row(item, views, stage)
        row["request_hash"] = sha_json({
            "query_id": item["query_id"],
            "candidate_ids": row["candidate_ids"],
            "selection_text_sha256": row["selection_text_sha256"],
            "instruction": DEFAULT_INSTRUCTION,
            "contract": "INDEPENDENT_RELEVANCE:v2",
            "batching": "20 candidates per direct scorer call",
        })
        before_forwards = forward_count["value"]
        before_captured = len(captured)
        started = time.perf_counter_ns()
        try:
            pairs = [reference.reference.format_mm_instruction(
                views[0].question_view, None, None, pointwise_pair(view)[1], None, None,
                instruction=DEFAULT_INSTRUCTION,
            ) for view in views]
            tokenized = reference.reference.tokenize(pairs).to(reference.reference.model.device)
            with torch.inference_mode():
                scores = [float(value) for value in reference.reference.compute_scores(tokenized)]
            actual_batches = captured[before_captured:]
            if len(scores) != 20 or any(not math.isfinite(value) for value in scores):
                raise RuntimeError("direct Qwen scorer returned invalid 20-score vector")
            row.update({
                "status": "ok",
                "real_inference": True,
                "scores": scores,
                "raw_response_sha256": sha_json(scores),
                "actual_model_input_sha256": sha_json(actual_batches),
                "tokenized_input_sha256": sha_json({key: tensor_fingerprint(value) for key, value in tokenized.items()}),
                "physical_forward_calls": forward_count["value"] - before_forwards,
                "physical_forward_evidence": bool(actual_batches) and forward_count["value"] > before_forwards,
                "input_forward_batches": len(actual_batches),
                "input_tokens": int(tokenized["attention_mask"].sum().item()),
                "duration_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
                "cache_source": "none",
            })
        except Exception as exc:
            row.update({"error_type": f"{type(exc).__name__}: {exc}", "physical_forward_calls": forward_count["value"] - before_forwards, "physical_forward_evidence": bool(captured[before_captured:])})
        rows.append(row)
        if index % 10 == 0 or index == len(items):
            write_jsonl(output_path, rows)
            ok = sum(row.get("status") == "ok" for row in rows)
            print(f"{stage} {index}/{len(items)} ok={ok} forwards={sum(int(row.get('physical_forward_calls', 0)) for row in rows if isinstance(row.get('physical_forward_calls'), int))}", flush=True)
    write_jsonl(output_path, rows)
    del reference
    torch.cuda.empty_cache()
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    strict_rows = [row for row in ok_rows if row.get("physical_forward_evidence") is True and int(row.get("physical_forward_calls", 0)) > 0]
    summary = {
        "stage": "N2_CAL" if stage == "n2" else "N3_FULL_HISTORICAL",
        "model": QWEN_ID,
        "revision": QWEN_REV,
        "rows": len(rows),
        "expected_rows": len(items),
        "unique_queries": len({row.get("query_id") for row in rows}),
        "ok_rows": len(ok_rows),
        "physical_forward_evidence_rows": len(strict_rows),
        "total_physical_forward_calls": sum(int(row.get("physical_forward_calls", 0)) for row in strict_rows),
        "input_and_response_hash_rows": sum(bool(row.get("actual_model_input_sha256") and row.get("raw_response_sha256")) for row in ok_rows),
        "valid_unique_20_candidate_rows": sum(len(row.get("candidate_ids", [])) == 20 and len(set(row.get("candidate_ids", []))) == 20 for row in ok_rows),
        "gold_in_inference_input_rows": sum(row.get("gold_in_inference_input") is True for row in rows),
        "cache_reuse_rows": sum(row.get("cache_reuse") is True for row in rows),
        "direct_checkpoint": True,
        "resident_service_restarted": False,
        "resident_service_modified": False,
        "gpu_snapshot_before": gpu_before,
        "gpu_snapshot_after": gpu_snapshot(),
        "trace_sha256": sha_file(output_path),
        "jev_calls": 0,
        "paid_api_calls": 0,
        "status": "PASS" if len(rows) == len(items) and len(ok_rows) == len(items) and len(strict_rows) == len(items) else "BLOCKED",
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("n2", "n3"), required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard selection")
    if args.merge and args.stage != "n3":
        raise SystemExit("--merge is only supported for n3")
    items = load_items(args.stage)
    if args.stage == "n2":
        output = OUT_N2 / "E2_N_N2_QWEN_DIRECT_CAL.jsonl"
        summary_path = OUT_N2 / "E2_N_N2_QWEN_DIRECT_CAL_SUMMARY.json"
    else:
        output = OUT_N3 / "E2_N_N3_QWEN_DIRECT_FULL.jsonl"
        summary_path = OUT_N3 / "E2_N_N3_QWEN_DIRECT_FULL_SUMMARY.json"
        if args.shard_count > 1:
            output = OUT_N3 / f"E2_N_N3_QWEN_DIRECT_FULL.shard{args.shard_index}.jsonl"
            summary_path = OUT_N3 / f"E2_N_N3_QWEN_DIRECT_FULL.shard{args.shard_index}.SUMMARY.json"
    if args.merge:
        shard_paths = [OUT_N3 / f"E2_N_N3_QWEN_DIRECT_FULL.shard{i}.jsonl" for i in range(args.shard_count)]
        shard_rows = [row for path in shard_paths for row in load_jsonl(path)]
        if len(shard_rows) != len(items) or len({row.get("query_id") for row in shard_rows}) != len(items):
            raise RuntimeError(f"shard merge population mismatch: {len(shard_rows)} rows for {len(items)} items")
        if any(row.get("status") != "ok" for row in shard_rows):
            raise RuntimeError("cannot merge shard traces containing errors")
        shard_rows.sort(key=lambda row: row["query_id"])
        write_jsonl(OUT_N3 / "E2_N_N3_QWEN_DIRECT_FULL.jsonl", shard_rows)
        merged = {
            "stage": "N3_FULL_HISTORICAL",
            "model": QWEN_ID,
            "revision": QWEN_REV,
            "rows": len(shard_rows),
            "expected_rows": len(items),
            "unique_queries": len({row["query_id"] for row in shard_rows}),
            "ok_rows": sum(row.get("status") == "ok" for row in shard_rows),
            "physical_forward_evidence_rows": sum(row.get("physical_forward_evidence") is True for row in shard_rows),
            "total_physical_forward_calls": sum(int(row.get("physical_forward_calls", 0)) for row in shard_rows),
            "input_and_response_hash_rows": sum(bool(row.get("actual_model_input_sha256") and row.get("raw_response_sha256")) for row in shard_rows),
            "valid_unique_20_candidate_rows": sum(len(row.get("candidate_ids", [])) == 20 and len(set(row.get("candidate_ids", []))) == 20 for row in shard_rows),
            "gold_in_inference_input_rows": sum(row.get("gold_in_inference_input") is True for row in shard_rows),
            "cache_reuse_rows": sum(row.get("cache_reuse") is True for row in shard_rows),
            "direct_checkpoint": True,
            "shard_count": args.shard_count,
            "shard_paths": [{"path": str(path.relative_to(ROOT)), "sha256": sha_file(path), "rows": len(load_jsonl(path))} for path in shard_paths],
            "resident_service_restarted": False,
            "resident_service_modified": False,
            "jev_calls": 0,
            "paid_api_calls": 0,
            "trace_sha256": sha_file(OUT_N3 / "E2_N_N3_QWEN_DIRECT_FULL.jsonl"),
            "status": "PASS",
        }
        (OUT_N3 / "E2_N_N3_QWEN_DIRECT_FULL_SUMMARY.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(merged, ensure_ascii=False, sort_keys=True))
        return 0
    if args.shard_count > 1:
        items = [item for pos, item in enumerate(items) if pos % args.shard_count == args.shard_index]
    _, summary = run_direct(items, args.stage, output, gpu_snapshot())
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
