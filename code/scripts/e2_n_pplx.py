#!/usr/bin/env python3
"""Native pplx-decider-v1.1-27b feasibility/smoke runner.

The checkpoint is loaded only from its downloaded local snapshot.  Its native
noncausal/readout interface is preserved as a single 20-choice slate decision;
this is not relabeled as pointwise relevance.
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
CHECKPOINT = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/14c9d90137962d300f3de95633f7b6535e80c5ce")
REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
MODEL_ID = "perplexity-ai/pplx-decider-v1.1-27b"
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


def gpu_snapshot() -> dict:
    cmd = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    return {"status": "PASS" if p.returncode == 0 else "UNAVAILABLE", "command": " ".join(cmd), "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_items(stage: str) -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_m1_experiment import load_full_items, load_local_cal_items
    from e2_n_n1_smoke import load_items as smoke_items
    if stage == "smoke":
        return smoke_items()
    return load_local_cal_items() if stage == "n2" else load_full_items()


def make_views(item: dict, tokenizer):
    sys.path.insert(0, str(ROOT / "src"))
    from decisionrag.selection import make_views
    return [make_views(item["question"], candidate, tokenizer) for candidate in item["candidates"]]


def tensor_fingerprint(value):
    if not hasattr(value, "detach"):
        return repr(value)
    tensor = value.detach().to("cpu").contiguous()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": sha_bytes(tensor.numpy().tobytes())}


def run(stage: str, run_items: list[dict], output: Path) -> dict:
    import torch
    if not CHECKPOINT.is_dir():
        raise RuntimeError(f"missing PPLX checkpoint {CHECKPOINT}")
    sys.path.insert(0, str(CHECKPOINT / "source" / "src"))
    from autojev.model import DecisionModel
    model = DecisionModel(CHECKPOINT, device="cuda", max_image_pixels=512 * 512)
    tokenizer = model.processor.tokenizer
    original_forward = model.backbone.forward
    forward_count = {"value": 0}
    captured = []

    def counted_forward(*args, **kwargs):
        forward_count["value"] += 1
        captured.append({"args": [tensor_fingerprint(v) for v in args], "kwargs": {k: tensor_fingerprint(v) for k, v in kwargs.items()}})
        return original_forward(*args, **kwargs)

    model.backbone.forward = counted_forward
    existing = {row["query_id"]: row for row in load_jsonl(output) if row.get("status") == "ok"}
    rows = []
    for index, item in enumerate(run_items, 1):
        if item["query_id"] in existing:
            rows.append(existing[item["query_id"]])
            continue
        v = make_views(item, tokenizer)
        if len(v) != 20 or len({x.doc_id for x in v}) != 20:
            raise RuntimeError(f"invalid frozen slate {item['query_id']}")
        ids = [x.doc_id for x in v]
        criteria = {doc_id: f"Title: {view.title}\nPassage: {view.passage_view}" for doc_id, view in zip(ids, v)}
        question = {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {v[0].question_view}", "criteria": criteria}
        state = f"Query: {v[0].question_view}\nEvaluate the supplied candidate passages using the option descriptions."
        request = {"state": state, "question": question}
        row = {"stage": "N1" if stage == "smoke" else "N2_CAL" if stage == "n2" else "N3_FULL_HISTORICAL", "seed": SEED, "query_id": item["query_id"], "cohort": item.get("cohort", "cal" if stage == "n2" else "historical"), "dataset": item["dataset"], "model_id": MODEL_ID, "revision": REV, "adapter": "official autojev DecisionModel native noncausal readout", "formulation": "native_slate_categorical", "score_contract": "NATIVE_SLATE_CATEGORICAL:v1", "candidate_ids": ids, "selection_text_sha256": [x.selection_text_sha256 for x in v], "score_direction": "higher_is_more_relevant", "temperature": float(model.temperature), "attention_mode": model.attention_mode, "pooling": model.pooling, "cache_reuse": False, "cache_provenance": "not_applicable_new_direct_checkpoint_request", "gold_in_inference_input": False, "physical_gpu_id": os.environ.get("E2_N_PHYSICAL_GPU_ID"), "visible_cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES"), "status": "error", "real_inference": False}
        row["request_hash"] = sha_json({"query_id": item["query_id"], "candidate_ids": ids, "selection_text_sha256": row["selection_text_sha256"], "request": request, "contract": "NATIVE_SLATE_CATEGORICAL:v1"})
        before = forward_count["value"]
        captured_before = len(captured)
        try:
            started = time.perf_counter_ns()
            probabilities = model.predict([request], batch_size=1)[0]
            actual = captured[captured_before:]
            scores = [float(value) for value in probabilities]
            if len(scores) != 20 or any(not math.isfinite(x) for x in scores):
                raise RuntimeError("PPLX native readout returned invalid 20-choice vector")
            row.update({"status": "ok", "real_inference": True, "scores": scores, "raw_response_sha256": sha_json(probabilities), "actual_model_input_sha256": sha_json(actual), "physical_forward_calls": forward_count["value"] - before, "physical_forward_evidence": bool(actual) and forward_count["value"] > before, "input_forward_batches": len(actual), "duration_ms": (time.perf_counter_ns() - started) / 1_000_000.0, "cache_source": "none", "input_tokens": "recorded_by_model_prepare"})
        except Exception as exc:
            row.update({"error_type": f"{type(exc).__name__}: {exc}", "physical_forward_calls": forward_count["value"] - before, "physical_forward_evidence": bool(captured[captured_before:])})
        rows.append(row)
        if index % 10 == 0 or index == len(run_items):
            write_jsonl(output, rows)
            print(f"pplx {stage} {index}/{len(run_items)} ok={sum(r.get('status') == 'ok' for r in rows)}", flush=True)
    write_jsonl(output, rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    strict = [r for r in ok if r.get("physical_forward_evidence") is True and int(r.get("physical_forward_calls", 0)) > 0]
    summary = {"stage": "N1" if stage == "smoke" else "N2_CAL" if stage == "n2" else "N3_FULL_HISTORICAL", "model": MODEL_ID, "revision": REV, "rows": len(rows), "expected_rows": len(run_items), "unique_queries": len({r.get("query_id") for r in rows}), "ok_rows": len(ok), "physical_forward_evidence_rows": len(strict), "total_physical_forward_calls": sum(int(r.get("physical_forward_calls", 0)) for r in strict), "input_and_response_hash_rows": sum(bool(r.get("actual_model_input_sha256") and r.get("raw_response_sha256")) for r in ok), "valid_unique_20_candidate_rows": sum(len(r.get("candidate_ids", [])) == 20 and len(set(r.get("candidate_ids", []))) == 20 for r in ok), "gold_in_inference_input_rows": sum(r.get("gold_in_inference_input") is True for r in rows), "cache_reuse_rows": sum(r.get("cache_reuse") is True for r in rows), "gpu_snapshot_after": gpu_snapshot(), "trace_sha256": sha_file(output), "jev_calls": 0, "paid_api_calls": 0, "status": "PASS" if len(ok) == len(run_items) and len(strict) == len(run_items) else "BLOCKED"}
    del model
    torch.cuda.empty_cache()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "n2", "n3"), required=True)
    args = parser.parse_args()
    run_items = load_items(args.stage)
    folder = ROOT / "results/non_jev/v2/e2_n" / ("n1" if args.stage == "smoke" else "n2" if args.stage == "n2" else "n3")
    stem = "E2_N_N1_PPLX_DECIDER_SMOKE" if args.stage == "smoke" else "E2_N_N2_PPLX_DECIDER_CAL" if args.stage == "n2" else "E2_N_N3_PPLX_DECIDER_FULL"
    output = folder / f"{stem}.jsonl"
    summary = run(args.stage, run_items, output)
    (folder / f"{stem}_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
