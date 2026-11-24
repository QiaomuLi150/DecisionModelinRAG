#!/usr/bin/env python3
"""Native Intern-Decision-4B adapter for E2-N smoke/CAL/historical traces."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HF_ROOT = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub")
REV = "0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd"
MODEL_ID = "internlm/Intern-Decision-4B"
MODEL_PATH = HF_ROOT / "models--internlm--Intern-Decision-4B" / "snapshots" / REV
TEMPERATURE = 1.99241824
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


def load_official():
    spec = importlib.util.spec_from_file_location("intern_decision_official", MODEL_PATH / "inference.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import official Intern-Decision inference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tensor_fingerprint(value):
    if not hasattr(value, "detach"):
        return repr(value)
    tensor = value.detach().to("cpu").contiguous()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": sha_bytes(tensor.numpy().tobytes())}


def run(stage: str, run_items: list[dict], output: Path) -> dict:
    import torch
    if not MODEL_PATH.is_dir():
        raise RuntimeError(f"missing Intern-Decision checkpoint: {MODEL_PATH}")
    module = load_official()
    backend = module.HFBackend(MODEL_PATH, max_length=8192, device="cuda", dtype="bfloat16", attn_implementation="sdpa")
    original_forward = backend.model.forward
    forward_count = {"value": 0}
    captured = []

    def counted_forward(*args, **kwargs):
        forward_count["value"] += 1
        captured.append({"args": [tensor_fingerprint(x) for x in args], "kwargs": {k: tensor_fingerprint(v) for k, v in kwargs.items()}})
        return original_forward(*args, **kwargs)

    backend.model.forward = counted_forward
    view_tokenizer = backend.tokenizer
    existing = {row["query_id"]: row for row in load_jsonl(output) if row.get("status") == "ok"}
    rows = []
    for index, item in enumerate(run_items, 1):
        if item["query_id"] in existing:
            rows.append(existing[item["query_id"]])
            continue
        views = make_views(item, view_tokenizer)
        if len(views) != 20 or len({v.doc_id for v in views}) != 20:
            raise RuntimeError(f"invalid frozen slate {item['query_id']}")
        ids = [v.doc_id for v in views]
        row = {"stage": "N1" if stage == "smoke" else "N2_CAL" if stage == "n2" else "N3_FULL_HISTORICAL", "seed": SEED, "query_id": item["query_id"], "cohort": item.get("cohort", "cal" if stage == "n2" else "historical"), "dataset": item["dataset"], "model_id": MODEL_ID, "revision": REV, "adapter": "official Intern-Decision-4B HFBackend", "formulation": "native_joint_binary", "score_contract": "JOINT_BINARY_RELEVANCE:v2", "candidate_ids": ids, "selection_text_sha256": [v.selection_text_sha256 for v in views], "score_direction": "higher_is_more_relevant", "temperature": TEMPERATURE, "cache_reuse": False, "cache_provenance": "not_applicable_new_direct_checkpoint_request", "gold_in_inference_input": False, "physical_gpu_id": os.environ.get("E2_N_PHYSICAL_GPU_ID"), "visible_cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES"), "status": "error", "real_inference": False}
        question_fields = {}
        for i, view in enumerate(views):
            question_fields[f"relevance_{i:02d}_{view.doc_id}"] = {"type": "noul", "instructions": f"For candidate_id={view.doc_id}, does the candidate passage contain factual evidence useful for answering the question?", "criteria": {"true": "The candidate contains useful evidence, including an intermediate multi-hop link.", "false": "The candidate does not contain useful evidence."}}
        state = {"question": views[0].question_view, "candidates": [{"candidate_id": v.doc_id, "original_rank": v.original_rank, "title": v.title, "passage": v.passage_view} for v in views]}
        request = {"id": f"{item['query_id']}::intern_joint", "state": state, "questions": question_fields}
        row["request_hash"] = sha_json({"query_id": item["query_id"], "candidate_ids": ids, "selection_text_sha256": row["selection_text_sha256"], "temperature": TEMPERATURE, "contract": "JOINT_BINARY_RELEVANCE:v2"})
        before = forward_count["value"]
        captured_before = len(captured)
        try:
            compiled, logits, length, inference_ms = backend.score(request)
            scores = []
            for field_index, field in enumerate(compiled.fields):
                token_ids = [backend.tokenizer.encode(symbol, add_special_tokens=False)[0] for symbol in compiled.symbols[field]]
                probs = torch.softmax(logits[field_index, token_ids].float(), dim=-1).cpu().tolist()
                # Intern's native API applies this fixed temperature exactly once.
                scaled = torch.tensor([math.log(max(p, 1e-45)) / TEMPERATURE for p in probs]).softmax(-1).tolist()
                scores.append(float(scaled[1]))
            actual = captured[captured_before:]
            if len(scores) != 20 or any(not math.isfinite(x) for x in scores):
                raise RuntimeError("Intern-Decision returned invalid K20 score vector")
            row.update({"status": "ok", "real_inference": True, "scores": scores, "raw_response_sha256": sha_json(scores), "actual_model_input_sha256": sha_json(actual), "physical_forward_calls": forward_count["value"] - before, "physical_forward_evidence": bool(actual) and forward_count["value"] > before, "input_forward_batches": len(actual), "input_tokens": length, "duration_ms": float(inference_ms), "cache_source": "none", "context_overflow": False})
        except Exception as exc:
            row.update({"error_type": f"{type(exc).__name__}: {exc}", "physical_forward_calls": forward_count["value"] - before, "physical_forward_evidence": bool(captured[captured_before:])})
        rows.append(row)
        if index % 10 == 0 or index == len(run_items):
            write_jsonl(output, rows)
            print(f"intern {stage} {index}/{len(run_items)} ok={sum(r.get('status') == 'ok' for r in rows)}", flush=True)
    write_jsonl(output, rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    strict = [r for r in ok if r.get("physical_forward_evidence") is True and int(r.get("physical_forward_calls", 0)) > 0]
    summary = {"stage": "N1" if stage == "smoke" else "N2_CAL" if stage == "n2" else "N3_FULL_HISTORICAL", "model": MODEL_ID, "revision": REV, "rows": len(rows), "expected_rows": len(run_items), "unique_queries": len({r.get("query_id") for r in rows}), "ok_rows": len(ok), "physical_forward_evidence_rows": len(strict), "total_physical_forward_calls": sum(int(r.get("physical_forward_calls", 0)) for r in strict), "input_and_response_hash_rows": sum(bool(r.get("actual_model_input_sha256") and r.get("raw_response_sha256")) for r in ok), "valid_unique_20_candidate_rows": sum(len(r.get("candidate_ids", [])) == 20 and len(set(r.get("candidate_ids", []))) == 20 for r in ok), "gold_in_inference_input_rows": sum(r.get("gold_in_inference_input") is True for r in rows), "cache_reuse_rows": sum(r.get("cache_reuse") is True for r in rows), "temperature": TEMPERATURE, "gpu_snapshot_after": gpu_snapshot(), "trace_sha256": sha_file(output), "jev_calls": 0, "paid_api_calls": 0, "status": "PASS" if len(ok) == len(run_items) and len(strict) == len(run_items) else "BLOCKED"}
    del backend
    torch.cuda.empty_cache()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "n2", "n3"), required=True)
    args = parser.parse_args()
    run_items = load_items(args.stage)
    folder = ROOT / "results/non_jev/v2/e2_n" / ("n1" if args.stage == "smoke" else "n2" if args.stage == "n2" else "n3")
    stem = "E2_N_N1_INTERN_DECISION_SMOKE" if args.stage == "smoke" else "E2_N_N2_INTERN_DECISION_CAL" if args.stage == "n2" else "E2_N_N3_INTERN_DECISION_FULL"
    output = folder / f"{stem}.jsonl"
    summary = run(args.stage, run_items, output)
    (folder / f"{stem}_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
