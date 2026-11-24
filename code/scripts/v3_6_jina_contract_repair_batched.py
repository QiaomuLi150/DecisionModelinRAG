#!/usr/bin/env python3
"""Reproduce the historical v1.5 Jina batch-8 scoring contract."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / "results/v3_6_essential/followup_v2_contract_repair"
MANIFEST = REPAIR / "JINA_CONTRACT_REPAIR_INPUT_MANIFEST.jsonl"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
JINA_PATH = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots") / JINA_REV


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def stable_top(ids, scores):
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i, ids[i]))[:5]]


def main() -> int:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    if args.batch_size != 8:
        raise SystemExit("This repair is frozen to historical batch_size=8")
    if not MANIFEST.exists():
        raise SystemExit("run contract-repair preflight first")
    repair = load_module(ROOT / "scripts/v3_6_jina_contract_repair.py", "v36_repair")
    old_runner = load_module(ROOT / "scripts/v1_5_c1_gpu3_runner.py", "v15_runner")
    rows = repair.read_jsonl(MANIFEST)
    receipt = json.loads((REPAIR / "CONTRACT_REPAIR_PREFLIGHT.json").read_text())
    if receipt.get("status") != "PASS":
        raise SystemExit("contract-repair preflight is not PASS")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from transformers import AutoModel, AutoTokenizer
    protocol = {
        "protocol_id": "v3.6_jina_contract_repair_batch8_v1",
        "historical_scoring_module": "scripts/v1_5_c1_gpu3_runner.py::score_many",
        "batch_size": 8,
        "jina_revision": JINA_REV,
        "input_manifest_sha256": repair.sha_file(MANIFEST),
        "input_contract": "query + candidate_ids + candidate_texts",
        "native_path": "format_docs_prompts_func -> tokenizer left padding -> model.forward -> outputs.scores",
        "selection": "stable descending score, then original index, then candidate_id; k=5",
        "labels_joined": False, "protected_holdouts_accessed": False,
        "no_retries": True, "gpu_requested": args.gpu, "frozen_before_inference": True,
    }
    repair.write_json(REPAIR / "PROTOCOL_BATCHED.json", protocol)
    tokenizer = AutoTokenizer.from_pretrained(str(JINA_PATH), local_files_only=True)
    model = AutoModel.from_pretrained(str(JINA_PATH), dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda:0").eval()
    out_rows = []
    physical_batches = 0
    batch_errors = 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        valid = [r for r in batch if r["eligible_for_native_replay"]]
        recs = {}
        for item in batch:
            recs[item["canonical_id"]] = {
                "canonical_id": item["canonical_id"], "domain": item["domain"], "query_id": item["query_id"],
                "policy": "jina_single_8192_contract_repair_batch8", "order_id": "original",
                "candidate_ids": item["candidate_ids"], "request_sha256": item["request_sha256"],
                "historical_request_sha256": item["historical_request_sha256"], "input_contract": "candidate_texts",
                "model_revision": JINA_REV, "labels_joined": False, "protected_holdouts_accessed": False,
                "batch_index": start // args.batch_size, "batch_size": len(valid), "native_forward_count": 0,
                "status": "FAIL", "valid": False, "selected_ids": [], "scores": [], "error_class": None, "error": None,
            }
            if not item["eligible_for_native_replay"]:
                recs[item["canonical_id"]].update({"error_class": "HISTORICAL_SOURCE_FAILURE", "error": "preserved historical failure; no retry", "replayed_historical_failure": True})
        if valid:
            requests = [(item["query"], [{"candidate_id": cid, "content": text} for cid, text in zip(item["candidate_ids"], item["candidate_texts"])]) for item in valid]
            started = time.perf_counter()
            try:
                torch.cuda.reset_peak_memory_stats()
                scored = old_runner.score_many(model, tokenizer, torch, requests)
                torch.cuda.synchronize()
                physical_batches += 1
                elapsed = round((time.perf_counter() - started) * 1000, 3)
                peak = round(float(torch.cuda.max_memory_allocated()) / 2**20, 3)
                for item, (scores, _meta) in zip(valid, scored):
                    selected = stable_top(item["candidate_ids"], scores)
                    recs[item["canonical_id"]].update({
                        "status": "PASS", "valid": len(selected) == 5 and len(set(selected)) == 5,
                        "selected_ids": selected, "scores": scores, "native_forward_count": 1,
                        "batch_elapsed_ms": elapsed, "peak_memory_allocated_mib": peak,
                        "selected_ids_exact_match": selected == item["historical_selected_ids"],
                        "selected_set_exact_match": set(selected) == set(item["historical_selected_ids"]),
                    })
            except Exception as exc:
                batch_errors += 1
                for item in valid:
                    recs[item["canonical_id"]].update({"error_class": type(exc).__name__, "error": str(exc)})
        out_rows.extend(recs[item["canonical_id"]] for item in batch)
        if (start + args.batch_size) % 80 == 0 or start + args.batch_size >= len(rows):
            print(json.dumps({"status": "RUNNING", "completed": min(start + args.batch_size, len(rows)), "total": len(rows), "physical_batches": physical_batches}), flush=True)
    out = REPAIR / "raw" / f"jina_contract_repair_batch8_gpu{args.gpu}.jsonl"
    repair.write_jsonl(out, out_rows)
    repair.write_json(out.with_suffix(".meta.json"), {"status": "PASS", "rows": len(out_rows), "batch_size": args.batch_size, "physical_batches": physical_batches, "batch_errors": batch_errors, "native_rows": sum(r["native_forward_count"] for r in out_rows), "sha256": repair.sha_file(out), "protocol_sha256": repair.sha_file(REPAIR / "PROTOCOL_BATCHED.json")})
    print(json.dumps({"status": "PASS", "output": str(out.relative_to(ROOT)), "rows": len(out_rows), "physical_batches": physical_batches, "batch_errors": batch_errors}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
