#!/usr/bin/env python3
"""Native label-blind runner for the frozen prospective validation.

One process owns one physical GPU and loads both native selector checkpoints
when the preflight memory gate permits it. It never reads qrels and never
retries a request. Pass 1 records quality-capable outputs; Pass 2 records
the same calls under reversed policy order for timing-order sensitivity.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/frozen_prospective_validation"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
PPLX = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")
JINA = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/{JINA_REV}")
PROTOCOL_SHA = "06ea012a3bd555454fe6a498ecada087f6affcb62bcab7915fe37a34d62df92c"


def canon(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_obj(value: Any) -> str:
    return hashlib.sha256(canon(value).encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def request_obj(query: str, docs: list[dict[str, str]]) -> dict[str, Any]:
    codes = [f"candidate_{i:03d}" for i in range(len(docs))]
    return {
        "state": f"Query: {query}\nEvaluate the supplied candidate passages using the option descriptions.",
        "question": {
            "type": "choice",
            "instructions": f"Select the candidate passage that best answers the query: {query}",
            "criteria": {codes[i]: f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(docs)},
        },
    }


def top5(ids: list[str], scores: list[float]) -> list[str]:
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i, ids[i]))[:5]]


def classify(exc: Exception) -> str:
    s = str(exc).lower()
    if "out of memory" in s or "cuda oom" in s:
        return "CUDA_OOM"
    if "timeout" in s:
        return "TIMEOUT"
    if "token" in s or "length" in s or "invalid" in s:
        return "INPUT_OR_OUTPUT_VALIDATION"
    return type(exc).__name__


def gpu_snapshot() -> dict[str, Any]:
    try:
        text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT)
        devices = []
        for line in text.splitlines():
            vals = [x.strip() for x in line.split(",")]
            if len(vals) >= 6:
                devices.append({"index": vals[0], "uuid": vals[1], "memory_total_mib": vals[2], "memory_used_mib": vals[3], "memory_free_mib": vals[4], "utilization_gpu": vals[5]})
        return {"status": "OK", "devices": devices, "captured_ns": time.time_ns()}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "error_class": type(exc).__name__, "error": str(exc), "captured_ns": time.time_ns()}


def load_inputs():
    slates = {x["canonical_id"]: x for x in read_jsonl(OUT / "ARGUANA_FROZEN_TOP20_SLATES.jsonl")}
    prepared = {L: {x["canonical_id"]: x for x in read_jsonl(OUT / f"prepared_L{L}.jsonl")} for L in (4096, 6144, 8192)}
    length = {x["canonical_id"]: x for x in read_jsonl(OUT / "ARGUANA_QUERY_LENGTHS.jsonl")}
    if len(slates) != 1406 or any(len(x) != 1406 for x in prepared.values()) or len(length) != 1406:
        raise RuntimeError("FROZEN_INPUT_COVERAGE_FAILURE")
    return slates, prepared, length


def load_models(torch):
    sys.path.insert(0, str(PPLX / "source/src"))
    from transformers import AutoProcessor, AutoModel, AutoTokenizer
    from autojev.model import DecisionModel
    proc = AutoProcessor.from_pretrained(str(PPLX), local_files_only=True)
    pplx = DecisionModel(PPLX, device="cuda:0", max_image_pixels=512 * 512)
    tokenizer = AutoTokenizer.from_pretrained(str(JINA), local_files_only=True)
    jina = AutoModel.from_pretrained(str(JINA), dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda:0").eval()
    torch.cuda.synchronize()
    return proc, pplx, tokenizer, jina


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--phase", choices=("dry", "pass1", "pass2"), required=True)
    ap.add_argument("--dry-count", type=int, default=20)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    OUT.joinpath("raw").mkdir(parents=True, exist_ok=True)
    protocol = OUT / "PROTOCOL_FROZEN.md"
    if sha_file(protocol) != PROTOCOL_SHA:
        raise RuntimeError("PROTOCOL_HASH_MISMATCH")
    slates, prepared, lengths = load_inputs()
    all_ids = sorted(slates)
    if args.phase == "dry":
        selected_ids = all_ids[:args.dry_count]
    else:
        selected_ids = [cid for i, cid in enumerate(all_ids) if i % args.shards == args.shard]
    policy_names = ["pplx_single_4096", "pplx_single_6144", "jina_full_text", "jina_trunc_8192"]
    out = OUT / "raw" / f"native_{args.phase}_gpu{args.gpu}_shard{args.shard}_of{args.shards}.jsonl"
    if out.exists() or out.with_suffix(out.suffix + ".part").exists():
        raise RuntimeError(f"OUTPUT_EXISTS:{out}")
    import torch
    torch.cuda.set_device(0)
    pre = gpu_snapshot()
    load_t0 = time.perf_counter()
    proc, pplx, jina_tok, jina = load_models(torch)
    load_ms = (time.perf_counter() - load_t0) * 1000.0
    post_load = gpu_snapshot()
    source_model = {"pplx_revision": PPLX_REV, "jina_revision": JINA_REV, "protocol_sha256": PROTOCOL_SHA}
    random_order = args.phase == "pass1"
    rows: list[dict[str, Any]] = []
    contention: list[dict[str, Any]] = []
    session_t0 = time.perf_counter()
    with out.with_suffix(out.suffix + ".part").open("w", encoding="utf-8") as f:
        for local_ix, cid in enumerate(selected_ids):
            s = slates[cid]
            by_policy = {"pplx_single_4096": prepared[4096][cid]["budgeted_candidates"], "pplx_single_6144": prepared[6144][cid]["budgeted_candidates"], "jina_full_text": s["candidates"], "jina_trunc_8192": prepared[8192][cid]["budgeted_candidates"]}
            order = list(policy_names)
            if random_order:
                random.Random(731 + int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16)).shuffle(order)
            else:
                order = list(reversed(order))
            if args.phase == "dry":
                order = policy_names
            for pidx, policy in enumerate(order):
                docs = [{"candidate_id": str(d["candidate_id"]), "content": str(d["content"])} for d in by_policy[policy]]
                ids = [d["candidate_id"] for d in docs]
                rec: dict[str, Any] = {
                    "dataset": "arguana", "canonical_id": cid, "query_id": s["query_id"], "length_tercile": lengths[cid]["length_tercile"], "policy": policy, "phase": args.phase, "policy_order_index": pidx, "candidate_ids": ids, "source_slate_sha256": s["slate_sha256"], "input_sha256": sha_obj(request_obj(str(s["query"]), docs)), "model_revision": PPLX_REV if policy.startswith("pplx") else JINA_REV, "gpu_index_requested": args.gpu, "gpu_visible_index": 0, "status": "FAIL", "valid": False, "selected_ids": [], "scores": [], "n_calls": 0, "physical_forward_count": 0, "error_class": None, "error": None, "latency_ms": None, "peak_memory_allocated_mib": None, "labels_joined": False, "input_label_blind": True, "quality_eligible": args.phase == "pass1",
                }
                t0 = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                try:
                    if len(docs) != 20 or len(set(ids)) != 20:
                        raise ValueError("INVALID_TOP20_INPUT")
                    if policy.startswith("pplx"):
                        scores = [float(v) for v in pplx.predict([request_obj(str(s["query"]), docs)], batch_size=1)[0]]
                    else:
                        result = jina.rerank(str(s["query"]), [d["content"] for d in docs])
                        score_by_index = {int(z["index"]): float(z["relevance_score"]) for z in result}
                        scores = [score_by_index[i] for i in range(20)]
                    if len(scores) != 20 or not all(math.isfinite(x) for x in scores):
                        raise ValueError("INVALID_SCORE_VECTOR")
                    chosen = top5(ids, scores)
                    if len(chosen) != 5 or len(set(chosen)) != 5:
                        raise ValueError("INVALID_TOP5")
                    rec.update({"status": "PASS", "valid": True, "selected_ids": chosen, "scores": scores, "n_calls": 1, "physical_forward_count": 1})
                    if args.phase == "pass2":
                        rec["quality_eligible"] = False
                except Exception as exc:
                    rec.update({"error_class": classify(exc), "error": f"{type(exc).__name__}: {exc}"})
                torch.cuda.synchronize()
                rec["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
                rec["peak_memory_allocated_mib"] = round(float(torch.cuda.max_memory_allocated()) / 2**20, 3)
                rows.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                f.flush()
            if (local_ix + 1) % 100 == 0:
                snap = gpu_snapshot()
                snap.update({"phase": args.phase, "gpu_index_requested": args.gpu, "completed_query_count": local_ix + 1, "elapsed_s": time.perf_counter() - session_t0})
                contention.append(snap)
                print(json.dumps({"phase": args.phase, "gpu": args.gpu, "completed_queries": local_ix + 1, "total_queries": len(selected_ids), "records": len(rows)}), flush=True)
    out.with_suffix(out.suffix + ".part").replace(out)
    raw_sha = sha_file(out)
    out.with_suffix(".SUMMARY.json").write_text(json.dumps({"status": "PASS", "phase": args.phase, "gpu_index_requested": args.gpu, "shard": args.shard, "shards": args.shards, "records": len(rows), "queries": len(selected_ids), "valid": sum(x["valid"] for x in rows), "failed": sum(not x["valid"] for x in rows), "raw_sha256": raw_sha, "pre_gpu": pre, "post_model_load_gpu": post_load, "model_load_ms": load_ms, "labels_joined": False, "policies": policy_names, "no_retries": True}, indent=2, sort_keys=True) + "\n")
    (OUT / "gpu_contention").mkdir(exist_ok=True)
    (OUT / "gpu_contention" / f"{args.phase}_gpu{args.gpu}_shard{args.shard}.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in contention), encoding="utf-8")
    print(json.dumps({"status": "PASS", "phase": args.phase, "gpu": args.gpu, "shard": args.shard, "queries": len(selected_ids), "records": len(rows), "raw_sha256": raw_sha}, indent=2), flush=True)
    del pplx, jina, proc, jina_tok
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
