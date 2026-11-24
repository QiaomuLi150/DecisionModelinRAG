#!/usr/bin/env python3
"""Label-blind native runner for the CQADupStack v2 frozen validation.

This runner reads only frozen slates and prepared text.  It never opens qrels,
never retries a request, and records failures rather than filling them in.
One process owns one physical GPU and loads the two pinned native selectors.
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
OUT = ROOT / "results/v3_6_essential/frozen_prospective_validation_v2"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
PPLX = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")
JINA = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/{JINA_REV}")
PROTOCOL_SHA = "f15cb0f6001327bfe47e0b1d605f73f7beefc3c6a5fc195689cd5ca1652ce86e"
POLICIES = ("pplx_single_4096", "pplx_single_6144", "jina_full_text", "jina_trunc_8192")


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
        text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.STDOUT,
        )
        devices = []
        for line in text.splitlines():
            vals = [x.strip() for x in line.split(",")]
            if len(vals) >= 6:
                devices.append({"index": vals[0], "uuid": vals[1], "memory_total_mib": vals[2], "memory_used_mib": vals[3], "memory_free_mib": vals[4], "utilization_gpu": vals[5]})
        return {"status": "OK", "devices": devices, "captured_ns": time.time_ns()}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "error_class": type(exc).__name__, "error": str(exc), "captured_ns": time.time_ns()}


def load_inputs() -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, dict[str, Any]]], dict[str, dict[str, Any]], set[str], list[str]]:
    slates = {x["canonical_id"]: x for x in read_jsonl(OUT / "CQADUPSTACK_FROZEN_TOP20_SLATES.jsonl")}
    prepared = {L: {x["canonical_id"]: x for x in read_jsonl(OUT / f"prepared_L{L}.jsonl")} for L in (4096, 6144, 8192)}
    lengths = {x["canonical_id"]: x for x in read_jsonl(OUT / "QUERY_LENGTHS.jsonl")}
    dry = {x["canonical_id"] for x in read_jsonl(OUT / "DRY_RUN_IDS.jsonl")}
    latency = [x["canonical_id"] for x in read_jsonl(OUT / "LATENCY_COHORT_IDS.jsonl")]
    if len(slates) != 13145 or any(len(x) != 13145 for x in prepared.values()) or len(lengths) != 13145 or len(dry) != 20 or len(latency) != 2000 or dry.intersection(latency):
        raise RuntimeError("FROZEN_INPUT_COVERAGE_FAILURE")
    if sha_file(OUT / "PROTOCOL_FROZEN.md") != PROTOCOL_SHA:
        raise RuntimeError("PROTOCOL_HASH_MISMATCH")
    return slates, prepared, lengths, dry, latency


def load_models(torch, model_group: str):
    if not PPLX.exists() or not JINA.exists():
        raise RuntimeError("PINNED_CHECKPOINT_MISSING")
    sys.path.insert(0, str(PPLX / "source/src"))
    from transformers import AutoProcessor, AutoModel, AutoTokenizer
    from autojev.model import DecisionModel
    proc = pplx = jina_tok = jina = None
    if model_group in ("paired", "pplx"):
        proc = AutoProcessor.from_pretrained(str(PPLX), local_files_only=True)
        pplx = DecisionModel(PPLX, device="cuda:0", max_image_pixels=512 * 512)
    if model_group in ("paired", "jina"):
        jina_tok = AutoTokenizer.from_pretrained(str(JINA), local_files_only=True)
        jina = AutoModel.from_pretrained(str(JINA), dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda:0").eval()
    torch.cuda.synchronize()
    return proc, pplx, jina_tok, jina


def choose_ids(slates, dry, latency, phase, shard, shards):
    if phase == "dry":
        ids = sorted(dry)
    elif phase == "passA":
        ids = [x for x in sorted(slates) if x not in dry]
    elif phase == "passB":
        ids = list(latency)
    else:
        raise RuntimeError(f"UNKNOWN_PHASE:{phase}")
    return [x for i, x in enumerate(ids) if i % shards == shard]


def run_one(policy, query, docs, pplx, jina, torch) -> tuple[list[float], int]:
    ids = [str(d["candidate_id"]) for d in docs]
    if len(docs) != 20 or len(set(ids)) != 20:
        raise ValueError("INVALID_TOP20_INPUT")
    if policy.startswith("pplx"):
        scores = [float(v) for v in pplx.predict([request_obj(query, docs)], batch_size=1)[0]]
    else:
        result = jina.rerank(query, [d["content"] for d in docs])
        score_by_index = {int(z["index"]): float(z["relevance_score"]) for z in result}
        scores = [score_by_index[i] for i in range(20)]
    if len(scores) != 20 or not all(math.isfinite(x) for x in scores):
        raise ValueError("INVALID_SCORE_VECTOR")
    chosen = top5(ids, scores)
    if len(chosen) != 5 or len(set(chosen)) != 5 or not set(chosen).issubset(set(ids)):
        raise ValueError("INVALID_TOP5")
    return scores, 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--phase", choices=("dry", "passA", "passB"), required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--pass-index", type=int, default=1)
    ap.add_argument("--model-group", choices=("paired", "pplx", "jina"), default="paired")
    ap.add_argument("--resume-from", type=Path, default=None)
    ap.add_argument("--run-label", default=None)
    args = ap.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("INVALID_SHARD")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    OUT.joinpath("raw").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("gpu_contention").mkdir(exist_ok=True)
    slates, prepared, lengths, dry, latency = load_inputs()
    selected_ids = choose_ids(slates, dry, latency, args.phase, args.shard, args.shards)
    resume_keys = set()
    if args.resume_from is not None:
        completed = set()
        for line in args.resume_from.open(encoding="utf-8"):
            if line.strip():
                old = json.loads(line)
                resume_keys.add((old["canonical_id"], old.get("policy"), old.get("anchor_stage")))
                completed.add(old["canonical_id"])
        if args.phase != "passB":
            selected_ids = [cid for cid in selected_ids if cid not in completed]
    if args.phase == "passB" and args.model_group != "paired":
        raise ValueError("PASSB_REQUIRES_PAIRED_MODEL_GROUP")
    active_policies = {"paired": POLICIES, "pplx": ("pplx_single_4096", "pplx_single_6144"), "jina": ("jina_full_text", "jina_trunc_8192")}[args.model_group]
    label = f"_{args.run_label}" if args.run_label else ""
    suffix = f"{args.phase}_{args.model_group}{label}_gpu{args.gpu}_shard{args.shard}_of{args.shards}"
    if args.phase == "passB":
        suffix += f"_pass{args.pass_index}"
    out = OUT / "raw" / f"native_{suffix}.jsonl"
    if out.exists() or out.with_suffix(out.suffix + ".part").exists():
        raise RuntimeError(f"OUTPUT_EXISTS:{out}")
    import torch
    torch.cuda.set_device(0)
    pre = gpu_snapshot()
    load_t0 = time.perf_counter()
    proc, pplx, jina_tok, jina = load_models(torch, args.model_group)
    load_ms = (time.perf_counter() - load_t0) * 1000.0
    post_load = gpu_snapshot()
    rows: list[dict[str, Any]] = []
    contention: list[dict[str, Any]] = []
    session_t0 = time.perf_counter()
    with out.with_suffix(out.suffix + ".part").open("w", encoding="utf-8") as f:
        def execute(cid: str, policy: str, docs: list[dict[str, Any]], anchor_stage: str | None = None, order_ix: int = 0):
            if (cid, policy, anchor_stage) in resume_keys:
                return
            s = slates[cid]
            clean_docs = [{"candidate_id": str(d["candidate_id"]), "content": str(d["content"])} for d in docs]
            rec: dict[str, Any] = {"dataset": "cqadupstack", "canonical_id": cid, "forum": s["forum"], "query_id": s["query_id"], "length_tercile": lengths[cid]["length_tercile"], "policy": policy, "phase": args.phase, "pass_index": args.pass_index if args.phase == "passB" else None, "anchor_stage": anchor_stage, "policy_order_index": order_ix, "candidate_ids": [d["candidate_id"] for d in clean_docs], "source_slate_sha256": s["slate_sha256"], "input_sha256": sha_obj(request_obj(str(s["query"]), clean_docs)), "model_revision": PPLX_REV if policy.startswith("pplx") else JINA_REV, "gpu_index_requested": args.gpu, "gpu_visible_index": 0, "status": "FAIL", "valid": False, "selected_ids": [], "scores": [], "n_calls": 0, "physical_forward_count": 0, "error_class": None, "error": None, "latency_ms": None, "peak_memory_allocated_mib": None, "labels_joined": False, "input_label_blind": True, "quality_eligible": args.phase in ("dry", "passA") and anchor_stage is None}
            t0 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            try:
                scores, ncall = run_one(policy, str(s["query"]), clean_docs, pplx, jina, torch)
                rec.update({"status": "PASS", "valid": True, "selected_ids": top5(rec["candidate_ids"], scores), "scores": scores, "n_calls": ncall, "physical_forward_count": ncall})
            except Exception as exc:
                rec.update({"error_class": classify(exc), "error": f"{type(exc).__name__}: {exc}", "n_calls": 1})
            torch.cuda.synchronize()
            rec["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            rec["peak_memory_allocated_mib"] = round(float(torch.cuda.max_memory_allocated()) / 2**20, 3)
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            rows.append(rec)

        for local_ix, cid in enumerate(selected_ids):
            s = slates[cid]
            views = {"pplx_single_4096": prepared[4096][cid]["budgeted_candidates"], "pplx_single_6144": prepared[6144][cid]["budgeted_candidates"], "jina_full_text": s["candidates"], "jina_trunc_8192": prepared[8192][cid]["budgeted_candidates"]}
            order = list(active_policies)
            if args.phase == "passB":
                seed = 731 + int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16)
                random.Random(seed).shuffle(order)
                if args.pass_index == 2:
                    order.reverse()
            for order_ix, policy in enumerate(order):
                execute(cid, policy, views[policy], order_ix=order_ix)
            if args.phase == "passB" and local_ix < 60:
                execute(cid, "jina_full_text", s["candidates"], anchor_stage="start", order_ix=999)
            if args.phase == "passB" and local_ix >= max(0, len(selected_ids) - 60):
                execute(cid, "jina_full_text", s["candidates"], anchor_stage="end", order_ix=999)
            if (local_ix + 1) % 100 == 0:
                snap = gpu_snapshot()
                snap.update({"phase": args.phase, "pass_index": args.pass_index, "gpu_index_requested": args.gpu, "completed_query_count": local_ix + 1, "elapsed_s": time.perf_counter() - session_t0})
                contention.append(snap)
                print(json.dumps({"phase": args.phase, "gpu": args.gpu, "pass": args.pass_index, "completed_queries": local_ix + 1, "total_queries": len(selected_ids), "records": len(rows)}), flush=True)
    out.with_suffix(out.suffix + ".part").replace(out)
    raw_sha = sha_file(out)
    summary = {"status": "PASS", "phase": args.phase, "model_group": args.model_group, "pass_index": args.pass_index, "gpu_index_requested": args.gpu, "shard": args.shard, "shards": args.shards, "records": len(rows), "queries": len(selected_ids), "valid": sum(x["valid"] for x in rows), "failed": sum(not x["valid"] for x in rows), "raw_sha256": raw_sha, "pre_gpu": pre, "post_model_load_gpu": post_load, "model_load_ms": load_ms, "labels_joined": False, "policies": list(active_policies), "no_retries": True, "protocol_sha256": PROTOCOL_SHA}
    out.with_suffix(".SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "gpu_contention" / f"{suffix}.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in contention), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    del pplx, jina, proc, jina_tok
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
