#!/usr/bin/env python3
"""B1 timing for the exact available frozen transfer views.

One process owns one GPU and one policy/session. Each session has a 60-query
full-text Jina anchor at the beginning and end and two serial target passes.
The runner is label blind and has no retry or fallback path.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/followup_targeted_audit/B_transfer_timing"
PREP = ROOT / "results/v3_6_essential/followup_v3_repro_fixes/R1_transfer_budget_grid"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
POLICIES = ("pplx_single_L4096", "pplx_single_L6144", "jina_full", "jina_truncated_L8192")
PPLX_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")
JINA_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/{JINA_REV}")

spec = importlib.util.spec_from_file_location("d2base", ROOT / "scripts/v1_6_d2_runner.py")
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def gpu_snapshot(gpu: int):
    q = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    p = subprocess.run(q, capture_output=True, text=True, check=False)
    rows = []
    for line in p.stdout.splitlines():
        vals = [x.strip() for x in line.split(",")]
        if vals and vals[0] == str(gpu):
            rows.append(vals)
    if len(rows) != 1:
        raise RuntimeError(f"GPU_SNAPSHOT_MISSING:{gpu}:{p.stderr.strip()}")
    row = rows[0]
    if int(row[5]) < 70000 or int(row[6]) != 0:
        raise RuntimeError(f"GPU_NOT_DEDICATED_OR_FREE:{row}")
    q2 = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"]
    p2 = subprocess.run(q2, capture_output=True, text=True, check=False)
    resident = [x for x in p2.stdout.splitlines() if row[1] in x]
    if resident:
        raise RuntimeError(f"GPU_HAS_COMPUTE_PROCESS:{gpu}:{resident}")
    return {"gpu_index": gpu, "gpu_uuid": row[1], "device_row": row, "compute_processes": [], "stderr": p.stderr.strip()}


def load_rows():
    out = {}
    for L in (4096, 6144, 8192):
        for x in read_jsonl(PREP / f"prepared_L{L}.jsonl"):
            out.setdefault(x["canonical_id"], {})[L] = x
    rows = []
    for cid in sorted(out):
        x = out[cid]
        if not all(L in x for L in (4096, 6144, 8192)):
            raise RuntimeError(f"MISSING_PREPARED_VIEW:{cid}")
        rows.append(x)
    return rows


def make_view(x, L):
    r = x[L]
    return {"canonical_id": r["canonical_id"], "query_id": r["query_id"], "dataset": r["dataset"], "group_id": r.get("group_id", ""), "query": r["query"], "source_slate_sha256": r["source_slate_sha256"], "original": r["candidates"], "budgeted": r["budgeted_candidates"], "serialized_tokens": r.get("serialized_tokens"), "length_limit": L}


def stable_top(ids, scores):
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i, str(ids[i])))[:5]]


def req_hash(query, docs):
    return base.sha(base.choice_request(query, docs))


def target_policy_run(policy, view, jina_model, pplx_model, proc, tok, decision_messages):
    if policy.startswith("pplx_single"):
        docs = view["budgeted"]
        req = base.choice_request(view["query"], docs)
        scores = base.pplx_score(pplx_model, req, len(docs))
        return {"candidate_ids": [d["candidate_id"] for d in docs], "scores": scores, "selected_ids": stable_top([d["candidate_id"] for d in docs], scores), "n_calls": 1, "input_sha256": base.sha(req), "input_view": f"exact_frozen_builder_L{view['length_limit']}"}
    docs = view["original"] if policy == "jina_full" else view["budgeted"]
    scores = base.jina_score(jina_model, view["query"], docs)
    ids = [d["candidate_id"] for d in docs]
    return {"candidate_ids": ids, "scores": scores, "selected_ids": stable_top(ids, scores), "n_calls": 1, "input_sha256": base.sha({"query": view["query"], "candidate_ids": ids, "candidate_texts": [d["content"] for d in docs]}), "input_view": "frozen original Top-20" if policy == "jina_full" else "exact_frozen_builder_L8192"}


def run_one(policy, view, attempt, gpu, session_id, pass_id, jina_model, pplx_model, proc, tok, decision_messages, torch):
    start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    rec = {"session": f"S{session_id}", "gpu_slot": f"GPU{gpu}", "cohort": "TRANSFER_AVAILABLE", "policy": policy, "canonical_id": view["canonical_id"], "dataset": view["dataset"], "attempt": attempt, "pass_id": pass_id, "status": "FAIL", "error_class": None, "latency_ms": None, "latency_with_fallback_ms": None, "alloc_peak_mib": None, "input_sha256": None, "candidate_ids": [], "selected_ids": [], "n_calls": 0, "labels_joined": False, "model_revision": PPLX_REV if policy.startswith("pplx") else JINA_REV}
    try:
        result = target_policy_run(policy, view, jina_model, pplx_model, proc, tok, decision_messages)
        if len(result["selected_ids"]) != 5 or len(set(result["selected_ids"])) != 5:
            raise ValueError("INVALID_TOP5")
        rec.update(result); rec["status"] = "PASS"; rec["valid"] = True
    except Exception as exc:
        rec.update({"valid": False, "error_class": type(exc).__name__, "error": f"{type(exc).__name__}: {exc}"})
    try: torch.cuda.synchronize()
    except Exception: pass
    rec["latency_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
    try: rec["alloc_peak_mib"] = round(float(torch.cuda.max_memory_allocated()) / 2**20, 3)
    except Exception: pass
    rec["latency_with_fallback_ms"] = rec["latency_ms"]
    return rec


def run_anchor(view, gpu, session_id, phase, jina_model, torch):
    start = time.perf_counter(); torch.cuda.reset_peak_memory_stats(); status = "PASS"; err = None
    try:
        ids = [d["candidate_id"] for d in view["original"]]
        scores = base.jina_score(jina_model, view["query"], view["original"])
        selected = stable_top(ids, scores)
    except Exception as exc:
        status = "FAIL"; err = f"{type(exc).__name__}: {exc}"; ids = []; selected = []
    try: torch.cuda.synchronize()
    except Exception: pass
    lat = round((time.perf_counter() - start) * 1000.0, 3)
    peak = None
    try: peak = round(float(torch.cuda.max_memory_allocated()) / 2**20, 3)
    except Exception: pass
    return {"session": f"S{session_id}", "gpu_slot": f"GPU{gpu}", "cohort": "TRANSFER_AVAILABLE", "policy": f"jina_full_anchor_{phase}", "canonical_id": view["canonical_id"], "dataset": view["dataset"], "attempt": 0, "pass_id": 0, "status": status, "valid": status == "PASS", "error_class": err.split(":", 1)[0] if err else None, "error": err, "latency_ms": lat, "latency_with_fallback_ms": lat, "alloc_peak_mib": peak, "input_sha256": base.sha({"query": view["query"], "candidate_ids": ids, "candidate_texts": [d["content"] for d in view["original"]]}) if ids else None, "candidate_ids": ids, "selected_ids": selected, "n_calls": 1, "labels_joined": False, "anchor": True, "model_revision": JINA_REV}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--policy", choices=POLICIES)
    ap.add_argument("--session", type=int, choices=(1, 2))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "raw").mkdir(exist_ok=True)
    rows = load_rows()
    by = {x[4096]["canonical_id"]: x for x in rows}
    dataset_counts = {}
    for x in rows: dataset_counts[x[4096]["dataset"]] = dataset_counts.get(x[4096]["dataset"], 0) + 1
    prepared_hashes = {str(L): file_sha(PREP / f"prepared_L{L}.jsonl") for L in (4096, 6144, 8192)}
    manifest = {"status": "FROZEN_PRE_OUTCOME", "stage": "B1_B2_targeted_clean_transfer_timing", "source": "v3.6 exact R1 builder artifacts", "requested_population": {"enterprise": 500, "musique": 2500, "total": 3000}, "available_frozen_population": dataset_counts, "coverage_blocker": {"status": "BLOCKED", "missing_musique_candidate_slate_count": 333, "reason": "No frozen candidate manifest or candidate content for the additional 333 MuSiQue requests was found; only MUSIQUE_QUESTION_IDS.jsonl and prepared R1 views for 2167 are available."}, "available_canonical_ids_sha256": hashlib.sha256("\n".join(sorted(by)).encode()).hexdigest(), "prepared_hashes": prepared_hashes, "policies": list(POLICIES), "sessions": 2, "passes_per_session": 2, "anchor_size": 60, "anchor_ids": sorted(by)[:60], "anchor_selection": "lexicographic canonical_id over the available frozen transfer cohort, frozen before outcome", "serial_requests": True, "no_retries": True, "labels_joined": False, "timing_boundary": "view selection/preparation + native inference + score parsing + ID mapping + Top-5 packaging", "models": {"pplx_revision": PPLX_REV, "jina_revision": JINA_REV}, "bootstrap": {"draws": 40000, "seed": 731, "unit": "canonical_id", "ratio_basis": "all target requests; no ratio is reported below 99% validity"}}
    write_json(OUT / "PROTOCOL.json", manifest)
    if args.prepare_only:
        print(json.dumps(manifest, indent=2)); return
    if args.gpu is None or args.policy is None or args.session is None:
        raise SystemExit("--gpu --policy --session required")
    gate = gpu_snapshot(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    proc = tok = decision_messages = pplx_model = jina_model = None
    # Jina anchors are loaded first and released before PPLX target work.
    proc = base.load_pplx_processor(); tok = proc.tokenizer
    jina_model, _ = base.load_jina(torch); torch.cuda.synchronize()
    target = []
    anchor_views = [make_view(by[cid], 8192) for cid in sorted(by)[:60]]
    for v in anchor_views: target.append(run_anchor(v, args.gpu, args.session, "start", jina_model, torch))
    if args.policy.startswith("pplx"):
        del jina_model; jina_model = None; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        pplx_model, decision_messages = base.load_pplx(torch); torch.cuda.synchronize()
    for pass_id in (1, 2):
        for cid in sorted(by):
            if args.policy == "pplx_single_L4096": v = make_view(by[cid], 4096)
            elif args.policy == "pplx_single_L6144": v = make_view(by[cid], 6144)
            else: v = make_view(by[cid], 8192)
            target.append(run_one(args.policy, v, 1, args.gpu, args.session, pass_id, jina_model, pplx_model, proc, tok, decision_messages, torch))
        print(json.dumps({"policy": args.policy, "session": args.session, "pass": pass_id, "target_rows": sum(x.get("policy") == args.policy for x in target), "valid": sum(x.get("policy") == args.policy and x.get("valid") for x in target)}), flush=True)
    if pplx_model is not None:
        del pplx_model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        jina_model, _ = base.load_jina(torch); torch.cuda.synchronize()
    for v in anchor_views: target.append(run_anchor(v, args.gpu, args.session, "end", jina_model, torch))
    if jina_model is not None: del jina_model
    gc.collect(); torch.cuda.empty_cache()
    raw = OUT / "raw" / f"{args.policy}__session_{args.session}.jsonl"
    raw.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in target), encoding="utf-8")
    meta = {"status": "PASS", "policy": args.policy, "session": args.session, "gpu_index": args.gpu, "gpu_uuid": gate["gpu_uuid"], "records": len(target), "target_records": 2 * len(by), "anchor_records": 120, "target_valid": sum(x.get("policy") == args.policy and x.get("valid") for x in target), "target_failed": sum(x.get("policy") == args.policy and not x.get("valid") for x in target), "raw_sha256": file_sha(raw), "labels_joined": False, "pre_gate": gate}
    write_json(OUT / "raw" / f"{args.policy}__session_{args.session}.SUMMARY.json", meta)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__": main()
