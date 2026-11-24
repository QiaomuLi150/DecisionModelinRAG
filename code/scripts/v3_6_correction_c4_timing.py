#!/usr/bin/env python3
"""v3.6 corrective C4 clean timing runner.

Each invocation owns one GPU and one target model.  Jina anchor requests are
run at both ends of every session on the first 60 canonical C360 IDs.  Raw
records are label-blind and append-only; no retry or fallback is performed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/followup_v2_correction/C4_clean_timing"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
STAGE0 = ROOT / "results/v3_6_essential/stage0_preflight"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
MODEL_PATHS = {
    "qwen3_reranker_0_6b": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473"),
    "qwen3_reranker_4b": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots/22e683669bc0f0bd69640a1354a6d0aebcfeede5"),
    "mxbai_rerank_large_v2": ROOT / "data/models/e12_o_checkpoints/mxbai_v2",
}


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"IMPORT_FAILED:{name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = load_mod("v36_c4_d2", ROOT / "scripts/v1_6_d2_runner.py")
n1 = load_mod("v36_c4_n1", ROOT / "scripts/v3_6_followup_n1.py")


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


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")


def make_views(proc, decision_messages, tok, cohort: str):
    views, _ = n1.load_all_views(proc, decision_messages, tok)
    if cohort == "C360":
        ids = [r["canonical_id"] for r in __import__("csv").DictReader((STAGE0 / "cohort_manifest.csv").open(encoding="utf-8"))]
        ids = ids[:360]
        by = {x["canonical_id"]: x for x in views}
        out = [by[x] for x in ids]
    elif cohort == "CALL":
        out = views
    else:
        raise ValueError(cohort)
    if len(out) != (360 if cohort == "C360" else 1384) or len({x["canonical_id"] for x in out}) != len(out):
        raise RuntimeError(f"COHORT_INVALID:{cohort}:{len(out)}")
    return out


def cross_score(model, docs, query):
    pairs = [(query, x["content"]) for x in docs]
    values = model.predict(pairs, batch_size=8, convert_to_tensor=True, show_progress_bar=False)
    return values.detach().float().cpu().tolist() if hasattr(values, "detach") else [float(x) for x in values]


def run_jina_4096(view, model, torch, policy):
    docs = view["single"].get("4096")
    if docs is None:
        raise RuntimeError("UNBUDGETABLE_L4096_VIEW")
    ids = [x["candidate_id"] for x in docs]
    scores = base.jina_score(model, view["query"], docs)
    return {"policy": policy, "candidate_ids": ids, "scores": scores, "selected_ids": base.stable_top(ids, scores), "n_calls": 1,
            "input_view": "frozen tokenizer-derived L4096 single view", "input_sha256": base.sha({"query": view["query"], "candidate_ids": ids, "candidate_texts": [x["content"] for x in docs]})}


def anchor(view, model, torch, phase, attempt):
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats(); status = "PASS"; err = None; scores = []
    try:
        scores = base.jina_score(model, view["query"], view["original"])
        selected = base.stable_top(view["original_ids"], scores)
    except Exception as exc:
        status = "FAIL"; err = f"{type(exc).__name__}: {exc}"; selected = []
    torch.cuda.synchronize()
    return {"policy": f"jina_full_anchor_{phase}", "canonical_id": view["canonical_id"], "domain": view["domain"], "attempt": attempt,
            "status": status, "valid": status == "PASS", "error_class": err.split(":", 1)[0] if err else None, "error": err,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3), "alloc_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 3),
            "candidate_ids": view["original_ids"], "selected_ids": selected, "scores": scores, "n_calls": 1,
            "labels_joined": False, "input_label_blind": True, "anchor": True, "anchor_phase": phase,
            "timed_boundary": "complete Top-20 native transaction through output packaging"}


def target_row(view, policy, attempt, gpu, result, torch, started):
    torch.cuda.synchronize()
    return {"policy": policy, "canonical_id": view["canonical_id"], "domain": view["domain"], "attempt": attempt,
            "status": "PASS", "valid": True, "error_class": None, "error": None,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "alloc_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 3), "gpu_index": gpu,
            "candidate_ids": result.get("candidate_ids", []), "selected_ids": result.get("selected_ids", []),
            "scores": result.get("scores", []), "stage_records": result.get("stage_records", []),
            "n_calls": result.get("n_calls", 1), "input_view": result.get("input_view"),
            "input_sha256": result.get("input_sha256") or result.get("native_request_sha256"),
            "labels_joined": False, "input_label_blind": True,
            "timed_boundary": "view preparation + native calls + parsing + ID mapping + Top-5 packaging"}


def failed_row(view, policy, attempt, gpu, exc, torch, started):
    try: torch.cuda.synchronize()
    except Exception: pass
    msg = f"{type(exc).__name__}: {exc}"
    return {"policy": policy, "canonical_id": view["canonical_id"], "domain": view["domain"], "attempt": attempt,
            "status": "FAIL", "valid": False, "error_class": type(exc).__name__, "error": msg,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "alloc_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 3), "gpu_index": gpu,
            "candidate_ids": [], "selected_ids": [], "scores": [], "stage_records": [], "n_calls": 0,
            "labels_joined": False, "input_label_blind": True,
            "timed_boundary": "view preparation + native calls + parsing + ID mapping + Top-5 packaging"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--mode", choices=["n2_mxbai", "n2_qwen06", "n2_qwen4", "call_pplx_two_stage_8192", "call_jina_single_4096"], required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    gate = base.gpu_gate(args.gpu)
    import torch
    torch.cuda.set_device(0)
    sys.path.insert(0, str(base.PPLX_PATH / "source" / "src"))
    from autojev.model import decision_messages
    proc = base.load_pplx_processor(); tok = proc.tokenizer
    views = make_views(proc, decision_messages, tok, "CALL" if args.mode.startswith("call_") else "C360")
    anchor_views = views[:60]
    target_policy = {"n2_mxbai": "mxbai_rerank_large_v2", "n2_qwen06": "qwen3_reranker_0_6b", "n2_qwen4": "qwen3_reranker_4b",
                     "call_pplx_two_stage_8192": "pplx_two_stage_8192", "call_jina_single_4096": "jina_single_4096"}[args.mode]
    outdir = OUT / args.mode; outdir.mkdir(parents=True, exist_ok=True)
    raw = outdir / "serving_perrequest_raw.jsonl"
    rows = read_jsonl(raw) if args.resume and raw.exists() else []
    done = {(x["policy"], x["canonical_id"], int(x["attempt"])) for x in rows}
    protocol = {"status": "FROZEN_PRE_OUTCOME", "stage": "C4_clean_timing", "mode": args.mode, "gpu_index": args.gpu,
                "gpu_uuid": gate.get("gpu_uuid"), "cohort": "CALL" if args.mode.startswith("call_") else "C360",
                "cohort_size": len(views), "attempts": 3 if args.mode.startswith("n2_") else 1,
                "anchor_size": 60, "anchor_policy": "Jina full text at start/end", "target_policy": target_policy,
                "models": {"jina_revision": JINA_REV, "pplx_revision": PPLX_REV}, "labels_joined": False,
                "protected_holdouts_accessed": False, "input_hashes": {"bright": file_sha(BRIGHT), "stage0": file_sha(STAGE0 / "PROTOCOL.json")},
                "failure_policy": "no retries; failures retain elapsed time; no imputation",
                "timed_boundary": "view preparation + all native calls + parsing + ID mapping + Top-5 packaging"}
    write_json(outdir / "PROTOCOL.json", protocol)
    model = None; decision = None
    # Jina is loaded first and used only for anchors, then released before the target.
    model, _ = base.load_jina(torch); torch.cuda.synchronize()
    for phase in ("start", "end"):
        if phase == "end":
            # target execution happens below; this loop branch is intentionally delayed
            continue
        for i, view in enumerate(anchor_views, 1):
            key = (f"jina_full_anchor_{phase}", view["canonical_id"], 1)
            if key not in done:
                x = anchor(view, model, torch, phase, 1); x["gpu_index"] = args.gpu; rows.append(x); done.add(key)
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    if args.mode.startswith("n2_"):
        from sentence_transformers import CrossEncoder
        model_key = {"n2_mxbai": "mxbai_rerank_large_v2", "n2_qwen06": "qwen3_reranker_0_6b", "n2_qwen4": "qwen3_reranker_4b"}[args.mode]
        model = CrossEncoder(str(MODEL_PATHS[model_key]), device="cuda", local_files_only=True)
        for attempt in (1, 2, 3):
            for view in views:
                key = (target_policy, view["canonical_id"], attempt)
                if key in done: continue
                torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
                try:
                    scores = cross_score(model, view["original"], view["query"]); ids = view["original_ids"]
                    result = {"candidate_ids": ids, "scores": scores, "selected_ids": base.stable_top(ids, scores), "n_calls": 1, "input_view": "frozen original Top-20"}
                    x = target_row(view, target_policy, attempt, args.gpu, result, torch, started)
                except Exception as exc: x = failed_row(view, target_policy, attempt, args.gpu, exc, torch, started)
                rows.append(x); done.add(key); raw.write_text("".join(json.dumps(z, ensure_ascii=False, sort_keys=True) + "\n" for z in rows), encoding="utf-8")
    else:
        if target_policy == "jina_single_4096":
            model, _ = base.load_jina(torch); torch.cuda.synchronize()
            for view in views:
                key = (target_policy, view["canonical_id"], 1)
                if key in done: continue
                torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
                try: x = target_row(view, target_policy, 1, args.gpu, run_jina_4096(view, model, torch, target_policy), torch, started)
                except Exception as exc: x = failed_row(view, target_policy, 1, args.gpu, exc, torch, started)
                rows.append(x); done.add(key); raw.write_text("".join(json.dumps(z, ensure_ascii=False, sort_keys=True) + "\n" for z in rows), encoding="utf-8")
        else:
            model, decision = base.load_pplx(torch); torch.cuda.synchronize()
            for view in views:
                key = (target_policy, view["canonical_id"], 1)
                if key in done: continue
                torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
                try:
                    result = base.run_request(view, {}, "pplx_two_stage", 1, args.gpu, torch, model, proc, tok, decision)
                    if not result.get("valid"): raise RuntimeError(result.get("error") or "PPLX_REQUEST_FAILED")
                    result["policy"] = target_policy
                    x = target_row(view, target_policy, 1, args.gpu, result, torch, started)
                except Exception as exc: x = failed_row(view, target_policy, 1, args.gpu, exc, torch, started)
                rows.append(x); done.add(key); raw.write_text("".join(json.dumps(z, ensure_ascii=False, sort_keys=True) + "\n" for z in rows), encoding="utf-8")
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    model, _ = base.load_jina(torch); torch.cuda.synchronize()
    for view in anchor_views:
        key = ("jina_full_anchor_end", view["canonical_id"], 1)
        if key in done: continue
        x = anchor(view, model, torch, "end", 1); x["gpu_index"] = args.gpu; rows.append(x); done.add(key)
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    write_jsonl(raw, rows)
    write_json(outdir / "TRACE_META.json", {"status": "PASS" if len(rows) == (len(views) * (3 if args.mode.startswith("n2_") else 1) + 120) else "INCOMPLETE", "records": len(rows), "expected_records": len(views) * (3 if args.mode.startswith("n2_") else 1) + 120, "valid": sum(x.get("valid") is True for x in rows), "failed": sum(x.get("valid") is not True for x in rows), "raw_sha256": file_sha(raw), "labels_joined": False, "gpu_index": args.gpu, "gpu_uuid": gate.get("gpu_uuid")})
    write_json(outdir / "GPU_GATE.json", gate)
    print(json.dumps({"mode": args.mode, "records": len(rows), "valid": sum(x.get("valid") is True for x in rows), "raw": str(raw)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
