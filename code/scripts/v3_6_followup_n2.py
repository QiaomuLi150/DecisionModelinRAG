#!/usr/bin/env python3
"""Bounded v3.6 N2 matched serving session on the frozen C360 cohort.

This runner is label-blind, no-retry, append-only, and deliberately records
only models with locally authenticated native adapters.  It includes the
required Jina and PPLX anchors in the same session as each available control.
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
OUT = ROOT / "results/v3_6_essential/followup_v1/stage2_n2/session_C"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
COHORT = ROOT / "results/v3_6_essential/stage0_preflight/cohort_manifest.csv"
T2 = ROOT / "scripts/v3_6_t2_serving.py"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
POLICIES = [
    "jina_full", "pplx_budget", "qwen3_reranker_0_6b",
    "qwen3_reranker_4b", "mxbai_rerank_large_v2",
]
MODEL_PATHS = {
    "qwen3_reranker_0_6b": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473"),
    "qwen3_reranker_4b": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots/22e683669bc0f0bd69640a1354a6d0aebcfeede5"),
    "mxbai_rerank_large_v2": ROOT / "data/models/e12_o_checkpoints/mxbai_v2",
}


def import_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"IMPORT_FAILED:{name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t2 = import_mod("v36_t2_n2", T2)
base = t2.base
n1 = import_mod("v36_n1_views", ROOT / "scripts/v3_6_followup_n1.py")


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def sha_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cohort_ids():
    import csv
    rows = list(csv.DictReader(COHORT.open(encoding="utf-8")))
    ids = [r["canonical_id"] for r in rows]
    if len(ids) != 360 or len(set(ids)) != 360:
        raise RuntimeError(f"C360_INVALID:{len(ids)}:{len(set(ids))}")
    return set(ids)


def load_views(proc, decision_messages, tok):
    # Reuse the frozen Stage-0 builder metadata and exact tokenizer path used by
    # N1.  Only the immutable C360 subset is retained for this N2 session.
    views, _ = n1.load_all_views(proc, decision_messages, tok)
    wanted = cohort_ids()
    out = [v for v in views if v["canonical_id"] in wanted]
    if len(out) != 360:
        raise RuntimeError(f"C360_VIEW_COVERAGE:{len(out)}")
    return out


def choose(ids, scores):
    return base.stable_top(ids, scores)


def row(view, policy, attempt, gpu, started, status, error, scores, selected, load_ms, interface, torch):
    try:
        torch.cuda.synchronize()
        alloc = round(torch.cuda.max_memory_allocated() / 2**20, 3)
    except Exception:
        alloc = None
    return {
        "session": "C", "gpu_slot": "C", "gpu_index": gpu,
        "policy": policy, "canonical_id": view["canonical_id"], "domain": view["domain"],
        "attempt": attempt, "cohort": "C360", "status": status,
        "error_class": error, "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "alloc_peak_mib": alloc, "model_load_ms": load_ms, "native_interface": interface,
        "model_revision": PPLX_REV if policy == "pplx_budget" else (JINA_REV if policy == "jina_full" else "local_registry"),
        "candidate_ids": view["original_ids"], "scores": scores, "selected_ids": selected,
        "native_request_sha256": sha({"query": view["query"], "candidate_ids": view["original_ids"]}),
        "labels_joined": False, "input_label_blind": True,
        "timed_boundary": "complete Top-20 transaction through native scoring, mapping and Top-5 packaging",
    }


def score_request(policy, view, model, tok=None, proc=None):
    q = view["query"]
    if policy == "jina_full":
        docs = view["original"]
        scores = base.jina_score(model, q, docs)
        return scores, choose(view["original_ids"], scores), "Jina native listwise/index-preserving scores"
    if policy == "pplx_budget":
        docs = view["single"]["8192"]
        if docs is None:
            raise RuntimeError("UNBUDGETABLE_QUERY_OR_OVERHEAD")
        scores = base.pplx_score(model, base.choice_request(q, docs), len(docs))
        ids = [x["candidate_id"] for x in docs]
        return scores, choose(ids, scores), "PPLX native choice probabilities on frozen L=8192 text"
    docs = view["original"]
    pairs = [(q, x["content"]) for x in docs]
    values = model.predict(pairs, batch_size=8, convert_to_tensor=True, show_progress_bar=False)
    scores = values.detach().float().cpu().tolist() if hasattr(values, "detach") else [float(x) for x in values]
    return scores, choose(view["original_ids"], scores), "official CrossEncoder pair logits; higher is better"


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # The isolation gate must run before importing/initializing CUDA so it does
    # not mistake this runner's own context for a resident workload.
    gate = base.gpu_gate(args.gpu)
    import torch
    torch.cuda.set_device(0)
    proc = base.load_pplx_processor()
    tok = proc.tokenizer
    sys.path.insert(0, str(base.PPLX_PATH / "source" / "src"))
    from autojev.model import decision_messages
    views = load_views(proc, decision_messages, tok)
    protocol = {
        "status": "FROZEN_PRE_OUTCOME", "stage": "N2_matched_C360", "session": "C", "gpu_slot": "C",
        "cohort": "C360", "cohort_size": 360, "attempts": 3, "concurrency": 1,
        "policy_order": POLICIES, "anchors": ["jina_full", "pplx_budget"],
        "available_native_controls": POLICIES[2:], "labels_joined": False,
        "protected_holdouts_accessed": False, "frozen_bright_sha256": sha_file(BRIGHT),
        "cohort_sha256": sha(sorted(v["canonical_id"] for v in views)), "bootstrap_seed": 731,
        "failure_policy": "no retries; failed elapsed time retained; no imputation",
    }
    write_json(OUT / "PROTOCOL_N2_C.json", protocol)
    checkpoint = OUT / "raw" / "serving_perrequest.checkpoint.jsonl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(checkpoint) if args.resume and checkpoint.exists() else []
    done = {(x["policy"], x["canonical_id"], int(x["attempt"])) for x in existing}
    fh = checkpoint.open("a", encoding="utf-8")
    all_rows = existing
    for policy in POLICIES:
        load_started = time.perf_counter(); model = None; interface = ""; load_error = None
        try:
            if policy == "jina_full":
                model, _ = base.load_jina(torch); interface = "Jina native listwise/index-preserving scores"
            elif policy == "pplx_budget":
                model, decision_messages = base.load_pplx(torch); interface = "PPLX native categorical choice probabilities"
            else:
                from sentence_transformers import CrossEncoder
                model = CrossEncoder(str(MODEL_PATHS[policy]), device="cuda", local_files_only=True)
                interface = "official CrossEncoder pair logits; higher is better"
            torch.cuda.synchronize()
        except Exception as exc:
            load_error = f"{type(exc).__name__}: {exc}"
        load_ms = round((time.perf_counter() - load_started) * 1000, 3)
        for attempt in (1, 2, 3):
            for i, view in enumerate(views, 1):
                key = (policy, view["canonical_id"], attempt)
                if key in done:
                    continue
                torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
                status, error, scores, selected = "FAIL", load_error, [], []
                if load_error is None:
                    try:
                        scores, selected, interface = score_request(policy, view, model, tok, proc)
                        if len(scores) != 20 or len(selected) != 5 or len(set(selected)) != 5:
                            raise ValueError("INVALID_TOP5_SELECTION")
                        status, error = "PASS", None
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                x = row(view, policy, attempt, args.gpu, started, status, error, scores, selected, load_ms, interface, torch)
                all_rows.append(x); done.add(key); fh.write(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n"); fh.flush()
            print(json.dumps({"session": "C", "policy": policy, "attempt": attempt, "rows": len(all_rows), "valid": sum(x["status"] == "PASS" for x in all_rows)}), flush=True)
        del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    fh.close()
    raw = OUT / "raw" / "serving_perrequest_raw.jsonl"
    raw.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in all_rows), encoding="utf-8")
    meta = {"status": "PASS" if len(all_rows) == len(POLICIES) * 360 * 3 else "INCOMPLETE", "records": len(all_rows), "expected_records": len(POLICIES) * 360 * 3, "valid": sum(x["status"] == "PASS" for x in all_rows), "failed": sum(x["status"] != "PASS" for x in all_rows), "sha256": sha_file(raw), "gpu_slot": "C", "gpu_uuid": gate.get("gpu_uuid"), "labels_joined": False}
    write_json(OUT / "TRACE_META.json", meta); write_json(OUT / "GPU_GATE.json", gate)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=5); ap.add_argument("--resume", action="store_true"); run(ap.parse_args())
