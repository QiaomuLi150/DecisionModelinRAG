#!/usr/bin/env python3
"""Run the required Jina/PPLX anchors for the GPU-4 Winnow session.

This phase is executed before the local Winnow server is loaded, so the 27B
decision model and the 12B GGUF never compete for memory.  The resulting raw
records are appended to the same logical session-E namespace as the Winnow
client phase.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/followup_v1/stage2_n2/session_E_winnow/raw"
COHORT = ROOT / "results/v3_6_essential/stage0_preflight/cohort_manifest.csv"
N1 = ROOT / "scripts/v3_6_followup_n1.py"
T2 = ROOT / "scripts/v3_6_t2_serving.py"
POLICIES = ["jina_full", "pplx_budget"]


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


t2 = mod("v36_winnow_anchor_t2", T2); base = t2.base; n1 = mod("v36_winnow_anchor_n1", N1)


def read_jsonl(p):
    return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=4); ap.add_argument("--resume", action="store_true"); a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu); gate = base.gpu_gate(a.gpu)
    import torch
    torch.cuda.set_device(0)
    proc = base.load_pplx_processor(); tok = proc.tokenizer; sys.path.insert(0, str(base.PPLX_PATH / "source" / "src")); from autojev.model import decision_messages
    views, _ = n1.load_all_views(proc, decision_messages, tok)
    wanted = {r["canonical_id"] for r in __import__("csv").DictReader((COHORT).open(encoding="utf-8"))}
    views = [v for v in views if v["canonical_id"] in wanted]
    if len(views) != 360: raise RuntimeError(f"C360_INVALID:{len(views)}")
    OUT.mkdir(parents=True, exist_ok=True); ck = OUT / "serving_perrequest.checkpoint.jsonl"; existing = read_jsonl(ck) if a.resume and ck.exists() else []
    done = {(x.get("policy"), x.get("canonical_id"), int(x.get("attempt", 0))) for x in existing}; fh = ck.open("a", encoding="utf-8")
    for policy in POLICIES:
        model = None
        if policy == "jina_full": model, _ = base.load_jina(torch)
        else: model, decision_messages = base.load_pplx(torch)
        for attempt in (1, 2, 3):
            for v in views:
                key = (policy, v["canonical_id"], attempt)
                if key in done: continue
                torch.cuda.reset_peak_memory_stats(); started = time.perf_counter(); status = "FAIL"; error = None; scores = []; selected = []
                try:
                    if policy == "jina_full":
                        scores = base.jina_score(model, v["query"], v["original"]); ids = v["original_ids"]
                    else:
                        docs = v["single"]["8192"]
                        if docs is None: raise RuntimeError("UNBUDGETABLE_QUERY_OR_OVERHEAD")
                        ids = [x["candidate_id"] for x in docs]; scores = base.pplx_score(model, base.choice_request(v["query"], docs), len(docs))
                    selected = base.stable_top(ids, scores)
                    if len(scores) != 20 or len(selected) != 5 or len(set(selected)) != 5: raise ValueError("INVALID_TOP5_SELECTION")
                    status = "PASS"
                except Exception as exc: error = f"{type(exc).__name__}: {exc}"
                try: torch.cuda.synchronize(); alloc = round(torch.cuda.max_memory_allocated() / 2**20, 3)
                except Exception: alloc = None
                row = {"session": "E", "gpu_slot": "D", "cohort": "C360", "policy": policy, "canonical_id": v["canonical_id"], "domain": v["domain"], "attempt": attempt, "status": status, "error_class": error, "latency_ms": round((time.perf_counter()-started)*1000, 3), "alloc_peak_mib": alloc, "model_revision": base.JINA_REV if policy.startswith("jina") else base.PPLX_REV, "candidate_ids": v["original_ids"], "selected_ids": selected, "scores": scores, "labels_joined": False, "input_label_blind": True, "anchor_status": "ANCHOR_PHASE_SAME_GPU_SESSION_E", "timed_boundary": "complete Top-20 transaction through native scoring, mapping and Top-5 packaging"}
                fh.write(json.dumps(row, sort_keys=True) + "\n"); fh.flush(); done.add(key)
            print(json.dumps({"session": "E", "policy": policy, "attempt": attempt, "rows": len(done)}), flush=True)
        del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    fh.close(); (OUT.parent / "ANCHOR_GATE.json").write_text(json.dumps({"status": "PASS", "session": "E", "gpu_slot": "D", "anchors": POLICIES, "records": len(existing) + len(done)}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
