#!/usr/bin/env python3
"""v3.6 Stage-2 matched serving runner.

This runner reuses the authenticated v1.6 native-call harness, but binds it to
the v3.6 C360 cohort and frozen v3.6 input artifacts.  Jina and PPLX are run
in separate model-resident phases so one model is released before the other is
loaded.  The checkpoint is append-only and records every request, including
failures; it never retries a request.
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
OUT = Path(os.environ.get("V36_SERVING_OUT", str(ROOT / "results/v3_6_essential/stage2_serving")))
STAGE0 = ROOT / "results/v3_6_essential/stage0_preflight"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
BUDGET = ROOT / "results/non_jev/v2/e11_v1_0/01_budget_policy/E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl"
D1 = ROOT / "results/v1_6/d1_perwindow_jina/D1_INPUT_MANIFEST.jsonl"
PROTO = OUT / "PROTOCOL.json"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
BRIGHT_SHA = "c304b2c35f445153181098d3a899271db6fc653b44aea50186c6efaeec8e6ea2"
POLICIES = {
    "jina": ["jina_full", "jina_pplx_budget", "jina_two_stage_pplxwindow"],
    "pplx": ["pplx_budget", "pplx_two_stage"],
}


def load_base():
    path = Path(__file__).with_name("v1_6_d2_runner.py")
    spec = importlib.util.spec_from_file_location("v16_d2_harness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("V16_HARNESS_IMPORT_FAILED")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = load_base()


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")


def load_views():
    slates = {f"{r['domain']}:{r['query_id']}": r for r in read_jsonl(BRIGHT)}
    budgets = {f"{r['domain']}:{r['query_id']}": r for r in read_jsonl(BUDGET)}
    cohort = [r["canonical_id"] for r in __import__("csv").DictReader((STAGE0 / "cohort_manifest.csv").open(encoding="utf-8"))]
    stage0 = {r["canonical_id"]: r for r in read_jsonl(STAGE0 / "BUDGET_VIEW_MANIFEST.jsonl")}
    d1 = {(str(r['canonical_id']), str(r['order_id'])): r for r in read_jsonl(D1)}
    if len(slates) != 1384 or file_sha(BRIGHT) != BRIGHT_SHA or len(cohort) != 360:
        raise RuntimeError("V36_SERVING_INPUT_COVERAGE_OR_HASH_FAILURE")
    if len(d1) != 9688:
        raise RuntimeError(f"V36_D1_MANIFEST_COVERAGE_FAILURE:{len(d1)}")
    views = []
    for cid in cohort:
        s = slates[cid]
        b = budgets[cid]
        original = [{"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in s["candidates"]]
        budgeted = [{"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in b.get("budgeted_candidates", [])] if str(b.get("status", "")).lower() in {"ok", "pass", "complete"} else None
        expected = stage0[cid]["single"]["8192"]
        if budgeted is not None:
            ids = [x["candidate_id"] for x in budgeted]
            hashes = [hashlib.sha256(x["content"].encode()).hexdigest() for x in budgeted]
            if ids != expected["candidate_ids"] or hashes != expected["content_sha256"]:
                raise RuntimeError(f"V36_8192_VIEW_MISMATCH:{cid}")
        views.append({"canonical_id": cid, "domain": str(s["domain"]), "query_id": str(s["query_id"]),
                      "query": str(s["query"]), "slate_sha256": s["slate_sha256"], "source_input_sha256": s["input_sha256"],
                      "original": original, "budgeted": budgeted, "budget_status": b.get("status"),
                      "original_ids": [x["candidate_id"] for x in original],
                      "budgeted_ids": [x["candidate_id"] for x in budgeted] if budgeted else [],
                      "budget_row_sha256": base.sha(b)})
    return views, d1


def protocol(gpu: int, gate: dict, views, phase: str, attempts: int):
    return {"status": "FROZEN_PRE_OUTCOME", "experiment": "DecisionRAG v3.6 Essential Verification",
            "stage": "B6.1_T1_matched_serving", "phase": phase, "gpu_index": gpu,
            "gpu_uuid": gate.get("gpu_uuid"), "cohort": "v3.6 frozen C360", "cohort_size": len(views),
            "attempts": attempts, "concurrency": 1, "policy_order": (["jina_two_stage_pplxwindow"] if phase == "jina_j2_repair" else POLICIES[phase]),
            "models": {"jina_revision": JINA_REV, "pplx_revision": PPLX_REV},
            "input_hashes": {"bright": file_sha(BRIGHT), "stage0_protocol": file_sha(STAGE0 / "PROTOCOL.json"),
                             "stage0_budget_manifest": file_sha(STAGE0 / "BUDGET_VIEW_MANIFEST.jsonl"), "d1_manifest": file_sha(D1)},
            "cohort_sha256": base.sha(sorted(x["canonical_id"] for x in views)),
            "timed_boundary": "view preparation + all native calls + score parsing + candidate-ID mapping + Top-5 packaging",
            "labels_joined": False, "protected_holdouts_accessed": False, "paid_api_calls": 0,
            "failure_policy": "no retries; failure elapsed time retained; no fallback in raw timing",
            "stage2_policy_mapping": {"jina_full": "J_full", "jina_pplx_budget": "J_8192", "jina_two_stage_pplxwindow": "J2_8192", "pplx_budget": "P_8192", "pplx_two_stage": "P2_8192"},
            "replication_deviation": "Only one isolated GPU was available at preflight; this is one T1 session, not three concurrent replicate sessions."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--phase", choices=["jina", "jina_j2_repair", "pplx"], required=True)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.attempts != 3:
        raise RuntimeError("V36_FIXED_ATTEMPTS_MUST_BE_3")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)
    gate = base.gpu_gate(args.gpu)
    views, d1 = load_views()
    if not PROTO.exists():
        write_json(PROTO, protocol(args.gpu, gate, views, args.phase, args.attempts))
    else:
        old = json.loads(PROTO.read_text(encoding="utf-8"))
        expected = {"bright": file_sha(BRIGHT), "stage0_protocol": file_sha(STAGE0 / "PROTOCOL.json"), "stage0_budget_manifest": file_sha(STAGE0 / "BUDGET_VIEW_MANIFEST.jsonl"), "d1_manifest": file_sha(D1)}
        if old.get("input_hashes") != expected:
            raise RuntimeError("V36_SERVING_PROTOCOL_INPUT_HASH_CHANGED")
    phase_dir = OUT / "raw" / args.phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = phase_dir / "serving_perrequest.checkpoint.jsonl"
    rows = read_jsonl(checkpoint) if args.resume and checkpoint.exists() else []
    policies = ["jina_two_stage_pplxwindow"] if args.phase == "jina_j2_repair" else POLICIES[args.phase]
    completed = {(x["policy"], x["canonical_id"], int(x["attempt"])) for x in rows}
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    torch.cuda.set_device(0)
    session = {"phase": args.phase, "gpu_index": args.gpu, "gpu_uuid": gate.get("gpu_uuid"), "cohort": "C360", "attempts": args.attempts, "policy_order": policies, "labels_joined": False, "model_loads": {}, "preflight": gate, "started_utc": base.now()}
    t0 = time.perf_counter()
    pplx_proc = base.load_pplx_processor(); pplx_tok = pplx_proc.tokenizer
    session["pplx_processor_load_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    model = None; decision_messages = None
    if args.phase in {"jina", "jina_j2_repair"}:
        t0 = time.perf_counter(); model, _tok = base.load_jina(torch); torch.cuda.synchronize(); session["jina_model_load_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    else:
        t0 = time.perf_counter(); model, decision_messages = base.load_pplx(torch); torch.cuda.synchronize(); session["pplx_model_load_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    for policy in policies:
        for attempt in range(1, args.attempts + 1):
            for view in views:
                key = (policy, view["canonical_id"], attempt)
                if key in completed:
                    continue
                row = base.run_request(view, d1, policy, attempt, args.gpu, torch, model, pplx_proc, pplx_tok, decision_messages)
                row["gpu_uuid"] = gate.get("gpu_uuid"); row["session_phase"] = args.phase; row["policy_contract"] = "v3.6_T1"
                rows.append(row); completed.add(key)
                checkpoint.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
            print(json.dumps({"phase": args.phase, "policy": policy, "attempt": attempt, "rows": len(rows), "valid": sum(x.get("valid") is True for x in rows)}), flush=True)
    rows.sort(key=lambda x: (policies.index(x["policy"]), int(x["attempt"]), x["canonical_id"]))
    raw = phase_dir / "serving_perrequest_raw.jsonl"
    write_jsonl(raw, rows)
    meta = {"status": "PASS" if len(rows) == len(policies) * len(views) * args.attempts else "INCOMPLETE", "phase": args.phase, "records": len(rows), "expected_records": len(policies) * len(views) * args.attempts, "valid": sum(x.get("valid") is True for x in rows), "failed": sum(x.get("valid") is not True for x in rows), "policy_counts": {p: sum(x["policy"] == p for x in rows) for p in policies}, "sha256": file_sha(raw), "labels_joined": False}
    write_json(phase_dir / "TRACE_META.json", meta)
    session.update({"status": meta["status"], "records": len(rows), "raw_sha256": meta["sha256"], "finished_utc": base.now()})
    write_json(OUT / f"SESSION_{args.phase}.json", session)
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
