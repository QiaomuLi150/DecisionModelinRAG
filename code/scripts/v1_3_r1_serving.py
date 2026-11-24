#!/usr/bin/env python3
"""DecisionRAG v1.3 R1 matched dedicated-GPU serving continuation.

This runner is label blind and additive.  It preserves the original R1
blocker and writes a versioned continuation namespace.  The timed boundary
is construction of the native request, model inference, score parsing,
candidate-ID mapping, and Top-5 packaging.  Model loading is outside the
request timer but is recorded separately.
"""
from __future__ import annotations

import argparse
import csv
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
OUT = ROOT / "results/v1_3/r1_serving_gate_v2"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
LOCK = ROOT / "results/non_jev/v2/e12_o_final_v1_0/00_protocol/E12O_COHORT_LOCK.json"
BUDGET = ROOT / "results/non_jev/v2/e11_v1_0/01_budget_policy/E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
PPLX_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")
JINA_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/{JINA_REV}")
EXPECTED_BRIGHT_SHA = ""  # filled from the authenticated source at runtime
L = 8192
RHO = 128


def canon(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(x: Any) -> str:
    return hashlib.sha256(canon(x).encode("utf-8")).hexdigest()


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def command(args: list[str]) -> dict[str, Any]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {"status": "PASS" if p.returncode == 0 else "ERROR", "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip(), "command": " ".join(args)}
    except Exception as exc:
        return {"status": "ERROR", "returncode": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}", "command": " ".join(args)}


def gpu_snapshot() -> dict[str, Any]:
    return command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"])


def process_snapshot() -> dict[str, Any]:
    return command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"])


def guard_gpu(gpu: int, allow_self: bool = False) -> dict[str, Any]:
    snap = gpu_snapshot()
    procs = process_snapshot()
    row = None
    for line in snap.get("stdout", "").splitlines():
        vals = [x.strip() for x in line.split(",")]
        if vals and vals[0] == str(gpu):
            row = vals
    if row is None:
        raise RuntimeError(f"GPU {gpu} is not observable: {snap}")
    resident = []
    uuid = row[1]
    for line in procs.get("stdout", "").splitlines():
        if uuid in line:
            resident.append(line)
    if allow_self:
        resident_external = [x for x in resident if f", {os.getpid()}," not in x]
    else:
        resident_external = resident
    if resident_external:
        raise RuntimeError(f"DEDICATED_GPU_BLOCKED gpu={gpu} uuid={uuid} processes={resident}")
    return {"gpu": gpu, "uuid": uuid, "device_row": row, "snapshot": snap, "process_snapshot": procs, "resident": resident, "resident_external": resident_external, "self_only": bool(resident) and not resident_external, "isolated": not resident_external}


def load_rows() -> list[dict[str, Any]]:
    slates = {}
    with BRIGHT.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                slates[f"{r['domain']}:{r['query_id']}"] = r
    budget = {}
    with BUDGET.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                budget[f"{r['domain']}:{r['query_id']}"] = r
    serving = set(json.loads(LOCK.read_text(encoding="utf-8"))["bright"]["serving_ids"])
    if len(serving) != 120 or len(slates) != 1384 or set(slates) != set(budget):
        raise RuntimeError(f"cohort/source mismatch slates={len(slates)} budget={len(budget)} serving={len(serving)}")
    rows = []
    for cid in sorted(serving):
        s, b = slates[cid], budget[cid]
        orig = [{"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in s["candidates"]]
        bud = None if "budgeted_candidates" not in b else [{"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in b["budgeted_candidates"]]
        if [x["candidate_id"] for x in orig] != [str(x) for x in b.get("original_candidate_ids", b.get("candidate_ids", []))]:
            raise RuntimeError(f"candidate identity mismatch {cid}")
        if len(orig) != 20 or (bud is not None and len(bud) != 20) or len({x["candidate_id"] for x in orig}) != 20:
            raise RuntimeError(f"invalid candidate cardinality {cid}")
        rows.append({
            "canonical_id": cid, "domain": str(s["domain"]), "query_id": str(s["query_id"]), "query": str(s["query"]),
            "slate_sha256": s["slate_sha256"], "source_input_sha256": s["input_sha256"],
            "original": orig, "budgeted": bud, "budget_status": b.get("status", "UNKNOWN"), "budget_error": b.get("error_class"),
            "original_ids": [x["candidate_id"] for x in orig],
            "budgeted_ids": [x["candidate_id"] for x in bud] if bud is not None else list(b.get("candidate_ids", [])), "budget_row_sha256": sha(b),
        })
    return rows


def choice_request(query: str, docs: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "state": f"Query: {query}\nEvaluate the supplied candidate passages using the option descriptions.",
        "question": {
            "type": "choice",
            "instructions": f"Select the candidate passage that best answers the query: {query}",
            "criteria": {f"candidate_{i:03d}": f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(docs)},
        },
    }


def fixed_nonselection_request() -> dict[str, Any]:
    return {
        "state": "Route this request to the appropriate processing queue.",
        "question": {
            "type": "choice",
            "instructions": "Choose the next processing action from the option descriptions.",
            "criteria": {
                "route_000": "Send the request to the standard text-processing queue.",
                "route_001": "Send the request to the specialist review queue.",
                "route_002": "Send the request to the deferred-processing queue.",
            },
        },
    }


def choose(ids: list[str], scores: list[float]) -> list[str]:
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i, ids[i]))[:5]]


def stats(torch) -> dict[str, Any]:
    torch.cuda.synchronize()
    return {
        "memory_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 3),
        "memory_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 3),
        "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 3),
        "max_memory_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 3),
    }


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def make_base(view: dict[str, Any], policy: str, rep: int) -> dict[str, Any]:
    return {
        "canonical_id": view["canonical_id"], "domain": view["domain"], "query_id": view["query_id"], "policy": policy, "repeat": rep,
        "slate_sha256": view["slate_sha256"], "source_input_sha256": view["source_input_sha256"], "status": "FAIL", "error": None,
        "candidate_ids": [], "candidate_ids_sha256": None, "selected_ids": [], "selected_ids_sha256": sha([]), "scores": [],
        "labels_joined": False, "physical_forward_count": "NOT_EXPOSED_BY_RUNTIME", "gpu_index": None, "gpu_uuid": None,
    }


def frozen_failure(view: dict[str, Any], policy: str, rep: int, reason: str, gpu: int, input_view: str) -> dict[str, Any]:
    out = make_base(view, policy, rep)
    ids = view["budgeted_ids"] if policy == "jina_pplx_budget" else []
    out.update({
        "candidate_ids": ids, "candidate_ids_sha256": sha(ids) if ids else None,
        "selected_ids_sha256": sha([]), "error": reason, "error_class": reason.split(":", 1)[0],
        "elapsed_ms": 0.0, "input_view": input_view, "model_revision": PPLX_REV if policy.startswith("pplx") else JINA_REV,
        "gpu_index": gpu, "timed_boundary": "not entered: frozen E11-A budget view unavailable",
    })
    return out


def invoke_pplx(model, torch, req: dict[str, Any], docs: list[dict[str, str]], view: dict[str, Any], policy: str, rep: int, gpu: int, marginal: bool = False) -> dict[str, Any]:
    out = make_base(view, policy, rep)
    ids = [d["candidate_id"] for d in docs]
    out.update({"candidate_ids": ids, "candidate_ids_sha256": sha(ids), "native_request_sha256": sha(req), "gpu_index": gpu})
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        scores = [float(x) for x in model.predict([req], batch_size=1)[0]]
        if len(scores) != len(ids) or not all(math.isfinite(x) for x in scores):
            raise ValueError(f"invalid score vector length={len(scores)} expected={len(ids)}")
        selected = choose(ids, scores)
        if len(set(selected)) != 5:
            raise ValueError("invalid Top-5 selection")
        out.update({"status": "PASS", "scores": scores, "selected_ids": selected, "selected_ids_sha256": sha(selected)})
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    torch.cuda.synchronize()
    out.update(stats(torch))
    out["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    out["timed_boundary"] = "request construction + native inference + score parsing + ID mapping + Top-5 packaging"
    return out


def invoke_jina(model, torch, query: str, docs: list[dict[str, str]], view: dict[str, Any], policy: str, rep: int, gpu: int) -> dict[str, Any]:
    ids = [d["candidate_id"] for d in docs]
    req = {"query": query, "candidate_ids": ids, "candidate_texts": [d["content"] for d in docs]}
    out = make_base(view, policy, rep)
    out.update({"candidate_ids": ids, "candidate_ids_sha256": sha(ids), "native_request_sha256": sha(req), "gpu_index": gpu})
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        result = model.rerank(query, [d["content"] for d in docs])
        mapping = {int(x["index"]): float(x["relevance_score"]) for x in result}
        if set(mapping) != set(range(len(docs))):
            raise ValueError("Jina indices do not cover all candidates")
        scores = [mapping[i] for i in range(len(docs))]
        selected = choose(ids, scores)
        if len(set(selected)) != 5:
            raise ValueError("invalid Top-5 selection")
        out.update({"status": "PASS", "scores": scores, "selected_ids": selected, "selected_ids_sha256": sha(selected)})
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    torch.cuda.synchronize()
    out.update(stats(torch))
    out["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    out["timed_boundary"] = "request construction + native inference + score parsing + ID mapping + Top-5 packaging"
    return out


def budget_fit(item: dict[str, Any], docs: list[dict[str, str]], proc, decision_messages, tok) -> tuple[list[dict[str, str]] | None, dict[str, Any]]:
    codes = [f"candidate_{i:03d}" for i in range(len(docs))]
    empty = [{"candidate_id": d["candidate_id"], "content": ""} for d in docs]
    def n_tokens(xs: list[dict[str, str]]) -> int:
        msg = decision_messages(choice_request(item["query"], xs), codes)
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = proc(text=[text], images=None, padding=True, return_tensors="pt")
        return int(enc["attention_mask"].sum())
    overhead = n_tokens(empty)
    available = L - RHO - overhead
    if available < len(docs):
        return None, {"error_class": "UNBUDGETABLE_QUERY_OR_OVERHEAD", "overhead_tokens": overhead, "effective_content_budget": available}
    cap = max(1, available // len(docs)); reductions = 0
    while cap >= 1:
        xs = [{"candidate_id": d["candidate_id"], "content": tok.decode(tok.encode(d["content"], add_special_tokens=False)[:cap], skip_special_tokens=False, clean_up_tokenization_spaces=False)} for d in docs]
        total = n_tokens(xs)
        if total <= L - RHO:
            return xs, {"overhead_tokens": overhead, "effective_content_budget": available, "final_cap": cap, "reductions": reductions, "serialized_tokens": total}
        cap -= 1; reductions += 1
    return None, {"error_class": "UNBUDGETABLE_QUERY_OR_OVERHEAD", "overhead_tokens": overhead, "effective_content_budget": available}


def load_pplx(torch):
    sys.path.insert(0, str(PPLX_PATH / "source" / "src"))
    from autojev.model import DecisionModel, decision_messages
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(str(PPLX_PATH), local_files_only=True)
    model = DecisionModel(PPLX_PATH, device="cuda:0", max_image_pixels=512 * 512)
    torch.cuda.synchronize()
    return model, proc, proc.tokenizer, decision_messages


def load_jina(torch):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(JINA_PATH), local_files_only=True)
    model = AutoModel.from_pretrained(str(JINA_PATH), dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda:0").eval()
    torch.cuda.synchronize()
    return model, tok


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")


def run_pplx_budget(views, model, torch, rep, gpu, proc, decision_messages, tok, policy="pplx_budget", with_marginal=True):
    rows = []
    baseline_req = fixed_nonselection_request()
    for i, view in enumerate(views, 1):
        if view["budgeted"] is None:
            rows.append(frozen_failure(view, policy, rep, "UNBUDGETABLE_QUERY_OR_OVERHEAD:frozen E11-A budgeted view unavailable", gpu, "E11-A frozen budgeted_candidates"))
            continue
        docs = view["budgeted"]
        req = choice_request(view["query"], docs)
        if with_marginal:
            torch.cuda.reset_peak_memory_stats(); b0 = time.perf_counter()
            try: model.predict([baseline_req], batch_size=1); torch.cuda.synchronize(); bstatus = "PASS"; berr = None
            except Exception as exc: torch.cuda.synchronize(); bstatus = "FAIL"; berr = f"{type(exc).__name__}: {exc}"
            bstats = stats(torch); bms = (time.perf_counter() - b0) * 1000
        else:
            bstatus, berr, bstats, bms = "NOT_RUN", None, {}, None
        row = invoke_pplx(model, torch, req, docs, view, policy, rep, gpu)
        row.update({"baseline_nonselection_status": bstatus, "baseline_nonselection_error": berr, "baseline_nonselection_elapsed_ms": round(bms, 3) if bms is not None else None, "baseline_nonselection_peak_allocated_mib": bstats.get("max_memory_allocated_mib"), "marginal_elapsed_ms": round(row["elapsed_ms"] - bms, 3) if bms is not None else None, "marginal_peak_allocated_mib": round(float(row.get("max_memory_allocated_mib", 0)) - float(bstats.get("max_memory_allocated_mib", 0)), 3) if bms is not None else None, "input_view": "E11-A frozen budgeted_candidates", "model_revision": PPLX_REV})
        rows.append(row)
        if i % 20 == 0: print(json.dumps({"policy": policy, "repeat": rep, "completed": i, "total": len(views)}), flush=True)
    return rows


def run_two_stage(views, model, torch, rep, gpu, proc, decision_messages, tok):
    rows = []
    for i, view in enumerate(views, 1):
        row = make_base(view, "pplx_two_stage", rep); row.update({"model_revision": PPLX_REV, "n_calls": 0, "stage_views": [], "stage_selected_ids": [], "errors": []})
        started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        try:
            stage_top = []
            for stage, docs0 in [("stage1_window1", view["original"][:10]), ("stage1_window2", view["original"][10:20])]:
                docs, info = budget_fit(view, docs0, proc, decision_messages, tok); row["n_calls"] += 1
                if docs is None: raise RuntimeError(info["error_class"])
                req = choice_request(view["query"], docs); scores = [float(x) for x in model.predict([req], batch_size=1)[0]]
                ids = [x["candidate_id"] for x in docs]
                if len(scores) != 10 or not all(math.isfinite(x) for x in scores): raise ValueError("invalid stage score vector")
                top = [docs[j] for j in sorted(range(10), key=lambda j: (-scores[j], j, ids[j]))[:5]]; stage_top.extend(top)
                row["stage_views"].append({"stage": stage, "candidate_ids": ids, "candidate_ids_sha256": sha(ids), "input_sha256": sha(req), "scores_sha256": sha(scores), "budget": info})
            docs, info = budget_fit(view, stage_top, proc, decision_messages, tok); row["n_calls"] += 1
            if docs is None: raise RuntimeError(info["error_class"])
            req = choice_request(view["query"], docs); scores = [float(x) for x in model.predict([req], batch_size=1)[0]]; ids = [x["candidate_id"] for x in docs]
            if len(scores) != 10 or not all(math.isfinite(x) for x in scores): raise ValueError("invalid stage2 score vector")
            row["stage_views"].append({"stage": "stage2", "candidate_ids": ids, "candidate_ids_sha256": sha(ids), "input_sha256": sha(req), "scores_sha256": sha(scores), "budget": info})
            row.update({"status": "PASS", "selected_ids": choose(ids, scores), "selected_ids_sha256": sha(choose(ids, scores)), "stage_selected_ids": [x["candidate_id"] for x in stage_top]})
        except Exception as exc:
            row["errors"].append({"error_class": type(exc).__name__, "message": str(exc)}); row["error"] = f"{type(exc).__name__}: {exc}"
        torch.cuda.synchronize(); row.update(stats(torch)); row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3); row["timed_boundary"] = "three complete native calls + budgeting + parsing + Top-5 packaging"
        rows.append(row)
        if i % 20 == 0: print(json.dumps({"policy": "pplx_two_stage", "repeat": rep, "completed": i, "total": len(views)}), flush=True)
    return rows


def run_jina(views, model, torch, rep, gpu, use_budgeted):
    rows = []
    policy = "jina_pplx_budget" if use_budgeted else "jina_full"
    for i, view in enumerate(views, 1):
        if use_budgeted and view["budgeted"] is None:
            rows.append(frozen_failure(view, policy, rep, "UNBUDGETABLE_QUERY_OR_OVERHEAD:frozen E11-A budgeted view unavailable", gpu, "E11-A frozen budgeted_candidates"))
            continue
        docs = view["budgeted"] if use_budgeted else view["original"]
        row = invoke_jina(model, torch, view["query"], docs, view, policy, rep, gpu); row["model_revision"] = JINA_REV; row["input_view"] = "E11-A frozen budgeted_candidates" if use_budgeted else "BRIGHT frozen original candidates"; rows.append(row)
        if i % 20 == 0: print(json.dumps({"policy": policy, "repeat": rep, "completed": i, "total": len(views)}), flush=True)
    return rows


def summary(rows, policy, load_ms, marginal=False):
    all_ms = [float(x["elapsed_ms"]) for x in rows]
    valid = [x for x in rows if x["status"] == "PASS"]
    fails = [x for x in rows if x["status"] != "PASS"]
    alloc = [float(x["max_memory_allocated_mib"]) for x in valid if x.get("max_memory_allocated_mib") is not None]
    marg = [float(x["marginal_elapsed_ms"]) for x in valid if x.get("marginal_elapsed_ms") is not None and x.get("baseline_nonselection_status") == "PASS"]
    marg_alloc = [float(x["marginal_peak_allocated_mib"]) for x in valid if x.get("marginal_peak_allocated_mib") is not None and x.get("baseline_nonselection_status") == "PASS"]
    return {"session": "r1_v2", "policy": policy, "attempts": len(rows), "valid": len(valid), "failed": len(fails), "fail_classes": {k: sum(1 for x in fails if (x.get("error") or "").startswith(k)) for k in sorted({(x.get("error") or "UNKNOWN").split(":", 1)[0] for x in fails})}, "p50_ms": percentile(all_ms, .5), "p95_ms": percentile(all_ms, .95), "fail_p50_ms": percentile([float(x["elapsed_ms"]) for x in fails], .5), "alloc_p95_mib": percentile(alloc, .95), "load_ms": load_ms, "marginal_p50_ms": percentile(marg, .5) if marginal else None, "marginal_alloc_mib": percentile(marg_alloc, .5) if marginal else None, "input_label_blind": True, "physical_forward_count": "NOT_EXPOSED_BY_RUNTIME", "latency_boundary": "request construction + native inference + score parsing + ID mapping + Top-5 packaging"}


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", default=None, help="append-only output root for a continuation run")
    args = ap.parse_args()
    if args.out_root:
        OUT = (ROOT / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "raw").mkdir(exist_ok=True); (OUT / "export").mkdir(exist_ok=True)
    guard_before = guard_gpu(args.gpu); views = load_rows(); views = views[: args.limit] if args.limit else views
    if len(views) != 120 and not args.smoke: raise RuntimeError(f"R1 requires 120 views, got {len(views)}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    results: dict[str, list[dict[str, Any]]] = {k: [] for k in ("pplx_budget", "pplx_two_stage", "jina_full", "jina_pplx_budget")}
    session = {"created_utc": now(), "gpu_requested": args.gpu, "gpu_pre": guard_before, "bright_sha256": file_sha(BRIGHT), "bright_records": 1384, "e12_lock_sha256": file_sha(LOCK), "budget_manifest_sha256": file_sha(BUDGET), "views": len(views), "repeats": args.repeats, "concurrency": 1, "models": {"pplx": PPLX_REV, "jina": JINA_REV}, "labels_joined": False, "protected_holdouts_accessed": False, "paid_api_calls": 0}
    # PPLX session: budgeted plus marginal-cost baseline and two-stage timings.
    load_t = time.perf_counter(); model, proc, tok, decision_messages = load_pplx(torch); load_ms = round((time.perf_counter() - load_t) * 1000, 3); torch.cuda.synchronize()
    for rep in range(1, args.repeats + 1):
        results["pplx_budget"].extend(run_pplx_budget(views, model, torch, rep, args.gpu, proc, decision_messages, tok, with_marginal=True))
        results["pplx_two_stage"].extend(run_two_stage(views, model, torch, rep, args.gpu, proc, decision_messages, tok))
    session["pplx_load_ms"] = load_ms
    del model, proc, tok, decision_messages; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    # Jina session: full original and Jina-on-identical PPLX-budgeted strings.
    load_t = time.perf_counter(); jmodel, jtok = load_jina(torch); load_jina_ms = round((time.perf_counter() - load_t) * 1000, 3); torch.cuda.synchronize()
    for rep in range(1, args.repeats + 1):
        results["jina_full"].extend(run_jina(views, jmodel, torch, rep, args.gpu, False))
        results["jina_pplx_budget"].extend(run_jina(views, jmodel, torch, rep, args.gpu, True))
    session["jina_load_ms"] = load_jina_ms
    del jmodel, jtok; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    guard_after = guard_gpu(args.gpu, allow_self=True); session["gpu_post_within_process"] = guard_after; session["status"] = "PASS" if guard_after["isolated"] else "BLOCKED"; session["raw_sha256"] = {}
    for policy, rows in results.items():
        path = OUT / "raw" / f"{policy}_R1_RAW_V2.jsonl"; save_jsonl(path, rows); session["raw_sha256"][str(path.relative_to(ROOT))] = file_sha(path)
        meta = summary(rows, policy, load_ms if policy.startswith("pplx") else load_jina_ms, marginal=(policy == "pplx_budget")); (OUT / "raw" / f"{policy}_R1_SUMMARY_V2.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    session["session_sha256"] = sha(session); (OUT / "SESSION_METADATA.json").write_text(json.dumps(session, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    export = []
    for policy, rows in results.items():
        export.append(summary(rows, policy, load_ms if policy.startswith("pplx") else load_jina_ms, marginal=(policy == "pplx_budget")))
    fields = ["session", "policy", "attempts", "valid", "p50_ms", "p95_ms", "fail_p50_ms", "alloc_p95_mib", "load_ms", "marginal_p50_ms", "marginal_alloc_mib"]
    with (OUT / "export/serving_r1.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows([{k: row.get(k) for k in fields} for row in export])
    (OUT / "export/R1_EXECUTION_RECEIPT.json").write_text(json.dumps({"status": session["status"], "attempts_by_policy": {p: len(v) for p, v in results.items()}, "valid_by_policy": {p: sum(x["status"] == "PASS" for x in v) for p, v in results.items()}, "failed_by_policy": {p: sum(x["status"] != "PASS" for x in v) for p, v in results.items()}, "labels_joined": False, "raw_sha256": session["raw_sha256"], "gpu_uuid": guard_before["uuid"], "gpu_isolated_before_process_exit": bool(guard_before["isolated"]), "gpu_post_in_process_external_isolation": bool(guard_after["isolated"]), "post_process_exit_check_required": True}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": session["status"], "gpu": args.gpu, "gpu_uuid": guard_before["uuid"], "views": len(views), "attempts": {p: len(v) for p, v in results.items()}, "valid": {p: sum(x["status"] == "PASS" for x in v) for p, v in results.items()}, "raw_sha256": session["raw_sha256"], "post_process_exit_check_required": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
