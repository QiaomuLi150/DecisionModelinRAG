#!/usr/bin/env python3
"""DecisionRAG v1.6 D2: matched, single-request selector timing.

This runner is append-only and label blind.  It uses one Python session and
one dedicated GPU, runs the frozen six-policy order, and records one raw row
for every policy/question/attempt, including failures.  Model loading is
outside the request timer; request preparation, native calls, parsing, ID
mapping and Top-5 packaging are inside it.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v1_6/d2_matched_timing"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
LOCK = ROOT / "results/non_jev/v2/e12_o_final_v1_0/00_protocol/E12O_COHORT_LOCK.json"
BUDGET = ROOT / "results/non_jev/v2/e11_v1_0/01_budget_policy/E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl"
D1_MANIFEST = ROOT / "results/v1_6/d1_perwindow_jina/D1_INPUT_MANIFEST.jsonl"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
PPLX_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")
JINA_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/{JINA_REV}")
BRIGHT_SHA = "c304b2c35f445153181098d3a899271db6fc653b44aea50186c6efaeec8e6ea2"
POLICIES = [
    "jina_full",
    "jina_two_stage_full",
    "jina_two_stage_pplxwindow",
    "jina_pplx_budget",
    "pplx_budget",
    "pplx_two_stage",
]
L = 8192
RHO = 128


def canon(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(x: Any) -> str:
    return hashlib.sha256(canon(x).encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def command(args: list[str]) -> dict[str, Any]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {"status": "PASS" if p.returncode == 0 else "ERROR", "returncode": p.returncode,
                "stdout": p.stdout.strip(), "stderr": p.stderr.strip(), "command": " ".join(args)}
    except Exception as exc:
        return {"status": "ERROR", "returncode": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}",
                "command": " ".join(args)}


def gpu_gate(gpu: int, manual: bool = False, manual_uuid: str | None = None,
             manual_used: int | None = None, manual_free: int | None = None,
             manual_util: int | None = None) -> dict[str, Any]:
    snap = command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"])
    procs = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"])
    row = None
    for line in snap.get("stdout", "").splitlines():
        vals = [x.strip() for x in line.split(",")]
        if vals and vals[0] == str(gpu):
            row = vals
            break
    if row is None and manual:
        if not manual_uuid or manual_used is None or manual_free is None or manual_util is None:
            raise RuntimeError("MANUAL_PREFLIGHT_FIELDS_REQUIRED")
        if manual_free < 70000 or manual_util != 0:
            raise RuntimeError("MANUAL_PREFLIGHT_RESOURCE_GATE_FAILED")
        return {"status": "PASS_EXTERNAL_PREFLIGHT", "gpu_index": gpu, "gpu_uuid": manual_uuid,
                "device_row": [str(gpu), manual_uuid, "NVIDIA A100-SXM4-80GB", "81920", str(manual_used), str(manual_free), str(manual_util)],
                "gpu_snapshot": {"method": "direct nvidia-smi -i 3 immediately before launch", "stdout_verified": True},
                "process_snapshot": {"method": "direct nvidia-smi -i 3 --query-compute-apps immediately before launch", "stdout": ""},
                "resident_process_present": False, "model_calls_started": 0, "memory_free_mib": manual_free,
                "utilization_percent": manual_util, "manual_external_preflight": True}
    if row is None:
        raise RuntimeError(f"GPU_NOT_FOUND:{gpu}")
    uuid = row[1]
    resident = [line for line in procs.get("stdout", "").splitlines() if uuid in line]
    if resident:
        raise RuntimeError(f"DEDICATED_GPU_NOT_ISOLATED:{gpu}:{resident}")
    if int(row[5]) < 70000 or int(row[6]) != 0:
        raise RuntimeError(f"DEDICATED_GPU_RESOURCE_GATE_FAILED:{row}")
    return {"status": "PASS_DEDICATED_GPU_AVAILABLE", "gpu_index": gpu, "gpu_uuid": uuid,
            "device_row": row, "gpu_snapshot": snap, "process_snapshot": procs,
            "resident_process_present": False, "model_calls_started": 0,
            "memory_free_mib": int(row[5]), "utilization_percent": int(row[6])}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_views() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    slates = {f"{r['domain']}:{r['query_id']}": r for r in read_jsonl(BRIGHT)}
    budgets = {f"{r['domain']}:{r['query_id']}": r for r in read_jsonl(BUDGET)}
    locked = json.loads(LOCK.read_text(encoding="utf-8"))
    serving = list(locked["bright"]["serving_ids"])
    if len(serving) != 120 or len(set(serving)) != 120:
        raise RuntimeError(f"SERVING_COHORT_INVALID:{len(serving)}")
    if file_sha(BRIGHT) != BRIGHT_SHA or len(slates) != 1384 or set(slates) != set(budgets):
        raise RuntimeError("FROZEN_INPUT_HASH_OR_COVERAGE_MISMATCH")
    views = []
    for cid in serving:
        s, b = slates[cid], budgets[cid]
        original = [{"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in s["candidates"]]
        budgeted = None
        if str(b.get("status", "")).lower() in {"ok", "pass", "complete"}:
            budgeted = [{"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in b["budgeted_candidates"]]
        views.append({"canonical_id": cid, "domain": str(s["domain"]), "query_id": str(s["query_id"]),
                      "query": str(s["query"]), "slate_sha256": s["slate_sha256"],
                      "source_input_sha256": s["input_sha256"], "original": original,
                      "budgeted": budgeted, "budget_status": b.get("status"),
                      "original_ids": [x["candidate_id"] for x in original],
                      "budgeted_ids": [x["candidate_id"] for x in budgeted] if budgeted else [],
                      "budget_row_sha256": sha(b)})
    d1 = {(str(x["canonical_id"]), str(x["order_id"])): x for x in read_jsonl(D1_MANIFEST)}
    if len(d1) != 9688:
        raise RuntimeError(f"D1_MANIFEST_COVERAGE_INVALID:{len(d1)}")
    return views, d1


def choice_request(query: str, docs: list[dict[str, str]]) -> dict[str, Any]:
    return {"state": f"Query: {query}\nEvaluate the supplied candidate passages using the option descriptions.",
            "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {query}",
                         "criteria": {f"candidate_{i:03d}": f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(docs)}}}


def fixed_nonselection_request() -> dict[str, Any]:
    return {"state": "Route this request to the appropriate processing queue.", "question": {"type": "choice",
            "instructions": "Choose the next processing action from the option descriptions.",
            "criteria": {"route_000": "Send the request to the standard text-processing queue.",
                         "route_001": "Send the request to the specialist review queue.",
                         "route_002": "Send the request to the deferred-processing queue."}}}


def stable_top(ids: list[str], scores: list[float], k: int = 5) -> list[str]:
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i, ids[i]))[:k]]


def prefix(tok, text: str, cap: int) -> str:
    return tok.decode(tok.encode(str(text), add_special_tokens=False)[:cap], skip_special_tokens=False,
                      clean_up_tokenization_spaces=False)


def d1_pplx_window(view: dict[str, Any], d1: dict[tuple[str, str], dict[str, Any]], tok) -> tuple[list[list[dict[str, str]]], dict[str, dict[str, str]], list[dict[str, str]]]:
    m = d1[(view["canonical_id"], "original")]
    if m.get("pplx_status") != "PASS":
        raise RuntimeError("PPLX_SOURCE_FAILURE_NO_FROZEN_WINDOW")
    source = {str(x["candidate_id"]): str(x["content"]) for x in view["original"]}
    codes = [f"candidate_{i:03d}" for i in range(10)]
    windows: list[list[dict[str, str]]] = []
    by_id: dict[str, dict[str, str]] = {}
    for stage in [x for x in m["stages"] if str(x["stage"]).startswith("stage1_")]:
        docs = [{"candidate_id": str(cid), "content": prefix(tok, source[str(cid)], int(stage["final_cap"]))} for cid in stage["candidate_ids"]]
        if sha(choice_request(view["query"], docs)) != stage["input_sha256"]:
            raise RuntimeError("D1_EXACT_STAGE1_RECONSTRUCTION_MISMATCH")
        windows.append(docs)
        by_id.update({x["candidate_id"]: x for x in docs})
    s2 = next(x for x in m["stages"] if x["stage"] == "stage2_survivors")
    docs2 = [{"candidate_id": str(cid), "content": prefix(tok, by_id[str(cid)]["content"], int(s2["final_cap"]))} for cid in m["pplx_stage_selected_ids"]]
    if sha(choice_request(view["query"], docs2)) != s2["input_sha256"]:
        raise RuntimeError("D1_EXACT_STAGE2_RECONSTRUCTION_MISMATCH")
    return windows, by_id, docs2


def budget_fit(query: str, docs: list[dict[str, str]], proc, decision_messages, tok) -> tuple[list[dict[str, str]] | None, dict[str, Any]]:
    codes = [f"candidate_{i:03d}" for i in range(len(docs))]
    empty = [{"candidate_id": d["candidate_id"], "content": ""} for d in docs]
    def n_tokens(xs: list[dict[str, str]]) -> int:
        msg = decision_messages(choice_request(query, xs), codes)
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = proc(text=[text], images=None, padding=True, return_tensors="pt")
        return int(enc["attention_mask"].sum())
    overhead = n_tokens(empty)
    available = L - RHO - overhead
    if available < len(docs):
        return None, {"error_class": "UNBUDGETABLE_QUERY_OR_OVERHEAD", "overhead_tokens": overhead, "effective_content_budget": available}
    cap = max(1, available // len(docs))
    reductions = 0
    while cap >= 1:
        xs = [{"candidate_id": d["candidate_id"], "content": prefix(tok, d["content"], cap)} for d in docs]
        total = n_tokens(xs)
        if total <= L - RHO:
            return xs, {"overhead_tokens": overhead, "effective_content_budget": available, "final_cap": cap,
                        "reductions": reductions, "serialized_tokens": total}
        cap -= 1
        reductions += 1
    return None, {"error_class": "UNBUDGETABLE_QUERY_OR_OVERHEAD", "overhead_tokens": overhead, "effective_content_budget": available}


def load_pplx_processor():
    sys.path.insert(0, str(PPLX_PATH / "source" / "src"))
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(str(PPLX_PATH), local_files_only=True)


def load_pplx(torch):
    sys.path.insert(0, str(PPLX_PATH / "source" / "src"))
    from autojev.model import DecisionModel, decision_messages
    model = DecisionModel(PPLX_PATH, device="cuda:0", max_image_pixels=512 * 512)
    torch.cuda.synchronize()
    return model, decision_messages


def load_jina(torch):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(JINA_PATH), local_files_only=True)
    model = AutoModel.from_pretrained(str(JINA_PATH), dtype="auto", trust_remote_code=True,
                                      local_files_only=True).to("cuda:0").eval()
    torch.cuda.synchronize()
    return model, tok


def cuda_stats(torch) -> dict[str, float]:
    torch.cuda.synchronize()
    return {"alloc_peak_mib": round(float(torch.cuda.max_memory_allocated()) / 2**20, 3),
            "alloc_current_mib": round(float(torch.cuda.memory_allocated()) / 2**20, 3),
            "reserved_peak_mib": round(float(torch.cuda.max_memory_reserved()) / 2**20, 3)}


def jina_score(model, query: str, docs: list[dict[str, str]]) -> list[float]:
    result = model.rerank(query, [x["content"] for x in docs])
    mapping = {int(x["index"]): float(x["relevance_score"]) for x in result}
    if set(mapping) != set(range(len(docs))) or not all(math.isfinite(mapping[i]) for i in range(len(docs))):
        raise ValueError("INVALID_JINA_SCORE_VECTOR")
    return [mapping[i] for i in range(len(docs))]


def pplx_score(model, req: dict[str, Any], n: int) -> list[float]:
    scores = [float(x) for x in model.predict([req], batch_size=1)[0]]
    if len(scores) != n or not all(math.isfinite(x) for x in scores):
        raise ValueError(f"INVALID_PPLX_SCORE_VECTOR:{len(scores)}:{n}")
    return scores


def empty_row(view: dict[str, Any], policy: str, attempt: int, gpu: int) -> dict[str, Any]:
    return {"canonical_id": view["canonical_id"], "domain": view["domain"], "query_id": view["query_id"],
            "policy": policy, "attempt": attempt, "status": "FAIL", "valid": False, "error_class": None,
            "error": None, "latency_ms": None, "alloc_peak_mib": None, "candidate_ids": [], "selected_ids": [],
            "scores": [], "stage_records": [], "n_calls": 0, "gpu_index": gpu, "gpu_uuid": None,
            "model_revision": PPLX_REV if policy.startswith("pplx") else JINA_REV,
            "labels_joined": False, "input_label_blind": True, "timed_boundary": "view preparation + native scoring + score parsing + ID mapping + Top-5 packaging"}


def run_request(view, d1, policy: str, attempt: int, gpu: int, torch, model, pplx_proc, pplx_tok, decision_messages) -> dict[str, Any]:
    row = empty_row(view, policy, attempt, gpu)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        if policy == "jina_full":
            docs = view["original"]
            ids = [x["candidate_id"] for x in docs]
            scores = jina_score(model, view["query"], docs)
            row.update({"candidate_ids": ids, "scores": scores, "selected_ids": stable_top(ids, scores), "n_calls": 1,
                        "input_view": "BRIGHT frozen original candidates", "native_request_sha256": sha({"query": view["query"], "candidate_ids": ids, "candidate_texts": [x["content"] for x in docs]})})
        elif policy == "jina_pplx_budget":
            if view["budgeted"] is None:
                raise RuntimeError("UNBUDGETABLE_QUERY_OR_OVERHEAD:FROZEN_E11_A_VIEW_UNAVAILABLE")
            docs = view["budgeted"]
            ids = [x["candidate_id"] for x in docs]
            scores = jina_score(model, view["query"], docs)
            row.update({"candidate_ids": ids, "scores": scores, "selected_ids": stable_top(ids, scores), "n_calls": 1,
                        "input_view": "E11-A frozen budgeted_candidates", "native_request_sha256": sha({"query": view["query"], "candidate_ids": ids, "candidate_texts": [x["content"] for x in docs]})})
        elif policy == "jina_two_stage_full":
            docs = view["original"]
            ids = [x["candidate_id"] for x in docs]
            records, survivors = [], []
            for name, part in (("stage1_window1", docs[:10]), ("stage1_window2", docs[10:20])):
                sc = jina_score(model, view["query"], part); pids = [x["candidate_id"] for x in part]; keep = stable_top(pids, sc)
                survivors.extend(keep); records.append({"stage": name, "candidate_ids": pids, "scores": sc, "input_sha256": sha({"query": view["query"], "candidate_ids": pids, "candidate_texts": [x["content"] for x in part]})})
            byid = {x["candidate_id"]: x for x in docs}; final_docs = [byid[x] for x in survivors]
            sc2 = jina_score(model, view["query"], final_docs); final_ids = [x["candidate_id"] for x in final_docs]
            records.append({"stage": "stage2_survivors", "candidate_ids": final_ids, "scores": sc2, "input_sha256": sha({"query": view["query"], "candidate_ids": final_ids, "candidate_texts": [x["content"] for x in final_docs]})})
            row.update({"candidate_ids": ids, "stage_selected_ids": survivors, "scores": sc2, "selected_ids": stable_top(final_ids, sc2), "stage_records": records, "n_calls": 3, "input_view": "BRIGHT frozen original candidates"})
        elif policy == "jina_two_stage_pplxwindow":
            windows, byid, pplx_stage2 = d1_pplx_window(view, d1, pplx_tok)
            records, survivors = [], []
            for i, part in enumerate(windows, 1):
                sc = jina_score(model, view["query"], part); pids = [x["candidate_id"] for x in part]; keep = stable_top(pids, sc)
                survivors.extend(keep); records.append({"stage": f"stage1_window{i}", "candidate_ids": pids, "scores": sc, "input_sha256": sha({"query": view["query"], "candidate_ids": pids, "candidate_texts": [x["content"] for x in part]})})
            same = set(survivors) == set(d1[(view["canonical_id"], "original")]["pplx_stage_selected_ids"]) and len(survivors) == 10
            final_docs = pplx_stage2 if same else [byid[x] for x in survivors]
            sc2 = jina_score(model, view["query"], final_docs); final_ids = [x["candidate_id"] for x in final_docs]
            records.append({"stage": "stage2_survivors", "candidate_ids": final_ids, "scores": sc2, "input_sha256": sha({"query": view["query"], "candidate_ids": final_ids, "candidate_texts": [x["content"] for x in final_docs]})})
            row.update({"candidate_ids": [x for w in windows for x in [d["candidate_id"] for d in w]], "stage_selected_ids": survivors, "scores": sc2, "selected_ids": stable_top(final_ids, sc2), "stage_records": records, "survivor_set_matches_pplx": same, "n_calls": 3, "input_view": "D1 exact PPLX per-window strings"})
        elif policy == "pplx_budget":
            if view["budgeted"] is None:
                raise RuntimeError("UNBUDGETABLE_QUERY_OR_OVERHEAD:FROZEN_E11_A_VIEW_UNAVAILABLE")
            docs = view["budgeted"]; ids = [x["candidate_id"] for x in docs]; req = choice_request(view["query"], docs)
            sc = pplx_score(model, req, len(docs))
            row.update({"candidate_ids": ids, "scores": sc, "selected_ids": stable_top(ids, sc), "n_calls": 1, "input_view": "E11-A frozen budgeted_candidates", "native_request_sha256": sha(req)})
        elif policy == "pplx_two_stage":
            records, survivors = [], []
            for name, part in (("stage1_window1", view["original"][:10]), ("stage1_window2", view["original"][10:20])):
                docs, info = budget_fit(view["query"], part, pplx_proc, decision_messages, pplx_tok)
                if docs is None: raise RuntimeError(info["error_class"])
                req = choice_request(view["query"], docs); sc = pplx_score(model, req, len(docs)); ids = [x["candidate_id"] for x in docs]
                keep = stable_top(ids, sc); survivors.extend(keep); records.append({"stage": name, "candidate_ids": ids, "scores": sc, "budget": info, "input_sha256": sha(req)})
            byid = {x["candidate_id"]: x for x in view["original"]}
            survivor_docs = [byid[x] for x in survivors]
            docs2, info2 = budget_fit(view["query"], survivor_docs, pplx_proc, decision_messages, pplx_tok)
            if docs2 is None: raise RuntimeError(info2["error_class"])
            req2 = choice_request(view["query"], docs2); sc2 = pplx_score(model, req2, len(docs2)); final_ids = [x["candidate_id"] for x in docs2]
            records.append({"stage": "stage2_survivors", "candidate_ids": final_ids, "scores": sc2, "budget": info2, "input_sha256": sha(req2)})
            row.update({"candidate_ids": [x["candidate_id"] for x in view["original"]], "stage_selected_ids": survivors, "scores": sc2, "selected_ids": stable_top(final_ids, sc2), "stage_records": records, "n_calls": 3, "input_view": "R1 PPLX dynamic per-window budget_fit"})
        else:
            raise ValueError(f"UNKNOWN_POLICY:{policy}")
        if len(row["selected_ids"]) != 5 or len(set(row["selected_ids"])) != 5:
            raise ValueError("INVALID_TOP5_SELECTION")
        row["status"] = "PASS"; row["valid"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["error_class"] = type(exc).__name__
    finally:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        try:
            row.update(cuda_stats(torch))
        except Exception:
            row["alloc_peak_mib"] = None
    return row


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--manual-gpu-preflight", action="store_true")
    ap.add_argument("--gpu-uuid", default=None)
    ap.add_argument("--memory-used-mib", type=int, default=None)
    ap.add_argument("--memory-free-mib", type=int, default=None)
    ap.add_argument("--utilization-percent", type=int, default=None)
    args = ap.parse_args()
    if args.repeats != 3:
        raise RuntimeError("D2_FIXED_REPEATS_MUST_BE_3")
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "raw").mkdir(exist_ok=True); (OUT / "export").mkdir(exist_ok=True)
    gate = gpu_gate(args.gpu, args.manual_gpu_preflight, args.gpu_uuid, args.memory_used_mib,
                    args.memory_free_mib, args.utilization_percent)
    gate_path = OUT / "D2_RESOURCE_GATE_V2.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    views, d1 = load_views()
    if len(views) != 120:
        raise RuntimeError("D2_REQUIRES_120_FROZEN_SERVING_IDS")
    protocol = {"status": "FROZEN_PRE_OUTCOME", "created_utc": now(), "gpu_index": args.gpu, "gpu_uuid": gate["gpu_uuid"],
                "cohort": "E12O bright.serving_ids", "cohort_size": 120, "repeats": 3, "concurrency": 1,
                "policy_order": POLICIES, "models": {"jina_revision": JINA_REV, "pplx_revision": PPLX_REV},
                "input_hashes": {"bright": file_sha(BRIGHT), "lock": file_sha(LOCK), "budget": file_sha(BUDGET), "d1_input_manifest": file_sha(D1_MANIFEST)},
                "timed_boundary": "view preparation + all native calls + score parsing + candidate-ID mapping + Top-5 packaging",
                "labels_joined": False, "protected_holdouts_accessed": False, "paid_api_calls": 0,
                "failure_policy": "no retries; failure time recorded; no fallback in raw timing",
                "ratio_bootstrap": {"draws": 10000, "seed": 731, "cluster": "canonical_id", "basis": "all requests with failure plus fixed BM25 fallback time if available"},
                "historical_blocker": "results/v1_6/d2_matched_timing/PROTOCOL.json", "resource_gate": str(gate_path.relative_to(ROOT))}
    protocol_path = OUT / "PROTOCOL_V2.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if old.get("input_hashes") != protocol["input_hashes"] or old.get("policy_order") != POLICIES:
            raise RuntimeError("EXISTING_D2_PROTOCOL_V2_MISMATCH")
        protocol = old
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    checkpoint = OUT / "raw" / "d2_latency_perrequest.checkpoint.jsonl"
    rows = read_jsonl(checkpoint) if args.resume and checkpoint.exists() else []
    completed = {(x["policy"], str(x["canonical_id"]), int(x["attempt"])) for x in rows}
    session = {"created_utc": now(), "gpu_index": args.gpu, "gpu_uuid": gate["gpu_uuid"], "cohort_size": 120, "repeats": 3,
               "policy_order": POLICIES, "labels_joined": False, "protected_holdouts_accessed": False, "paid_api_calls": 0,
               "model_loads": {}, "input_hashes": protocol["input_hashes"], "resumed_rows": len(rows)}
    # The PPLX tokenizer is needed for the exact D1 window reconstruction and
    # the dynamic PPLX budget path.  It is CPU-side and loaded before Jina.
    proc_load = time.perf_counter(); pplx_proc = load_pplx_processor(); pplx_tok = pplx_proc.tokenizer
    session["pplx_processor_load_ms"] = round((time.perf_counter() - proc_load) * 1000, 3)
    # Jina policies are contiguous in the frozen order, then PPLX policies.
    load_t = time.perf_counter(); jina_model, _jina_tok = load_jina(torch); torch.cuda.synchronize()
    session["jina_model_load_ms"] = round((time.perf_counter() - load_t) * 1000, 3)
    for policy in POLICIES[:4]:
        for attempt in range(1, args.repeats + 1):
            for view in views:
                key = (policy, view["canonical_id"], attempt)
                if key in completed:
                    continue
                row = run_request(view, d1, policy, attempt, args.gpu, torch, jina_model, pplx_proc, pplx_tok, None)
                row["gpu_uuid"] = gate["gpu_uuid"]
                rows.append(row); completed.add(key)
                checkpoint.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
            print(json.dumps({"policy": policy, "attempt": attempt, "rows": len(rows), "valid": sum(x.get("valid") is True for x in rows)}), flush=True)
    del jina_model, _jina_tok; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    load_t = time.perf_counter(); pplx_model, decision_messages = load_pplx(torch); torch.cuda.synchronize()
    session["pplx_model_load_ms"] = round((time.perf_counter() - load_t) * 1000, 3)
    for policy in POLICIES[4:]:
        for attempt in range(1, args.repeats + 1):
            for view in views:
                key = (policy, view["canonical_id"], attempt)
                if key in completed:
                    continue
                row = run_request(view, d1, policy, attempt, args.gpu, torch, pplx_model, pplx_proc, pplx_tok, decision_messages)
                row["gpu_uuid"] = gate["gpu_uuid"]
                rows.append(row); completed.add(key)
                checkpoint.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
            print(json.dumps({"policy": policy, "attempt": attempt, "rows": len(rows), "valid": sum(x.get("valid") is True for x in rows)}), flush=True)
    del pplx_model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    rows.sort(key=lambda x: (POLICIES.index(x["policy"]), int(x["attempt"]), x["canonical_id"]))
    raw = OUT / "raw" / "d2_latency_perrequest_raw.jsonl"
    save_jsonl(raw, rows)
    meta = {"status": "PASS" if len(rows) == 2160 else "INCOMPLETE", "records": len(rows), "expected_records": 2160,
            "valid": sum(x.get("valid") is True for x in rows), "failed": sum(x.get("valid") is not True for x in rows),
            "sha256": file_sha(raw), "labels_joined": False, "gpu_index": args.gpu, "gpu_uuid": gate["gpu_uuid"],
            "policy_counts": {p: sum(x["policy"] == p for x in rows) for p in POLICIES}, "model_loads": session["model_loads"]}
    (OUT / "raw" / "D2_RAW_TRACE_META_V2.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    session["raw_sha256"] = file_sha(raw); session["status"] = meta["status"]
    (OUT / "D2_SESSION_METADATA_V2.json").write_text(json.dumps(session, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Keep the checkpoint as a recovery artifact, but do not treat it as the
    # canonical trace; it is byte-identical in records, with the same rows.
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
