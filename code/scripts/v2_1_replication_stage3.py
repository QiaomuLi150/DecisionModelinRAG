#!/usr/bin/env python3
"""Stage 3 matched serving timing for the V2-HYBRID replication.

This runner is label-blind and append-only.  It freezes the deterministic
serving cohorts and protocol before loading a model, then measures the five
declared selector policies in one serial session on one dedicated GPU.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/v2_1_replication/stage0_preflight_v2_hybrid/INPUT_MANIFEST_V2_HYBRID.jsonl"
PROTOCOL = ROOT / "results/v2_1_replication/PROTOCOL_V2.json"
OUT = ROOT / "results/v2_1_replication/stage3_serving_v2_hybrid"
RAW = OUT / "raw/stage3_perrequest.jsonl"
PART = RAW.with_suffix(RAW.suffix + ".part")
COHORT = OUT / "STAGE3_COHORT.json"
GATE = OUT / "STAGE3_RESOURCE_GATE.json"
STAGE3_PROTOCOL = OUT / "PROTOCOL_STAGE3.json"
STAGE3_PROTOCOL_SHA = OUT / "PROTOCOL_STAGE3.sha256"
POLICIES = ["jina_full", "jina_budget", "jina_two_stage_matched", "pplx_budget", "pplx_two_stage"]
DATASETS = ["enterprise", "musique"]
SEEDS = {"enterprise": 20261020, "musique": 20261021}
REPEATS = 3


def canon(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(x: Any) -> str:
    return hashlib.sha256(canon(x).encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def command(args: list[str]) -> dict[str, Any]:
    p = subprocess.run(args, capture_output=True, text=True, timeout=30)
    return {"status": "PASS" if p.returncode == 0 else "ERROR", "returncode": p.returncode,
            "stdout": p.stdout.strip(), "stderr": p.stderr.strip(), "command": " ".join(args)}


def resource_gate(gpu: int) -> dict[str, Any]:
    snap = command(["nvidia-smi", "-i", str(gpu), "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"])
    procs = command(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"])
    if snap["returncode"] != 0 or not snap["stdout"]:
        raise RuntimeError("GPU_QUERY_FAILED")
    row = [x.strip() for x in snap["stdout"].splitlines()[0].split(",")]
    if int(row[0]) != gpu:
        raise RuntimeError("GPU_INDEX_MISMATCH")
    if procs["stdout"].strip():
        raise RuntimeError(f"DEDICATED_GPU_NOT_ISOLATED:{procs['stdout']}")
    if int(row[5]) < 70000 or int(row[6]) != 0:
        raise RuntimeError(f"DEDICATED_GPU_RESOURCE_GATE_FAILED:{row}")
    return {"status": "PASS_DEDICATED_GPU_AVAILABLE", "gpu_index": gpu, "device_row": row,
            "process_snapshot": procs, "resident_process_present": False,
            "memory_free_mib": int(row[5]), "utilization_percent": int(row[6]), "checked_utc": now()}


def freeze_cohorts(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for dataset in DATASETS:
        ids = sorted({str(r["canonical_id"]) for r in rows if r["dataset"] == dataset})
        if len(ids) < 120:
            raise RuntimeError(f"INSUFFICIENT_SERVING_COHORT:{dataset}:{len(ids)}")
        rng = random.Random(SEEDS[dataset])
        result[dataset] = sorted(rng.sample(ids, 120))
    return result


def choice_request(query: str, docs: list[dict[str, str]]) -> dict[str, Any]:
    return {"state": f"Query: {query}\nEvaluate the supplied candidate passages using the option descriptions.",
            "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {query}",
                         "criteria": {f"candidate_{i:03d}": f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(docs)}}}


def stable_top(ids: list[str], scores: list[float], k: int = 5) -> list[str]:
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i, ids[i]))[:k]]


def model_sha_request(query: str, docs: list[dict[str, str]]) -> str:
    return sha({"query": query, "candidate_ids": [x["candidate_id"] for x in docs],
                "candidate_texts": [x["content"] for x in docs]})


def load_views(rows: list[dict[str, Any]], cohort: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    original_rows = [r for r in rows if r.get("order_id") == "original"]
    if len(original_rows) != 2667 or len({str(r["canonical_id"]) for r in original_rows}) != 2667:
        raise RuntimeError(f"ORIGINAL_ORDER_MANIFEST_INVALID:{len(original_rows)}")
    by_id = {str(r["canonical_id"]): r for r in original_rows}
    views: dict[str, list[dict[str, Any]]] = {d: [] for d in DATASETS}
    for dataset, ids in cohort.items():
        for cid in ids:
            r = by_id[cid]
            views[dataset].append({
                "canonical_id": cid, "dataset": dataset, "query": str(r["query"]),
                "group_id": str(r.get("group_id", "")), "source_slate_sha256": r["source_slate_sha256"],
                "original": [{"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in r["candidates"]],
                "budgeted": [{"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in r["budgeted_candidates"]],
                "window_a": [{"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in r["window_a_candidates"]],
                "window_b": [{"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in r["window_b_candidates"]],
            })
    return views


def jina_score(model, query: str, docs: list[dict[str, str]]) -> list[float]:
    result = model.rerank(query, [x["content"] for x in docs])
    mapping = {int(x["index"]): float(x["relevance_score"]) for x in result}
    if set(mapping) != set(range(len(docs))):
        raise ValueError("INVALID_JINA_SCORE_VECTOR")
    return [mapping[i] for i in range(len(docs))]


def pplx_score(model, req: dict[str, Any], n: int) -> list[float]:
    vals = [float(x) for x in model.predict([req], batch_size=1)[0]]
    if len(vals) != n:
        raise ValueError(f"INVALID_PPLX_SCORE_VECTOR:{len(vals)}:{n}")
    return vals


def cuda_stats(torch) -> dict[str, float | None]:
    try:
        torch.cuda.synchronize()
        return {"alloc_peak_mib": round(float(torch.cuda.max_memory_allocated()) / 2**20, 3),
                "reserved_peak_mib": round(float(torch.cuda.max_memory_reserved()) / 2**20, 3)}
    except Exception:
        return {"alloc_peak_mib": None, "reserved_peak_mib": None}


def run_request(view: dict[str, Any], policy: str, attempt: int, gpu: int, torch, model, proc, decision_messages) -> dict[str, Any]:
    row: dict[str, Any] = {"dataset": view["dataset"], "canonical_id": view["canonical_id"], "policy": policy,
                           "attempt": attempt, "status": "FAIL", "valid": False, "error_class": None,
                           "error": None, "latency_ms": None, "alloc_peak_mib": None, "reserved_peak_mib": None,
                           "candidate_ids": [], "selected_ids": [], "scores": [], "stage_records": [],
                           "n_calls": 0, "model_revision": "14c9d90137962d300f3de95633f7b6535e80c5ce" if policy.startswith("pplx") else "e8a93f33f0b22108f8c2364f8484ce3422552fbc",
                           "source_slate_sha256": view["source_slate_sha256"], "labels_joined": False,
                           "input_label_blind": True, "gpu_index_requested": gpu,
                           "timed_boundary": "view selection + native scoring + parsing + ID mapping + Top-5 packaging"}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        if policy == "jina_full":
            docs = view["original"]; ids = [x["candidate_id"] for x in docs]; scores = jina_score(model, view["query"], docs)
            row.update({"candidate_ids": ids, "scores": scores, "selected_ids": stable_top(ids, scores), "n_calls": 1,
                        "native_input_sha256": model_sha_request(view["query"], docs)})
        elif policy == "jina_budget":
            docs = view["budgeted"]; ids = [x["candidate_id"] for x in docs]; scores = jina_score(model, view["query"], docs)
            row.update({"candidate_ids": ids, "scores": scores, "selected_ids": stable_top(ids, scores), "n_calls": 1,
                        "native_input_sha256": model_sha_request(view["query"], docs)})
        elif policy == "jina_two_stage_matched":
            windows = [view["window_a"], view["window_b"]]; by_id = {x["candidate_id"]: x for w in windows for x in w}; survivors = []; records = []
            for i, docs in enumerate(windows, 1):
                ids = [x["candidate_id"] for x in docs]; scores = jina_score(model, view["query"], docs); keep = stable_top(ids, scores); survivors.extend(keep)
                records.append({"stage": f"stage1_window{i}", "candidate_ids": ids, "scores": scores, "input_sha256": model_sha_request(view["query"], docs)})
            final_docs = [by_id[x] for x in survivors]; final_ids = [x["candidate_id"] for x in final_docs]; scores = jina_score(model, view["query"], final_docs)
            records.append({"stage": "stage2_survivors", "candidate_ids": final_ids, "scores": scores, "input_sha256": model_sha_request(view["query"], final_docs)})
            row.update({"candidate_ids": [x["candidate_id"] for w in windows for x in w], "stage_selected_ids": survivors, "scores": scores,
                        "selected_ids": stable_top(final_ids, scores), "stage_records": records, "n_calls": 3,
                        "native_input_sha256": sha([x["input_sha256"] for x in records])})
        elif policy == "pplx_budget":
            docs = view["budgeted"]; ids = [x["candidate_id"] for x in docs]; req = choice_request(view["query"], docs); scores = pplx_score(model, req, len(docs))
            row.update({"candidate_ids": ids, "scores": scores, "selected_ids": stable_top(ids, scores), "n_calls": 1,
                        "native_input_sha256": sha(req)})
        elif policy == "pplx_two_stage":
            windows = [view["window_a"], view["window_b"]]; survivors = []; records = []; by_id = {x["candidate_id"]: x for w in windows for x in w}
            for i, docs in enumerate(windows, 1):
                req = choice_request(view["query"], docs); ids = [x["candidate_id"] for x in docs]; scores = pplx_score(model, req, len(docs)); keep = stable_top(ids, scores); survivors.extend(keep)
                records.append({"stage": f"stage1_window{i}", "candidate_ids": ids, "scores": scores, "input_sha256": sha(req)})
            final_docs = [by_id[x] for x in survivors]; final_ids = [x["candidate_id"] for x in final_docs]; req2 = choice_request(view["query"], final_docs); scores = pplx_score(model, req2, len(final_docs))
            records.append({"stage": "stage2_survivors", "candidate_ids": final_ids, "scores": scores, "input_sha256": sha(req2)})
            row.update({"candidate_ids": [x["candidate_id"] for w in windows for x in w], "stage_selected_ids": survivors, "scores": scores,
                        "selected_ids": stable_top(final_ids, scores), "stage_records": records, "n_calls": 3,
                        "native_input_sha256": sha([x["input_sha256"] for x in records])})
        else:
            raise ValueError(f"UNKNOWN_POLICY:{policy}")
        if len(row["selected_ids"]) != 5 or len(set(row["selected_ids"])) != 5:
            raise ValueError("INVALID_TOP5_SELECTION")
        row["status"] = "PASS"; row["valid"] = True
    except Exception as exc:
        row["error_class"] = type(exc).__name__; row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try: torch.cuda.synchronize()
        except Exception: pass
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        row.update(cuda_stats(torch))
    return row


def load_jina(torch, jina_path: Path):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(jina_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(jina_path), dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda:0").eval()
    torch.cuda.synchronize()
    return model


def load_pplx(torch, pplx_path: Path):
    sys.path.insert(0, str(pplx_path / "source" / "src"))
    from autojev.model import DecisionModel, decision_messages
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(str(pplx_path), local_files_only=True)
    model = DecisionModel(pplx_path, device="cuda:0", max_image_pixels=512 * 512)
    torch.cuda.synchronize()
    return proc, model, decision_messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    if args.repeats != REPEATS:
        raise RuntimeError("STAGE3_REPEATS_MUST_BE_3")
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "raw").mkdir(exist_ok=True)
    rows = read_jsonl(MANIFEST)
    if len(rows) != 10668:
        raise RuntimeError(f"MANIFEST_ROW_COUNT:{len(rows)}")
    gate = resource_gate(args.gpu)
    GATE.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    cohort = freeze_cohorts(rows)
    cohort_doc = {"status": "FROZEN_PRE_OUTCOME", "created_utc": now(), "seed_by_dataset": SEEDS,
                  "cohort_size_by_dataset": {d: len(v) for d, v in cohort.items()}, "canonical_ids": cohort,
                  "selection_rule": "sorted canonical IDs, deterministic random.sample, label-blind, before Stage3 model loading",
                  "input_manifest_sha256": file_sha(MANIFEST), "protocol_v2_sha256": file_sha(PROTOCOL), "labels_joined": False}
    if COHORT.exists():
        old = json.loads(COHORT.read_text())
        if old.get("canonical_ids") != cohort or old.get("input_manifest_sha256") != cohort_doc["input_manifest_sha256"]:
            raise RuntimeError("EXISTING_STAGE3_COHORT_MISMATCH")
        cohort_doc = old
    else:
        COHORT.write_text(json.dumps(cohort_doc, indent=2, sort_keys=True) + "\n")
    protocol = {"status": "FROZEN_PRE_OUTCOME", "created_utc": now(), "gpu_index": args.gpu,
                "gpu_uuid": gate["device_row"][1], "repeats": REPEATS, "concurrency": 1,
                "policy_order": POLICIES, "datasets": DATASETS, "cohort_sha256": file_sha(COHORT),
                "input_manifest_sha256": file_sha(MANIFEST), "protocol_v2_sha256": file_sha(PROTOCOL),
                "models": {"jina_revision": "e8a93f33f0b22108f8c2364f8484ce3422552fbc", "pplx_revision": "14c9d90137962d300f3de95633f7b6535e80c5ce"},
                "timed_boundary": "view selection + native scoring + parsing + ID mapping + Top-5 packaging",
                "model_loading_excluded": True, "labels_joined": False, "no_retries": True,
                "protected_holdouts_accessed": False, "paid_api_calls": 0,
                "ratio_bootstrap": {"draws": 10000, "seed": 731, "cluster": "canonical_id", "basis": "all requests"}}
    if STAGE3_PROTOCOL.exists():
        old = json.loads(STAGE3_PROTOCOL.read_text())
        if old.get("cohort_sha256") != protocol["cohort_sha256"] or old.get("policy_order") != POLICIES:
            raise RuntimeError("EXISTING_STAGE3_PROTOCOL_MISMATCH")
        protocol = old
    else:
        STAGE3_PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
        STAGE3_PROTOCOL_SHA.write_text(file_sha(STAGE3_PROTOCOL) + "  PROTOCOL_STAGE3.json\n")
    if PART.exists() or RAW.exists():
        raise RuntimeError("STAGE3_RAW_ALREADY_EXISTS_RUNNER_IS_APPEND_ONLY")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    views = load_views(rows, cohort)
    all_rows: list[dict[str, Any]] = []
    with PART.open("w", encoding="utf-8") as out:
        proc = None; jina = None; pplx = None; decision_messages = None
        pplx_path = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/14c9d90137962d300f3de95633f7b6535e80c5ce")
        jina_path = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/e8a93f33f0b22108f8c2364f8484ce3422552fbc")
        for policy in POLICIES:
            if policy.startswith("jina") and jina is None:
                jina = load_jina(torch, jina_path)
            if policy.startswith("pplx") and pplx is None:
                del jina; jina = None; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
                proc, pplx, decision_messages = load_pplx(torch, pplx_path)
            model = jina if policy.startswith("jina") else pplx
            for dataset in DATASETS:
                for attempt in range(1, REPEATS + 1):
                    for view in views[dataset]:
                        row = run_request(view, policy, attempt, args.gpu, torch, model, proc, decision_messages)
                        out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"); out.flush(); all_rows.append(row)
                    print(json.dumps({"policy": policy, "dataset": dataset, "attempt": attempt, "completed": len(all_rows), "valid": sum(x["valid"] for x in all_rows), "failures": sum(not x["valid"] for x in all_rows)}), flush=True)
        if pplx is not None:
            del pplx
        if jina is not None:
            del jina
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    os.replace(PART, RAW)
    summary = {"status": "COMPLETE", "created_utc": now(), "n_rows": len(all_rows), "valid": sum(x["valid"] for x in all_rows),
               "failed": sum(not x["valid"] for x in all_rows), "policies": POLICIES, "cohort_sizes": {d: len(v) for d, v in cohort.items()},
               "raw_sha256": file_sha(RAW), "protocol_sha256": file_sha(STAGE3_PROTOCOL), "labels_joined": False,
               "protected_holdouts_accessed": False, "paid_api_calls": 0}
    (OUT / "STAGE3_RAW_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
