#!/usr/bin/env python3
"""Versioned repair rerun for the v3.6 N2-E Winnow branch.

This script is append-only.  It creates a new pre-outcome receipt before any
native calls, runs four matched physical-GPU shards, and leaves the original
followup_v1 traces untouched.  Labels are never loaded by this runner.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("N2E_REPAIR_OUT", str(ROOT / "results/v3_6_essential/followup_v2_n2e_repair")))
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
STAGE0 = ROOT / "results/v3_6_essential/stage0_preflight"
RUNTIME = ROOT / "data/models/e12_o_runtimes/winnow-inference"
WINNOW_MODEL = ROOT / "data/models/e12_o_checkpoints/winnow12/gguf/Winnow-12B-Q8_0.gguf"
WINNOW_SERVER = RUNTIME / ".build-e12o/bin/winnow-server"
WINNOW_PY = RUNTIME / "scripts/serve.py"
PYTHON = ROOT.parents[2] / "MultiModalRAG/venv/bin/python"
GPUS = (3, 4, 5, 6)
PORTS = {3: 18103, 4: 18104, 5: 18105, 6: 18106}
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
WINNOW_REV = "b2b14213dfa252e6d6b543c8b334762e51772d29"
WINNOW_SHA = "b710efc4c0d048ee61eed92c5fef5ce323a4d17e7c51f9f0533cc72ae50818ea"
RUNTIME_LOCK_SHA = "21e6e69d8e43e318089e77459d102f05ee2cad266999c7860e64048f69c89e84"


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"IMPORT_FAILED:{name}")
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_obj(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def load_views_cpu():
    v36 = module("v36_n2e_views", ROOT / "scripts/v3_6_serving.py")
    views, _ = v36.load_views()
    views.sort(key=lambda x: x["canonical_id"])
    if len(views) != 360 or len({x["canonical_id"] for x in views}) != 360:
        raise RuntimeError(f"C360_INVALID:{len(views)}")
    return views


def input_manifest(views):
    rows = []
    for v in views:
        rows.append({
            "canonical_id": v["canonical_id"],
            "domain": v["domain"],
            "query_sha256": hashlib.sha256(v["query"].encode()).hexdigest(),
            "slate_sha256": v["slate_sha256"],
            "original_candidate_ids": v["original_ids"],
            "original_content_sha256": [hashlib.sha256(x["content"].encode()).hexdigest() for x in v["original"]],
            "budgeted_candidate_ids": v["budgeted_ids"],
            "budgeted_content_sha256": [hashlib.sha256(x["content"].encode()).hexdigest() for x in (v["budgeted"] or [])],
            "budget_status": v["budget_status"],
            "labels_joined": False,
        })
    return rows


def freeze() -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    views = load_views_cpu()
    rows = input_manifest(views)
    shards = {str(gpu): [r["canonical_id"] for r in rows[i * 90:(i + 1) * 90]] for i, gpu in enumerate(GPUS)}
    protocol = {
        "status": "FROZEN_PRE_OUTCOME",
        "experiment": "DecisionRAG v3.6 N2-E blocker-resolution rerun",
        "protocol_version": "v3.6-followup-v2-n2e-repair",
        "supersedes": "v3.6-followup-v1 N2 session E only; original artifacts immutable",
        "population": {"cohort": "C360", "queries": 360, "candidate_count": 20, "attempts": 3, "concurrency": 1},
        "candidate_policy": "original frozen BRIGHT Top-20 identities and text; PPLX anchor uses frozen L=8192 budgeted view",
        "policies": ["jina_full", "pplx_budget", "winnow_native"],
        "policy_order_per_shard": ["jina_full", "pplx_budget", "winnow_native"],
        "model_revisions": {"jina": JINA_REV, "pplx": PPLX_REV, "winnow": WINNOW_REV},
        "native_contracts": {
            "jina_full": "Jina native listwise/index-preserving scores on original Top-20",
            "pplx_budget": "PPLX native typed choice probabilities on frozen L=8192 strings",
            "winnow_native": "Winnow native POST /v1/systemone typed choice probabilities on original Top-20",
        },
        "source_hashes": {
            "bright_top20": sha_file(BRIGHT),
            "stage0_protocol": sha_file(STAGE0 / "PROTOCOL.json"),
            "stage0_budget_manifest": sha_file(STAGE0 / "BUDGET_VIEW_MANIFEST.jsonl"),
            "winnow_model": sha_file(WINNOW_MODEL),
            "winnow_runtime_lock": sha_file(RUNTIME / "runtime.lock.json"),
            "winnow_server": sha_file(WINNOW_SERVER),
        },
        "shards": shards,
        "statistics": {"labels_joined": False, "bootstrap_seed": 731, "failure_policy": "no retries; failure elapsed time retained; no imputation"},
        "resource_policy": {"approved_gpu_indices": list(GPUS), "one_server_per_gpu": True, "foreign_processes_untouched": True},
        "protected_holdouts_accessed": False,
        "paid_api_calls": 0,
    }
    write_json(OUT / "PROTOCOL_N2E_REPAIR.json", protocol)
    (OUT / "PROTOCOL_N2E_REPAIR.sha256").write_text(sha_file(OUT / "PROTOCOL_N2E_REPAIR.json") + "\n", encoding="utf-8")
    write_jsonl(OUT / "INPUT_MANIFEST_N2E_REPAIR.jsonl", rows)
    write_json(OUT / "GPU_SHARD_PLAN.json", {"status": "FROZEN_PRE_OUTCOME", "shards": shards, "gpu_indices": list(GPUS)})
    return views


def http_post(url: str, body: dict, timeout: float = 300.0):
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    req = urllib.request.Request(url, data=raw, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read(), raw


def wait_ready(url: str, deadline: float = 240.0):
    end = time.time() + deadline
    last = None
    while time.time() < end:
        try:
            req = urllib.request.Request(url + "/health")
            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last = repr(exc)
        time.sleep(1)
    raise RuntimeError(f"WINNOW_SERVER_NOT_READY:{last}")


def unload(model, torch):
    try:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


def worker(gpu: int, shard_views: list[dict]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t2 = module(f"v36_n2e_t2_{gpu}", ROOT / "scripts/v3_6_t2_serving.py")
    base = t2.base
    outdir = OUT / "shards" / f"gpu{gpu}"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    gate = base.gpu_gate(gpu)
    write_json(outdir / "GPU_GATE.json", gate)
    if not gate.get("isolated", True):
        write_json(outdir / "STATUS.json", {"status": "BLOCKED_GPU_GATE", "gpu": gpu})
        return
    import torch
    torch.cuda.set_device(0)
    proc = base.load_pplx_processor()
    tok = proc.tokenizer
    sys.path.insert(0, str(base.PPLX_PATH / "source" / "src"))
    from autojev.model import decision_messages

    def record(view, policy, attempt, started, status, error, scores, selected, request_hash, interface):
        try:
            torch.cuda.synchronize()
            alloc = round(torch.cuda.max_memory_allocated() / 2**20, 3)
        except Exception:
            alloc = None
        return {
            "session": f"E2E-gpu{gpu}", "gpu": gpu, "policy": policy, "canonical_id": view["canonical_id"],
            "domain": view["domain"], "attempt": attempt, "status": status, "error_class": error,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3), "alloc_peak_mib": alloc,
            "model_revision": {"jina_full": JINA_REV, "pplx_budget": PPLX_REV, "winnow_native": WINNOW_REV}[policy],
            "candidate_ids": view["original_ids"], "selected_ids": selected, "scores": scores,
            "native_request_sha256": request_hash, "native_interface": interface,
            "labels_joined": False, "input_label_blind": True,
            "timed_boundary": "native scoring/HTTP transaction through parsing, ID mapping and Top-5 packaging",
        }

    def run_local(policy, model):
        for attempt in (1, 2, 3):
            for view in shard_views:
                started = time.perf_counter(); status = "FAIL"; error = None; scores = []; selected = []; docs = []; ids = []
                try:
                    docs = view["original"] if policy == "jina_full" else view["budgeted"]
                    if docs is None:
                        raise RuntimeError("UNBUDGETABLE_QUERY_OR_OVERHEAD")
                    ids = [x["candidate_id"] for x in docs]
                    if policy == "jina_full":
                        scores = base.jina_score(model, view["query"], docs)
                        interface = "Jina native listwise/index-preserving scores"
                    else:
                        scores = base.pplx_score(model, base.choice_request(view["query"], docs), len(docs))
                        interface = "PPLX native categorical choice probabilities"
                    selected = base.stable_top(ids, scores)
                    if len(scores) != 20 or len(selected) != 5 or len(set(selected)) != 5:
                        raise ValueError("INVALID_TOP5_SELECTION")
                    status = "PASS"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    interface = "Jina native listwise/index-preserving scores" if policy == "jina_full" else "PPLX native categorical choice probabilities"
                request_hash = sha_obj({"query": view["query"], "candidate_ids": ids, "candidate_text_sha256": [hashlib.sha256(x["content"].encode()).hexdigest() for x in (docs or [])]})
                rows.append(record(view, policy, attempt, started, status, error, scores, selected, request_hash, interface))

    for policy in ("jina_full", "pplx_budget"):
        model = None
        try:
            model = base.load_jina(torch)[0] if policy == "jina_full" else base.load_pplx(torch)[0]
            run_local(policy, model)
        except BaseException as exc:
            write_json(outdir / f"FATAL_{policy}.json", {
                "status": "FATAL_MODEL_OR_RUNNER_ERROR",
                "policy": policy,
                "error_class": type(exc).__name__,
                "error": str(exc),
                "rows_recorded_before_fatal": sum(x["policy"] == policy for x in rows),
                "unattempted_rows_not_filled": True,
            })
        finally:
            unload(model, torch)
            model = None

    port = PORTS[gpu]
    log_path = outdir / "winnow_server.log"
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [str(PYTHON), str(WINNOW_PY), "--model", str(WINNOW_MODEL), "--context", "8192", "--decision-context", "8192", "--cache", "q8_0", "--decision-parallel", "4", "--chat-parallel", "1", "--memory", "exclusive", "--text-only", "--gpu", str(gpu), "--server", str(WINNOW_SERVER), "--host", "<REDACTED_IP>", "--port", str(port)]
    with log_path.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(command, cwd=str(RUNTIME), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    try:
        wait_ready(f"http://service.example.invalid:{port}")
        for attempt in (1, 2, 3):
            for view in shard_views:
                started = time.perf_counter(); status = "FAIL"; error = None; scores = []; selected = []; response = b""; req_hash = None
                ids = view["original_ids"]
                req = {"model": "Winnow-12B", "state": {"query": str(view["query"])}, "questions": {"candidate": {"type": "choice", "instructions": "Which candidate passage is most relevant to the query?", "criteria": {str(x["candidate_id"]): str(x["content"]) for x in view["original"]}}}}
                try:
                    response, req_raw = http_post(f"http://service.example.invalid:{port}/v1/systemone", req)
                    req_hash = hashlib.sha256(req_raw).hexdigest(); obj = json.loads(response)
                    probs = (obj.get("answers") or {}).get("candidate", {}).get("probabilities", {})
                    scores = [float(probs[x]) for x in ids]
                    selected = base.stable_top(ids, scores)
                    if len(selected) != 5 or len(set(selected)) != 5:
                        raise ValueError("INVALID_TOP5_SELECTION")
                    status = "PASS"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                rows.append(record(view, "winnow_native", attempt, started, status, error, scores, selected, req_hash, "Winnow native POST /v1/systemone typed choice probabilities"))
    finally:
        try:
            server.send_signal(signal.SIGTERM); server.wait(timeout=30)
        except Exception:
            server.kill(); server.wait(timeout=10)
    rows.sort(key=lambda x: (x["policy"], x["attempt"], x["canonical_id"]))
    write_jsonl(outdir / "raw.jsonl", rows)
    write_json(outdir / "TRACE_META.json", {"status": "PASS" if len(rows) == 810 else "INCOMPLETE", "records": len(rows), "expected_records": 810, "valid": sum(x["status"] == "PASS" for x in rows), "failed": sum(x["status"] != "PASS" for x in rows), "gpu": gpu, "labels_joined": False, "raw_sha256": sha_file(outdir / "raw.jsonl")})


def aggregate(views):
    all_rows = []
    for gpu in GPUS:
        path = OUT / "shards" / f"gpu{gpu}" / "raw.jsonl"
        if path.exists(): all_rows.extend(read_jsonl(path))
    write_jsonl(OUT / "RAW_N2E_REPAIR.jsonl", sorted(all_rows, key=lambda x: (x["policy"], x["attempt"], x["canonical_id"])))
    expected = 4 * 810
    summary = {"status": "PASS" if len(all_rows) == expected else "INCOMPLETE", "records": len(all_rows), "expected_records": expected, "valid": sum(x["status"] == "PASS" for x in all_rows), "failed": sum(x["status"] != "PASS" for x in all_rows), "policy_counts": {p: sum(x["policy"] == p for x in all_rows) for p in ("jina_full", "pplx_budget", "winnow_native")}, "labels_joined": False, "protected_holdouts_accessed": False, "raw_sha256": sha_file(OUT / "RAW_N2E_REPAIR.jsonl")}
    write_json(OUT / "TRACE_META.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpus", nargs="+", type=int, default=list(GPUS)); ap.add_argument("--freeze-only", action="store_true")
    args = ap.parse_args()
    if tuple(sorted(args.gpus)) != GPUS:
        raise SystemExit(f"FIXED_GPU_SET_REQUIRED:{GPUS}")
    views = freeze()
    if args.freeze_only:
        print(json.dumps({"status": "FROZEN_PRE_OUTCOME", "protocol": str(OUT / "PROTOCOL_N2E_REPAIR.json")}, indent=2)); return
    chunks = {gpu: views[i * 90:(i + 1) * 90] for i, gpu in enumerate(GPUS)}
    ctx = mp.get_context("spawn")
    processes = [ctx.Process(target=worker, args=(gpu, chunks[gpu]), name=f"n2e-gpu{gpu}") for gpu in GPUS]
    for p in processes: p.start()
    for p in processes: p.join()
    result = aggregate(views)
    write_json(OUT / "WORKER_EXIT_CODES.json", {p.name: p.exitcode for p in processes})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
