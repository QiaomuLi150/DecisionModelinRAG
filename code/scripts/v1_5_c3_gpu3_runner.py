#!/usr/bin/env python3
"""DecisionRAG v1.5 C3 bounded concurrency continuation.

This runner is label-blind.  It uses the exact R1 fixed routing request and
the frozen E11 budgeted selection views.  A single FIFO worker owns each
native model instance, matching the R1 batch_size=1 execution contract.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import queue
import random
import subprocess
import threading
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v1_5/c3_concurrency_gpu3"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
BUDGET = ROOT / "results/non_jev/v2/e11_v1_0/01_budget_policy/E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
PPLX_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")
JINA_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/{JINA_REV}")


def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(x):
    return hashlib.sha256(canon(x).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gpu_state(gpu):
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    for line in rows:
        v = [x.strip() for x in line.split(",")]
        if v and v[0] == str(gpu):
            return {"index": int(v[0]), "uuid": v[1], "memory_used_mib": int(float(v[2])), "memory_free_mib": int(float(v[3])), "utilization_percent": int(float(v[4]))}
    raise RuntimeError(f"GPU {gpu} not found")


def compute_apps(gpu_uuid):
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], text=True)
    return [x for x in raw.splitlines() if gpu_uuid in x]


def isolated(gpu):
    s = gpu_state(gpu)
    apps = compute_apps(s["uuid"])
    if apps or s["memory_used_mib"] > 64 or s["utilization_percent"] != 0:
        raise RuntimeError(f"GPU not isolated: {s}; apps={apps}")
    return {"state": s, "apps": apps}


def read_jsonl(path):
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def routing_request():
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


def choice_request(query, docs):
    return {
        "state": f"Query: {query}\nEvaluate the supplied candidate passages using the option descriptions.",
        "question": {
            "type": "choice",
            "instructions": f"Select the candidate passage that best answers the query: {query}",
            "criteria": {f"candidate_{i:03d}": f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(docs)},
        },
    }


def input_lock():
    slates = read_jsonl(BRIGHT)
    budgets = read_jsonl(BUDGET)
    if len(slates) != 1384 or len(budgets) != 1384:
        raise RuntimeError("frozen source counts do not match C3")
    views = []
    for row in sorted(slates, key=lambda x: (x["domain"], str(x["query_id"]))):
        key = f"{row['domain']}:{row['query_id']}"
        b = next(x for x in budgets if f"{x['domain']}:{x['query_id']}" == key)
        if str(b.get("status", "")).lower() not in {"ok", "pass", "complete"}:
            continue
        docs = [{"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in b["budgeted_candidates"]]
        views.append({"canonical_id": key, "query": str(row["query"]), "docs": docs})
    routing = [
        {"routing_id": f"routing_{i:03d}", "request": routing_request()}
        for i in range(50)
    ]
    lock = {
        "protocol": "decisionrag-v1.5-c3-v2",
        "bright_path": str(BRIGHT.relative_to(ROOT)),
        "bright_sha256": file_sha(BRIGHT),
        "budget_path": str(BUDGET.relative_to(ROOT)),
        "budget_sha256": file_sha(BUDGET),
        "routing_source": "scripts/v1_3_r1_serving.py::fixed_nonselection_request",
        "routing_population": 50,
        "routing_request_sha256": sha(routing_request()),
        "routing_pool_sha256": sha(routing),
        "routing_pool_note": "The frozen R1 source contains one fixed label-free routing request; C3 repeats that exact request with stable request IDs rather than fabricating prompt variants.",
        "selection_views": len(views),
        "selection_view_ids_sha256": sha([x["canonical_id"] for x in views]),
        "labels_joined": False,
    }
    return lock, routing, views


class Worker:
    def __init__(self, kind, model, torch):
        self.kind, self.model, self.torch = kind, model, torch
        self.q = queue.Queue(maxsize=512)
        self.stop = object()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is self.stop:
                self.q.task_done()
                return
            req, submitted, done = item
            started = time.perf_counter()
            row = {"status": "FAIL", "error": None, "queue_ms": (started - submitted) * 1000.0}
            try:
                if self.kind == "pplx":
                    scores = [float(x) for x in self.model.predict([req], batch_size=1)[0]]
                    if not scores or not all(math.isfinite(x) for x in scores):
                        raise ValueError("non-finite native scores")
                else:
                    result = self.model.rerank(req["query"], req["texts"])
                    scores = [float(x["relevance_score"]) for x in sorted(result, key=lambda x: int(x["index"]))]
                    if not scores or not all(math.isfinite(x) for x in scores):
                        raise ValueError("non-finite native scores")
                self.torch.cuda.synchronize()
                row.update({"status": "PASS", "service_ms": (time.perf_counter() - started) * 1000.0, "scores_n": len(scores)})
            except Exception as exc:
                row.update({"error": f"{type(exc).__name__}: {exc}", "service_ms": (time.perf_counter() - started) * 1000.0})
            row["elapsed_ms"] = (time.perf_counter() - submitted) * 1000.0
            done["row"] = row
            done["event"].set()
            self.q.task_done()

    def submit(self, req):
        done = {"event": threading.Event(), "row": None}
        try:
            submitted = time.perf_counter()
            self.q.put_nowait((req, submitted, done))
        except queue.Full:
            return None
        return done

    def close(self):
        self.q.join()
        self.q.put(self.stop)
        self.thread.join(timeout=30)


def percentile(xs, q):
    if not xs:
        return None
    ys = sorted(xs)
    p = (len(ys) - 1) * q
    lo, hi = int(p), min(int(p) + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (p - lo)


def summarize(config, rr, sr, typ, records, peak_mem, measure_seconds):
    good = [x for x in records if x["status"] == "PASS"]
    vals = [x["elapsed_ms"] for x in good]
    return {"config": config, "routing_rate_rps": rr, "selection_rate_rps": sr, "type": typ, "n": len(records), "p50_ms": percentile(vals, .5), "p95_ms": percentile(vals, .95), "p99_ms": percentile(vals, .99), "throughput_rps": len(good) / float(measure_seconds), "timeouts": sum(x["status"] == "TIMEOUT" for x in records), "peak_mem_gib": peak_mem}


def load_pplx(torch):
    path = str(PPLX_PATH / "source" / "src")
    import sys
    sys.path.insert(0, path)
    from autojev.model import DecisionModel
    model = DecisionModel(PPLX_PATH, device="cuda:0", max_image_pixels=512 * 512)
    torch.cuda.synchronize()
    return model


def load_jina(torch):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(JINA_PATH), local_files_only=True)
    model = AutoModel.from_pretrained(str(JINA_PATH), dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda:0").eval()
    torch.cuda.synchronize()
    return model


def run_rmax(worker, routing, duration=20):
    rows = []
    for concurrency in (1, 2, 4, 8):
        stop_at = time.monotonic() + duration
        lock = threading.Lock()
        completed = []
        def client(offset):
            i = offset
            while time.monotonic() < stop_at:
                done = worker.submit(routing[i % len(routing)]["request"])
                if done is None:
                    i += concurrency
                    continue
                done["event"].wait(30)
                if done["row"] and done["row"]["status"] == "PASS":
                    with lock: completed.append(done["row"])
                i += concurrency
        threads = [threading.Thread(target=client, args=(i,)) for i in range(concurrency)]
        for t in threads: t.start()
        for t in threads: t.join()
        rows.append({"concurrency": concurrency, "duration_s": duration, "completed": len(completed), "throughput_rps": len(completed) / duration, "p50_ms": percentile([x["elapsed_ms"] for x in completed], .5), "p95_ms": percentile([x["elapsed_ms"] for x in completed], .95)})
    return max(rows, key=lambda x: x["throughput_rps"]), rows


def run_cell(config, rr, sr, routing, views, pplx_worker, jina_worker, warmup, measure, seed):
    # The arrival schedule is deterministic and label blind.  The worker queue
    # is FIFO, matching R1's native batch_size=1 contract.
    rng = random.Random(seed)
    jobs = []
    def producer(kind, rate, start, end):
        t = start
        i = 0
        while t < end:
            if rate <= 0: break
            t += rng.expovariate(rate)
            if t >= end: break
            time.sleep(max(0.0, t - time.monotonic()))
            if kind == "routing": req = routing[i % len(routing)]["request"]; worker = pplx_worker
            else:
                view = views[i % len(views)]
                req = {"query": view["query"], "texts": [d["content"] for d in view["docs"]], "candidate_ids": [d["candidate_id"] for d in view["docs"]]}
                if config == "B_consolidated": req = choice_request(view["query"], view["docs"]); worker = pplx_worker
                else: worker = jina_worker
            done = worker.submit(req)
            if done is not None: jobs.append((kind, done, time.monotonic()))
            i += 1
    total = warmup + measure
    start = time.monotonic()
    threads = []
    if rr > 0: threads.append(threading.Thread(target=producer, args=("routing", rr, start, start + total)))
    if sr > 0: threads.append(threading.Thread(target=producer, args=("selection", sr, start, start + total)))
    for t in threads: t.start()
    for t in threads: t.join()
    end_measure = start + total
    records = []
    for kind, done, arrival in jobs:
        deadline = end_measure + 30
        done["event"].wait(max(0.0, deadline - time.monotonic()))
        row = done["row"]
        if row is None:
            records.append({"type": kind, "status": "TIMEOUT", "elapsed_ms": 30000.0, "queue_ms": 30000.0})
        elif arrival >= start + warmup:
            row = dict(row); row["type"] = kind; records.append(row)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--warmup-sec", type=int, default=60)
    ap.add_argument("--measure-sec", type=int, default=300)
    ap.add_argument("--saturation-sec", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "raw").mkdir(exist_ok=True); (OUT / "export").mkdir(exist_ok=True)
    pre = isolated(args.gpu)
    lock, routing, views = input_lock()
    (OUT / "INPUT_LOCK.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    load_start = time.perf_counter(); pplx = load_pplx(torch); pplx_load_ms = (time.perf_counter() - load_start) * 1000
    pworker = Worker("pplx", pplx, torch)
    rmax, rmax_rows = run_rmax(pworker, routing, args.saturation_sec)
    rows = [{"config": "A_routing_only", "routing_rate_rps": None, "selection_rate_rps": None, "type": "capacity", "n": rmax["completed"], "p50_ms": rmax["p50_ms"], "p95_ms": rmax["p95_ms"], "p99_ms": None, "throughput_rps": rmax["throughput_rps"], "timeouts": 0, "peak_mem_gib": None}]
    # Selection at 1 rps is intentionally tested only after the lower-rate cell
    # has been observed; the resulting queue divergence is retained explicitly.
    cells = [("A_routing_only", 0.3, 0.0), ("A_routing_only", 0.6, 0.0), ("B_consolidated", 0.3, 0.25), ("B_consolidated", 0.6, 0.25), ("B_consolidated", 0.3, 0.5), ("B_consolidated", 0.6, 0.5)]
    for config, rr, sr in cells:
        rec = run_cell(config, rr, sr, routing, views, pworker, None, args.warmup_sec, args.measure_sec, 731 + int(rr * 100) + int(sr * 1000))
        for typ in {x["type"] for x in rec}:
            rows.append(summarize(config, rr, sr, typ, [x for x in rec if x["type"] == typ], None, args.measure_sec))
        (OUT / "raw/C3_CHECKPOINT.json").write_text(json.dumps({"completed_cells": [x["config"] + ":" + str(x["routing_rate_rps"]) + ":" + str(x["selection_rate_rps"]) for x in rows], "labels_joined": False}, indent=2) + "\n")
    pworker.close()
    del pworker, pplx; torch.cuda.empty_cache(); torch.cuda.synchronize()
    # C uses both validated models resident on the same GPU.  It is attempted
    # only after A/B so a memory admission failure remains visible.
    c_status = "PASS"
    try:
        jload = time.perf_counter(); jina = load_jina(torch); jina_load_ms = (time.perf_counter() - jload) * 1000
        pload = time.perf_counter(); pplx = load_pplx(torch); pplx_reload_ms = (time.perf_counter() - pload) * 1000
        pworker = Worker("pplx", pplx, torch); jworker = Worker("jina", jina, torch)
        for config, rr, sr in [("C_colocated", 0.3, 0.25), ("C_colocated", 0.6, 0.25), ("C_colocated", 0.3, 0.5), ("C_colocated", 0.6, 0.5)]:
            rec = run_cell(config, rr, sr, routing, views, pworker, jworker, args.warmup_sec, args.measure_sec, 1731 + int(rr * 100) + int(sr * 1000))
            for typ in {x["type"] for x in rec}:
                rows.append(summarize(config, rr, sr, typ, [x for x in rec if x["type"] == typ], None, args.measure_sec))
            (OUT / "raw/C3_CHECKPOINT.json").write_text(json.dumps({"completed_cells": [x["config"] + ":" + str(x["routing_rate_rps"]) + ":" + str(x["selection_rate_rps"]) for x in rows], "labels_joined": False}, indent=2) + "\n")
        pworker.close(); jworker.close(); del pworker, jworker, pplx, jina; torch.cuda.empty_cache(); torch.cuda.synchronize()
    except Exception as exc:
        c_status = f"BLOCKED:{type(exc).__name__}:{exc}"
    lock.update({"pplx_load_ms": round(pplx_load_ms, 3), "c_status": c_status, "models_loaded": {"pplx": PPLX_REV, "jina": JINA_REV}, "labels_joined": False})
    (OUT / "C3_EXECUTION_RECEIPT.json").write_text(json.dumps({"status": "PASS_WITH_D_BLOCKED" if c_status == "PASS" else "PARTIAL", "gpu": args.gpu, "gpu_uuid": pre["state"]["uuid"], "pre": pre, "rmax": rmax_rows, "c_status": c_status, "pplx_load_ms": pplx_load_ms, "labels_joined": False, "deviations": ["D_separate skipped: no second dedicated A100 was available", "C3 uses the exact single R1 routing request repeated 50 times because no 50-variant frozen pool artifact exists"]}, indent=2, sort_keys=True) + "\n")
    with (OUT / "export/c3_concurrency.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["config", "routing_rate_rps", "selection_rate_rps", "type", "n", "p50_ms", "p95_ms", "p99_ms", "throughput_rps", "timeouts", "peak_mem_gib"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows({k: row.get(k) for k in fields} for row in rows)
    (OUT / "export/R_MAX.json").write_text(json.dumps(rmax, indent=2, sort_keys=True) + "\n")
    post = gpu_state(args.gpu)
    (OUT / "RESOURCE_CLOSURE.json").write_text(json.dumps({"post_state": post, "post_apps": compute_apps(post["uuid"])}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS_WITH_D_BLOCKED" if c_status == "PASS" else "PARTIAL", "rows": len(rows), "rmax": rmax, "c_status": c_status}, indent=2), flush=True)


if __name__ == "__main__":
    main()
