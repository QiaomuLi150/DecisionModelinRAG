#!/usr/bin/env python3
"""Robust bounded C3 queueing diagnostic with genuine main-thread forwards.

The failed thread-based harness is preserved by its traceback/session.  This
version uses the validated R1 native adapters in the main thread and a
deterministic FIFO virtual-arrival schedule, so no unvalidated model
thread-safety assumption enters the result.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v1_5/c3_concurrency_gpu3"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
BUDGET = ROOT / "results/non_jev/v2/e11_v1_0/01_budget_policy/E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
PPLX_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")
JINA_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/{JINA_REV}")


def sha(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fsha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def gpu_state(gpu):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    for line in raw.splitlines():
        v = [x.strip() for x in line.split(",")]
        if v and v[0] == str(gpu):
            return {"index": int(v[0]), "uuid": v[1], "memory_used_mib": int(float(v[2])), "memory_free_mib": int(float(v[3])), "utilization_percent": int(float(v[4]))}
    raise RuntimeError("GPU missing")


def apps(uuid):
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], text=True)
    return [x for x in raw.splitlines() if uuid in x]


def fixed_route():
    return {"state": "Route this request to the appropriate processing queue.", "question": {"type": "choice", "instructions": "Choose the next processing action from the option descriptions.", "criteria": {"route_000": "Send the request to the standard text-processing queue.", "route_001": "Send the request to the specialist review queue.", "route_002": "Send the request to the deferred-processing queue."}}}


def choice(query, docs):
    return {"state": f"Query: {query}\nEvaluate the supplied candidate passages using the option descriptions.", "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {query}", "criteria": {f"candidate_{i:03d}": f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(docs)}}}


def load_views():
    slates = [json.loads(x) for x in BRIGHT.open(encoding="utf-8") if x.strip()]
    budgets = {f"{x['domain']}:{x['query_id']}": x for x in (json.loads(x) for x in BUDGET.open(encoding="utf-8") if x.strip())}
    views = []
    for s in sorted(slates, key=lambda x: (x["domain"], str(x["query_id"]))):
        key = f"{s['domain']}:{s['query_id']}"; b = budgets[key]
        if str(b.get("status", "")).lower() not in {"ok", "pass", "complete"}: continue
        views.append({"canonical_id": key, "query": str(s["query"]), "docs": [{"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in b["budgeted_candidates"]]})
    return views


def percentile(xs, q):
    if not xs: return None
    ys = sorted(xs); p = (len(ys)-1)*q; lo, hi = int(p), min(int(p)+1, len(ys)-1)
    return ys[lo] + (ys[hi]-ys[lo])*(p-lo)


def events(rate, total, kind, seed):
    if rate <= 0: return []
    rng = random.Random(seed); t = 0.0; out = []; i = 0
    while True:
        t += rng.expovariate(rate)
        if t >= total: return out
        out.append((t, kind, i)); i += 1


def call_pplx(model, torch, req):
    started = time.perf_counter(); status = "PASS"; error = None
    try:
        scores = [float(x) for x in model.predict([req], batch_size=1)[0]]
        if not scores or not all(math.isfinite(x) for x in scores): raise ValueError("invalid native scores")
    except Exception as exc:
        status = "FAIL"; error = f"{type(exc).__name__}: {exc}"
    torch.cuda.synchronize(); return status, error, (time.perf_counter()-started)*1000.0


def call_jina(model, torch, req):
    started = time.perf_counter(); status = "PASS"; error = None
    try:
        scores = [float(x["relevance_score"]) for x in sorted(model.rerank(req["query"], req["texts"]), key=lambda x: int(x["index"]))]
        if not scores or not all(math.isfinite(x) for x in scores): raise ValueError("invalid native scores")
    except Exception as exc:
        status = "FAIL"; error = f"{type(exc).__name__}: {exc}"
    torch.cuda.synchronize(); return status, error, (time.perf_counter()-started)*1000.0


def run_cell(config, rr, sr, routing, views, pplx, jina, torch, warmup, measure, seed):
    total = warmup + measure
    schedule = events(rr, total, "routing", seed) + events(sr, total, "selection", seed + 100003)
    schedule.sort(key=lambda x: (x[0], x[1], x[2]))
    virtual_available = 0.0; measured = []
    for arrival, kind, idx in schedule:
        if kind == "routing":
            req = routing[idx % len(routing)]
            status, error, service_ms = call_pplx(pplx, torch, req)
        else:
            v = views[idx % len(views)]; docs = v["docs"]; req = choice(v["query"], docs)
            if config == "B_consolidated": status, error, service_ms = call_pplx(pplx, torch, req)
            else: status, error, service_ms = call_jina(jina, torch, {"query": v["query"], "texts": [x["content"] for x in docs]})
        start = max(arrival, virtual_available); queue_ms = max(0.0, start-arrival)*1000.0; virtual_available = start + service_ms/1000.0
        if arrival >= warmup:
            measured.append({"type": kind, "status": status, "error": error, "queue_ms": queue_ms, "service_ms": service_ms, "elapsed_ms": queue_ms+service_ms})
    return measured, virtual_available


def row(config, rr, sr, typ, records, measure, mem):
    good = [x for x in records if x["status"] == "PASS"]; vals = [x["elapsed_ms"] for x in good]
    return {"config": config, "routing_rate_rps": rr, "selection_rate_rps": sr, "type": typ, "n": len(records), "p50_ms": percentile(vals,.5), "p95_ms": percentile(vals,.95), "p99_ms": percentile(vals,.99), "throughput_rps": len(good)/measure, "timeouts": sum(x["status"] == "TIMEOUT" for x in records), "peak_mem_gib": mem}


def main():
    ap = __import__("argparse").ArgumentParser(); ap.add_argument("--gpu", type=int, default=3); ap.add_argument("--warmup-sec", type=int, default=10); ap.add_argument("--measure-sec", type=int, default=60); ap.add_argument("--saturation-sec", type=int, default=10); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); (OUT/"export").mkdir(exist_ok=True); (OUT/"raw").mkdir(exist_ok=True)
    pre = gpu_state(args.gpu); pre_apps = apps(pre["uuid"])
    if pre["memory_used_mib"] > 64 or pre["utilization_percent"] != 0 or pre_apps: raise RuntimeError(f"GPU3 not isolated: {pre} apps={pre_apps}")
    views = load_views(); routing = [fixed_route() for _ in range(50)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    sys.path.insert(0, str(PPLX_PATH/"source"/"src")); from autojev.model import DecisionModel
    t0=time.perf_counter(); pplx=DecisionModel(PPLX_PATH, device="cuda:0", max_image_pixels=512*512); torch.cuda.synchronize(); load_ms=(time.perf_counter()-t0)*1000
    rmax_rows=[]
    for conc in (1,2,4,8):
        vals=[]; t=time.perf_counter()
        for i in range(max(4, conc*4)):
            st, er, ms = call_pplx(pplx, torch, routing[i%50]);
            if st == "PASS": vals.append(ms)
        rmax_rows.append({"concurrency": conc, "duration_s": args.saturation_sec, "completed": len(vals), "throughput_rps": len(vals)/max(time.perf_counter()-t,1e-9), "p50_ms": percentile(vals,.5), "p95_ms": percentile(vals,.95)})
    rmax=max(rmax_rows, key=lambda x:x["throughput_rps"]); result_rows=[]; event_rows=[]
    cells=[("A_routing_only",.3,0.0),("A_routing_only",.6,0.0),("B_consolidated",.3,.25),("B_consolidated",.6,.25),("B_consolidated",.3,.5),("B_consolidated",.6,.5)]
    for config,rr,sr in cells:
        rec, vc=run_cell(config,rr,sr,routing,views,pplx,None,torch,args.warmup_sec,args.measure_sec,731+int(rr*100)+int(sr*1000))
        for typ in sorted({x["type"] for x in rec}): result_rows.append(row(config,rr,sr,typ,[x for x in rec if x["type"]==typ],args.measure_sec,None))
        event_rows += [{"config":config,"routing_rate_rps":rr,"selection_rate_rps":sr,**x} for x in rec]
    del pplx; torch.cuda.empty_cache(); torch.cuda.synchronize(); jina=None
    c_status="PASS"
    try:
        from transformers import AutoModel, AutoTokenizer
        jt=AutoTokenizer.from_pretrained(str(JINA_PATH),local_files_only=True); jina=AutoModel.from_pretrained(str(JINA_PATH),dtype="auto",trust_remote_code=True,local_files_only=True).to("cuda:0").eval(); torch.cuda.synchronize()
        t0=time.perf_counter(); sys.path.insert(0,str(PPLX_PATH/"source"/"src")); pplx=DecisionModel(PPLX_PATH,device="cuda:0",max_image_pixels=512*512); torch.cuda.synchronize(); reload_ms=(time.perf_counter()-t0)*1000
        for config,rr,sr in [("C_colocated",.3,.25),("C_colocated",.6,.25),("C_colocated",.3,.5),("C_colocated",.6,.5)]:
            rec,vc=run_cell(config,rr,sr,routing,views,pplx,jina,torch,args.warmup_sec,args.measure_sec,1731+int(rr*100)+int(sr*1000))
            for typ in sorted({x["type"] for x in rec}): result_rows.append(row(config,rr,sr,typ,[x for x in rec if x["type"]==typ],args.measure_sec,None))
            event_rows += [{"config":config,"routing_rate_rps":rr,"selection_rate_rps":sr,**x} for x in rec]
    except Exception as exc: c_status=f"BLOCKED:{type(exc).__name__}:{exc}"
    with (OUT/"export/c3_concurrency.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["config","routing_rate_rps","selection_rate_rps","type","n","p50_ms","p95_ms","p99_ms","throughput_rps","timeouts","peak_mem_gib"]; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows({k:x.get(k) for k in fields} for x in result_rows)
    (OUT/"raw/c3_events.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in event_rows),encoding="utf-8")
    receipt={"status":"PASS_WITH_D_BLOCKED" if c_status=="PASS" else "PARTIAL","gpu":args.gpu,"gpu_uuid":pre["uuid"],"pre":pre,"pre_apps":pre_apps,"rmax":rmax_rows,"c_status":c_status,"pplx_load_ms":load_ms,"warmup_seconds":args.warmup_sec,"measure_seconds":args.measure_sec,"arrival_schedule":"deterministic virtual FIFO; genuine main-thread native forwards; no thread-safety assumption","labels_joined":False,"D_separate":"BLOCKED_NO_SECOND_DEDICATED_A100","deviations":["bounded 10/60-second cells rather than v2 60/300 schedule","single fixed R1 routing request repeated with stable IDs","Rmax row is serial native-call capacity diagnostic rather than concurrent server throughput"]}
    (OUT/"C3_EXECUTION_RECEIPT.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    post=gpu_state(args.gpu); (OUT/"RESOURCE_CLOSURE.json").write_text(json.dumps({"post":post,"apps":apps(post["uuid"])},indent=2,sort_keys=True)+"\n")
    del jina,pplx; torch.cuda.empty_cache(); torch.cuda.synchronize(); print(json.dumps(receipt,indent=2,sort_keys=True),flush=True)


if __name__=="__main__": main()
