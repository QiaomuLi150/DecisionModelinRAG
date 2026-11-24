#!/usr/bin/env python3
"""R2 label-blind native order-sensitivity runner.

The original order is reused only from authenticated historical raw traces;
all requested non-original orders are physically scored by the selected
native adapter.  This script never reads relevance labels.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V13 = ROOT / "results/v1_3/r2_order_sensitivity"
E11 = ROOT / "results/non_jev/v2/e11_v1_0"
E9 = ROOT / "results/non_jev/v2/e9_bright_v1_0"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
QWEN_REV = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
PPLX_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")
JINA_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/{JINA_REV}")

def canon(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def file_sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read_jsonl(p): return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]

def gpu_snapshot():
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        out = []
        for line in raw.splitlines():
            vals = [v.strip() for v in line.split(",")]
            if len(vals) >= 5:
                out.append({"index": int(vals[0]), "uuid": vals[1], "memory_used_mib": int(vals[2]), "memory_free_mib": int(vals[3]), "utilization_percent": int(vals[4])})
        return out
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}:{exc}"}]

def source_rows():
    slates = {f"{x['domain']}:{x['query_id']}": x for x in read_jsonl(E9 / "01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl")}
    budget = {f"{x['domain']}:{x['query_id']}": x for x in read_jsonl(E11 / "01_budget_policy/E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl")}
    if len(slates) != 1384 or set(slates) != set(budget): raise RuntimeError("frozen population mismatch")
    return slates, budget

def order_list(ids, order_id):
    if order_id == "original": return list(ids)
    if order_id == "reversed": return list(reversed(ids))
    seed = int(order_id.rsplit("_", 1)[1]); p = list(ids); import random; random.Random(seed).shuffle(p); return p

def native_request_pplx(row, ordered):
    codes = [f"candidate_{i:03d}" for i in range(len(ordered))]
    by_id = {str(x["candidate_id"]): str(x["content"]) for x in row["budgeted_candidates"]}
    criteria = {codes[i]: f"Candidate ID {cid}. Passage: {by_id[cid]}" for i,cid in enumerate(ordered)}
    return {"state": f"Query: {row['query']}\nEvaluate the supplied candidate passages using the option descriptions.", "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {row['query']}", "criteria": criteria}}

def native_request_jina(slate, ordered):
    by_id = {str(x["candidate_id"]): str(x["content"]) for x in slate["candidates"]}
    return {"query": str(slate["query"]), "documents": [by_id[x] for x in ordered], "candidate_ids": list(ordered)}

def base(canonical_id, slate, ordered, policy, order_id):
    return {"canonical_id": canonical_id, "domain": slate["domain"], "query_id": str(slate["query_id"]), "policy": policy, "order_id": order_id, "candidate_ids": list(ordered), "candidate_ids_sha256": sha(list(ordered)), "source_slate_sha256": slate["slate_sha256"], "source_input_sha256": slate["input_sha256"], "labels_joined": False, "status": "FAIL", "selected_ids": [], "scores": [], "error": None}

def reuse_pplx(canonical_id, slate, budget, ordered, order_id):
    rawpath = E11 / "02_native/E11_A_PPLX_BUDGETED_RAW.jsonl"
    old = {f"{x['domain']}:{x['query_id']}": x for x in read_jsonl(rawpath)}[canonical_id]
    by_id = {cid: score for cid,score in zip(old["candidate_ids"], old.get("scores", []))}
    scores = [float(by_id[cid]) for cid in ordered]
    out = base(canonical_id, slate, ordered, "pplx_budget", order_id)
    out.update({"status": old["status"], "scores": scores, "selected_ids": [ordered[i] for i in sorted(range(len(ordered)), key=lambda i: (-scores[i], i, ordered[i]))[:5]], "reused_from": str(rawpath.relative_to(ROOT)), "historical_raw_sha256": file_sha(rawpath), "native_request_sha256": budget["pplx_request_sha256"], "physical_forward_count": "HISTORICAL_TRACE_NOT_REEXECUTED", "elapsed_ms": old.get("duration_ms")})
    return out

def reuse_jina(canonical_id, slate, ordered, order_id):
    rawpath = E9 / "02_native/BRIGHT_NATIVE_RAW_OUTPUTS_jina_v3_5.jsonl"
    old = {f"{x['domain']}:{x['query_id']}": x for x in read_jsonl(rawpath)}[canonical_id]
    by_id = {cid: score for cid,score in zip(old["candidate_ids"], old.get("scores", []))}
    scores = [float(by_id[cid]) for cid in ordered]
    out = base(canonical_id, slate, ordered, "jina_full", order_id)
    out.update({"status": old["status"], "scores": scores, "selected_ids": [ordered[i] for i in sorted(range(len(ordered)), key=lambda i: (-scores[i], i, ordered[i]))[:5]], "reused_from": str(rawpath.relative_to(ROOT)), "historical_raw_sha256": file_sha(rawpath), "native_request_sha256": old.get("native_input_sha256"), "physical_forward_count": "HISTORICAL_TRACE_NOT_REEXECUTED", "elapsed_ms": old.get("duration_ms")})
    return out

def reuse_qwen(canonical_id, slate, ordered, order_id):
    rawpath = E9 / "02_native/BRIGHT_NATIVE_RAW_OUTPUTS_qwen3_reranker_0_6b.jsonl"
    old = {f"{x['domain']}:{x['query_id']}": x for x in read_jsonl(rawpath)}[canonical_id]
    by_id = {cid: score for cid,score in zip(old["candidate_ids"], old.get("scores", []))}
    scores = [float(by_id[cid]) for cid in ordered]
    out = base(canonical_id, slate, ordered, "qwen3_reranker_0_6b", order_id)
    out.update({"status": old["status"], "scores": scores, "selected_ids": [ordered[i] for i in sorted(range(len(ordered)), key=lambda i: (-scores[i], i, ordered[i]))[:5]], "reused_from": str(rawpath.relative_to(ROOT)), "historical_raw_sha256": file_sha(rawpath), "native_request_sha256": old.get("native_input_sha256"), "physical_forward_count": "HISTORICAL_POINTWISE_TRACE_REUSED", "elapsed_ms": old.get("duration_ms")})
    return out

def run_native(policy, order_ids, gpu, limit=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    slates, budgets = source_rows()
    ordered_keys = sorted(slates)
    if limit:
        ordered_keys = ordered_keys[:limit]
    if policy == "pplx_budget":
        sys.path.insert(0, str(PPLX_PATH / "source" / "src")); from autojev.model import DecisionModel
        load_t = time.perf_counter(); adapter = DecisionModel(PPLX_PATH, device="cuda:0", max_image_pixels=512*512); torch.cuda.synchronize(); load_ms=(time.perf_counter()-load_t)*1000
    elif policy == "jina_full":
        from transformers import AutoModel, AutoTokenizer
        load_t=time.perf_counter(); tok=AutoTokenizer.from_pretrained(str(JINA_PATH), local_files_only=True); adapter=AutoModel.from_pretrained(str(JINA_PATH), dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda:0").eval(); torch.cuda.synchronize(); load_ms=(time.perf_counter()-load_t)*1000
    else: raise ValueError(policy)
    out=[]
    for n,cid in enumerate(ordered_keys,1):
        slate, budget = slates[cid], budgets[cid]; original_ids=[str(x["candidate_id"]) for x in slate["candidates"]]
        for order_id in order_ids:
            ordered=order_list(original_ids, order_id)
            if order_id == "original":
                out.append(reuse_pplx(cid,slate,budget,ordered,order_id) if policy=="pplx_budget" else reuse_jina(cid,slate,ordered,order_id)); continue
            rec=base(cid,slate,ordered,policy,order_id); t=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            try:
                if policy=="pplx_budget":
                    req=native_request_pplx(budget, ordered); scores=[float(x) for x in adapter.predict([req],batch_size=1)[0]]; rec["native_request_sha256"]=sha(req)
                else:
                    req=native_request_jina(slate, ordered); result=adapter.rerank(slate["query"], [x["content"] for x in sorted(slate["candidates"],key=lambda d: ordered.index(str(d["candidate_id"])))])
                    mapping={int(x["index"]):float(x["relevance_score"]) for x in result}; scores=[mapping[i] for i in range(20)]; rec["native_request_sha256"]=sha(req)
                if len(scores)!=20 or not all(math.isfinite(x) for x in scores): raise ValueError("invalid score vector")
                idx=sorted(range(20),key=lambda i:(-scores[i],i,ordered[i])); rec.update({"status":"PASS","scores":scores,"selected_ids":[ordered[i] for i in idx[:5]],"elapsed_ms":(time.perf_counter()-t)*1000,"peak_memory_allocated_mib":torch.cuda.max_memory_allocated()/2**20,"physical_forward_count":"NOT_EXPOSED_BY_RUNTIME"})
            except Exception as exc:
                rec.update({"status":"FAIL","error":f"{type(exc).__name__}:{exc}","elapsed_ms":(time.perf_counter()-t)*1000,"peak_memory_allocated_mib":torch.cuda.max_memory_allocated()/2**20})
            out.append(rec)
        if n % 25 == 0: print(json.dumps({"policy":policy,"completed_queries":n,"orders":order_ids}),flush=True)
    outdir=V13/"raw"; outdir.mkdir(parents=True,exist_ok=True); path=outdir/f"{policy}_gpu{gpu}_orders_{'_'.join(order_ids)}.jsonl"; path.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in out),encoding="utf-8")
    meta={"policy":policy,"gpu":gpu,"orders":order_ids,"n_rows":len(out),"valid":sum(x["status"] in ("PASS","ok") for x in out),"failed":sum(x["status"] not in ("PASS","ok") for x in out),"model_load_ms":load_ms,"gpu_before":gpu_snapshot(),"raw_sha256":file_sha(path),"labels_joined":False,"historical_original_order_reused":True}
    (outdir/f"{policy}_gpu{gpu}_orders_{'_'.join(order_ids)}.meta.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta,indent=2),flush=True)
    del adapter; gc.collect(); torch.cuda.empty_cache()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--policy",choices=["pplx_budget","jina_full"] ,required=True); ap.add_argument("--gpu",type=int,required=True); ap.add_argument("--orders",required=True); ap.add_argument("--limit",type=int,default=0); args=ap.parse_args(); run_native(args.policy,args.orders.split(","),args.gpu,args.limit)

if __name__ == "__main__": main()
