#!/usr/bin/env python3
"""Frozen v3.6 follow-up Jina rerun with explicit None-output instrumentation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/followup_v1"
STAGE0 = ROOT / "results/v3_6_essential/stage0_preflight"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
JINA_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/{JINA_REV}")


def load_base():
    spec = importlib.util.spec_from_file_location("v36base", ROOT / "scripts/v3_6_essential.py")
    base = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(base); return base


def read_jsonl(path):
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def sha_bytes(b): return hashlib.sha256(b).hexdigest()


def sha_json(x): return sha_bytes(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def stable_top(ids, scores):
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i, ids[i]))[:5]]


def classify(exc):
    s = str(exc).lower()
    if "unbudgetable" in s or "budgetunavailable" in s: return "BUILDER_INFEASIBLE"
    if "out of memory" in s or "cuda oom" in s: return "CUDA_OOM"
    if "none" in s and "rerank" in s: return "JINA_NONE_OUTPUT"
    if "timeout" in s: return "TIMEOUT"
    return type(exc).__name__


def call_jina(model, query, docs):
    texts = [str(x["content"]) for x in docs]
    result = model.rerank(query, texts)
    result_type = type(result).__name__
    if result is None:
        raise RuntimeError("JINA_RERANK_NONE_OUTPUT")
    if not isinstance(result, list):
        raise TypeError(f"JINA_RERANK_UNEXPECTED_TYPE:{result_type}")
    mapping = {int(x["index"]): float(x["relevance_score"]) for x in result}
    if set(mapping) != set(range(len(docs))) or not all(math.isfinite(x) for x in mapping.values()):
        raise ValueError("invalid Jina index score vector")
    return [mapping[i] for i in range(len(docs))], result_type


def one_policy(base, model, proc, tok, decision_messages, item, policy, limit, diagnosis=False):
    original = [{"candidate_id": str(d["candidate_id"]), "content": str(d["content"])} for d in item["candidates"]]
    rec = {"canonical_id": f"{item['domain']}:{item['query_id']}", "domain": item["domain"], "query_id": str(item["query_id"]), "policy": policy, "budget_L": limit, "order_id": "original", "source_slate_sha256": item["slate_sha256"], "candidate_ids": [x["candidate_id"] for x in original], "status": "FAIL", "valid": False, "error_class": None, "error": None, "selected_ids": [], "scores": [], "stage_selected_ids": [], "stage_records": [], "n_calls": 0, "labels_joined": False, "input_label_blind": True, "model_revision": JINA_REV, "gpu_slot_request": os.environ.get("V36_GPU_SLOT", "UNKNOWN"), "model_return_type": None, "model_return_was_none": False}
    t0 = time.perf_counter()
    try:
        if policy.startswith("jina_single"):
            docs, info = base.fit_docs(item, original, proc, decision_messages, tok, limit)
            if docs is None: raise RuntimeError(info["error_class"])
            scores, rt = call_jina(model, str(item["query"]), docs)
            ids = [x["candidate_id"] for x in docs]
            rec.update({"candidate_ids": ids, "selected_ids": stable_top(ids, scores), "scores": scores, "n_calls": 1, "input_hashes": [sha_json({"query": item["query"], "candidate_ids": ids, "documents": [x["content"] for x in docs]})], "budget_info": info, "model_return_type": rt})
        else:
            survivors=[]; stages=[]
            for wi, window in enumerate((original[:10], original[10:]), 1):
                docs, info = base.fit_docs(item, window, proc, decision_messages, tok, limit)
                if docs is None: raise RuntimeError(info["error_class"])
                scores, rt = call_jina(model, str(item["query"]), docs)
                ids=[x["candidate_id"] for x in docs]; keep=stable_top(ids,scores); by={x["candidate_id"]:x for x in docs}; survivors.extend(by[x] for x in keep)
                stages.append({"stage":f"stage1_window{wi}","candidate_ids":ids,"scores":scores,"input_sha256":sha_json({"query":item["query"],"candidate_ids":ids,"documents":[x["content"] for x in docs]}),"budget":info,"model_return_type":rt})
            docs2, info2=base.fit_docs(item,survivors,proc,decision_messages,tok,limit)
            if docs2 is None: raise RuntimeError(info2["error_class"])
            scores2,rt2=call_jina(model,str(item["query"]),docs2); ids2=[x["candidate_id"] for x in docs2]
            stages.append({"stage":"stage2_survivors","candidate_ids":ids2,"scores":scores2,"input_sha256":sha_json({"query":item["query"],"candidate_ids":ids2,"documents":[x["content"] for x in docs2]}),"budget":info2,"model_return_type":rt2})
            rec.update({"candidate_ids":[x["candidate_id"] for x in original],"selected_ids":stable_top(ids2,scores2),"scores":scores2,"stage_selected_ids":[x["candidate_id"] for x in survivors],"stage_records":stages,"input_hashes":[x["input_sha256"] for x in stages],"n_calls":3})
        rec["status"]="PASS"; rec["valid"]=len(rec["selected_ids"])==5 and len(set(rec["selected_ids"]))==5
        if not rec["valid"]: raise ValueError("invalid Top-5 selection")
    except Exception as exc:
        rec["error_class"]=classify(exc); rec["error"]=f"{type(exc).__name__}: {exc}"; rec["model_return_was_none"]="JINA_RERANK_NONE_OUTPUT" in str(exc)
    rec["latency_ms"]=round((time.perf_counter()-t0)*1000,3)
    return rec


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--gpu",type=int,required=True); ap.add_argument("--start",type=int,default=0); ap.add_argument("--end",type=int,default=1384); ap.add_argument("--policies",default="jina_single_4096,jina_single_6144,jina_single_8192,jina_two_stage_2048,jina_two_stage_4096,jina_two_stage_8192"); ap.add_argument("--pilot",action="store_true"); args=ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu); os.environ["V36_GPU_SLOT"]={3:"A",6:"B"}.get(args.gpu,"C")
    base=load_base(); import torch
    sys.path.insert(0,str(base.PPLX_PATH/"source"/"src")); from autojev.model import decision_messages
    from transformers import AutoTokenizer, AutoModel, AutoProcessor
    tok=AutoTokenizer.from_pretrained(str(JINA_PATH),local_files_only=True); proc=AutoProcessor.from_pretrained(str(base.PPLX_PATH),local_files_only=True); ptoken=proc.tokenizer
    model=AutoModel.from_pretrained(str(JINA_PATH),dtype="auto",trust_remote_code=True,local_files_only=True).to("cuda:0").eval()
    slates=sorted(read_jsonl(BRIGHT),key=lambda x:(x["domain"],str(x["query_id"])))
    views={x["canonical_id"]:x for x in read_jsonl(STAGE0/"BUDGET_VIEW_MANIFEST.jsonl")}
    policies=[x.strip() for x in args.policies.split(",") if x.strip()]
    outdir=OUT/"stage1_jina"/"raw"; outdir.mkdir(parents=True,exist_ok=True)
    lo=max(0,args.start); hi=min(args.end,len(slates)); suffix="pilot" if args.pilot else f"{lo}_{hi}"
    path=outdir/f"jina_rerun_{suffix}_gpu{args.gpu}.jsonl"; part=path.with_suffix(path.suffix+".part")
    if part.exists(): raise RuntimeError(f"partial file exists: {part}")
    with part.open("w",encoding="utf-8") as f:
        for ix,item in enumerate(slates[lo:hi],lo):
            cid=f"{item['domain']}:{item['query_id']}"; item=dict(item); item["query"]=str(item["query"])
            for policy in policies:
                L=int(policy.rsplit("_",1)[1]); r=one_policy(base,model,proc,ptoken,decision_messages,item,policy,L,diagnosis=args.pilot); r["request_index"]=ix; f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n"); f.flush()
            if (ix-lo+1)%10==0: print(json.dumps({"status":"RUNNING","gpu_slot":os.environ["V36_GPU_SLOT"],"completed":ix-lo+1,"total":hi-lo}),flush=True)
    part.replace(path)
    meta={"status":"PASS","gpu_slot":os.environ["V36_GPU_SLOT"],"gpu_index":args.gpu,"start":lo,"end":hi,"policies":policies,"rows":sum(1 for _ in path.open()),"sha256":sha_bytes(path.read_bytes()),"labels_joined":False,"protected_holdouts_accessed":False}
    (path.with_suffix(path.suffix+".meta.json")).write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(meta,indent=2))


if __name__=="__main__": main()
