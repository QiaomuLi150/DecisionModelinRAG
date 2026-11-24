#!/usr/bin/env python3
"""Label-blind Jina controls for the frozen LeetCode sub-slates.

Runs Jina v3.5 on the exact candidate strings used by the frozen Kev/Drex
sub-slate preparation: untrimmed native strings and the deterministic
equal-prefix budget strings.  Gold labels are intentionally not imported.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
E9=ROOT/"results/non_jev/v2/e9_bright_v1_0"
SUB=ROOT/"results/v1_3/r6_leetcode_trim/leetcode_subslate_manifest.jsonl"
OUT=ROOT/"results/v1_3/r6_leetcode_trim/raw"
JINA_REV="e8a93f33f0b22108f8c2364f8484ce3422552fbc"
JINA_PATH=Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/{JINA_REV}")
def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def fsha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def readj(p): return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--gpu",type=int,required=True); args=ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    import torch
    from transformers import AutoModel,AutoTokenizer
    slates={f"{x['domain']}:{x['query_id']}":x for x in readj(E9/"01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl")}
    subs=readj(SUB)
    load_t=time.perf_counter(); tok=AutoTokenizer.from_pretrained(str(JINA_PATH),local_files_only=True); model=AutoModel.from_pretrained(str(JINA_PATH),dtype="auto",trust_remote_code=True,local_files_only=True).to("cuda:0").eval(); torch.cuda.synchronize(); load_ms=(time.perf_counter()-load_t)*1000
    rows=[]
    for n,s in enumerate(subs,1):
        slate=slates[s["canonical_id"]]; by_id={str(c["candidate_id"]):str(c["content"]) for c in slate["candidates"]}
        for base_name in ("kev_native","kev_budget","drex_native","drex_budget"):
            mv=s["model_views"]["kev_native" if base_name.startswith("kev") else "drex_native"]
            ids=[str(x) for x in mv["candidate_ids"]]
            # The native view is the frozen untrimmed prefix; the budget view
            # is the same prefix with frozen E11-A token truncation.
            if base_name.endswith("native"):
                docs=[by_id[x] for x in ids]; view=base_name
            else:
                # Prefix strings are not stored in the R6 manifest; token
                # counts alone cannot reconstruct the exact budgeted text.
                # Record the control as blocked rather than inventing text.
                docs=None; view=base_name
            rec={"canonical_id":s["canonical_id"],"domain":"leetcode","query_id":s["query_id"],"model":"jina_v3_5","policy":"jina_"+view+"_sub","source_slate_sha256":s["source_slate_sha256"],"labels_joined":False,"candidate_ids":ids,"status":"FAIL","selected_ids":[],"scores":[],"native_interface":"Jina v3.5 rerank","model_revision":JINA_REV,"error":None}
            if docs is None:
                rec["error"]="BUDGET_STRING_NOT_PERSISTED_FOR_JINA_CONTROL"; rows.append(rec); continue
            req={"query":str(slate["query"]),"documents":docs,"candidate_ids":ids}; rec["input_sha256"]=sha(req); torch.cuda.reset_peak_memory_stats(); t=time.perf_counter()
            try:
                result=model.rerank(slate["query"],docs); mapping={int(x["index"]):float(x["relevance_score"]) for x in result}; scores=[mapping[i] for i in range(len(ids))]
                order=sorted(range(len(ids)),key=lambda i:(-scores[i],i,ids[i])); torch.cuda.synchronize(); rec.update({"status":"PASS","scores":scores,"selected_ids":[ids[i] for i in order[:5]],"elapsed_ms":(time.perf_counter()-t)*1000,"peak_memory_allocated_mib":torch.cuda.max_memory_allocated()/2**20,"physical_forward_count":"NOT_EXPOSED_BY_RUNTIME"})
            except Exception as exc: rec.update({"error":f"{type(exc).__name__}:{exc}","elapsed_ms":(time.perf_counter()-t)*1000,"peak_memory_allocated_mib":torch.cuda.max_memory_allocated()/2**20})
            rows.append(rec)
        if n%20==0: print(json.dumps({"completed":n,"total":len(subs)}),flush=True)
    OUT.mkdir(parents=True,exist_ok=True); p=OUT/"jina_r6_native_budgeted_raw.jsonl"; p.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in rows),encoding="utf-8"); meta={"rows":len(rows),"valid":sum(x["status"]=="PASS" for x in rows),"failed":sum(x["status"]!="PASS" for x in rows),"load_ms":load_ms,"raw_sha256":fsha(p),"labels_joined":False,"note":"Budget controls are explicitly blocked because the frozen sub-slate manifest stores token metadata but not budgeted strings."}; (p.with_suffix(".meta.json")).write_text(json.dumps(meta,indent=2)+"\n"); print(json.dumps(meta,indent=2)); del model,tok; gc.collect(); torch.cuda.empty_cache()
if __name__=="__main__": main()
