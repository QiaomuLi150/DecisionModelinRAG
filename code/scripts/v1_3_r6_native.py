#!/usr/bin/env python3
"""R6 native/budgeted Kev or Drex selector run on frozen LeetCode sub-slates."""
from __future__ import annotations
import argparse, gc, hashlib, json, math, os, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
E9=ROOT/"results/non_jev/v2/e9_bright_v1_0"
SUB=ROOT/"results/v1_3/r6_leetcode_trim/leetcode_subslate_manifest.jsonl"
OUT=ROOT/"results/v1_3/r6_leetcode_trim/raw"
DREX=ROOT/"data/models/e12_o_checkpoints/drex_dlm8"
KEV_BASE=ROOT/"data/models/e12_o_checkpoints/kev9_base"
KEV_ADAPTER=ROOT/"data/models/e12_o_checkpoints/kev9_adapter"
KEV_RUNTIME=ROOT/"data/models/e12_o_runtimes/kev"
L=8192; RHO=128
def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def fsha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def make_req(row, docs):
    ids=[str(c["candidate_id"]) for c in docs]
    return {"state":{"query":str(row["query"])},"model":"v1_3_r6_native","questions":{"candidate":{"type":"choice","instructions":"Which candidate passage is most relevant to the query?","criteria":{cid:f"Candidate ID {cid}. Passage: {str(c['content'])}" for cid,c in zip(ids,docs)}}}}
def prefix(tok,text,cap): return tok.decode(tok.encode(text,add_special_tokens=False)[:cap],skip_special_tokens=False,clean_up_tokenization_spaces=False)
def budget_row(row,docs,tok,to_record,encode):
    k=len(docs); empty=[{"candidate_id":str(c["candidate_id"]),"content":""} for c in docs]
    # Use the native request encoder for the empty-passage overhead.
    from kev.api import SystemOneRequest
    def n_tokens(xs):
        rec,_=to_record(SystemOneRequest.model_validate(make_req(row,xs))); return len(encode(tok,rec,max_state=L,max_branch=L,strict=True)["ids"])
    overhead=n_tokens(empty); available=L-RHO-overhead
    if available < k: return None,{"error_class":"UNBUDGETABLE_QUERY_OR_OVERHEAD","overhead_tokens":overhead,"effective_content_budget":available}
    cap=max(1,available//k); reductions=0; selected=None; total=None
    while cap>=1:
        xs=[{"candidate_id":str(c["candidate_id"]),"content":prefix(tok,str(c["content"]),cap)} for c in docs]
        try: total=n_tokens(xs)
        except Exception: total=None
        if total is not None and total<=L-RHO: selected=xs; break
        cap-=1; reductions+=1
    if selected is None: return None,{"error_class":"UNBUDGETABLE_QUERY_OR_OVERHEAD","overhead_tokens":overhead,"effective_content_budget":available}
    return selected,{"overhead_tokens":overhead,"effective_content_budget":available,"final_cap":cap,"cap_reductions":reductions,"serialized_tokens":total}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",choices=["kev9","drex_dlm"],required=True); ap.add_argument("--gpu",type=int,required=True); args=ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    # Keep the two official `kev` packages isolated: Drex's checkpoint loader
    # accepts dtype/device, while the Kev runtime exposes a different contract.
    sys.path.insert(0,str(DREX/"code" if args.model=="drex_dlm" else KEV_RUNTIME))
    import torch
    from kev.api import SystemOneRequest,to_record
    from kev.model import encode
    slates={f"{x['domain']}:{x['query_id']}":x for x in (json.loads(v) for v in (E9/"01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl").open(encoding="utf-8") if v.strip())}
    subs=[json.loads(v) for v in SUB.open(encoding="utf-8") if v.strip()]
    if args.model=="drex_dlm":
        from kev.checkpoint import load
        load_start=time.perf_counter(); tok,model=load(str(DREX),device="cuda",dtype=torch.bfloat16); interface="Drex-DLM official System-One pointer"; forwards=1
    else:
        from kev.checkpoint import Checkpoint,LoadOptions
        ck=Checkpoint(str(KEV_ADAPTER)); ck.meta.base=str(KEV_BASE); ck.meta.base_revision=None; opts=LoadOptions(dtype=torch.bfloat16,merge=True,backend="torch",cuda_graphs=False,fused=False)
        load_start=time.perf_counter(); tok,model=ck.load("cuda",opts); interface="Kev-9B official System-One pointer"; forwards=2
    torch.cuda.synchronize(); load_ms=(time.perf_counter()-load_start)*1000; rows=[]
    for n,s in enumerate(subs,1):
        row=slates[s["canonical_id"]]
        for view_name in ("native","budget"):
            mv=s["model_views"]["kev_native" if args.model=="kev9" else "drex_native"]
            k=int(mv.get("k",0)); docs=row["candidates"][:k] if k else []
            if view_name=="budget" and docs: docs,info=budget_row(row,docs,tok,to_record,encode)
            else: info={"overhead_tokens":None,"effective_content_budget":None} if docs else {"error_class":"NO_PREFIX_AT_LEAST_6"}
            policy=f"{args.model}_{view_name}_sub"; rec={"canonical_id":s["canonical_id"],"domain":"leetcode","query_id":s["query_id"],"model":args.model,"view":view_name,"policy":policy,"k":len(docs) if docs else 0,"candidate_ids":[str(c["candidate_id"]) for c in docs] if docs else [],"source_slate_sha256":s["source_slate_sha256"],"labels_joined":False,"status":"FAIL","scores":[],"selected_ids":[],"error":info.get("error_class"),"input_sha256":sha(make_req(row,docs)) if docs else None,"native_interface":interface,"physical_forward_count":forwards}
            if not docs or info.get("error_class"): rows.append(rec); continue
            try:
                req=make_req(row,docs); rr,_=to_record(SystemOneRequest.model_validate(req)); enc=model.encode(tok,rr,max_state=L,max_branch=L,strict=True); torch.cuda.reset_peak_memory_stats(); t=time.perf_counter()
                with torch.inference_mode(): probs=model.probs(enc)
                torch.cuda.synchronize(); scores=[float(x) for x in probs[0].tolist()]; ids=rec["candidate_ids"]
                if len(scores)!=len(ids) or not all(math.isfinite(x) for x in scores): raise ValueError("invalid score vector")
                idx=sorted(range(len(ids)),key=lambda i:(-scores[i],i,ids[i])); rec.update({"status":"PASS","scores":scores,"selected_ids":[ids[i] for i in idx[:5]],"elapsed_ms":(time.perf_counter()-t)*1000,"peak_memory_allocated_mib":torch.cuda.max_memory_allocated()/2**20,"input_tokens":len(enc["ids"]),"native_request_sha256":sha(req),"error":None,"budget_meta":info})
            except Exception as exc: rec.update({"status":"FAIL","error":f"{type(exc).__name__}:{exc}","elapsed_ms":(time.perf_counter()-t)*1000 if 't' in locals() else None,"peak_memory_allocated_mib":torch.cuda.max_memory_allocated()/2**20})
            rows.append(rec)
        if n%20==0: print(json.dumps({"model":args.model,"completed":n,"total":len(subs)}),flush=True)
    OUT.mkdir(parents=True,exist_ok=True); p=OUT/f"{args.model}_r6_native_budgeted_raw.jsonl"; p.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in rows),encoding="utf-8"); meta={"model":args.model,"rows":len(rows),"valid":sum(x["status"]=="PASS" for x in rows),"failed":sum(x["status"]!="PASS" for x in rows),"load_ms":load_ms,"raw_sha256":fsha(p),"labels_joined":False,"model_revision":"E12 authenticated local checkpoint"}; (p.with_suffix(".meta.json")).write_text(json.dumps(meta,indent=2)+"\n"); print(json.dumps(meta,indent=2)); del model,tok; gc.collect(); torch.cuda.empty_cache()
if __name__=="__main__": main()
