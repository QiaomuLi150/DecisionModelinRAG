#!/usr/bin/env python3
"""R3 label-blind PPLX two-stage selector on the frozen BRIGHT population."""
from __future__ import annotations
import argparse, gc, hashlib, json, math, os, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
E9=ROOT/"results/non_jev/v2/e9_bright_v1_0"
OUT=ROOT/"results/v1_3/r3_multipass"
PPLX_REV="14c9d90137962d300f3de95633f7b6535e80c5ce"
PPLX_PATH=Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")
L=8192; RHO=128
def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def fsha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def token_count(proc,dm,row,codes):
    msg=dm(row,codes); text=proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False); e=proc(text=[text],images=None,padding=True,return_tensors="pt"); return int(e["attention_mask"].sum())
def make_row(item,docs,codes):
    return {"state":f"Query: {item['query']}\nEvaluate the supplied candidate passages using the option descriptions.","question":{"type":"choice","instructions":f"Select the candidate passage that best answers the query: {item['query']}","criteria":{codes[i]:f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i,d in enumerate(docs)}}}
def prefix(tok,text,cap): return tok.decode(tok.encode(text,add_special_tokens=False)[:cap],skip_special_tokens=False,clean_up_tokenization_spaces=False)
def fit_docs(item,docs,proc,dm,tok,codes):
    empty=[{"candidate_id":str(d["candidate_id"]),"content":""} for d in docs]; overhead=token_count(proc,dm,make_row(item,empty,codes),codes); available=L-RHO-overhead; k=len(docs)
    if available<k:return None,{"error_class":"UNBUDGETABLE_QUERY_OR_OVERHEAD","overhead_tokens":overhead,"effective_content_budget":available}
    cap=max(1,available//k); reductions=0
    while cap>=1:
        xs=[{"candidate_id":str(d["candidate_id"]),"content":prefix(tok,str(d["content"]),cap)} for d in docs]; total=token_count(proc,dm,make_row(item,xs,codes),codes)
        if total<=L-RHO:return xs,{"overhead_tokens":overhead,"effective_content_budget":available,"final_cap":cap,"reductions":reductions,"serialized_tokens":total}
        cap-=1; reductions+=1
    return None,{"error_class":"UNBUDGETABLE_QUERY_OR_OVERHEAD","overhead_tokens":overhead,"effective_content_budget":available}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--gpu",type=int,required=True); ap.add_argument("--limit",type=int,default=0); ap.add_argument("--start",type=int,default=0); args=ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu); sys.path.insert(0,str(PPLX_PATH/"source"/"src"))
    import torch
    from autojev.model import DecisionModel,decision_messages
    from transformers import AutoProcessor
    slates=[json.loads(v) for v in (E9/"01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl").open(encoding="utf-8") if v.strip()]; slates=sorted(slates,key=lambda x:(x["domain"],x["query_id"]));
    slates=slates[args.start:args.start+args.limit] if args.limit else slates[args.start:]
    load_t=time.perf_counter(); proc=AutoProcessor.from_pretrained(str(PPLX_PATH),local_files_only=True); tok=proc.tokenizer; model=DecisionModel(PPLX_PATH,device="cuda:0",max_image_pixels=512*512); torch.cuda.synchronize(); load_ms=(time.perf_counter()-load_t)*1000; codes10=[f"candidate_{i:03d}" for i in range(10)]; rows=[]
    for n,item in enumerate(slates,1):
        rec={"canonical_id":f"{item['domain']}:{item['query_id']}","domain":item["domain"],"query_id":str(item["query_id"]),"policy":"pplx_two_stage","source_slate_sha256":item["slate_sha256"],"status":"FAIL","selected_ids":[],"stage_selected_ids":[],"n_calls":0,"total_ms":None,"errors":[],"labels_joined":False}
        t0=time.perf_counter(); all_stage_docs=[]; failed=False
        try:
            for win in (item["candidates"][:10],item["candidates"][10:20]):
                docs,info=fit_docs(item,win,proc,decision_messages,tok,codes10); rec["n_calls"]+=1
                if docs is None: rec["errors"].append(info); failed=True; break
                req=make_row(item,docs,codes10); scores=[float(x) for x in model.predict([req],batch_size=1)[0]]; ids=[str(d["candidate_id"]) for d in docs]
                if len(scores)!=10 or not all(math.isfinite(x) for x in scores): raise ValueError("invalid stage-1 score vector")
                idx=sorted(range(10),key=lambda i:(-scores[i],i,ids[i])); top=[docs[i] for i in idx[:5]]; all_stage_docs.extend(top); rec.setdefault("stage1_views",[]).append({"candidate_ids":ids,"input_sha256":sha(req),"scores_sha256":sha(scores),"budget":info})
            if not failed:
                docs,info=fit_docs(item,all_stage_docs,proc,decision_messages,tok,codes10); rec["n_calls"]+=1
                if docs is None: rec["errors"].append(info); failed=True
                else:
                    req=make_row(item,docs,codes10); scores=[float(x) for x in model.predict([req],batch_size=1)[0]]; ids=[str(d["candidate_id"]) for d in docs]
                    if len(scores)!=10 or not all(math.isfinite(x) for x in scores): raise ValueError("invalid stage-2 score vector")
                    idx=sorted(range(10),key=lambda i:(-scores[i],i,ids[i])); rec["stage2_view"]={"candidate_ids":ids,"input_sha256":sha(req),"scores_sha256":sha(scores),"budget":info}; rec.update({"status":"PASS","selected_ids":[ids[i] for i in idx[:5]],"stage_selected_ids":[x["candidate_id"] for x in all_stage_docs]})
        except Exception as exc: rec["errors"].append({"error_class":type(exc).__name__,"message":str(exc)}); failed=True
        rec["total_ms"]=(time.perf_counter()-t0)*1000; rows.append(rec)
        if n%25==0: print(json.dumps({"completed":n,"total":len(slates)}),flush=True)
    OUT.mkdir(parents=True,exist_ok=True); suffix=f"_start{args.start}_n{len(rows)}" if args.start or args.limit else ""; p=OUT/f"multipass_raw{suffix}.jsonl"; p.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in rows),encoding="utf-8"); meta={"policy":"pplx_two_stage","gpu":args.gpu,"start":args.start,"n_rows":len(rows),"valid":sum(x["status"]=="PASS" for x in rows),"failed":sum(x["status"]!="PASS" for x in rows),"load_ms":load_ms,"raw_sha256":fsha(p),"labels_joined":False,"model_revision":PPLX_REV}; (OUT/f"multipass_meta{suffix}.json").write_text(json.dumps(meta,indent=2)+"\n"); print(json.dumps(meta,indent=2)); del model; gc.collect(); torch.cuda.empty_cache()
if __name__=="__main__": main()
