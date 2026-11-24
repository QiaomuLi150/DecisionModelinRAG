#!/usr/bin/env python3
"""Post-freeze 20-record D1 determinism check; no labels and no retries."""
from __future__ import annotations
import argparse, gc, json, os, random, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/v1_6/d1_perwindow_jina"
BRIGHT=ROOT/"results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
MANIFEST=OUT/"D1_INPUT_MANIFEST.jsonl"
RAW_FILES=sorted((OUT/"raw").glob("d1_jina_two_stage_pplxwindow__*.jsonl"))
JINA_REV="e8a93f33f0b22108f8c2364f8484ce3422552fbc"
JINA_PATH=Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots")/JINA_REV

def read(p): return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--gpu",type=int,required=True); ap.add_argument("--sample-size",type=int,default=20); ap.add_argument("--seed",type=int,default=731); a=ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    sys.path.insert(0,str(ROOT/"scripts"))
    import v1_6_d1_runner as d1
    allm=read(MANIFEST); slates={f"{x['domain']}:{x['query_id']}":x for x in read(BRIGHT)}
    rng=random.Random(a.seed); idx=rng.sample(range(len(allm)),min(a.sample_size,len(allm))); sm=[allm[i] for i in idx]
    p=Path("<REDACTED_PATH><REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--jina-reranker-v3.5/snapshots")/JINA_REV
    from transformers import AutoProcessor,AutoTokenizer,AutoModel
    import torch
    pplx_path=Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/14c9d90137962d300f3de95633f7b6535e80c5ce")
    pplx_tok=AutoProcessor.from_pretrained(str(pplx_path),local_files_only=True).tokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(JINA_PATH),local_files_only=True)
    model=AutoModel.from_pretrained(str(JINA_PATH),dtype="auto",trust_remote_code=True,local_files_only=True).to("cuda:0").eval()
    plans=d1.make_plans_batch(sm,slates,pplx_tok,[f"candidate_{i:03d}" for i in range(10)])
    first=d1.process_batch(plans,model,tokenizer,torch,max(1,len(plans)),a.gpu)
    second=d1.process_batch(plans,model,tokenizer,torch,max(1,len(plans)),a.gpu)
    def signature(r): return {"selected_ids":r.get("selected_ids"),"stage_records":r.get("stage_records"),"status":r.get("status")}
    mism=[]
    for x,y in zip(first,second):
        if signature(x)!=signature(y): mism.append({"canonical_id":x["canonical_id"],"order_id":x["order_id"],"first":signature(x),"second":signature(y)})
    receipt={"seed":a.seed,"requested":a.sample_size,"evaluated":len(first),"selected_or_score_trace_mismatches":len(mism),"status":"PASS" if not mism else "FAIL","records":[{"canonical_id":x["canonical_id"],"order_id":x["order_id"],"status":x["status"],"selected_ids_equal":signature(x)==signature(y)} for x,y in zip(first,second)],"mismatches":mism}
    (OUT/"D1_DETERMINISM_RECHECK.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    del model,tokenizer,pplx_tok;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();print(json.dumps(receipt,indent=2,sort_keys=True))
    if mism: raise SystemExit(2)
if __name__=="__main__":main()
