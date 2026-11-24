#!/usr/bin/env python3
"""C1 native selector runner on the prepared v2.1 hybrid transfer views.

The runner is label-blind and writes append-only JSONL traces.  Each process
owns one device and one model; the prepared views already encode the frozen
per-query b(q) text, so no alternate truncation is introduced here.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, math, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v3_6_essential/followup_v2_correction/C1_transfer_confirmation/raw'
PPLX=Path('<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/14c9d90137962d300f3de95633f7b6535e80c5ce')
JINA=Path('<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/e8a93f33f0b22108f8c2364f8484ce3422552fbc')
def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def read(p): return [json.loads(x) for x in p.open(encoding='utf-8') if x.strip()]
def req(q,docs): return {'state':f'Query: {q}\nEvaluate the supplied candidate passages using the option descriptions.','question':{'type':'choice','instructions':f'Select the candidate passage that best answers the query: {q}','criteria':{f'candidate_{i:03d}':f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i,d in enumerate(docs)}}}
def top(ids,s): return [ids[i] for i in sorted(range(len(ids)),key=lambda i:(-float(s[i]),i,ids[i]))[:5]]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model',choices=['pplx','jina'],required=True); ap.add_argument('--gpu',type=int,required=True); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--shards',type=int,default=1); ap.add_argument('--Ls',default='2048,4096'); args=ap.parse_args()
 os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
 import torch
 if args.model=='pplx':
  sys.path.insert(0,str(PPLX/'source/src')); from transformers import AutoProcessor; from autojev.model import DecisionModel
  proc=AutoProcessor.from_pretrained(str(PPLX),local_files_only=True); model=DecisionModel(PPLX,device='cuda:0',max_image_pixels=512*512)
 else:
  from transformers import AutoTokenizer,AutoModel
  tok=AutoTokenizer.from_pretrained(str(JINA),local_files_only=True); model=AutoModel.from_pretrained(str(JINA),dtype='auto',trust_remote_code=True,local_files_only=True).to('cuda:0').eval()
 torch.cuda.synchronize(); OUT.mkdir(parents=True,exist_ok=True)
 Ls=[int(x) for x in args.Ls.split(',') if x]
 rows=[]
 for L in Ls:
  rows.extend((L,x) for x in read(ROOT/f'results/v3_6_essential/followup_v2_correction/C1_transfer_confirmation/prepared_L{L}.jsonl'))
 rows=[x for i,x in enumerate(rows) if i%args.shards==args.shard]
 path=OUT/f'{args.model}_shard{args.shard}_of{args.shards}_L{"_".join(map(str,Ls))}.jsonl'; part=path.with_suffix('.jsonl.part')
 if path.exists() or part.exists(): raise RuntimeError(f'OUTPUT_EXISTS:{path}')
 failures={}; valid=0; start_all=time.perf_counter()
 with part.open('w',encoding='utf-8') as f:
  for ix,(L,x) in enumerate(rows,1):
   docs=x['candidates']; ids=[d['candidate_id'] for d in docs]; rec={'canonical_id':x['canonical_id'],'dataset':x['dataset'],'query_id':x['query_id'],'group_id':x.get('group_id',''),'policy':f'{args.model}_single_matched','budget_L':L,'order_id':'original','candidate_ids':ids,'source_slate_sha256':x['source_slate_sha256'],'input_sha256':x['input_sha256'],'full_input_sha256':x['full_input_sha256'],'labels_joined':False,'input_label_blind':True,'model_revision':str(PPLX.name if args.model=='pplx' else JINA.name),'gpu_index_requested':args.gpu,'status':'FAIL','valid':False,'selected_ids':[],'scores':[],'n_calls':0,'physical_forward_count':0,'error_class':None,'error':None,'latency_ms':None,'peak_memory_allocated_mib':None}
   torch.cuda.reset_peak_memory_stats(); t=time.perf_counter()
   try:
    if args.model=='pplx':
     vals=model.predict([req(str(x['query']),docs)],batch_size=1)[0]; scores=[float(v) for v in vals]
    else:
     result=model.rerank(str(x['query']),[d['content'] for d in docs]); mp={int(z['index']):float(z['relevance_score']) for z in result}; scores=[mp[i] for i in range(20)]
    if len(scores)!=20 or not all(math.isfinite(z) for z in scores): raise ValueError('INVALID_SCORE_VECTOR')
    sel=top(ids,scores)
    if len(sel)!=5 or len(set(sel))!=5: raise ValueError('INVALID_TOP5')
    rec.update({'status':'PASS','valid':True,'selected_ids':sel,'scores':scores,'n_calls':1,'physical_forward_count':1})
    valid+=1
   except Exception as e:
    cls='CUDA_OOM' if 'out of memory' in str(e).lower() else ('TIMEOUT' if 'timeout' in str(e).lower() else type(e).__name__); rec.update({'error_class':cls,'error':f'{type(e).__name__}: {e}'}); failures[cls]=failures.get(cls,0)+1
   torch.cuda.synchronize(); rec['latency_ms']=round((time.perf_counter()-t)*1000,3); rec['peak_memory_allocated_mib']=round(float(torch.cuda.max_memory_allocated())/2**20,3); f.write(json.dumps(rec,ensure_ascii=False,sort_keys=True)+'\n'); f.flush()
   if ix%50==0: print(json.dumps({'model':args.model,'gpu':args.gpu,'completed':ix,'total':len(rows),'valid':valid,'failures':failures}),flush=True)
 part.replace(path); summary={'status':'PASS','model':args.model,'gpu':args.gpu,'shard':args.shard,'shards':args.shards,'rows':len(rows),'valid':valid,'failures':len(rows)-valid,'failure_classes':failures,'elapsed_s':time.perf_counter()-start_all,'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'labels_joined':False,'Ls':Ls}; (path.with_suffix('.SUMMARY.json')).write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
 del model; gc.collect(); torch.cuda.empty_cache()
if __name__=='__main__':main()
