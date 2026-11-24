#!/usr/bin/env python3
"""C2 two fresh deterministic Jina diagnostics, batch-size one, fp32 load."""
from __future__ import annotations
import argparse,gc,hashlib,importlib.util,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/v3_6_essential/followup_v2_correction/C2_jina_diagnostic/raw'
BRIGHT=ROOT/'results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl'; JINA=Path('<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/e8a93f33f0b22108f8c2364f8484ce3422552fbc'); PPLX=Path('<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/14c9d90137962d300f3de95633f7b6535e80c5ce')
def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def sha(x):return hashlib.sha256(canon(x).encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--run-id',required=True);a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
 spec=importlib.util.spec_from_file_location('v36',ROOT/'scripts/v3_6_essential.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
 import torch; torch.set_float32_matmul_precision('high'); sys.path.insert(0,str(PPLX/'source/src')); from autojev.model import decision_messages; from transformers import AutoProcessor,AutoTokenizer,AutoModel
 proc=AutoProcessor.from_pretrained(str(PPLX),local_files_only=True); ptoken=proc.tokenizer; tok=AutoTokenizer.from_pretrained(str(JINA),local_files_only=True); model=AutoModel.from_pretrained(str(JINA),dtype=torch.float32,trust_remote_code=True,local_files_only=True).to('cuda:0').eval();torch.cuda.synchronize()
 sl=sorted([json.loads(x) for x in BRIGHT.open() if x.strip()],key=lambda x:(x['domain'],str(x['query_id']))); OUT.mkdir(parents=True,exist_ok=True); path=OUT/f'jina_fp32_batch1_{a.run_id}.jsonl';part=path.with_suffix('.jsonl.part')
 with part.open('w',encoding='utf-8') as f:
  for ix,item in enumerate(sl):
   rec={'canonical_id':f"{item['domain']}:{item['query_id']}",'domain':item['domain'],'query_id':str(item['query_id']),'policy':'jina_single_8192_fp32_batch1','budget_L':8192,'order_id':'original','model_revision':JINA.name,'dtype':'float32','batch_size':1,'candidate_ids':[str(x['candidate_id']) for x in item['candidates']],'source_slate_sha256':item['slate_sha256'],'status':'FAIL','valid':False,'selected_ids':[],'scores':[],'labels_joined':False,'input_label_blind':True,'error_class':None,'error':None,'n_calls':0,'latency_ms':None,'peak_memory_allocated_mib':None}
   t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
   try:
    original=[{'candidate_id':str(d['candidate_id']),'content':str(d['content'])} for d in item['candidates']];docs,info=base.fit_docs(item,original,proc,decision_messages,ptoken,8192)
    if docs is None:raise RuntimeError(info['error_class'])
    result=model.rerank(str(item['query']),[d['content'] for d in docs]);mp={int(x['index']):float(x['relevance_score']) for x in result};scores=[mp[i] for i in range(len(docs))]
    if len(scores)!=20 or not all(math.isfinite(x) for x in scores):raise ValueError('INVALID_SCORE_VECTOR')
    ids=[d['candidate_id'] for d in docs];sel=[ids[i] for i in sorted(range(20),key=lambda i:(-scores[i],i,ids[i]))[:5]];rec.update({'candidate_ids':ids,'selected_ids':sel,'scores':scores,'status':'PASS','valid':True,'n_calls':1,'input_sha256':sha({'query':item['query'],'candidate_ids':ids,'documents':[d['content'] for d in docs]}),'budget_info':info})
   except Exception as e:rec.update({'error_class':'CUDA_OOM' if 'out of memory' in str(e).lower() else type(e).__name__,'error':f'{type(e).__name__}: {e}'})
   torch.cuda.synchronize();rec['latency_ms']=round((time.perf_counter()-t)*1000,3);rec['peak_memory_allocated_mib']=round(float(torch.cuda.max_memory_allocated())/2**20,3);f.write(json.dumps(rec,ensure_ascii=False,sort_keys=True)+'\n');f.flush()
   if (ix+1)%50==0:print(json.dumps({'run_id':a.run_id,'completed':ix+1,'total':len(sl)}),flush=True)
 part.replace(path);meta={'status':'PASS','run_id':a.run_id,'rows':len(sl),'valid':sum(1 for x in path.open() if '"status": "PASS"' in x),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model_revision':JINA.name,'dtype':'float32','batch_size':1,'labels_joined':False};path.with_suffix('.meta.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2));del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
