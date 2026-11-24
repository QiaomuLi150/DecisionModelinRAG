#!/usr/bin/env python3
"""E11-V V2 native selector evaluation on the frozen CRAG cohort.

The script has a preparation mode that creates a compact content-only view and
an execution mode for the four frozen policies.  Gold answers never enter
selector requests or selection policies.  Model code is reused from the
authenticated E11 native contracts; no checkpoint download is performed.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path
from typing import Any, Iterable

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/non_jev/v2/e11v_v1_0"
ADM=OUT/"01_admission"
SEL=OUT/"02_selection"
CRAG=ROOT/"data/e11v/crag_task_3/extracted/crag_task_3_dev_v4"
PPLX_REV="14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV="e8a93f33f0b22108f8c2364f8484ce3422552fbc"
PPLX_PATH=Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")
JINA_PATH=Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/{JINA_REV}")
MAX_CONTEXT=8192
SAFETY_MARGIN=128
TOPK=20

def sha_bytes(b:bytes)->str: return hashlib.sha256(b).hexdigest()
def sha_json(x:Any)->str: return sha_bytes(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",", ":")).encode())
def utc(): return time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
def sha_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def gpu_snapshot():
    try:
        env=os.environ.copy(); env.pop('CUDA_VISIBLE_DEVICES',None)
        raw=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True,env=env)
        dev=[]
        for line in raw.splitlines():
            x=[v.strip() for v in line.split(',')]
            if len(x)>=7: dev.append({"index":int(x[0]),"uuid":x[1],"name":x[2],"memory_total_mib":int(x[3]),"memory_used_mib":int(x[4]),"memory_free_mib":int(x[5]),"utilization_percent":int(x[6])})
        apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory","--format=csv,noheader,nounits"],text=True,env=env).strip()
        return {"status":"ok","devices":dev,"compute_apps_raw":apps,"captured_utc":utc()}
    except Exception as e: return {"status":"unavailable","error":repr(e),"captured_utc":utc()}
def iter_rows()->Iterable[dict[str,Any]]:
    for p in sorted(CRAG.glob('crag_task_3_dev_v4_*.jsonl')):
        with p.open(encoding='utf-8') as f:
            for line in f:
                try: yield json.loads(line)
                except Exception: continue
def load_cohort_manifest():
    x=json.loads((ADM/"V1_COHORT_MANIFEST.json").read_text())
    rows=x.get('representative',[])+x.get('oversized_diagnostic',[])
    return {r['question_id']:r for r in rows}, rows
def content_rows():
    selected, rows=load_cohort_manifest()
    path=SEL/"V1_COHORT_CONTENT.jsonl"; path.parent.mkdir(parents=True,exist_ok=True)
    seen=set()
    writer=path.open('w',encoding='utf-8')
    for row in iter_rows():
        qid=str(row.get('interaction_id') or '')
        if qid not in selected: continue
        pages=[]
        for rank,page in enumerate(row.get('search_results') or [],1):
            if not isinstance(page,dict): continue
            url=str(page.get('page_url') or '').strip()
            html=page.get('page_result')
            text=html if isinstance(html,str) and html.strip() else page.get('page_snippet')
            if url and isinstance(text,str) and text.strip():
                pages.append({"candidate_id":f"crag_task3:{qid}::archived_rank_{rank:02d}","archived_rank":rank,"page_url":url,"page_name":str(page.get('page_name') or ''),"content":text})
                if len(pages) == TOPK: break
        if len(pages)==TOPK:
            base=selected[qid]
            obj={"question_id":qid,"cohort":base['cohort'],"domain":base['domain'],"question_type":base['question_type'],"source_component":base['source_component'],"query":str(row.get('query') or ''),"candidate_ids":[p['candidate_id'] for p in pages],"candidates":pages}
            writer.write(json.dumps(obj,ensure_ascii=False,sort_keys=True)+'\n'); seen.add(qid)
    writer.close()
    if seen!=set(selected):
        missing=sorted(set(selected)-seen); raise RuntimeError(f"missing selected content rows: {len(missing)}")
    return path,None
def iter_content(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def make_row(item,docs,codes):
    criteria={codes[i]:f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i,d in enumerate(docs)}
    q=item['query']
    return {"state":f"Query: {q}\nEvaluate the supplied candidate passages using the option descriptions.","question":{"type":"choice","instructions":f"Select the candidate passage that best answers the query: {q}","criteria":criteria}}
def token_count(processor,decision_messages,row,codes):
    msgs=decision_messages(row,codes)
    text=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    enc=processor(text=[text],images=None,padding=True,return_tensors='pt')
    return int(enc['attention_mask'].sum())
def prefix(tokenizer,text,cap):
    # Ask the fast tokenizer for only the deterministic prefix.  This is
    # equivalent to encode-then-slice but avoids materializing 300k-token HTML
    # pages in Python memory.
    ids=tokenizer(text,add_special_tokens=False,truncation=True,max_length=cap,return_attention_mask=False)['input_ids']
    return tokenizer.decode(ids[:cap],skip_special_tokens=False,clean_up_tokenization_spaces=False)
def budget_item(item,processor,decision_messages,tokenizer):
    codes=[f'candidate_{i:03d}' for i in range(TOPK)]
    originals=[{"candidate_id":d['candidate_id'],"content":d['content']} for d in item['candidates']]
    overhead=token_count(processor,decision_messages,make_row(item,[{"candidate_id":d['candidate_id'],"content":""} for d in originals],codes),codes)
    avail=MAX_CONTEXT-SAFETY_MARGIN-overhead
    if avail<20: return {"status":"UNBUDGETABLE_QUERY_OR_OVERHEAD","question_id":item['question_id'],"candidate_ids":item['candidate_ids'],"overhead_tokens":overhead,"effective_content_budget":avail}
    cap=max(1,avail//TOPK); reductions=0
    while cap>=1:
        encoded_prefixes=tokenizer([d['content'] for d in originals],add_special_tokens=False,truncation=True,max_length=cap,padding=False,return_attention_mask=False)['input_ids']
        docs=[{"candidate_id":d['candidate_id'],"content":tokenizer.decode(ids[:cap],skip_special_tokens=False,clean_up_tokenization_spaces=False)} for d,ids in zip(originals,encoded_prefixes)]
        total=token_count(processor,decision_messages,make_row(item,docs,codes),codes)
        if total<=MAX_CONTEXT-SAFETY_MARGIN: break
        cap-=1; reductions+=1
    if cap<1: return {"status":"UNBUDGETABLE_QUERY_OR_OVERHEAD","question_id":item['question_id'],"candidate_ids":item['candidate_ids'],"overhead_tokens":overhead,"effective_content_budget":avail}
    return {"status":"ok","question_id":item['question_id'],"domain":item['domain'],"cohort":item['cohort'],"query":item['query'],"candidate_ids":item['candidate_ids'],"original_content_sha256":[sha_bytes(d['content'].encode()) for d in originals],"budgeted_candidates":docs,"budgeted_content_sha256":[sha_bytes(d['content'].encode()) for d in docs],"question_overhead_tokens":overhead,"effective_content_budget":avail,"initial_per_passage_cap":max(1,avail//TOPK),"final_per_passage_cap":cap,"deterministic_cap_reductions":reductions,"pplx_serialized_token_count":total,"pplx_request_sha256":sha_json(make_row(item,docs,codes)),"jina_budget_request_sha256":sha_json({"query":item['query'],"documents":[d['content'] for d in docs],"candidate_ids":item['candidate_ids']}),"policy_id":"E11-A-EQUAL-PREFIX-128-MARGIN-V1","label_blind":True}
def prepare(path):
    import sys
    sys.path.insert(0,str(PPLX_PATH/'source/src'))
    from autojev.model import decision_messages
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(str(PPLX_PATH),local_files_only=True)
    rows=[]
    for item in iter_content(path): rows.append(budget_item(item,processor,decision_messages,processor.tokenizer))
    with (SEL/'V2_BUDGET_MANIFEST.jsonl').open('w',encoding='utf-8') as f:
        for x in rows: f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n')
    return rows
def native(model,rows,gpu,full_jina=False):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    import torch
    before=gpu_snapshot(); load_error=None; adapter=None; processor=None
    tload=time.perf_counter_ns()
    try:
        if model=='pplx_budgeted':
            sys.path.insert(0,str(PPLX_PATH/'source/src'))
            from autojev.model import DecisionModel
            adapter=DecisionModel(PPLX_PATH,device='cuda:0',max_image_pixels=512*512)
        else:
            from transformers import AutoModel,AutoTokenizer
            processor=AutoTokenizer.from_pretrained(str(JINA_PATH),local_files_only=True)
            adapter=AutoModel.from_pretrained(str(JINA_PATH),dtype='auto',trust_remote_code=True,local_files_only=True).to('cuda:0').eval()
        torch.cuda.synchronize()
    except Exception as e: load_error=f'{type(e).__name__}: {e}'
    load_ms=(time.perf_counter_ns()-tload)/1e6
    out=[]; codes=[f'candidate_{i:03d}' for i in range(TOPK)]
    for item in rows:
        ids=item['candidate_ids']
        if model=='pplx_budgeted': docs=item.get('budgeted_candidates',[]); req_hash=item.get('pplx_request_sha256')
        elif full_jina: docs=item['_full_candidates']; req_hash=sha_json({"query":item['query'],"documents":[d['content'] for d in docs],"candidate_ids":ids})
        else: docs=item.get('budgeted_candidates',[]); req_hash=item.get('jina_budget_request_sha256')
        result={"policy":"P1_PPLX_BUDGETED" if model=='pplx_budgeted' else ('S1_JINA_FULL' if full_jina else 'S2_JINA_BUDGETED'),"question_id":item['question_id'],"domain":item.get('domain'),"cohort":item.get('cohort'),"candidate_ids":ids,"candidate_ids_sha256":sha_json(ids),"input_sha256":req_hash,"status":"ERROR","selected_ids":[],"scores":[],"native_call_count":0,"physical_forward_count":"NOT_EXPOSED_BY_RUNTIME","gpu_id":gpu,"load_ms":load_ms,"failure":load_error,"failure_class":"MODEL_LOAD_ERROR" if load_error else None,"started_utc":utc()}
        if load_error or item.get('status')!='ok':
            if not load_error: result.update(failure=item.get('status'),failure_class=item.get('status'))
            out.append(result); continue
        try:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter_ns()
            if model=='pplx_budgeted':
                request=make_row(item,docs,codes); scores=[float(x) for x in adapter.predict([request],batch_size=1)[0]]; result['native_input_sha256']=sha_json(request); result['readout']='native choice probabilities'
            else:
                request={"query":item['query'],"documents":[d['content'] for d in docs],"candidate_ids":ids}; rr=adapter.rerank(item['query'],[d['content'] for d in docs]); mapping={int(x['index']):float(x['relevance_score']) for x in rr}; scores=[mapping[i] for i in range(TOPK)]; result['native_input_sha256']=sha_json(request); result['readout']='native listwise score by original index'
            if len(scores)!=TOPK or not all(math.isfinite(x) for x in scores): raise RuntimeError('invalid score vector')
            order=sorted(range(TOPK),key=lambda i:(-scores[i],i,ids[i]))
            torch.cuda.synchronize()
            result.update(status='ok',scores=scores,selected_ids=[ids[i] for i in order[:5]],duration_ms=(time.perf_counter_ns()-t0)/1e6,peak_memory_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,native_call_count=1,failure=None,failure_class=None,finished_utc=utc())
        except Exception as e:
            msg=f'{type(e).__name__}: {e}'; cls='CONTEXT_LIMIT' if 'context' in msg.lower() or '8192' in msg else 'CUDA_OOM' if 'out of memory' in msg.lower() or 'oom' in msg.lower() else 'NATIVE_ERROR'; result.update(duration_ms=(time.perf_counter_ns()-t0)/1e6 if 't0' in locals() else None,failure=msg,failure_class=cls,finished_utc=utc())
        out.append(result)
    return out,{"model":model,"full_jina":full_jina,"gpu_id":gpu,"gpu_pre":before,"gpu_post":gpu_snapshot(),"load_ms":load_ms,"load_error":load_error}
def main():
    global SEL
    ap=argparse.ArgumentParser(); ap.add_argument('--prepare-only',action='store_true'); ap.add_argument('--execute',action='store_true'); ap.add_argument('--gpu',type=int,default=5); ap.add_argument('--output-dir',default=None); args=ap.parse_args()
    base_sel=SEL
    source_content=base_sel/'V1_COHORT_CONTENT.jsonl'
    source_budget=base_sel/'V2_BUDGET_MANIFEST.jsonl'
    if args.output_dir: SEL=OUT/args.output_dir
    SEL.mkdir(parents=True,exist_ok=True)
    content=source_content if args.execute else SEL/'V1_COHORT_CONTENT.jsonl'
    if not content.exists(): content,_=content_rows()
    if args.prepare_only:
        rows=prepare(content); print(json.dumps({"content_path":str(content),"budget_rows":len(rows),"budgetable":sum(x.get('status')=='ok' for x in rows),"gpu_calls":0},indent=2)); return 0
    if not args.execute: raise SystemExit('use --prepare-only or --execute')
    budget_path=source_budget
    if not budget_path.exists(): prepare(content)
    rows=[json.loads(x) for x in budget_path.read_text(encoding='utf-8').splitlines() if x.strip()]
    # Full Jina receives the exact original archived strings; PPLX/S2 receive E11 budgeted strings.
    originals={x['question_id']:x for x in iter_content(content)}
    for x in rows: x['_full_candidates']=originals[x['question_id']]['candidates']
    # Freeze R0 before model calls; it uses original archived order only.
    r0=[]
    for x in rows: r0.append({"policy":"R0_RETRIEVAL_TOP5","question_id":x['question_id'],"domain":x.get('domain'),"cohort":x.get('cohort'),"candidate_ids":x['candidate_ids'],"selected_ids":x['candidate_ids'][:5],"status":"ok","native_call_count":0,"input_sha256":sha_json({"query":x['query'],"candidate_ids":x['candidate_ids']})})
    (SEL/'V2_R0_RETRIEVAL_TOP5.jsonl').write_text('\n'.join(json.dumps(x,sort_keys=True) for x in r0)+'\n')
    pre={"protocol":"E11-V V2 pre-outcome freeze","cohort_manifest":sha_file(ADM/'V1_COHORT_MANIFEST.json'),"content_manifest":sha_file(content),"budget_policy":"E11-A-EQUAL-PREFIX-128-MARGIN-V1","pplx_revision":PPLX_REV,"jina_revision":JINA_REV,"candidate_scope":"official CRAG archived positions 1..20","policies":["R0_RETRIEVAL_TOP5","P1_PPLX_BUDGETED","S1_JINA_FULL","S2_JINA_BUDGETED"],"gpu_pre":gpu_snapshot(),"pplx_path_present":PPLX_PATH.exists(),"jina_path_present":JINA_PATH.exists(),"gold_in_selector_inputs":False,"answers_in_selector_inputs":False,"created_utc":utc()}
    (SEL/'V2_PRE_OUTCOME_FREEZE.json').write_text(json.dumps(pre,indent=2)+'\n')
    for model,full in [('pplx_budgeted',False),('jina',True),('jina',False)]:
        rows_out,meta=native(model,rows,args.gpu,full_jina=full)
        name='V2_P1_PPLX_BUDGETED_RAW.jsonl' if model=='pplx_budgeted' else ('V2_S1_JINA_FULL_RAW.jsonl' if full else 'V2_S2_JINA_BUDGETED_RAW.jsonl')
        path=SEL/name
        if path.exists(): raise RuntimeError(f'refusing overwrite {path}')
        with path.open('w',encoding='utf-8') as f:
            for x in rows_out: f.write(json.dumps(x,sort_keys=True)+'\n')
        (SEL/name.replace('.jsonl','.RUN.json')).write_text(json.dumps({**meta,"raw_sha256":sha_file(path),"rows":len(rows_out)},indent=2)+'\n')
        del rows_out,meta; gc.collect()
    print(json.dumps({"gate":"E11V_V2_SELECTION_RAW_FROZEN","attempts":len(rows),"gpu_post":gpu_snapshot()},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
