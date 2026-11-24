#!/usr/bin/env python3
"""E11-V V3 fixed-reader validation and V4 answer-reference analysis."""
from __future__ import annotations
import base64, concurrent.futures, hashlib, json, os, re, time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/non_jev/v2/e11v_v1_0'
REPAIR_DIR=os.environ.get('E11V_REPAIR_SELECTION_DIR','')
if REPAIR_DIR:
    SEL=OUT/'02_selection'/REPAIR_DIR
    READER_OUT=OUT/'03_reader'/REPAIR_DIR
    ANALYSIS=OUT/'04_analysis'/REPAIR_DIR
    CONTENT_PATH=SEL/'V1_COHORT_CONTENT_VISIBLE_TEXT.jsonl'
    POLICIES=['R0_RETRIEVAL_TOP5_VISIBLE_TEXT','P1_PPLX_BUDGETED_VISIBLE_TEXT','S1_JINA_FULL_VISIBLE_TEXT','S2_JINA_BUDGETED_VISIBLE_TEXT']
else:
    SEL=OUT/'02_selection/gpu5_retry_1'
    READER_OUT=OUT/'03_reader'
    ANALYSIS=OUT/'04_analysis'
    CONTENT_PATH=OUT/'02_selection/V1_COHORT_CONTENT.jsonl'
    POLICIES=['R0_RETRIEVAL_TOP5','P1_PPLX_BUDGETED','S2_JINA_BUDGETED']
READER_URL='http://service.example.invalid/v1/chat/completions'
READER_MODEL='Qwen/Qwen3-VL-8B-Instruct'
READER_REV='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
PROMPT_REV='E11V_FIXED_READER_QWEN_V1'
MAX_OUTPUT=64
TOKEN_CAP=256
HEADERS={'Cache-Control':'no-cache, no-store','Pragma':'no-cache','X-DecisionRAG-Cache':'disabled','Content-Type':'application/json'}

def sha_bytes(x:bytes)->str: return hashlib.sha256(x).hexdigest()
def sha_json(x): return sha_bytes(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
def sha_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def load_jsonl(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def words(x): return re.findall(r"[\w]+",str(x or '').casefold(),flags=re.UNICODE)
def f1(pred,ref):
    a=words(pred); b=words(ref)
    if not a or not b: return float(a==b)
    ca={x:a.count(x) for x in set(a)}; cb={x:b.count(x) for x in set(b)}
    overlap=sum(min(ca.get(x,0),cb.get(x,0)) for x in ca)
    if not overlap:return 0.0
    p=overlap/len(a); r=overlap/len(b); return 2*p*r/(p+r)
def cap_text(tokenizer,title,text):
    ids=tokenizer(str(text),add_special_tokens=False,truncation=True,max_length=TOKEN_CAP+1,return_attention_mask=False)['input_ids']
    view=tokenizer.decode(ids[:TOKEN_CAP],skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
    return f"{title}\n{view}",min(len(ids),TOKEN_CAP),len(ids)>TOKEN_CAP
def load_selection_rows():
    if REPAIR_DIR:
        files={'R0_RETRIEVAL_TOP5_VISIBLE_TEXT':SEL/'V2_R0_RETRIEVAL_TOP5_VISIBLE_TEXT.jsonl','P1_PPLX_BUDGETED_VISIBLE_TEXT':SEL/'V2_P1_PPLX_BUDGETED_VISIBLE_TEXT_RAW.jsonl','S1_JINA_FULL_VISIBLE_TEXT':SEL/'V2_S1_JINA_FULL_VISIBLE_TEXT_RAW.jsonl','S2_JINA_BUDGETED_VISIBLE_TEXT':SEL/'V2_S2_JINA_BUDGETED_VISIBLE_TEXT_RAW.jsonl'}
    else:
        files={'R0_RETRIEVAL_TOP5':SEL/'V2_R0_RETRIEVAL_TOP5.jsonl','P1_PPLX_BUDGETED':SEL/'V2_P1_PPLX_BUDGETED_RAW.jsonl','S2_JINA_BUDGETED':SEL/'V2_S2_JINA_BUDGETED_RAW.jsonl'}
    out={}
    for pol,p in files.items():
        rows=load_jsonl(p); out[pol]=rows
        if len(rows)!=300 or any(r.get('status')!='ok' for r in rows): raise RuntimeError(f'{pol} lacks 300 valid rows')
    return out,files
def build_tasks():
    selections,files=load_selection_rows()
    needed={p:{r['question_id']:r['selected_ids'] for r in rows} for p,rows in selections.items()}
    manifest=json.loads((OUT/'01_admission/V1_COHORT_MANIFEST.json').read_text())
    qmeta={r['question_id']:r for r in manifest['representative']+manifest['oversized_diagnostic']}
    wanted=set(qmeta)
    content={}
    for line in CONTENT_PATH.open(encoding='utf-8'):
        row=json.loads(line); qid=row['question_id']
        if qid in wanted:
            by={d['candidate_id']:d for d in row['candidates']}
            content[qid]={'row':row,'by':by}
    if set(content)!=wanted: raise RuntimeError(f'missing content rows {len(wanted-set(content))}')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/'+READER_REV,local_files_only=True,trust_remote_code=True)
    tasks=[]
    view_cache={}
    for pol in POLICIES:
        for qid in sorted(wanted):
            sids=needed[pol][qid]; row=content[qid]['row']; by=content[qid]['by']
            pieces=[]; views=[]
            for i,sid in enumerate(sids):
                d=by[sid]
                if sid not in view_cache: view_cache[sid]=cap_text(tok,d['page_name'],d['content'])
                view,n,trunc=view_cache[sid]; pieces.append(f"[P{i} | {d['page_name']} | {sid}] {view}"); views.append({'candidate_id':sid,'reader_tokens':n,'truncated':trunc,'view_sha256':sha_bytes(view.encode())})
            prompt=("You must answer using only the supplied passages.\nIf the answer is not supported, output exactly: INSUFFICIENT_EVIDENCE.\nProvide one short answer, not reasoning. Do not use external knowledge.\n"+f"Question: {row['query']}\nPassage<redacted-path>"+'\n'.join(pieces)+'\nAnswer:')
            payload={'model':READER_MODEL,'messages':[{'role':'user','content':prompt}],'temperature':0.0,'top_p':1.0,'max_tokens':MAX_OUTPUT,'seed':0,'stream':False}
            body=json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode()
            tasks.append({'task_id':f'{pol}::{qid}','policy':pol,'question_id':qid,'cohort':row['cohort'],'domain':row['domain'],'selected_ids':sids,'candidate_ids':row['candidate_ids'],'prompt':prompt,'prompt_sha256':sha_bytes(prompt.encode()),'prompt_views':views,'payload':payload,'request_body':body,'request_body_sha256':sha_bytes(body),'gold_in_prompt':False})
    return tasks,files
def call_one(task):
    started=time.perf_counter_ns(); r={'task_id':task['task_id'],'policy':task['policy'],'question_id':task['question_id'],'cohort':task['cohort'],'domain':task['domain'],'selected_ids':task['selected_ids'],'candidate_ids':task['candidate_ids'],'reader_model':READER_MODEL,'reader_revision':READER_REV,'prompt_revision':PROMPT_REV,'prompt_sha256':task['prompt_sha256'],'actual_model_input_sha256':task['request_body_sha256'],'prompt_views':task['prompt_views'],'generation':{'temperature':0.0,'top_p':1.0,'max_tokens':MAX_OUTPUT,'seed':0,'stream':False},'cache_disabled':True,'cache_reuse':False,'cache_provenance':'no_application_cache; fresh_http_request','gold_in_reader_input':False,'reader_service_url':READER_URL,'status':'error','answer':'','raw_response_sha256':None,'raw_response_text':'','http_status':None,'input_tokens':None,'output_tokens':None,'duration_ms':None,'error_type':None,'attempts':0}
    for attempt in range(1,3):
        r['attempts']=attempt
        try:
            resp=requests.post(READER_URL,data=task['request_body'],headers=HEADERS,timeout=300)
            raw=resp.content; r['http_status']=resp.status_code; r['raw_response_sha256']=sha_bytes(raw); r['raw_response_text']=raw.decode('utf-8','replace'); data=resp.json(); resp.raise_for_status(); r['response_json']=data
            choice=(data.get('choices') or [{}])[0]; msg=choice.get('message') or {}; r['answer']=str(msg.get('content') or '').strip(); usage=data.get('usage') or {}; r['input_tokens']=usage.get('prompt_tokens'); r['output_tokens']=usage.get('completion_tokens'); r['status']='ok'; r['real_inference']=True; r['duration_ms']=(time.perf_counter_ns()-started)/1e6; return r
        except Exception as e: r['error_type']=f'{type(e).__name__}: {e}'
    r['duration_ms']=(time.perf_counter_ns()-started)/1e6; r['real_inference']=False; return r
def main():
    READER_OUT.mkdir(parents=True,exist_ok=True); ANALYSIS.mkdir(parents=True,exist_ok=True)
    tasks,files=build_tasks()
    freeze={'protocol':'E11-V V3 fixed-reader validation','reader_model':READER_MODEL,'reader_revision':READER_REV,'reader_service_url':READER_URL,'prompt_revision':PROMPT_REV,'token_cap_per_selected_page':TOKEN_CAP,'generation':{'temperature':0.0,'top_p':1.0,'max_tokens':MAX_OUTPUT,'seed':0,'stream':False},'policies':POLICIES,'cohort_n':300,'tasks':len(tasks),'selection_raw_hashes':{p:sha_file(path) for p,path in files.items()},'gold_in_reader_inputs':False,'application_cache_disabled':True,'service_restart':False}
    (READER_OUT/'V3_PRE_OUTCOME_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
    (READER_OUT/'V3_TASK_MANIFEST.jsonl').write_text('\n'.join(json.dumps({k:v for k,v in t.items() if k not in ('payload','request_body')},sort_keys=True) for t in tasks)+'\n')
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(call_one,tasks): results.append(r)
    rawpath=READER_OUT/'V3_READER_RAW_OUTPUTS.jsonl'
    with rawpath.open('w',encoding='utf-8') as f:
        for r in results: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
    (READER_OUT/'V3_READER_RUN_SUMMARY.json').write_text(json.dumps({'tasks':len(tasks),'ok':sum(r['status']=='ok' for r in results),'errors':sum(r['status']!='ok' for r in results),'raw_sha256':sha_file(rawpath),'service':READER_URL},indent=2)+'\n')
    # Offline post-outcome reference scoring. These are deterministic proxies.
    manifest=json.loads((OUT/'01_admission/V1_COHORT_MANIFEST.json').read_text()); gold={r['question_id']:([r.get('answer')]+list(r.get('alt_ans') or [])) for r in manifest['representative']+manifest['oversized_diagnostic']}
    scored=[]
    for r in results:
        refs=gold.get(r['question_id'],[]); scores=[f1(r.get('answer',''),ref) for ref in refs if str(ref).strip()]; best=max(scores,default=0.0); exact=any(' '.join(words(r.get('answer','')))==' '.join(words(ref)) for ref in refs if str(ref).strip())
        scored.append({'task_id':r['task_id'],'policy':r['policy'],'question_id':r['question_id'],'cohort':r['cohort'],'domain':r['domain'],'status':r['status'],'answer':r.get('answer',''),'reference_exact_proxy':exact,'reference_token_f1_proxy':best,'reference_count':len(refs),'real_inference':r.get('real_inference',False)})
    sp=ANALYSIS/'V4_REFERENCE_PROXY_SCORES.jsonl'; sp.write_text('\n'.join(json.dumps(x,sort_keys=True) for x in scored)+'\n')
    scorecard=[]
    for p in POLICIES:
        xs=[x for x in scored if x['policy']==p]; ok=[x for x in xs if x['status']=='ok']; scorecard.append({'policy':p,'n':len(xs),'reader_ok_n':len(ok),'reader_error_n':len(xs)-len(ok),'exact_proxy_rate_all':sum(x['reference_exact_proxy'] for x in xs)/len(xs),'exact_proxy_rate_ok':sum(x['reference_exact_proxy'] for x in ok)/len(ok) if ok else None,'token_f1_proxy_all':sum(x['reference_token_f1_proxy'] for x in xs)/len(xs),'token_f1_proxy_ok':sum(x['reference_token_f1_proxy'] for x in ok)/len(ok) if ok else None})
    (ANALYSIS/'V4_REFERENCE_PROXY_SCORECARD.json').write_text(json.dumps({'scorecard':scorecard,'endpoint':'deterministic reference agreement/token-F1 proxy; not factual correctness'},indent=2)+'\n')
    s1_note='The corrected visible-text S1 arm completed 300/300 valid native selections.' if REPAIR_DIR else 'S1 is unavailable because the original raw-content Jina selection contract failed its native feasibility probe.'
    (ANALYSIS/'V4_REFERENCE_PROXY_SCORECARD.md').write_text('# E11-V V4 Reference Proxy Scorecard\n\n'+json.dumps(scorecard,indent=2)+'\n\nThese deterministic answer/reference metrics are proxies and do not establish factual correctness, grounding, completeness, or citation validity. '+s1_note+'\n')
    print(json.dumps({'gate':'E11V_V3_READER_AND_V4_PROXY_COMPLETE','tasks':len(tasks),'reader_ok':sum(r['status']=='ok' for r in results),'reader_errors':sum(r['status']!='ok' for r in results),'scorecard':scorecard},indent=2))
if __name__=='__main__': main()
