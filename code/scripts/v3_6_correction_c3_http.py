#!/usr/bin/env python3
"""C3 HTTP rerun for Winnow or Drex, retaining first five HTTP 400 bodies."""
from __future__ import annotations
import argparse,csv,hashlib,json,time,urllib.error,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BRIGHT=ROOT/'results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl'
COHORT=ROOT/'results/v3_6_essential/stage0_preflight/cohort_manifest.csv'
def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def sha(x):return hashlib.sha256(canon(x).encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['winnow','drex'],required=True);ap.add_argument('--url',required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--revision',required=True);a=ap.parse_args()
 wanted={x['canonical_id'] for x in csv.DictReader(COHORT.open())}; sl=[json.loads(x) for x in BRIGHT.open() if x.strip()]; sl=[x for x in sl if f"{x['domain']}:{x['query_id']}" in wanted];sl.sort(key=lambda x:f"{x['domain']}:{x['query_id']}"); assert len(sl)==360
 out=ROOT/a.out_dir; (out/'raw').mkdir(parents=True,exist_ok=True); path=out/'raw'/'serving_perrequest_raw.jsonl'; bodylog=out/'HTTP_400_BODIES.jsonl'; rows=[]; four=[]
 with path.open('w',encoding='utf-8') as f,bodylog.open('w',encoding='utf-8') as bf:
  for attempt in (1,2,3):
   for v in sl:
    cid=f"{v['domain']}:{v['query_id']}"; ids=[str(x['candidate_id']) for x in v['candidates']]; req={'model':'Winnow-12B' if a.model=='winnow' else 'drex-v2.0','state':{'query':str(v['query'])},'questions':{'candidate':{'type':'choice','instructions':'Which candidate passage is most relevant to the query?','criteria':{str(x['candidate_id']):str(x['content']) for x in v['candidates']}}}}
    t=time.perf_counter(); status='FAIL';err=None;http=None;raw=b'';scores=[];selected=[]
    try:
     q=urllib.request.Request(a.url,data=canon(req).encode(),headers={'Content-Type':'application/json'})
     with urllib.request.urlopen(q,timeout=900) as resp:http=resp.status;raw=resp.read()
     obj=json.loads(raw); probs=(obj.get('answers') or {}).get('candidate',{}).get('probabilities',{}); scores=[float(probs[x]) for x in ids]; selected=[ids[i] for i in sorted(range(len(ids)),key=lambda i:(-scores[i],i,ids[i]))[:5]]; status='PASS'
    except urllib.error.HTTPError as e:
     http=e.code;raw=e.read();err=f'HTTPError: {e}';
     if e.code==400 and len(four)<5:
      z={'model':a.model,'attempt':attempt,'canonical_id':cid,'http_status':e.code,'body':raw.decode('utf-8','replace')};four.append(z);bf.write(json.dumps(z,ensure_ascii=False,sort_keys=True)+'\n');bf.flush()
    except Exception as e:err=f'{type(e).__name__}: {e}'
    row={'model':a.model,'canonical_id':cid,'domain':v['domain'],'attempt':attempt,'status':status,'error_class':err,'http_status':http,'latency_ms':round((time.perf_counter()-t)*1000,3),'candidate_ids':ids,'selected_ids':selected,'scores':scores,'native_request_sha256':sha(req),'raw_response_sha256':hashlib.sha256(raw).hexdigest() if raw else None,'model_revision':a.revision,'labels_joined':False,'input_label_blind':True,'context_settings':'original 8K text-only Q8 direct; server launch receipt recorded separately'};rows.append(row);f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n');f.flush()
   print(json.dumps({'model':a.model,'attempt':attempt,'records':len(rows),'valid':sum(x['status']=='PASS' for x in rows),'failed':sum(x['status']!='PASS' for x in rows)}),flush=True)
 (out/'TRACE_META.json').write_text(json.dumps({'status':'PASS','model':a.model,'records':len(rows),'expected_records':1080,'valid':sum(x['status']=='PASS' for x in rows),'failed':sum(x['status']!='PASS' for x in rows),'http_400_bodies':len(four),'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'labels_joined':False},indent=2)+'\n');print(json.dumps({'status':'PASS','records':len(rows),'valid':sum(x['status']=='PASS' for x in rows),'http400':len(four)},indent=2))
if __name__=='__main__':main()
