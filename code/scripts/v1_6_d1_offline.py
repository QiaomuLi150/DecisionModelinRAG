#!/usr/bin/env python3
"""D1 offline label join, metric computation, statistics and exports."""
from __future__ import annotations
import csv, hashlib, itertools, json, math
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/v1_6/d1_perwindow_jina"; RAW=OUT/"raw"; EXPORT=OUT/"export"
BRIGHT=ROOT/"results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
GOLD=ROOT/"results/non_jev/v2/e9_bright_v1_0/00_admission/BRIGHT_EXAMPLE_ID_MANIFEST.json"
ORDERS=["original","reversed","perm_seed_1","perm_seed_2","perm_seed_3","perm_seed_4","perm_seed_5"]
SEED=731; DRAWS=40_000
SHARDS=[
 "d1_jina_two_stage_pplxwindow__original_reversed.jsonl",
 "d1_jina_two_stage_pplxwindow__perm_seed_1_perm_seed_2.jsonl",
 "d1_jina_two_stage_pplxwindow__perm_seed_3_perm_seed_4.jsonl",
 "d1_jina_two_stage_pplxwindow__perm_seed_5.jsonl",
]

def sha_file(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def read(p:Path): return [json.loads(x) for x in p.open(encoding='utf-8') if x.strip()]
def valid(r):
 ids=[str(x) for x in r.get('selected_ids',[])]
 return r.get('status')=='PASS' and bool(r.get('valid')) and len(ids)==5 and len(set(ids))==5
def ndcg5(sel,gold):
 if not gold:return 0.0
 dcg=sum((1.0 if x in gold else 0.0)/math.log2(i+2) for i,x in enumerate(sel[:5]))
 ideal=sum(1.0/math.log2(i+2) for i in range(min(5,len(gold))))
 return dcg/ideal if ideal else 0.0
def bootstrap(domain_deltas):
 domains=sorted(domain_deltas); dm=np.asarray([np.mean(domain_deltas[d]) for d in domains],dtype=float)
 rng=np.random.Generator(np.random.PCG64(SEED)); draws=dm[rng.integers(0,len(domains),size=(DRAWS,len(domains)))].mean(axis=1)
 return dm,draws
def signflip(dm):
 vals=np.asarray([np.mean(dm*np.asarray(s)) for s in itertools.product((-1.0,1.0),repeat=len(dm))])
 return float(np.mean(np.abs(vals)>=abs(float(np.mean(dm)))-1e-15))

def main():
 EXPORT.mkdir(parents=True,exist_ok=True)
 slates=read(BRIGHT); slate={f"{x['domain']}:{x['query_id']}":x for x in slates}
 if len(slate)!=1384 or sha_file(BRIGHT)!="c304b2c35f445153181098d3a899271db6fc653b44aea50186c6efaeec8e6ea2": raise RuntimeError('BRIGHT integrity failure')
 gm=json.loads(GOLD.read_text(encoding='utf-8')); gold={f"{x['domain']}:{x['id']}":set(map(str,x.get('gold_ids',[]))) for x in gm['items']}
 if set(gold)!=set(slate): raise RuntimeError('gold population mismatch')
 shards=[]; rows=[]; seen=set()
 for name in SHARDS:
  p=RAW/name; m=p.with_suffix('.meta.json')
  if not p.exists(): raise RuntimeError(f'missing shard {name}')
  rs=read(p)
  if len(rs)!=1384*len(set(r['order_id'] for r in rs)): raise RuntimeError(f'bad shard coverage {name}')
  for r in rs:
   key=(r['canonical_id'],r['order_id'])
   if key in seen or key[0] not in slate or r.get('policy')!='jina_two_stage_pplxwindow' or r.get('labels_joined') is not False: raise RuntimeError(f'raw identity/label failure {name} {key}')
   if r['order_id'] not in ORDERS or not set(r.get('selected_ids',[])).issubset({str(x['candidate_id']) for x in slate[key[0]]['candidates']}):
    if valid(r): raise RuntimeError(f'selected-ID failure {name} {key}')
   seen.add(key); rows.append(r)
  shards.append({'path':str(p.relative_to(ROOT)),'sha256':sha_file(p),'records':len(rs),'valid':sum(valid(r) for r in rs),'failed':sum(not valid(r) for r in rs),'orders':sorted(set(r['order_id'] for r in rs))})
 if len(rows)!=9688 or len(seen)!=9688: raise RuntimeError(f'D1 total coverage failure {len(rows)}/{len(seen)}')
 unified=RAW/'d1_jina_two_stage_pplxwindow_all.jsonl'; unified.write_text(''.join((json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n') for r in sorted(rows,key=lambda r:(r['canonical_id'],ORDERS.index(r['order_id'])))),encoding='utf-8')
 # Authenticated historical PPLX anchors, still label blind in their original traces.
 ppx=[]
 for p in sorted((ROOT/'results/v1_3/r3_multipass').glob('multipass_raw_start*_n*.jsonl')): ppx += read(p)
 pplx_orig={r['canonical_id']:r for r in ppx}; ppx_shuf={o:{} for o in ORDERS[1:]}
 for p in sorted((ROOT/'results/v1_3/r3_multipass_ordered/raw').glob('*.jsonl')):
  for r in read(p): ppx_shuf.setdefault(r['order_id'],{})[r['canonical_id']]=r
 if len(pplx_orig)!=1384 or any(len(ppx_shuf[o])!=1384 for o in ORDERS[1:]): raise RuntimeError('PPLX anchor coverage failure')
 def score(cid,r): return ndcg5([str(x) for x in r.get('selected_ids',[])],gold[cid]) if r.get('status')=='PASS' and len(set(map(str,r.get('selected_ids',[]))))==5 else 0.0
 jmap={(r['canonical_id'],r['order_id']):r for r in rows}; jscore={(cid,o):score(cid,jmap[(cid,o)]) for cid in slate for o in ORDERS}
 pscore={(cid,'original'):score(cid,pplx_orig[cid]) for cid in slate}; pscore.update({(cid,o):score(cid,ppx_shuf[o][cid]) for cid in slate for o in ORDERS[1:]})
 # Required per-query export includes both sides of each paired contrast.
 with (EXPORT/'d1_perquery.csv').open('w',newline='',encoding='utf-8') as f:
  fields=['canonical_id','domain','policy','order_id','valid','metric','survivor_set_matches_pplx']; w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for cid in sorted(slate):
   for o in ORDERS:
    r=jmap[(cid,o)];w.writerow({'canonical_id':cid,'domain':slate[cid]['domain'],'policy':'jina_two_stage_pplxwindow','order_id':o,'valid':str(valid(r)).lower(),'metric':jscore[(cid,o)],'survivor_set_matches_pplx':str(bool(r.get('survivor_set_matches_pplx'))).lower() if valid(r) else 'false'})
    pr=pplx_orig[cid] if o=='original' else ppx_shuf[o][cid]
    w.writerow({'canonical_id':cid,'domain':slate[cid]['domain'],'policy':'pplx_two_stage','order_id':o,'valid':str(pr.get('status')=='PASS' and len(set(map(str,pr.get('selected_ids',[]))))==5).lower(),'metric':pscore[(cid,o)],'survivor_set_matches_pplx':'true' if pr.get('status')=='PASS' else 'false'})
 with (EXPORT/'d1_domain_summary.csv').open('w',newline='',encoding='utf-8') as f:
  fields=['policy','order_id','domain','n','valid','mean_ndcg5'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for policy in ['pplx_two_stage','jina_two_stage_pplxwindow']:
   for o in ORDERS:
    for d in sorted({x['domain'] for x in slates}):
     cids=[cid for cid in slate if slate[cid]['domain']==d]; vals=[(pscore if policy=='pplx_two_stage' else jscore)[(cid,o)] for cid in cids]; valids=sum((pplx_orig[cid] if o=='original' else ppx_shuf[o][cid]).get('status')=='PASS' if policy=='pplx_two_stage' else valid(jmap[(cid,o)]) for cid in cids)
     w.writerow({'policy':policy,'order_id':o,'domain':d,'n':len(cids),'valid':valids,'mean_ndcg5':float(np.mean(vals))})
 # Primary contrasts and exact 2^12 sign-flip p values.
 contrasts=[]
 def add(name,left,right):
  delta={cid:left(cid)-right(cid) for cid in slate}; groups=defaultdict(list)
  for cid,x in delta.items():groups[slate[cid]['domain']].append(x)
  dm,draws=bootstrap(groups);p=signflip(dm); contrasts.append({'contrast':name,'delta_query_weighted':float(np.mean(list(delta.values()))),'lo95':float(np.quantile(draws,.025)),'hi95':float(np.quantile(draws,.975)),'lo90':float(np.quantile(draws,.05)),'hi90':float(np.quantile(draws,.95)),'p_signflip':p,'domains_positive':int(sum(x>0 for x in dm)),'n_queries':len(delta),'n_domains':len(dm),'domain_effects':{d:float(np.mean(v)) for d,v in sorted(groups.items())}})
 add('pplx_two_stage_minus_jina_two_stage_pplxwindow__original',lambda c:pscore[(c,'original')],lambda c:jscore[(c,'original')])
 add('pplx_two_stage_minus_jina_two_stage_pplxwindow__six_shuffled_mean',lambda c:float(np.mean([pscore[(c,o)] for o in ORDERS[1:]])),lambda c:float(np.mean([jscore[(c,o)] for o in ORDERS[1:]])))
 raw=[x['p_signflip'] for x in contrasts]; order=sorted(range(2),key=lambda i:raw[i]); running=0; holm=[0,0]
 for rank,i in enumerate(order): running=max(running,min(1,raw[i]*(2-rank)));holm[i]=running
 for x,p in zip(contrasts,holm):x['p_holm']=p
 with (EXPORT/'d1_contrasts.csv').open('w',newline='',encoding='utf-8') as f:
  fields=['contrast','delta_query_weighted','lo95','hi95','lo90','hi90','p_signflip','p_holm','domains_positive'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:x[k] for k in fields} for x in contrasts])
 (RAW/'d1_jina_two_stage_pplxwindow_all.meta.json').write_text(json.dumps({'path':str(unified.relative_to(ROOT)),'sha256':sha_file(unified),'records':len(rows),'valid':sum(valid(r) for r in rows),'failed':sum(not valid(r) for r in rows),'labels_joined':False},indent=2,sort_keys=True)+'\n')
 trace_manifest={'status':'COMPLETE_RAW_TRACE_MANIFEST','traces':shards+[{'path':str(unified.relative_to(ROOT)),'sha256':sha_file(unified),'records':len(rows)}]+[{'path':str(p.relative_to(ROOT)),'sha256':sha_file(p),'records':len(read(p))} for p in sorted((ROOT/'results/v1_3/r3_multipass').glob('multipass_raw_start*_n*.jsonl'))]+[{'path':str(p.relative_to(ROOT)),'sha256':sha_file(p),'records':len(read(p))} for p in sorted((ROOT/'results/v1_3/r3_multipass_ordered/raw').glob('*.jsonl'))]}
 (OUT/'D1_RAW_TRACE_SHA256_MANIFEST.json').write_text(json.dumps(trace_manifest,indent=2,sort_keys=True)+'\n')
 summary={'status':'COMPLETE_OFFLINE_ANALYSIS','d1_records':len(rows),'d1_valid':sum(valid(r) for r in rows),'d1_failed':sum(not valid(r) for r in rows),'pplx_original_valid':sum(pplx_orig[c].get('status')=='PASS' for c in slate),'survivor_set_matches':{o:{'n_valid':sum(valid(jmap[(c,o)]) for c in slate),'matches':sum(bool(jmap[(c,o)].get('survivor_set_matches_pplx')) for c in slate if valid(jmap[(c,o)]))} for o in ORDERS},'contrasts':contrasts,'input_manifest_sha256':sha_file(OUT/'D1_INPUT_MANIFEST.jsonl'),'raw_trace_manifest_sha256':sha_file(OUT/'D1_RAW_TRACE_SHA256_MANIFEST.json'),'labels_joined_after_raw_freeze':True}
 (OUT/'D1_OFFLINE_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':main()
