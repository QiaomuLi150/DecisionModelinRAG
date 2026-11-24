#!/usr/bin/env python3
"""Offline R3 score join and domain-bootstrap analysis."""
from __future__ import annotations
import csv, hashlib, json, math, itertools
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; E9=ROOT/"results/non_jev/v2/e9_bright_v1_0"; V=ROOT/"results/v1_3/r3_multipass"; SEED=731; DRAWS=40000
def sha(x): return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def ndcg(ids,gold):
    n=min(5,len(gold))
    if not n:return 0.0
    dcg=sum((1.0 if x in gold else 0.0)/math.log2(i+2) for i,x in enumerate(ids[:5]))
    return dcg/sum(1.0/math.log2(i+2) for i in range(n))
def load(p): return [json.loads(l) for l in p.open(encoding='utf-8') if l.strip()]
def file_sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def historical(rel,policy):
    p=ROOT/rel; out=[]
    for x in load(p):out.append({'canonical_id':f"{x['domain']}:{x['query_id']}",'domain':x['domain'],'query_id':str(x['query_id']),'policy':policy,'status':x.get('status'),'selected_ids':x.get('selected_ids',[]),'elapsed_ms':x.get('duration_ms')})
    return out
def main():
    man=json.loads((E9/'00_admission/BRIGHT_EXAMPLE_ID_MANIFEST.json').read_text())
    gold={f"{x['domain']}:{x['id']}":set(map(str,x.get('gold_ids',[]))) for x in man['items']}
    shards=sorted(V.glob('multipass_raw_start*_n*.jsonl')); rows=[x for p in shards for x in load(p)]
    if len(rows)!=1384: raise RuntimeError(f'R3 shard rows {len(rows)} != 1384')
    if len({x['canonical_id'] for x in rows})!=1384: raise RuntimeError('R3 duplicate or missing canonical IDs')
    out=[]
    for x in rows:
        g=gold[x['canonical_id']]; ok=x.get('status')=='PASS' and len(x.get('selected_ids',[]))==5 and len(set(x.get('selected_ids',[])))==5
        out.append({'canonical_id':x['canonical_id'],'domain':x['domain'],'policy':'pplx_two_stage','valid':str(ok).lower(),'status':x.get('status'),'ndcg5':ndcg(x.get('selected_ids',[]) if ok else [],g) if ok else None,'n_calls':x.get('n_calls'),'total_ms':x.get('total_ms'),'selected_ids':x.get('selected_ids',[]),'selected_ids_hash':sha(x.get('selected_ids',[])) if ok else ''})
    base=historical('results/non_jev/v2/e11_v1_0/02_native/E11_A_PPLX_BUDGETED_RAW.jsonl','pplx_budget')
    jina=historical('results/non_jev/v2/e9_bright_v1_0/02_native/BRIGHT_NATIVE_RAW_OUTPUTS_jina_v3_5.jsonl','jina_full')
    metrics=[]
    for name,xs in [('pplx_two_stage',out),('pplx_budget',base),('jina_full',jina)]:
        vals=[]
        for x in xs:
            g=gold.get(x['canonical_id'],set()); ok=x.get('status') in ('PASS','ok') and len(x.get('selected_ids',[]))==5 and len(set(x.get('selected_ids',[])))==5
            if ok: vals.append(ndcg(x['selected_ids'],g))
        metrics.append({'policy':name,'rows':len(xs),'valid':len(vals),'failed':len(xs)-len(vals),'mean_ndcg5_valid':float(np.mean(vals)) if vals else None,'all_attempt_mean_ndcg5':float(np.mean([ndcg(x.get('selected_ids',[]),gold[x['canonical_id']]) if x.get('status') in ('PASS','ok') else 0.0 for x in xs])) if xs else None})
    V.joinpath('export').mkdir(parents=True,exist_ok=True)
    with (V/'export/multipass_perquery_detailed.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    with (V/'export/multipass_perquery.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['canonical_id','domain','policy','metric','valid','n_calls','total_ms']); w.writeheader()
        for x in out:w.writerow({'canonical_id':x['canonical_id'],'domain':x['domain'],'policy':x['policy'],'metric':(x['ndcg5'] if x['valid']=='true' else 0.0),'valid':x['valid'],'n_calls':x['n_calls'],'total_ms':x['total_ms']})
    with (V/'export/multipass_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0])); w.writeheader(); w.writerows(metrics)
    # Query-weighted and equal-domain bootstrap for multipass minus historical controls.
    by={x['canonical_id']:x for x in out}; boot={}
    for name,ref in [('pplx_budget',base),('jina_full',jina)]:
        refmap={x['canonical_id']:x for x in ref}; grouped=defaultdict(list)
        for cid,x in by.items():
            y=refmap.get(cid); g=gold[cid]
            if not y:continue
            ok=x['valid']=='true' and y.get('status') in ('PASS','ok') and len(y.get('selected_ids',[]))==5
            d=(ndcg(x.get('selected_ids',[]),g) if x['valid']=='true' else 0.0)-(ndcg(y.get('selected_ids',[]),g) if y.get('status') in ('PASS','ok') else 0.0)
            grouped[cid.split(':',1)[0]].append(d)
        ds=sorted(grouped); means=np.array([np.mean(grouped[d]) for d in ds]); rng=np.random.Generator(np.random.PCG64(SEED)); idx=rng.integers(0,len(ds),size=(DRAWS,len(ds))); draws=np.mean(means[idx],axis=1)
        signs=[float(np.mean(np.asarray(s)*means)) for s in itertools.product((-1.0,1.0),repeat=len(ds))]
        boot[name]={'n':sum(len(x) for x in grouped.values()),'query_weighted':float(np.mean([z for d in ds for z in grouped[d]])),'domain_weighted':float(np.mean(means)),'ci95_whole_domain_bootstrap':[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],'exact_2_to_12_sign_flip_p_two_sided':float(np.mean(np.abs(signs)>=abs(float(np.mean(means)))-1e-15)),'seed':SEED,'draws':DRAWS,'domains':{d:{'n':len(grouped[d]),'effect':float(np.mean(grouped[d]))} for d in ds}}
    (V/'export/multipass_bootstrap.json').write_text(json.dumps(boot,indent=2)+'\n')
    (V/'export/multipass_provenance.json').write_text(json.dumps({'shards':[{"path":str(p.relative_to(ROOT)),"sha256":file_sha(p),"rows":len(load(p))} for p in shards],'labels_joined_offline':True},indent=2)+'\n')
    print(json.dumps({'shards':len(shards),'rows':len(rows),'metrics':metrics,'boot':boot},indent=2))
if __name__=='__main__':main()
