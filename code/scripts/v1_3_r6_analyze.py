#!/usr/bin/env python3
"""Offline LeetCode R6 score join; labels are imported only in this step."""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
E9=ROOT/"results/non_jev/v2/e9_bright_v1_0"
V=ROOT/"results/v1_3/r6_leetcode_trim"
def sha(x): return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def ndcg(ids,gold):
    n=min(len(gold),5)
    if not n:return 0.0
    dcg=sum((1.0 if x in gold else 0.0)/math.log2(i+2) for i,x in enumerate(ids[:5]))
    return dcg/sum(1.0/math.log2(i+2) for i in range(n))
def main():
    man=json.loads((E9/"00_admission/BRIGHT_EXAMPLE_ID_MANIFEST.json").read_text())
    gold={f"{x['domain']}:{x['id']}":set(map(str,x.get('gold_ids',[]))) for x in man['items'] if x['domain']=='leetcode'}
    paths=list((V/'raw').glob('*r6_native_budgeted_raw.jsonl'))
    rows=[json.loads(l) for p in paths for l in p.open(encoding='utf-8') if l.strip()]
    out=[]; by=defaultdict(list)
    for x in rows:
        cid=x['canonical_id']; g=gold.get(cid,set()); ok=x.get('status')=='PASS' and len(x.get('selected_ids',[]))==5
        score=ndcg(x.get('selected_ids',[]),g) if ok and g else None
        y={k:x.get(k) for k in ['canonical_id','domain','query_id','model','view','policy','status','error','k','elapsed_ms','peak_memory_allocated_mib','input_tokens','physical_forward_count']}
        y.update({'gold_count':len(g),'ndcg5':score,'selected_ids_hash':sha(x.get('selected_ids',[])) if ok else ''})
        out.append(y); by[(cid,x['policy'])].append(x)
    V.joinpath('export').mkdir(parents=True,exist_ok=True)
    fields=list(out[0].keys()) if out else []
    with (V/'export/leetcode_sub_perquery_detailed.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(out)
    with (V/'export/leetcode_sub_perquery.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['canonical_id','model','view','k','metric','valid']); w.writeheader()
        for x in out:w.writerow({'canonical_id':x['canonical_id'],'model':x['model'],'view':x['view'],'k':x['k'],'metric':(x['ndcg5'] if x['ndcg5'] is not None else 0.0),'valid':str(x['status']=='PASS').lower()})
    summaries=[]
    for pol in sorted({x['policy'] for x in rows}):
        ys=[x['ndcg5'] for x in out if x['policy']==pol and x['ndcg5'] is not None]
        summaries.append({'policy':pol,'rows':sum(x['policy']==pol for x in out),'valid':sum(x['policy']==pol and x['status']=='PASS' for x in out),'scored':len(ys),'mean_ndcg5':float(np.mean(ys)) if ys else None,'failed':sum(x['policy']==pol and x['status']!='PASS' for x in out)})
    with (V/'export/leetcode_sub_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]) if summaries else ['policy']); w.writeheader(); w.writerows(summaries)
    paired=[]
    for model in ('kev9','drex_dlm'):
        for view in ('native_sub','budget_sub'):
            a=f'{model}_{view}'; b=f'jina_{"kev" if model=="kev9" else "drex"}_{view}'
            d=[]
            for cid in gold:
                xa=next((x for x in out if x['canonical_id']==cid and x['policy']==a and x['ndcg5'] is not None),None)
                xb=next((x for x in out if x['canonical_id']==cid and x['policy']==b and x['ndcg5'] is not None),None)
                if xa and xb:d.append(float(xa['ndcg5'])-float(xb['ndcg5']))
            paired.append({'contrast':a+' - '+b,'n':len(d),'mean_delta':float(np.mean(d)) if d else None,'positive':sum(x>0 for x in d),'negative':sum(x<0 for x in d),'unchanged':sum(x==0 for x in d)})
    (V/'export/leetcode_sub_paired.json').write_text(json.dumps(paired,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'gold_queries':len(gold),'summaries':summaries,'paired':paired},indent=2))
if __name__=='__main__':main()
