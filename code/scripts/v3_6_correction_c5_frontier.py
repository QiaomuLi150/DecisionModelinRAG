#!/usr/bin/env python3
"""Rebuild C5 frontier_v2 from frozen quality and N1 serving exports only."""
from __future__ import annotations
import csv, json, math, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v3_6_essential/followup_v2_correction/C5_frontier_v2'
Q=ROOT/'results/v3_6_essential/followup_v1/export/quality_v2'
R=ROOT/'results/v3_6_essential/followup_v1/export/serving_full/serving_ratios_pooled.csv'
BRIGHT=ROOT/'results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl'
GOLD=ROOT/'results/non_jev/v2/e9_bright_v1_0/00_admission/BRIGHT_EXAMPLE_ID_MANIFEST.json'
SEED=731; DRAWS=40000

def rows(p):
    with p.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
def ndcg(sel,gold):
    g=set(gold)
    if not g:return 0.0
    d=sum((1.0 if x in g else 0.0)/math.log2(i+2) for i,x in enumerate(sel[:5]))
    z=sum(1.0/math.log2(i+2) for i in range(min(5,len(g))))
    return d/z if z else 0.0
def ci_delta(a,b):
    x=np.asarray(a,float)-np.asarray(b,float); rng=np.random.default_rng(SEED)
    idx=rng.integers(0,len(x),size=(DRAWS,len(x))); v=x[idx].mean(axis=1)
    return float(x.mean()),float(np.quantile(v,.025)),float(np.quantile(v,.975))
def main():
    q=rows(Q/'q_perquery_v2.csv'); s=rows(Q/'q_summary_v2.csv'); sr=rows(R)
    by={(x['policy'],x['budget_L']):x for x in s}
    pp=defaultdict(dict)
    for x in q:
        pp[(x['policy'],x['budget_L'])][x['canonical_id']]=x
    sl=[json.loads(x) for x in BRIGHT.open() if x.strip()]
    gm=json.loads(GOLD.read_text())['items']; gold={(str(x['domain']),str(x['id'])):set(map(str,x.get('gold_ids',[]))) for x in gm}
    bm={f"{x['domain']}:{x['query_id']}":ndcg([str(y) for y in x['candidate_ids'][:5]],gold.get((str(x['domain']),str(x['query_id'])),set())) for x in sl}
    ratio={}
    for x in sr:
        if x['basis']=='all_request': ratio[x['policy']]=x
    fields=['policy','budget_L','valid_rate_strict','valid_rate_lenient','ndcg5','delta_vs_jina','delta_lo','delta_hi','p95_ms','p95_ratio','ratio_lo','ratio_hi','verdict_R','verdict_Q','verdict_S']
    out=[]
    specs=[('jina_full',''),('jina_single','2048'),('jina_single','4096'),('jina_single','6144'),('jina_single','8192'),('jina_two_stage','2048'),('jina_two_stage','4096'),('jina_two_stage','8192'),('pplx_single','2048'),('pplx_single','4096'),('pplx_single','6144'),('pplx_single','8192'),('pplx_two_stage','2048'),('pplx_two_stage','4096'),('pplx_two_stage','8192')]
    for pol,L in specs:
        x=by.get((pol,L))
        if x is None: continue
        ids=pp[(pol,L)]
        valid=float(x['model_valid_pct']); lenient=float(sum(1 for z in ids.values() if z.get('ndcg5_bm25_fallback','')!='')/len(ids))
        nd=float(x['mean_ndcg5_nofallback'])
        delta=lo=hi='NOT_MEASURED'; qv='NOT_MEASURED'
        # PPLX-vs-Jina same budget is the frozen declared quality contrast.
        if pol.startswith('pplx'):
            jpol='jina_single' if pol=='pplx_single' else 'jina_two_stage'
            pair=[z for z in rows(Q/'q_contrasts_v2.csv') if z['contrast']==f'{pol}_{L}-{jpol}_{L}' and z['basis']=='strict']
            if pair:
                z=pair[0]; delta=z['delta_query_weighted']; lo=z['lo95']; hi=z['hi95']
            a=[float(ids[k]['ndcg5']) for k in sorted(ids) if k in pp.get((jpol,L),{})]; b=[float(pp[(jpol,L)][k]['ndcg5']) for k in sorted(ids) if k in pp.get((jpol,L),{})]
            qj=ci_delta(a,b) if len(a)==len(b) and a else None
            # BM25 comparison uses the same query IDs and the same strict metric.
            ab=[float(ids[k]['ndcg5'])-bm[k] for k in sorted(ids) if k in bm]
            qb=(float(np.mean(ab)),)+tuple(float(x) for x in (np.quantile(np.random.default_rng(SEED).choice(np.asarray(ab),size=(DRAWS,len(ab)),replace=True).mean(axis=1),[.025,.975]))) if ab else None
            if qj and qb and float(qj[1])>-0.02 and qb[1]>0: qv='PASS'
            elif qj and qb: qv='FAIL'
        elif pol=='jina_full':
            delta=0.0; lo=0.0; hi=0.0; qv='REFERENCE'
        rr=ratio.get(pol)
        p50=rr['p50_ms'] if rr else 'NOT_MEASURED'; p95=rr['p95_ms'] if rr else 'NOT_MEASURED'; pr=rr['p95_ratio_point'] if rr else 'NOT_MEASURED'; rlo=rr['bootstrap_lo95'] if rr else 'NOT_MEASURED'; rhi=rr['bootstrap_hi95'] if rr else 'NOT_MEASURED'
        Rv='PASS' if valid>=.99 else 'FAIL'
        if isinstance(pr,str): Sv='NOT_MEASURED'
        else: Sv='PASS' if float(rhi)<=2 else ('FAIL' if float(rlo)>2 else 'INCONCLUSIVE')
        out.append({'policy':pol,'budget_L':L or 'FULL','valid_rate_strict':valid,'valid_rate_lenient':lenient,'ndcg5':nd,'delta_vs_jina':delta,'delta_lo':lo,'delta_hi':hi,'p95_ms':p95,'p95_ratio':pr,'ratio_lo':rlo,'ratio_hi':rhi,'verdict_R':Rv,'verdict_Q':qv,'verdict_S':Sv})
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'frontier_v2.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(out)
    receipt={'status':'PASS','source_quality':str((Q/'q_perquery_v2.csv').relative_to(ROOT)),'source_summary':str((Q/'q_summary_v2.csv').relative_to(ROOT)),'source_serving':str(R.relative_to(ROOT)),'bm25_source':str(BRIGHT.relative_to(ROOT)),'rows':len(out),'draws':DRAWS,'seed':SEED,'notes':'PPLX Q verdicts use same-budget paired contrast and independently recomputed BM25 bootstrap; lenient means an offline fallback score is available, not model validity.'}
    (OUT/'FRONTIER_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':'PASS','rows':len(out),'path':str((OUT/'frontier_v2.csv').relative_to(ROOT))},indent=2))
if __name__=='__main__': main()
