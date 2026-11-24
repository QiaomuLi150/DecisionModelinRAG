#!/usr/bin/env python3
"""CPU-only C1 label join, metrics and R/Q/S summary."""
from __future__ import annotations
import csv,importlib.util,json,math,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; C1=ROOT/'results/v3_6_essential/followup_v2_correction/C1_transfer_confirmation'; RAW=C1/'raw'; MAN=ROOT/'results/v2_1_replication/stage0_preflight_v2_hybrid/INPUT_MANIFEST_V2_HYBRID.jsonl'; SEED=731; DRAWS=40000
def read(p):return [json.loads(x) for x in p.open(encoding='utf-8') if x.strip()]
def ndcg(sel,g):
 g=set(g)
 if not g:return 0.0
 d=sum((1.0 if x in g else 0.0)/math.log2(i+2) for i,x in enumerate(sel[:5]));z=sum(1.0/math.log2(i+2) for i in range(min(5,len(g))));return d/z if z else 0.0
def boot(d):
 d=np.asarray(d,float);rng=np.random.default_rng(SEED);idx=rng.integers(0,len(d),size=(DRAWS,len(d)));v=d[idx].mean(axis=1);return float(d.mean()),float(np.quantile(v,.025)),float(np.quantile(v,.975))
def main():
 spec=importlib.util.spec_from_file_location('ana',ROOT/'scripts/v2_1_replication_analysis.py');ana=importlib.util.module_from_spec(spec);spec.loader.exec_module(ana);m=ana.load_manifest(); labs=ana.load_labels(m)
 sl={ (r['dataset'],r['canonical_id']):r for r in read(MAN) if r.get('order_id')=='original'}
 raw=[]
 for p in sorted(RAW.glob('*.jsonl')):raw.extend(read(p))
 expected_lengths={2048,4096,6144,8192}
 if len(raw) not in {10668,21336}:raise RuntimeError(f'C1_RAW_COVERAGE:{len(raw)}')
 observed_lengths={int(x['budget_L']) for x in raw}
 if not observed_lengths.issubset(expected_lengths):raise RuntimeError(f'C1_LENGTH_SET:{sorted(observed_lengths)}')
 rec=[]
 for r in raw:
  lab=labs.get((r['dataset'],r['canonical_id']),{'gold':set(),'labeled':False,'status':'MISSING'});g=set(lab.get('gold',set()));ok=bool(r.get('status')=='PASS' and r.get('valid') and len(r.get('selected_ids',[]))==5); rec.append({'canonical_id':r['canonical_id'],'dataset':r['dataset'],'L':r['budget_L'],'policy':r['policy'],'valid':int(ok),'error_class':'' if ok else str(r.get('error_class') or 'UNKNOWN'),'metric':ndcg(r.get('selected_ids',[]) if ok else [],g) if g else '','gold_count':len(g),'reachable':int(bool(g and g.issubset(set(r.get('candidate_ids',[]))))),'input_sha256':r.get('input_sha256'),'source_slate_sha256':r.get('source_slate_sha256'),'n_calls':r.get('n_calls',0),'latency_ms':r.get('latency_ms'),'peak_memory_allocated_mib':r.get('peak_memory_allocated_mib')})
 C1.mkdir(parents=True,exist_ok=True);fields=list(rec[0]);
 with (C1/'c1_perquery.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rec)
 # BM25 and pairwise model summaries, query-weighted within each dataset.
 by=defaultdict(list)
 for r in rec:by[(r['dataset'],int(r['L']),r['policy'])].append(r)
 rows=[];contrast=[]
 for ds in ['enterprise','musique']:
  for L in sorted(observed_lengths):
   for pol in ['pplx_single_matched','jina_single_matched']:
    z=by[(ds,L,pol)];vals=[float(x['metric']) for x in z if x['metric']!=''];valid=sum(x['valid'] for x in z)/len(z);rows.append({'dataset':ds,'L':L,'policy':pol,'n':len(z),'valid_rate':valid,'ndcg5':float(np.mean(vals)) if vals else None,'gold_labeled_n':len(vals),'verdict_R':'PASS' if valid>=.99 else 'FAIL','verdict_Q':'REFERENCE' if pol.startswith('jina') else 'NOT_MEASURED','verdict_S':'NOT_MEASURED'})
   a={x['canonical_id']:x for x in by[(ds,L,'pplx_single_matched')]};b={x['canonical_id']:x for x in by[(ds,L,'jina_single_matched')]};d=[float(a[k]['metric'])-float(b[k]['metric']) for k in sorted(a) if a[k]['metric']!='' and b[k]['metric']!=''];c=boot(d) if d else (None,None,None);contrast.append({'dataset':ds,'L':L,'contrast':'pplx_single_matched-jina_single_matched','n':len(d),'delta':c[0],'lo95':c[1],'hi95':c[2],'verdict_Q':'PASS' if c[1]>-0.02 else 'FAIL' if c[0] is not None else 'NOT_MEASURED'})
 with (C1/'c1_summary.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (C1/'c1_contrasts.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(contrast[0]));w.writeheader();w.writerows(contrast)
 meta={'status':'PASS','raw_rows':len(raw),'perquery_rows':len(rec),'summary_rows':len(rows),'contrast_rows':len(contrast),'observed_lengths':sorted(observed_lengths),'labels_joined_after_raw_freeze':True,'seed':SEED,'draws':DRAWS,'source_manifest':str(MAN.relative_to(ROOT)),'note':'C1 S is NOT_MEASURED because this is a quality run; C1 Q compares PPLX/Jina only and does not claim equivalence.'};(C1/'C1_JOIN_SUMMARY.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
