#!/usr/bin/env python3
"""Offline comparison of the frozen/corrected and two fresh C2 Jina runs."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v3_6_essential/followup_v2_correction/C2_jina_diagnostic'
BRIGHT=ROOT/'results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl'
GOLD=ROOT/'results/non_jev/v2/e9_bright_v1_0/00_admission/BRIGHT_EXAMPLE_ID_MANIFEST.json'
FROZEN=ROOT/'results/v3_6_essential/stage1_quality/raw/jina_single_8192__8192_reuse.jsonl'
CORR=ROOT/'results/v3_6_essential/followup_v1/stage1_jina/raw/jina_rerun_combined.jsonl'
def read(p): return [json.loads(x) for x in p.open(encoding='utf-8') if x.strip()]
def ndcg(sel,g):
 g=set(g)
 if not g:return 0.0
 d=sum((1.0 if x in g else 0.0)/math.log2(i+2) for i,x in enumerate(sel[:5])); z=sum(1.0/math.log2(i+2) for i in range(min(5,len(g))))
 return d/z if z else 0.0
def main():
 sl={f"{x['domain']}:{x['query_id']}":x for x in read(BRIGHT)}
 gold={(str(x['domain']),str(x['id'])):set(map(str,x.get('gold_ids',[]))) for x in json.loads(GOLD.read_text())['items']}
 runs=[]
 for name in ('run1','run2'):
  runs.append((name,{x['canonical_id']:x for x in read(OUT/f'raw/jina_fp32_batch1_{name}.jsonl')}))
 fr={x['canonical_id']:x for x in read(FROZEN)}; cr={x['canonical_id']:x for x in read(CORR) if x.get('policy')=='jina_single_8192'}
 pairs=[]; diffs=[]
 for cid in sorted(sl):
  x=sl[cid]; g=gold.get((str(x['domain']),str(x['query_id'])),set())
  a=runs[0][1].get(cid,{}); b=runs[1][1].get(cid,{})
  valid_a=a.get('status')=='PASS'; valid_b=b.get('status')=='PASS'; la=a.get('selected_ids') or []; lb=b.get('selected_ids') or []
  pairs.append({'canonical_id':cid,'domain':x['domain'],'run1_valid':int(valid_a),'run2_valid':int(valid_b),'list_equal':int(la==lb),'set_equal':int(set(la)==set(lb)),'ndcg5_run1':ndcg(la,g) if valid_a else 0.0,'ndcg5_run2':ndcg(lb,g) if valid_b else 0.0})
  if la!=lb: diffs.append({'canonical_id':cid,'run1_selected_ids':la,'run2_selected_ids':lb,'run1_scores':a.get('scores',[]),'run2_scores':b.get('scores',[])})
 def pop(name, rows):
  return {'name':name,'n':len(rows),'valid':sum(r.get(f'{name}_valid',0) for r in rows) if name in ('run1','run2') else None,'list_equal_n':sum(r['list_equal'] for r in rows),'set_equal_n':sum(r['set_equal'] for r in rows),'list_equal_rate':sum(r['list_equal'] for r in rows)/len(rows),'set_equal_rate':sum(r['set_equal'] for r in rows)/len(rows)}
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'fresh_runs_perquery.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
 (OUT/'fresh_run_first20_differences.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in diffs[:20]),encoding='utf-8')
 valid_both=[r for r in pairs if r['run1_valid'] and r['run2_valid']]
 d=[r['ndcg5_run1']-r['ndcg5_run2'] for r in valid_both]
 summary={'status':'PASS','n_queries':len(pairs),'run1_valid':sum(r['run1_valid'] for r in pairs),'run2_valid':sum(r['run2_valid'] for r in pairs),'fresh_both_valid_n':len(valid_both),'fresh_list_equal_n':sum(r['list_equal'] for r in pairs),'fresh_set_equal_n':sum(r['set_equal'] for r in pairs),'fresh_list_equal_rate':sum(r['list_equal'] for r in pairs)/len(pairs),'fresh_set_equal_rate':sum(r['set_equal'] for r in pairs)/len(pairs),'fresh_list_differences_n':len(diffs),'fresh_ndcg_spread_max':max([abs(x) for x in d],default=0.0),'fresh_ndcg_mean_run1':sum(r['ndcg5_run1'] for r in valid_both)/len(valid_both),'fresh_ndcg_mean_run2':sum(r['ndcg5_run2'] for r in valid_both)/len(valid_both),'frozen_vs_corrected_summary':json.loads((OUT/'C2_OFFLINE_SUMMARY.json').read_text()),'cause_note':'Fresh-run disagreement identifies output instability only; this trace does not identify whether padding, dtype, batching or truncation caused it.'}
 (OUT/'C2_FRESH_RUN_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
