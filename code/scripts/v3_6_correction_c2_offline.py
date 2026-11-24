#!/usr/bin/env python3
"""Offline C2 comparison of the frozen Jina run and the prior corrected rerun."""
from __future__ import annotations
import csv,json,math
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
def cause(a,b):
 if a.get('status')!=b.get('status'): return 'status_or_failure_difference'
 if a.get('candidate_ids')!=b.get('candidate_ids'): return 'candidate_view_difference'
 sa=a.get('scores') or []; sb=b.get('scores') or []
 if sa==sb:return 'tie_order_or_unexplained_selection_difference'
 if len(sa)!=len(sb):return 'score_vector_length_difference'
 eq=[i for i,(x,y) in enumerate(zip(sa,sb)) if float(x)==float(y)]
 if any(float(x)==float(y) for x,y in zip(sa,sb)):
  # A boundary tie is an evidential tie only if the changed output is tied at the cutoff.
  aa=sorted([(float(x),i) for i,x in enumerate(sa)],reverse=True); bb=sorted([(float(x),i) for i,x in enumerate(sb)],reverse=True)
  if len(aa)>=5 and len(bb)>=5 and (aa[4][0]==aa[5][0] or bb[4][0]==bb[5][0]): return 'tie_order'
 return 'score_difference_cause_not_identifiable_from_trace'
def main():
 sl={f"{x['domain']}:{x['query_id']}":x for x in read(BRIGHT)}
 gold={(str(x['domain']),str(x['id'])):set(map(str,x.get('gold_ids',[]))) for x in json.loads(GOLD.read_text())['items']}
 fr={x['canonical_id']:x for x in read(FROZEN)}
 cr={x['canonical_id']:x for x in read(CORR) if x.get('policy')=='jina_single_8192'}
 out=[]; diffs=[]
 for cid in sorted(sl):
  a,b=fr.get(cid,{}),cr.get(cid,{})
  x=sl[cid]; g=gold.get((str(x['domain']),str(x['query_id'])),set()); sa=a.get('selected_ids') or []; sb=b.get('selected_ids') or []
  le=sa==sb; se=set(sa)==set(sb); d=not le
  row={'canonical_id':cid,'domain':x['domain'],'valid_frozen':int(a.get('status')=='PASS'),'valid_corrected':int(b.get('status')=='PASS'),'ndcg5_frozen':ndcg(sa,g) if a.get('status')=='PASS' else 0.0,'ndcg5_corrected':ndcg(sb,g) if b.get('status')=='PASS' else 0.0,'list_equal':int(le),'set_equal':int(se),'difference_cause': '' if le else cause(a,b)}
  out.append(row)
  if d: diffs.append({'canonical_id':cid,'cause':row['difference_cause'],'frozen_selected_ids':sa,'corrected_selected_ids':sb,'frozen_scores':a.get('scores',[]),'corrected_scores':b.get('scores',[])})
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'frozen_vs_corrected_perquery.csv').open('w',newline='',encoding='utf-8') as f:
  fields=list(out[0]); w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
 (OUT/'first20_differences.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in diffs[:20]),encoding='utf-8')
 summary={'status':'PASS','n_queries':len(out),'list_equal_n':sum(x['list_equal'] for x in out),'set_equal_n':sum(x['set_equal'] for x in out),'list_difference_n':len(diffs),'set_difference_n':sum(not x['set_equal'] for x in out),'first20_path':str((OUT/'first20_differences.jsonl').relative_to(ROOT)),'fresh_replicates':'PENDING_NATIVE_RUN','note':'Causes are evidence-bounded; a raw score difference alone does not identify dtype, padding, batch or truncation as the cause.'}
 (OUT/'C2_OFFLINE_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
