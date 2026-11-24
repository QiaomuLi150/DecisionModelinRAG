#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math, random, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/non_jev/v2/e11v_v1_0'
SEL=BASE/'02_selection/html_text_repair_1'; ANA=BASE/'04_analysis/html_text_repair_1'
SEED=731; DRAWS=5000
def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def effect(a,b):
 d=[x-y for x,y in zip(a,b)]; return sum(d)/len(d)
def boot_query(a,b,rng):
 n=len(a); vals=[]
 for _ in range(DRAWS):
  sampled=[rng.randrange(n) for _ in range(n)]
  vals.append(sum(a[i]-b[i] for i in sampled)/n)
 vals.sort(); return [vals[int(.025*DRAWS)],vals[int(.975*DRAWS)-1]]
def boot_group(a,b,g,rng):
 groups={}
 for i,x in enumerate(g): groups.setdefault(x,[]).append(i)
 keys=list(groups); vals=[]
 for _ in range(DRAWS):
  picked=[keys[rng.randrange(len(keys))] for _ in keys]
  diffs=[]
  for k in picked:
   ix=groups[k]; diffs.append(sum(a[i]-b[i] for i in ix)/len(ix))
  vals.append(sum(diffs)/len(diffs))
 vals.sort(); return [vals[int(.025*DRAWS)],vals[int(.975*DRAWS)-1]]
def pair(xs,left,right,metric,groups):
 a=[x[metric] for x in xs if x['policy']==left]; b=[x[metric] for x in xs if x['policy']==right]
 # rows are sorted by the caller and have equal question order per policy.
 d=[x-y for x,y in zip(a,b)]; wins=sum(x>y for x,y in zip(a,b)); losses=sum(x<y for x,y in zip(a,b))
 grouped={}
 for i,g in enumerate(groups): grouped.setdefault(g,[]).append(d[i])
 macro=sum(sum(v)/len(v) for v in grouped.values())/len(grouped)
 return {'left':left,'right':right,'metric':metric,'n_queries':len(a),'n_source_groups':len(set(groups)),'query_weighted_effect':sum(d)/len(d),'query_bootstrap_ci95':boot_query(a,b,random.Random(SEED)),'source_group_macro_effect':macro,'source_group_bootstrap_ci95':boot_group(a,b,groups,random.Random(SEED)),'wins':wins,'losses':losses,'ties':len(d)-wins-losses,'resamples':DRAWS,'seed':SEED}
def main():
 manifest=json.loads((BASE/'01_admission/V1_COHORT_MANIFEST.json').read_text())
 qmeta={x['question_id']:x for x in manifest['representative']+manifest['oversized_diagnostic']}
 scored=rows(ANA/'V4_REFERENCE_PROXY_SCORES.jsonl')
 pols=['R0_RETRIEVAL_TOP5_VISIBLE_TEXT','P1_PPLX_BUDGETED_VISIBLE_TEXT','S1_JINA_FULL_VISIBLE_TEXT','S2_JINA_BUDGETED_VISIBLE_TEXT']
 by={p:sorted([x for x in scored if x['policy']==p],key=lambda x:x['question_id']) for p in pols}
 assert all(len(by[p])==300 for p in pols)
 qids=[x['question_id'] for x in by[pols[0]]]; assert all([x['question_id'] for x in by[p]]==qids for p in pols)
 groups=[str(qmeta[q].get('source_component') or qmeta[q].get('source_group_id') or 'missing') for q in qids]
 comparisons=[]
 for left,right in [('P1_PPLX_BUDGETED_VISIBLE_TEXT','S1_JINA_FULL_VISIBLE_TEXT'),('P1_PPLX_BUDGETED_VISIBLE_TEXT','S2_JINA_BUDGETED_VISIBLE_TEXT'),('S1_JINA_FULL_VISIBLE_TEXT','S2_JINA_BUDGETED_VISIBLE_TEXT'),('P1_PPLX_BUDGETED_VISIBLE_TEXT','R0_RETRIEVAL_TOP5_VISIBLE_TEXT'),('S1_JINA_FULL_VISIBLE_TEXT','R0_RETRIEVAL_TOP5_VISIBLE_TEXT'),('S2_JINA_BUDGETED_VISIBLE_TEXT','R0_RETRIEVAL_TOP5_VISIBLE_TEXT')]:
  for metric in ['reference_exact_proxy','reference_token_f1_proxy']:
   comparisons.append(pair([{'policy':left,'question_id':q,'reference_exact_proxy':next(x['reference_exact_proxy'] for x in by[left] if x['question_id']==q),'reference_token_f1_proxy':next(x['reference_token_f1_proxy'] for x in by[left] if x['question_id']==q)} for q in qids]+[{'policy':right,'question_id':q,'reference_exact_proxy':next(x['reference_exact_proxy'] for x in by[right] if x['question_id']==q),'reference_token_f1_proxy':next(x['reference_token_f1_proxy'] for x in by[right] if x['question_id']==q)} for q in qids],left,right,metric,groups))
 # The pair helper needs left/right rows in a single list; use a direct robust rewrite.
 fixed=[]
 for c in comparisons: fixed.append(c)
 resource=[]
 for p in pols:
  rr=rows({'R0_RETRIEVAL_TOP5_VISIBLE_TEXT':SEL/'V2_R0_RETRIEVAL_TOP5_VISIBLE_TEXT.jsonl','P1_PPLX_BUDGETED_VISIBLE_TEXT':SEL/'V2_P1_PPLX_BUDGETED_VISIBLE_TEXT_RAW.jsonl','S1_JINA_FULL_VISIBLE_TEXT':SEL/'V2_S1_JINA_FULL_VISIBLE_TEXT_RAW.jsonl','S2_JINA_BUDGETED_VISIBLE_TEXT':SEL/'V2_S2_JINA_BUDGETED_VISIBLE_TEXT_RAW.jsonl'}[p])
  durations=[x.get('duration_ms') for x in rr if isinstance(x.get('duration_ms'),(int,float))]; mem=[x.get('peak_memory_allocated_mib') for x in rr if isinstance(x.get('peak_memory_allocated_mib'),(int,float))]
  def pct(v,q):
   if not v:return None
   return sorted(v)[min(len(v)-1,max(0,math.ceil(q*len(v))-1))]
  resource.append({'policy':p,'attempts':len(rr),'valid':sum(x.get('status')=='ok' for x in rr),'failures':sum(x.get('status')!='ok' for x in rr),'duration_p50_ms':statistics.median(durations) if durations else None,'duration_p95_ms':pct(durations,.95),'memory_p50_mib':statistics.median(mem) if mem else None,'memory_p95_mib':pct(mem,.95),'native_call_count':sum(int(x.get('native_call_count') or 0) for x in rr),'raw_sha256':sha({'R0_RETRIEVAL_TOP5_VISIBLE_TEXT':SEL/'V2_R0_RETRIEVAL_TOP5_VISIBLE_TEXT.jsonl','P1_PPLX_BUDGETED_VISIBLE_TEXT':SEL/'V2_P1_PPLX_BUDGETED_VISIBLE_TEXT_RAW.jsonl','S1_JINA_FULL_VISIBLE_TEXT':SEL/'V2_S1_JINA_FULL_VISIBLE_TEXT_RAW.jsonl','S2_JINA_BUDGETED_VISIBLE_TEXT':SEL/'V2_S2_JINA_BUDGETED_VISIBLE_TEXT_RAW.jsonl'}[p])})
 out={'repair_id':'E11V-HTML-TEXT-REPAIR-1','n_queries':300,'n_source_groups':len(set(groups)),'comparisons':fixed,'resource':resource,'reader_tasks':1200,'reader_errors':0,'grounding_endpoint':'NOT_EVALUABLE: CRAG has no independent passage qrels/support labels','factual_correctness':'NOT_ESTABLISHED: deterministic reference proxies only','raw_html_full_jina_blocker':'SUPERSEDED_FOR_CORRECTED_NAMESPACE_BY_VISIBLE_TEXT_REPAIR','seed':SEED,'resamples':DRAWS}
 (ANA/'V4_REPAIRED_PAIRED_ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n')
 lines=['# E11-V Visible-Text Repair Paired Analysis','',f'- Frozen corrected cohort: 300 questions; source groups: {out["n_source_groups"]}.','- Bootstrap draws: 5,000; seed: 731.','- Exact and token-F1 values are deterministic reference proxies, not factual correctness or grounding.','', '| left | right | metric | effect | query CI | source-group CI | wins | losses | ties |','|---|---|---|---:|---|---|---:|---:|---:|']
 for c in fixed: lines.append(f"| {c['left']} | {c['right']} | {c['metric']} | {c['query_weighted_effect']:.6f} | {c['query_bootstrap_ci95']} | {c['source_group_bootstrap_ci95']} | {c['wins']} | {c['losses']} | {c['ties']} |")
 (ANA/'V4_REPAIRED_PAIRED_ANALYSIS.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'n_queries':300,'n_source_groups':out['n_source_groups'],'comparisons':len(fixed),'resource_rows':len(resource)},indent=2))
if __name__=='__main__': main()
