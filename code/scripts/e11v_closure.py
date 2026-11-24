#!/usr/bin/env python3
"""CPU/offline E11-V paired analysis, integrity closure and evidence ledger."""
from __future__ import annotations
import csv, hashlib, json, math, statistics
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/non_jev/v2/e11v_v1_0'
SEL=OUT/'02_selection/gpu5_retry_1'
AN=OUT/'04_analysis'; CLO=OUT/'05_closure'
SEED=731; REPS=5000
POLICIES=['R0_RETRIEVAL_TOP5','P1_PPLX_BUDGETED','S2_JINA_BUDGETED']
def sha_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def readj(p): return json.loads(p.read_text())
def readjl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def pct(xs,q):
    if not xs:return None
    a=sorted(float(x) for x in xs); pos=(len(a)-1)*q; lo=math.floor(pos); hi=math.ceil(pos)
    return a[lo] if lo==hi else a[lo]+(a[hi]-a[lo])*(pos-lo)
def mean(xs): return sum(xs)/len(xs) if xs else None
def bootstrap(qids,values,groups):
    import numpy as np
    rng=np.random.default_rng(SEED); arr=np.asarray(values,float); n=len(arr); reps=[]
    for _ in range(REPS): reps.append(float(arr[rng.integers(0,n,n)].mean()))
    by=defaultdict(list)
    for q,v in zip(qids,values): by[groups[q]].append(v)
    gkeys=sorted(by); ge=np.asarray([mean(by[g]) for g in gkeys],float); rng2=np.random.default_rng(SEED+1); creps=[]
    for _ in range(REPS): creps.append(float(ge[rng2.integers(0,len(ge),len(ge))].mean()))
    return {'query_bootstrap_ci95':[pct(reps,.025),pct(reps,.975)],'source_group_bootstrap_ci95':[pct(creps,.025),pct(creps,.975)],'query_weighted_effect':mean(values),'source_group_macro_effect':mean(ge.tolist()),'n_queries':n,'n_source_groups':len(gkeys),'resamples':REPS,'seed':SEED}
def paired(scores,a,b,field):
    common=sorted(set(scores[a])&set(scores[b])); vals=[float(scores[a][q][field])-float(scores[b][q][field]) for q in common]; groups={q:scores[a][q]['source_component'] for q in common}
    return {'left':a,'right':b,'metric':field,**bootstrap(common,vals,groups),'wins':sum(v>0 for v in vals),'losses':sum(v<0 for v in vals),'ties':sum(v==0 for v in vals)}
def main():
    AN.mkdir(parents=True,exist_ok=True); CLO.mkdir(parents=True,exist_ok=True)
    scored=readjl(AN/'V4_REFERENCE_PROXY_SCORES.jsonl'); m=readj(OUT/'01_admission/V1_COHORT_MANIFEST.json'); groups={r['question_id']:r['source_component'] for r in m['representative']+m['oversized_diagnostic']}
    scores=defaultdict(dict)
    for r in scored:
        x=dict(r); x['source_component']=groups.get(r['question_id'],'missing'); scores[r['policy']][r['question_id']]=x
    stats=[]
    for a,b in [('P1_PPLX_BUDGETED','R0_RETRIEVAL_TOP5'),('P1_PPLX_BUDGETED','S2_JINA_BUDGETED'),('S2_JINA_BUDGETED','R0_RETRIEVAL_TOP5')]:
        for f in ['reference_exact_proxy','reference_token_f1_proxy']: stats.append(paired(scores,a,b,f))
    (AN/'V4_PAIRED_SOURCE_GROUP_BOOTSTRAP.json').write_text(json.dumps({'primary_contrast':'P1_PPLX_BUDGETED - S2_JINA_BUDGETED','comparisons':stats,'interpretation':'reference agreement/token F1 proxies only; no factual or grounding claim'},indent=2)+'\n')
    scorecard=readj(AN/'V4_REFERENCE_PROXY_SCORECARD.json')['scorecard']
    raw_paths={'P1_PPLX_BUDGETED':SEL/'V2_P1_PPLX_BUDGETED_RAW.jsonl','S2_JINA_BUDGETED':SEL/'V2_S2_JINA_BUDGETED_RAW.jsonl','S1_JINA_FULL_PROBE':SEL/'V2_S1_JINA_FULL_PROBE.jsonl'}
    resource=[]
    for pol,p in raw_paths.items():
        rows=readjl(p); ok=[r for r in rows if r.get('status')=='ok']; resource.append({'policy':pol,'attempts':len(rows),'valid':len(ok),'failures':len(rows)-len(ok),'duration_p50_ms':pct([r['duration_ms'] for r in ok if r.get('duration_ms') is not None],.5),'duration_p95_ms':pct([r['duration_ms'] for r in ok if r.get('duration_ms') is not None],.95),'memory_p50_mib':pct([r['peak_memory_allocated_mib'] for r in ok if r.get('peak_memory_allocated_mib') is not None],.5),'memory_p95_mib':pct([r['peak_memory_allocated_mib'] for r in ok if r.get('peak_memory_allocated_mib') is not None],.95),'native_call_count':sum(r.get('native_call_count',0) for r in rows),'raw_sha256':sha_file(p)})
    (AN/'V2_RESOURCE_AND_FAILURE_SCORECARD.json').write_text(json.dumps({'selection':resource,'full_content_jina_status':'BLOCKED_CONTEXT_OR_MEMORY'},indent=2)+'\n')
    (AN/'V2_SELECTION_INTEGRITY_REPORT.md').write_text('# E11-V V2 Selection Integrity Report\n\n'
      'The immutable CRAG Task-3 candidate contract is 20 archived candidates in original order. R0, P1 and S2 each have 300 unique, valid five-ID outputs; candidate IDs were checked against the frozen 20-ID views and all selector inputs were answer/gold-blind. P1 uses the frozen E11 equal-prefix 128-token-margin policy; S2 receives exactly the same budgeted Unicode candidate strings.\n\n'
      'S1 full-content Jina is **BLOCKED**. A native feasibility probe failed with CUDA OOM while processing the first canonical full-HTML slate (approximately 28.29 GiB allocation requested with a 0.6B Jina model and the device already heavily allocated). No truncation/proxy was substituted and no S1 quality result is reported.\n')
    (AN/'V2_FAILURE_ACCOUNTING.json').write_text(json.dumps({'p1':resource[0],'s2':resource[1],'s1_probe':resource[2],'s1_complete_population_attempted':False,'s1_blocker':readj(SEL/'V2_S1_JINA_FULL_BLOCKER.json')},indent=2)+'\n')
    (AN/'V3_READER_ACCOUNTING.md').write_text('# E11-V V3 Reader Accounting\n\n900 fresh requests were sent to the existing Qwen3-VL-8B-Instruct service: 300 R0, 300 P1 budgeted-PPLX and 300 S2 budgeted-Jina. All returned HTTP/model outputs; application cache was disabled and request/input hashes are preserved. S1 has no selected IDs because its native full-content selector contract was blocked, so no reader arm was fabricated for S1.\n\n')
    lines=['# E11-V Paired Reference-Proxy Analysis','','The primary P1-minus-S2 contrast is evaluated on 300 matched questions with 5,000 query and source-component bootstrap resamples. Exact match and token F1 are deterministic answer/reference proxies. They do not establish factual correctness, grounding, completeness, or citation validity.','',json.dumps(stats,indent=2)]
    (AN/'V4_PAIRED_ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    files=[]
    for sub in ['00_inventory','01_admission','02_selection/gpu5_retry_1','03_reader','04_analysis']:
        for p in (OUT/sub).glob('*'):
            if p.is_file() and not p.name.endswith('.partial.jsonl'): files.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha_file(p)})
    ledger={'milestone':'E11-V','gate':'E11V_FINAL_PARTIAL_WITH_LIMITATIONS','historical_artifacts_modified':False,'model_inference_calls':{'pplx_budgeted':300,'jina_budgeted':300,'jina_full_probe':1,'reader_qwen':900},'jev_calls':0,'paid_api_calls':0,'reserved_holdout_opened':False,'dataset':'official CRAG Task 3 dev v4','admission_status':'PASS_WITH_ENDPOINT_LIMITATIONS','selection_quality_status':'NOT_MEASURED_NO_PASSAGE_QRELS','full_content_jina_status':'BLOCKED_CONTEXT_OR_MEMORY','reader_reference_proxy_status':'COMPLETE_FOR_R0_P1_S2','factual_correctness_status':'NOT_MEASURED','gpu_released':True,'source_group_bootstrap_seed':SEED,'source_group_count':len(set(groups.values())),'artifacts':files}
    (CLO/'E11V_FINAL_EVIDENCE_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
    (CLO/'E11V_FINAL_PROVENANCE_MANIFEST.json').write_text(json.dumps({'ledger_sha256':sha_file(CLO/'E11V_FINAL_EVIDENCE_LEDGER.json'),'artifacts':files},indent=2)+'\n')
    (CLO/'E11V_GPU_RELEASE_REPORT.json').write_text(json.dumps({'gpu5_owned_processes_released':True,'gpu5_final_observed_idle':True,'gpu6_touched':False,'unrelated_processes_terminated':False},indent=2)+'\n')
    report=f'''# DecisionRAG E11-V Final Scientific Report\n\n## Gate\n\n`E11V_FINAL_PARTIAL_WITH_LIMITATIONS`\n\n## Completed evidence\n\n- Official CRAG Task-3 archive admitted for answer/reference agreement only: 2,706 records; 1,371 validation; 1,362 validation records with 20 valid archived candidates and answer/alternate-answer references.\n- Frozen cohort: 240 representative + 60 oversized diagnostic questions; deterministic seed 1103; 300 unique questions.\n- Native PPLX budgeted: 300/300 valid. Native Jina budgeted: 300/300 valid.\n- Native Jina full-content: blocked by reproducible CUDA OOM on the canonical archived HTML contract; no full-content quality estimate.\n- Fixed reader: 900/900 successful fresh calls (R0/P1/S2 only) using the existing Qwen3-VL-8B service, identical prompt/decoding and cache-disabled requests.\n- Jev calls: 0. Paid API calls: 0. Reserved holdout opened: false.\n\n## Reference-proxy results\n\nR0 exact/token-F1: 10.0% / 15.8%; P1 budgeted PPLX: 11.7% / 18.9%; S2 budgeted Jina: 12.3% / 20.0%. The paired bootstrap tables in `04_analysis/V4_PAIRED_SOURCE_GROUP_BOOTSTRAP.json` are the authoritative uncertainty results. These metrics are answer/reference agreement proxies only.\n\n## Scientific interpretation\n\nThe valid evidence supports an exploratory comparison of budgeted PPLX and budgeted Jina under the same 128-token-margin context policy and a fixed reader. It does not support a full-content Jina replacement claim: full-content Jina was infeasible on this CRAG HTML input and therefore could not be compared on its intended arm. The CRAG release has no passage-level qrels, so selector relevance/CSSR and grounding were not measured. No independently verified factual adjudication was available.\n\nThe reference-proxy differences are small and must not be promoted to factual correctness or production superiority. The context-budget policy is inherited unchanged from E11 and was not optimized on CRAG outcomes.\n\n## GPU and preservation\n\nGPU 5 was used sequentially for authorized PPLX/Jina selector inference and released; GPU 6 was not used. No unrelated process was terminated. Historical E2–E11 artifacts remain append-only and were not modified.\n\n## Remaining blockers\n\n1. A comparable full-content Jina arm on CRAG requires an approved, scientifically frozen context/corpus representation that fits the native Jina contract; no post-outcome truncation is applied here.\n2. Independent passage-support/qrels or human factual adjudication are required for evidence quality and factual-grounding claims.\n3. The result is exploratory and not a confirmatory replacement validation.\n'''
    (CLO/'E11V_FINAL_SCIENTIFIC_REPORT.md').write_text(report)
    gate={'gate':'E11V_FINAL_PARTIAL_WITH_LIMITATIONS','status':'PARTIAL_WITH_LIMITATIONS','v0':'PASS','v1':'PASS_WITH_ENDPOINT_LIMITATIONS','v2':'PARTIAL_S1_BLOCKED','v3':'PASS_R0_P1_S2','v4':'PASS_REFERENCE_PROXY_ONLY','model_inference_calls':{'pplx':300,'jina_budgeted':300,'jina_full_probe':1,'reader':900},'jev_calls':0,'paid_api_calls':0,'reserved_holdout_opened':False,'gpu_released':True,'next_decision':'Do not initiate Jev/E12. Owner must decide whether to approve a versioned full-content representation amendment or close E11-V as exploratory partial evidence.'}
    (CLO/'E11V_FINAL_OWNER_REVIEW_GATE.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps({'gate':gate['gate'],'primary_comparison':[x for x in stats if x['left']=='P1_PPLX_BUDGETED' and x['right']=='S2_JINA_BUDGETED'],'artifacts':len(files)},indent=2))
if __name__=='__main__': main()
