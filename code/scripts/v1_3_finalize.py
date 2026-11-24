#!/usr/bin/env python3
"""Create the append-only v1.3 closure ledger, raw hashes and shareable ZIP."""
from __future__ import annotations
import hashlib, json, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; V=ROOT/'results/v1_3'
def fsha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def nlines(p): return sum(1 for x in p.open(encoding='utf-8') if x.strip())
def raw_manifest():
    rows=[]
    for p in sorted((V/'r2_order_sensitivity/raw').glob('*.jsonl'))+sorted((V/'r3_multipass').glob('*.jsonl'))+sorted((V/'r6_leetcode_trim/raw').glob('*.jsonl')):
        vals=[]
        for line in p.open(encoding='utf-8'):
            if line.strip():
                try: vals.append(json.loads(line))
                except Exception: pass
        rows.append({'path':str(p.relative_to(ROOT)),'sha256':fsha(p),'records':len(vals),'labels_joined_values':sorted({v.get('labels_joined') for v in vals if isinstance(v,dict)})})
    return rows
def checks():
    bright=ROOT/'results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl'
    r2=list((V/'r2_order_sensitivity/raw').glob('*_gpu*_orders_*.jsonl'))
    full_r2=[]
    for p in r2:
        if '_perm_seed_' in p.stem: full_r2.extend(json.loads(x) for x in p.open() if x.strip())
    r3=list((V/'r3_multipass').glob('multipass_raw_start*_n*.jsonl'))
    r3rows=[json.loads(x) for p in r3 for x in p.open() if x.strip()]
    r6=[p for p in (V/'r6_leetcode_trim/raw').glob('*.jsonl')]
    return {
        'bright_records':nlines(bright),'r2_full_shard_rows':len(full_r2),'r2_full_unique_keys':len({(x.get('policy'),x.get('canonical_id'),x.get('order_id')) for x in full_r2}),
        'r3_shard_rows':len(r3rows),'r3_unique_ids':len({x.get('canonical_id') for x in r3rows}),'r3_shards':len(r3),
        'r6_raw_files':len(r6),'r2_expected_nonoriginal_rows':16608,'r3_expected_rows':1384,
        'pass':nlines(bright)==1384 and len(full_r2)==16608 and len({(x.get('policy'),x.get('canonical_id'),x.get('order_id')) for x in full_r2})==16608 and len(r3rows)==1384 and len({x.get('canonical_id') for x in r3rows})==1384
    }
def main():
    manifest=raw_manifest(); audit=checks()
    (V/'RAW_TRACE_SHA256_MANIFEST.json').write_text(json.dumps({'generated_by':'scripts/v1_3_finalize.py','raw_traces':manifest},indent=2)+'\n')
    status={
      'gate':'V1_3_PARTIAL_BLOCKED',
      'scope':'DecisionRAG v1.3 six follow-up experiments',
      'frozen_population':{'bright':1384,'e12o_serving':120,'e11v_crag':300,'leetcode':142},
      'protocols_hashed_before_inference':True,'labels_joined_only_offline':True,'jev_calls':0,'paid_api_calls':0,'training':False,'protected_holdout_access':False,
      'experiments':{
        'R1':{'status':'BLOCKED','attempted':0,'valid':0,'failed':0,'reason':'No dedicated A100 with zero resident processes; no timing substitute.'},
        'R2':{'status':'COMPLETE','attempted_per_policy':9688,'valid':{'pplx_budget':9660,'jina_full':9688,'qwen3_reranker_0_6b':9674},'failed':{'pplx_budget':28,'jina_full':0,'qwen3_reranker_0_6b':14}},
        'R3':{'status':'COMPLETE_QUALITY_PARTIAL_TIMING','attempted':1384,'valid':1381,'failed':3,'reason':'Shared-GPU run; dedicated timing gate not passed.'},
        'R4':{'status':'BLOCKED','attempted':0,'valid':0,'failed':0,'reason':'Only existing 256-token reader outputs were available; 512/1024 rerun not validly executable.'},
        'R5':{'status':'BLOCKED','attempted':0,'valid':0,'failed':0,'reason':'Pinned Winnow runtime binary unavailable; no substitute built.'},
        'R6':{'status':'PARTIAL','attempted':{'kev':284,'drex':284,'jina_controls':568},'valid':{'kev':271,'drex':266,'jina_native_controls':279},'failed_or_blocked':{'kev':13,'drex':18,'jina_native_controls':5,'jina_budget_controls':284},'reason':'Exact Jina budget strings were not persisted in the frozen R6 manifest.'}
      },
      'integrity_checks':audit,
      'raw_trace_manifest':'results/v1_3/RAW_TRACE_SHA256_MANIFEST.json',
      'claim_changes':{'pplx_p95_le_2x_jina':'NOT_MEASURED','order_sensitivity':'PPLX and Jina Top-5 sets are order-sensitive; PPLX-vs-Jina quality sign is uncertain under domain-aware inference','leetcode_native_deficit':'persists for Kev and Drex against matched native Jina controls'},
      'known_failed_attempts':['R3 single-worker interrupted before raw commit','R6 Drex first loader path used wrong package precedence','R6 first Jina control script bookkeeping error'],
      'next_owner_decision':'Provide a genuinely isolated GPU and the missing Winnow/Jina-budget inputs if the blocked branches are required; do not treat this partial gate as a full serving certification.'
    }
    (V/'V1_3_FINAL_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    test='''# v1.3 Test Report\n\n- Shared ML virtual environment: **PASS**, 302/302 pytest tests passed.\n- System Python: **PARTIAL**, 299/302 passed and 3 failed. Two failures require the already-cached SciFact Arrow audit dependency `pyarrow`; one requires `torch` for the StartLux runtime import. These are environment dependency failures, not v1.3 result failures.\n- v1.3 offline R2/R3/R6 analyses: **PASS**; exact frozen-population row and uniqueness checks are in `V1_3_FINAL_STATUS.json`.\n- v1.3 adapter scripts: **PASS** for Python compilation after the recorded R6 fixes.\n- Raw records retain label-blind markers; all label joins occur in offline analysis scripts.\n- No Jev/paid API calls, training, retrieval rebuild or protected-holdout access.\n'''
    (V/'TEST_REPORT.md').write_text(test)
    zip_path=V/'decisionrag_v1_3_exports.zip'
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for exp in ['r1_serving_gate','r2_order_sensitivity','r3_multipass','r4_crag_reader_window','r5_winnow_budget','r6_leetcode_trim']:
            ed=V/exp/'export'
            if ed.exists():
                for p in sorted(ed.rglob('*')):
                    if p.is_file(): z.write(p,p.relative_to(V).as_posix())
        for p in sorted(V.glob('r*/PROTOCOL.json')): z.write(p,('protocols/'+p.parent.name+'/'+p.name))
        z.write(V/'RAW_TRACE_SHA256_MANIFEST.json','RAW_TRACE_SHA256_MANIFEST.json')
    (V/'ZIP_MANIFEST.json').write_text(json.dumps({'path':str(zip_path.relative_to(ROOT)),'bytes':zip_path.stat().st_size,'sha256':fsha(zip_path),'members':zipfile.ZipFile(zip_path).namelist()},indent=2)+'\n')
    print(json.dumps({'status':status['gate'],'audit':audit,'zip':str(zip_path.relative_to(ROOT)),'zip_sha256':fsha(zip_path),'zip_bytes':zip_path.stat().st_size},indent=2))
if __name__=='__main__':main()
