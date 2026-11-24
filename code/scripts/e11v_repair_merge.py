#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/non_jev/v2/e11v_v1_0/02_selection/html_text_repair_1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
    p0=rows(OUT/'V2_S1_JINA_FULL_VISIBLE_TEXT_RAW.part0.jsonl'); p1=rows(OUT/'V2_S1_JINA_FULL_VISIBLE_TEXT_RAW.part1.jsonl')
    merged=p0+p1
    ids=[x['question_id'] for x in merged]
    assert len(merged)==300 and len(set(ids))==300
    assert all(x.get('status')=='ok' and len(x.get('selected_ids',[]))==5 and len(set(x.get('selected_ids',[])))==5 for x in merged)
    out=OUT/'V2_S1_JINA_FULL_VISIBLE_TEXT_RAW.jsonl'
    if out.exists(): raise RuntimeError(f'refusing overwrite {out}')
    out.write_text('\n'.join(json.dumps(x,sort_keys=True) for x in merged)+'\n')
    expected=rows(OUT/'V2_BUDGET_MANIFEST_VISIBLE_TEXT.jsonl')
    assert ids==[x['question_id'] for x in expected]
    report={'repair_id':'E11V-HTML-TEXT-REPAIR-1','full_jina_rows':len(merged),'full_jina_valid':sum(x.get('status')=='ok' for x in merged),'full_jina_failures':sum(x.get('status')!='ok' for x in merged),'question_ids_unique':len(set(ids))==300,'canonical_order_matches_budget_manifest':True,'part0_sha256':sha(OUT/'V2_S1_JINA_FULL_VISIBLE_TEXT_RAW.part0.jsonl'),'part1_sha256':sha(OUT/'V2_S1_JINA_FULL_VISIBLE_TEXT_RAW.part1.jsonl'),'merged_sha256':sha(out),'pplx_raw_sha256':sha(OUT/'V2_P1_PPLX_BUDGETED_VISIBLE_TEXT_RAW.jsonl'),'jina_budget_raw_sha256':sha(OUT/'V2_S2_JINA_BUDGETED_VISIBLE_TEXT_RAW.jsonl'),'native_gold_or_qrels_used':False}
    (OUT/'V2_REPAIRED_SELECTOR_INTEGRITY.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
