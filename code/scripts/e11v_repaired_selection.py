#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import e11v_selection_eval as ev

def write_jsonl(p, rows, refuse=True):
    if refuse and p.exists(): raise RuntimeError(f'refusing overwrite {p}')
    with p.open('w',encoding='utf-8') as f:
        for x in rows: f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n')

def read_jsonl(p):
    with p.open(encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['prepare','freeze','pplx','jina-full','jina-budget'],required=True)
    ap.add_argument('--gpu',type=int,default=5)
    ap.add_argument('--start',type=int,default=None)
    ap.add_argument('--end',type=int,default=None)
    ap.add_argument('--suffix',default='')
    args=ap.parse_args()
    out=ROOT/'results/non_jev/v2/e11v_v1_0/02_selection/html_text_repair_1'
    out.mkdir(parents=True,exist_ok=True); ev.SEL=out
    content=out/'V1_COHORT_CONTENT_VISIBLE_TEXT.jsonl'; budget=out/'V2_BUDGET_MANIFEST_VISIBLE_TEXT.jsonl'
    if args.mode in {'prepare','freeze'}:
        if args.mode=='prepare':
            rows=ev.prepare(content)
            # prepare() writes the default filename under ev.SEL; publish the
            # explicitly versioned copy without changing the original manifest.
            default=out/'V2_BUDGET_MANIFEST.jsonl'
            if default != budget and not budget.exists(): default.replace(budget)
        else:
            rows=read_jsonl(budget)
        originals={x['question_id']:x for x in read_jsonl(content)}
        rows=read_jsonl(budget)
        for x in rows: x['_full_candidates']=originals[x['question_id']]['candidates']
        r0=[{'policy':'R0_RETRIEVAL_TOP5','question_id':x['question_id'],'domain':x.get('domain'),'cohort':x.get('cohort'),'candidate_ids':x['candidate_ids'],'selected_ids':x['candidate_ids'][:5],'status':'ok','native_call_count':0,'input_sha256':ev.sha_json({'query':x['query'],'candidate_ids':x['candidate_ids']})} for x in rows]
        r0_path=out/'V2_R0_RETRIEVAL_TOP5_VISIBLE_TEXT.jsonl'
        if args.mode=='prepare' and not r0_path.exists(): write_jsonl(r0_path,r0)
        elif args.mode=='prepare' and r0_path.exists(): pass
        freeze={'repair_id':'E11V-HTML-TEXT-REPAIR-1','parser_manifest':str((out/'V1_HTML_TEXT_REPAIR_MANIFEST.json').relative_to(ROOT)),'content_sha256':ev.sha_file(content),'budget_sha256':ev.sha_file(budget),'policies':['R0_RETRIEVAL_TOP5_VISIBLE_TEXT','P1_PPLX_BUDGETED_VISIBLE_TEXT','S1_JINA_FULL_VISIBLE_TEXT','S2_JINA_BUDGETED_VISIBLE_TEXT'],'gold_or_qrels_in_selector_inputs':False,'source_candidate_ids_and_order_preserved':True}
        freeze['freeze_receipt']='V2_VISIBLE_TEXT_PRE_OUTCOME_FREEZE_R2'
        (out/'V2_VISIBLE_TEXT_PRE_OUTCOME_FREEZE_R2.json').write_text(json.dumps(freeze,indent=2)+'\n')
        print(json.dumps({'mode':args.mode,'rows':len(rows),'budgetable':sum(x.get('status')=='ok' for x in rows),'budget_sha256':freeze['budget_sha256']},indent=2)); return 0
    rows=read_jsonl(budget)
    if args.start is not None or args.end is not None:
        rows=rows[args.start or 0:args.end]
    suffix=args.suffix
    if suffix and not suffix.startswith('.'): suffix='.'+suffix
    if args.mode=='pplx': model='pplx_budgeted'; full=False; name=f'V2_P1_PPLX_BUDGETED_VISIBLE_TEXT_RAW{suffix}.jsonl'; run=f'V2_P1_PPLX_BUDGETED_VISIBLE_TEXT.RUN{suffix}.json'
    elif args.mode=='jina-full': model='jina'; full=True; name=f'V2_S1_JINA_FULL_VISIBLE_TEXT_RAW{suffix}.jsonl'; run=f'V2_S1_JINA_FULL_VISIBLE_TEXT.RUN{suffix}.json'
    else: model='jina'; full=False; name=f'V2_S2_JINA_BUDGETED_VISIBLE_TEXT_RAW{suffix}.jsonl'; run=f'V2_S2_JINA_BUDGETED_VISIBLE_TEXT.RUN{suffix}.json'
    if full:
        originals={x['question_id']:x for x in read_jsonl(content)}
        for x in rows: x['_full_candidates']=originals[x['question_id']]['candidates']
    rows_out,meta=ev.native(model,rows,args.gpu,full_jina=full)
    raw=out/name; write_jsonl(raw,rows_out)
    (out/run).write_text(json.dumps({**meta,'rows':len(rows_out),'raw_sha256':ev.sha_file(raw)},indent=2)+'\n')
    print(json.dumps({'mode':args.mode,'rows':len(rows_out),'valid':sum(x.get('status')=='ok' for x in rows_out),'failures':sum(x.get('status')!='ok' for x in rows_out)},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
