#!/usr/bin/env python3
import json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/non_jev/v2/e11v_v1_0/02_selection/gpu5_retry_1'
BASE=ROOT/'results/non_jev/v2/e11v_v1_0/02_selection'
sys.path.insert(0,str(ROOT/'scripts'))
import e11v_selection_eval as ev

def write_rows(path, rows):
    if path.exists(): raise RuntimeError(f'refusing overwrite {path}')
    with path.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,sort_keys=True)+'\n')

def main():
    global OUT
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',default=None)
    ap.add_argument('--s1-probe',action='store_true')
    ap.add_argument('--s1-full',action='store_true')
    ap.add_argument('--s2',action='store_true')
    ap.add_argument('--gpu',type=int,default=5)
    args=ap.parse_args()
    if args.output_dir: OUT=BASE/args.output_dir
    OUT.mkdir(parents=True,exist_ok=True)
    ev.SEL=OUT
    content_path=BASE/'V1_COHORT_CONTENT.jsonl'
    budget_path=BASE/'V2_BUDGET_MANIFEST.jsonl'
    budgets=[json.loads(x) for x in budget_path.read_text(encoding='utf-8').splitlines() if x.strip()]
    if args.s1_probe:
        first=json.loads(content_path.open(encoding='utf-8').readline())
        row=budgets[0]
        row['_full_candidates']=first['candidates']
        rows,meta=ev.native('jina',[row],args.gpu,full_jina=True)
        path=OUT/'V2_S1_JINA_FULL_PROBE.jsonl'; write_rows(path,rows)
        (OUT/'V2_S1_JINA_FULL_BLOCKER.json').write_text(json.dumps({
            'status':'BLOCKED_CONTEXT_OR_MEMORY',
            'policy':'S1_JINA_FULL',
            'probe_rows':len(rows),
            'probe_status':rows[0].get('status'),
            'probe_failure':rows[0].get('failure'),
            'model_revision':ev.JINA_REV,
            'gpu':5,
            'raw_probe_sha256':ev.sha_file(path),
            'interpretation':'Full archived HTML content was passed under the native Jina contract; the probe failed before a valid score vector. No context truncation or proxy was substituted.',
        },indent=2)+'\n')
        print(json.dumps({'arm':'S1_JINA_FULL','status':rows[0].get('status'),'failure_class':rows[0].get('failure_class')},indent=2))
        return 0
    if args.s1_full:
        originals={x['question_id']:x for x in ev.iter_content(content_path)}
        rows_in=[]
        for row in budgets:
            x=dict(row); x['_full_candidates']=originals[row['question_id']]['candidates']; rows_in.append(x)
        rows,meta=ev.native('jina',rows_in,args.gpu,full_jina=True)
        path=OUT/'V2_S1_JINA_FULL_RAW.jsonl'; write_rows(path,rows)
        (OUT/'V2_S1_JINA_FULL.RUN.json').write_text(json.dumps({**meta,'raw_sha256':ev.sha_file(path),'rows':len(rows)},indent=2)+'\n')
        print(json.dumps({'arm':'S1_JINA_FULL','rows':len(rows),'valid':sum(r.get('status')=='ok' for r in rows),'failures':sum(r.get('status')!='ok' for r in rows)},indent=2))
        return 0
    if args.s2:
        rows,meta=ev.native('jina',budgets,args.gpu,full_jina=False)
        path=OUT/'V2_S2_JINA_BUDGETED_RAW.jsonl'; write_rows(path,rows)
        (OUT/'V2_S2_JINA_BUDGETED.RUN.json').write_text(json.dumps({**meta,'raw_sha256':ev.sha_file(path),'rows':len(rows)},indent=2)+'\n')
        print(json.dumps({'arm':'S2_JINA_BUDGETED','rows':len(rows),'valid':sum(r.get('status')=='ok' for r in rows),'failures':sum(r.get('status')!='ok' for r in rows)},indent=2))
        return 0
    raise SystemExit('use --s1-probe, --s1-full or --s2')
if __name__=='__main__': raise SystemExit(main())
