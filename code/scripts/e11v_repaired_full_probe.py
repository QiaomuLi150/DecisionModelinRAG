#!/usr/bin/env python3
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import e11v_selection_eval as ev

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--gpu',type=int,default=5); ap.add_argument('--output-dir',default='html_text_repair_1'); ap.add_argument('--source-dir',default=None)
    args=ap.parse_args()
    out=ROOT/'results/non_jev/v2/e11v_v1_0/02_selection'/args.output_dir; out.mkdir(parents=True,exist_ok=True)
    content_dir=ROOT/'results/non_jev/v2/e11v_v1_0/02_selection'/args.source_dir if args.source_dir else out
    content=content_dir/'V1_COHORT_CONTENT_VISIBLE_TEXT.jsonl'
    row=json.loads(content.open(encoding='utf-8').readline())
    row['status']='ok'
    row['_full_candidates']=row['candidates']
    rows,meta=ev.native('jina',[row],args.gpu,full_jina=True)
    raw=out/'V2_S1_JINA_FULL_REPAIRED_PROBE.jsonl'
    with raw.open('w',encoding='utf-8') as f:
        for x in rows: f.write(json.dumps(x,sort_keys=True)+'\n')
    (out/'V2_S1_JINA_FULL_REPAIRED_PROBE.RUN.json').write_text(json.dumps({**meta,'rows':len(rows),'raw_sha256':ev.sha_file(raw)},indent=2)+'\n')
    print(json.dumps({'status':rows[0].get('status'),'failure_class':rows[0].get('failure_class'),'duration_ms':rows[0].get('duration_ms'),'peak_memory_allocated_mib':rows[0].get('peak_memory_allocated_mib')},indent=2))
if __name__=='__main__': main()
