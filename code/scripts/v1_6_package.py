#!/usr/bin/env python3
"""Build the sanitized, path-free v1.6 continuation export package."""
from __future__ import annotations
import csv, hashlib, re, shutil, tempfile, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/v1_6"; ZIP=OUT/"decisionrag_v1_6_continuation_exports.zip"
FILES={
 "README.md": OUT/"README.md",
 "d1_perwindow_jina/PROTOCOL.json": OUT/"d1_perwindow_jina/PROTOCOL.json",
 "d1_perwindow_jina/D1_PERWINDOW_REPORT.md": OUT/"d1_perwindow_jina/D1_PERWINDOW_REPORT.md",
 "d1_perwindow_jina/D1_INPUT_MANIFEST_LOCK.json": OUT/"d1_perwindow_jina/D1_INPUT_MANIFEST_LOCK.json",
 "d1_perwindow_jina/D1_RAW_TRACE_SHA256_MANIFEST.json": OUT/"d1_perwindow_jina/D1_RAW_TRACE_SHA256_MANIFEST.json",
 "d1_perwindow_jina/D1_DETERMINISM_RECHECK.json": OUT/"d1_perwindow_jina/D1_DETERMINISM_RECHECK.json",
 "d1_perwindow_jina/D1_OFFLINE_SUMMARY.json": OUT/"d1_perwindow_jina/D1_OFFLINE_SUMMARY.json",
 "d1_perwindow_jina/export/d1_perquery.csv": OUT/"d1_perwindow_jina/export/d1_perquery.csv",
 "d1_perwindow_jina/export/d1_domain_summary.csv": OUT/"d1_perwindow_jina/export/d1_domain_summary.csv",
 "d1_perwindow_jina/export/d1_contrasts.csv": OUT/"d1_perwindow_jina/export/d1_contrasts.csv",
 "d2_matched_timing/PROTOCOL.json": OUT/"d2_matched_timing/PROTOCOL.json",
 "d2_matched_timing/D2_RESOURCE_GATE.json": OUT/"d2_matched_timing/D2_RESOURCE_GATE.json",
 "D2_MATCHED_TIMING_BLOCKER.md": OUT/"D2_MATCHED_TIMING_BLOCKER.md",
}
MISSING=[
 "d2_matched_timing/export/d2_latency_perrequest.csv (not created: D2 dedicated-GPU gate failed before inference)",
 "d2_matched_timing/export/d2_latency_ratios.csv (not created: D2 dedicated-GPU gate failed before inference)",
]

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def sanitize(s:str)->str:
 s=re.sub(r"/(?:mnt|home|root)/[^\s\"']+", "<redacted>", s)
 s=re.sub(r"GPU-[0-9A-Fa-f-]{8,}", "<redacted>", s)
 s=re.sub(r"\b(?:<REDACTED_USER>|<REDACTED_USER>)\b", "<redacted>", s)
 s=re.sub(r"(?i)<redacted>", "<redacted>", s)
 return s
def row_count(p:Path):
 if p.suffix.lower()=='.csv':
  with p.open(newline='',encoding='utf-8') as f:return max(0,sum(1 for _ in csv.reader(f))-1)
 if p.suffix.lower()=='.jsonl':return sum(1 for x in p.open(encoding='utf-8') if x.strip())
 return 'n/a'

def main():
 stage=Path(tempfile.mkdtemp(prefix='decisionrag_v1_6_stage_',dir='/tmp'))
 entries=[]
 for rel,src in FILES.items():
  if not src.exists(): raise SystemExit(f'missing required file: {src}')
  dst=stage/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  raw=src.read_text(encoding='utf-8');dst.write_text(sanitize(raw),encoding='utf-8')
  entries.append({'path':rel,'rows':row_count(dst),'sha256':sha(dst),'description':rel.split('/')[-1]})
 contents=['# DecisionRAG v1.6 continuation export contents','', 'All paths are relative and export copies are sanitized. Raw traces are not included; their hashes and record counts are in `d1_perwindow_jina/D1_RAW_TRACE_SHA256_MANIFEST.json`.','']
 contents.append('| path | rows | SHA-256 | description |');contents.append('|---|---:|---|---|')
 for e in entries:contents.append(f"| `{e['path']}` | {e['rows']} | `{e['sha256']}` | {e['description']} |")
 contents += ['', '## Requested files not present', '']
 contents += [f'- {x}' for x in MISSING]
 (stage/'CONTENTS.md').write_text('\n'.join(contents)+'\n',encoding='utf-8')
 # Strict package hygiene check.
 bad=[]
 for p in stage.rglob('*'):
  if p.is_file():
   text=p.read_text(encoding='utf-8',errors='replace')
   if re.search(r"<REDACTED_PATH><REDACTED_USER>|<REDACTED_USER>|<redacted>|GPU-[0-9A-Fa-f-]{8,}",text,re.I):bad.append(str(p.relative_to(stage)))
 if bad: raise SystemExit(f'export hygiene failure: {bad}')
 if ZIP.exists(): raise SystemExit(f'refusing to overwrite existing package: {ZIP}')
 with zipfile.ZipFile(ZIP,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for p in sorted(stage.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(stage).as_posix())
 print({'zip':str(ZIP.relative_to(ROOT)),'size_bytes':ZIP.stat().st_size,'sha256':sha(ZIP),'staging':str(stage),'files':len(entries)+1})
 print('\n'.join(contents))
if __name__=='__main__':main()
