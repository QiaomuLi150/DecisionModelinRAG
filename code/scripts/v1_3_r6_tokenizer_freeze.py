#!/usr/bin/env python3
"""CPU-only R6 native tokenizer feasibility and label-blind sub-slate freeze."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
E9=ROOT/"results/non_jev/v2/e9_bright_v1_0"
OUT=ROOT/"results/v1_3/r6_leetcode_trim"
KEV=ROOT/"data/models/e12_o_checkpoints/kev9_base"
DREX=ROOT/"data/models/e12_o_checkpoints/drex_dlm8"
LIMIT=8064
def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def file_sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def make_req(row, candidates):
    ids=[str(c["candidate_id"]) for c in candidates]
    return {"state":{"query":str(row["query"])},"model":"v1_3_r6_native_fit","questions":{"candidate":{"type":"choice","instructions":"Which candidate passage is most relevant to the query?","criteria":{cid:str(c["content"]) for cid,c in zip(ids,candidates)}}}}
def main():
    sys.path.insert(0,str(DREX/"code"))
    from kev.api import SystemOneRequest,to_record
    from kev.model import encode,load_tokenizer
    kev_tok=load_tokenizer(str(KEV)); drex_tok=load_tokenizer(str(DREX))
    rows=[json.loads(x) for x in (E9/"01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl").open(encoding="utf-8") if x.strip() and json.loads(x)["domain"]=="leetcode"]
    out=[]
    for row in sorted(rows,key=lambda x:x["query_id"]):
        result={"canonical_id":f"leetcode:{row['query_id']}","domain":"leetcode","query_id":str(row["query_id"]),"source_slate_sha256":row["slate_sha256"],"candidate_ids":[str(c["candidate_id"]) for c in row["candidates"]],"labels_joined":False,"model_views":{}}
        for name,tok in (("kev_native",kev_tok),("drex_native",drex_tok)):
            selected=None; token_len=None; attempts=[]
            for k in range(20,5,-1):
                cand=row["candidates"][:k]; req=make_req(row,cand)
                try:
                    rec,_=to_record(SystemOneRequest.model_validate(req)); enc=encode(tok,rec,max_state=LIMIT,max_branch=LIMIT,strict=True); n=len(enc["ids"]); attempts.append({"k":k,"tokens":n,"fits":n<=LIMIT})
                    if n<=LIMIT: selected={"k":k,"candidate_ids":[str(c["candidate_id"]) for c in cand],"candidate_ids_sha256":sha([str(c["candidate_id"]) for c in cand]),"tokens":n,"fit_limit":LIMIT,"request_sha256":sha(req),"attempts":attempts}; break
                except Exception as exc:
                    attempts.append({"k":k,"tokens":None,"fits":False,"error_class":type(exc).__name__})
            result["model_views"][name]=selected or {"k":0,"candidate_ids":[],"tokens":None,"fit_limit":LIMIT,"attempts":attempts,"status":"NO_PREFIX_AT_LEAST_6"}
        out.append(result)
    OUT.mkdir(parents=True,exist_ok=True); p=OUT/"leetcode_subslate_manifest.jsonl"; p.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in out),encoding="utf-8")
    summary={"status":"FROZEN_LABEL_BLIND_SUBSLATES","records":len(out),"kev_at_least_6":sum(x["model_views"]["kev_native"].get("k",0)>=6 for x in out),"drex_at_least_6":sum(x["model_views"]["drex_native"].get("k",0)>=6 for x in out),"manifest_sha256":file_sha(p),"tokenizer_sources":{"kev":str(KEV.relative_to(ROOT)),"drex":str(DREX.relative_to(ROOT))},"fit_limit":LIMIT,"labels_joined":False,"weights_loaded":False}
    (OUT/"leetcode_subslate_manifest.lock.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
