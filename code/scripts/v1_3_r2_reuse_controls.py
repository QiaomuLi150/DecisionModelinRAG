#!/usr/bin/env python3
"""Materialize label-blind Qwen pointwise order controls from authenticated raw scores."""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
V13=ROOT/"results/v1_3/r2_order_sensitivity"
E9=ROOT/"results/non_jev/v2/e9_bright_v1_0"
MAN=V13/"permutation_manifest.jsonl"
RAW=E9/"02_native/BRIGHT_NATIVE_RAW_OUTPUTS_qwen3_reranker_0_6b.jsonl"
def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def fsha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    slates={f"{x['domain']}:{x['query_id']}":x for x in (json.loads(v) for v in (E9/"01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl").open(encoding="utf-8") if v.strip())}
    old={f"{x['domain']}:{x['query_id']}":x for x in (json.loads(v) for v in RAW.open(encoding="utf-8") if v.strip())}
    out=[]
    for line in MAN.open(encoding="utf-8"):
        m=json.loads(line); cid=m["canonical_id"]; r=slates[cid]; o=old[cid]; by={str(k):float(v) for k,v in zip(o["candidate_ids"],o.get("scores",[]))}; scores=[by[x] for x in m["candidate_ids"]] if len(o.get("scores",[])) == 20 else []; idx=sorted(range(20),key=lambda i:(-scores[i],i,m["candidate_ids"][i])) if scores else []
        out.append({"canonical_id":cid,"domain":m["domain"],"query_id":m["query_id"],"policy":"qwen3_reranker_0_6b","order_id":m["order_id"],"candidate_ids":m["candidate_ids"],"candidate_ids_sha256":m["candidate_ids_sha256"],"source_slate_sha256":m["source_slate_sha256"],"source_input_sha256":m["source_input_sha256"],"status":o["status"],"scores":scores,"selected_ids":[m["candidate_ids"][i] for i in idx[:5]],"reused_from":str(RAW.relative_to(ROOT)),"historical_raw_sha256":fsha(RAW),"native_request_sha256":o.get("native_input_sha256"),"physical_forward_count":"HISTORICAL_POINTWISE_TRACE_REUSED","labels_joined":False})
    path=V13/"raw/qwen3_reranker_0_6b_reused.jsonl"; path.parent.mkdir(parents=True,exist_ok=True); path.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in out),encoding="utf-8")
    meta={"policy":"qwen3_reranker_0_6b","rows":len(out),"valid":sum(x["status"] in ("PASS","ok") for x in out),"failed":sum(x["status"] not in ("PASS","ok") for x in out),"source_raw_sha256":fsha(RAW),"raw_sha256":fsha(path),"labels_joined":False,"physical_model_calls":0,"reason":"pointwise score reuse is order-safe; IDs and scores are remapped offline"}
    (path.with_suffix(".meta.json")).write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
