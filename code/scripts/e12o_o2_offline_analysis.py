#!/usr/bin/env python3
"""Offline E12-O2 scorer.  Raw native traces are frozen before this script joins labels."""
from __future__ import annotations
import csv, hashlib, json, math, random, statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RANK=ROOT/"results/non_jev/v2/e12_o_final_v1_0/02_ranking/e12o_o2_attempt_01"
BRIGHT=ROOT/"results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
BRIGHT_COV=ROOT/"results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_POSTFREEZE_KNOWN_POSITIVE_COVERAGE.jsonl"
GARAGE=ROOT/"results/non_jev/v2/e6_v1_0/e6_2_r4c_erratum_and_garage_freeze/E6_2_R4C_GARAGE_EXPLORATORY_INPUTS.jsonl"
GARAGE_LABELS=ROOT/"results/non_jev/v2/e6_v1_0/e6_2_r4c_erratum_and_garage_freeze/E6_2_R4C_GARAGE_LABEL_PROJECTION.json"
E9=ROOT/"results/non_jev/v2/e9_bright_v1_0/02_native"
E6=ROOT/"results/non_jev/v2/e6_v1_0/e6_2_r4d_garage_native_inference"
OUT=ROOT/"results/non_jev/v2/e12_o_final_v1_0/02_ranking/offline_analysis_v2"

def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def sha_file(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def read_jsonl(p): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def candidate_ids(view):
    """Return canonical candidate order across the frozen dataset schemas."""
    return list(view.get("candidate_ids", view.get("candidate_order", [])))
def bright_views(): return read_jsonl(BRIGHT)
def load_bright_gold():
    import pyarrow as pa, pyarrow.ipc as ipc
    root=ROOT/"data/external/e9_bright_v1_0/hf_cache/xlangai___bright/examples/0.0.0/3066d29c9651a576c8aba4832d249807b181ecae"
    out={}
    for p in root.glob("bright-*.arrow"):
        domain=p.stem.removeprefix("bright-")
        tab=ipc.open_stream(pa.memory_map(str(p),"r")).read_all()
        for row in tab.to_pylist(): out[f"{domain}:{row['id']}"]={"gold_ids":set(row.get("gold_ids") or []),"gold_count":len(row.get("gold_ids") or [])}
    return out
def garage_views(): return read_jsonl(GARAGE)
def load_garage_labels(): return {r["sample_id"]:r for r in json.loads(GARAGE_LABELS.read_text())["rows"] if r.get("partition")=="EXPLORATORY"}
def read_outputs():
    out={}
    for p in RANK.glob("*_NATIVE_RAW_V2.jsonl"):
        policy=p.name.split("_NATIVE_RAW_V2")[0]
        out[policy]={r["canonical_id"]:r for r in read_jsonl(p)}
    return out
def hist_bright():
    paths={"jina_v3_5":E9/"BRIGHT_NATIVE_RAW_OUTPUTS_jina_v3_5.jsonl","qwen3_reranker_0_6b":E9/"BRIGHT_NATIVE_RAW_OUTPUTS_qwen3_reranker_0_6b.jsonl","pplx_decider":E9/"BRIGHT_NATIVE_RAW_OUTPUTS_pplx_decider.jsonl","liquid_d1_3b":E9/"BRIGHT_NATIVE_RAW_OUTPUTS_liquid_d1_3b.jsonl"}
    return {k:{f"{r['domain']}:{r['query_id']}":r for r in read_jsonl(p)} for k,p in paths.items() if p.exists()}
def hist_garage():
    p=E6/"E6_2_R4D_NATIVE_SELECTIONS.jsonl"
    if not p.exists(): return {}
    rows=read_jsonl(p); out=defaultdict(dict)
    for r in rows:
        pol=r.get("model",r.get("policy_id", "unknown")); cid=r.get("sample_id",r.get("canonical_id")); out[pol][cid]=r
    return out
def valid_selected(r): return r and r.get("status") in ("PASS","ok","ATTEMPTED_VALID") and len(r.get("selected_ids",[]))==5 and len(set(r.get("selected_ids",[])))==5
def ndcg(selected,gold):
    if not gold: return 0.0
    dcg=sum((1.0/math.log2(i+2)) for i,x in enumerate(selected) if x in gold)
    idcg=sum(1.0/math.log2(i+2) for i in range(min(5,len(gold))))
    return dcg/idcg if idcg else 0.0
def boot(values,domains=None,n=5000,seed=731):
    rng=random.Random(seed); vals=list(values)
    if not vals: return [None,None]
    if domains:
        groups=defaultdict(list)
        for v,d in zip(vals,domains): groups[d].append(v)
        keys=list(groups); samples=[]
        for _ in range(n): samples.append(sum(statistics.mean(groups[k]) for k in (rng.choice(keys) for _ in keys))/len(keys))
    else: samples=[statistics.mean(rng.choices(vals,k=len(vals))) for _ in range(n)]
    samples.sort(); return [samples[int(.025*n)],samples[int(.975*n)]]
def main():
    OUT.mkdir(parents=True,exist_ok=True); views={f"BRIGHT:{v['domain']}:{v['query_id']}":v for v in bright_views()}; views.update({f"GaRAGe:{v['sample_id']}":v for v in garage_views()}); bg=load_bright_gold(); gl=load_garage_labels(); new=read_outputs(); hb=hist_bright(); hg=hist_garage(); allout={**hb,**new}
    # Historical GaRAGe outputs use the same sample_id identity but live in a
    # separate manifest. Merge them by policy without replacing new E12-O
    # traces, so paired GaRAGe comparisons remain explicit historical reuse.
    for policy, byid in hg.items():
        allout.setdefault(policy, {}).update(byid)
    # Historical names may differ; preserve them as a separate explicit source.
    per=[]; summary=[]
    for policy,byid in sorted(allout.items()):
      for dataset in ("BRIGHT","GaRAGe"):
        items=[]
        for key,v in views.items():
          if not key.startswith(dataset+":"): continue
          cid=key.split(":",1)[1] if dataset=="BRIGHT" else key.split(":",1)[1]
          r=byid.get(cid)
          if not r: continue
          sel=r.get("selected_ids",[]) if valid_selected(r) else []
          if dataset=="BRIGHT":
            g=bg.get(cid,{"gold_ids":set(),"gold_count":0}); metric=ndcg(sel,g["gold_ids"]); known=bool(set(v["candidate_ids"]) & g["gold_ids"]); domain=v["domain"]; support=None
          else:
            lab=gl.get(cid); order=candidate_ids(v); rel=(lab or {}).get("evidence_relevant",[]); cor=(lab or {}).get("evidence_correct",[]); idx={x:i for i,x in enumerate(order)}; chosen=[idx[x] for x in sel if x in idx]; rel5=sum(1 for i in chosen if i<len(rel) and rel[i]=="YES")/5.0; support=1.0 if any(i<len(cor) and cor[i]=="ANSWER-THE-QUESTION" for i in chosen) else 0.0; metric=rel5; known=None; domain="garage"
          row={"policy_id":policy,"dataset":dataset,"canonical_id":cid,"domain":domain,"status":r.get("status","MISSING"),"valid":bool(sel),"metric":metric,"known_positive_in_top20":known,"selected_ids":sel,"latency_ms":r.get("elapsed_ms"),"grounding_support_at5":support}
          per.append(row); items.append(row)
        if items:
          vals=[x["metric"] for x in items]; domains=[x["domain"] for x in items]; valid=sum(x["valid"] for x in items); summary.append({"policy_id":policy,"dataset":dataset,"n":len(items),"valid":valid,"failure_rate":1-valid/len(items),"mean_metric_all_attempts":statistics.mean(vals),"mean_metric_valid_only":statistics.mean([x["metric"] for x in items if x["valid"]]) if valid else None,"ci_query_or_domain_bootstrap":boot(vals,domains if dataset=="BRIGHT" else None),"grounding_support_at5_all_attempts":statistics.mean([x["grounding_support_at5"] or 0 for x in items]) if dataset=="GaRAGe" else None,"source":"new_raw_or_authenticated_historical_reuse"})
    # Pair new policies to Jina on common IDs, but do not pretend a missing
    # historical view is a zero score.
    paired=[]
    for dataset in ("BRIGHT","GaRAGe"):
      jina={x["canonical_id"]:x for x in per if x["policy_id"]=="jina_v3_5" and x["dataset"]==dataset}
      for policy in sorted(set(x["policy_id"] for x in per)-{"jina_v3_5"}):
        cur={x["canonical_id"]:x for x in per if x["policy_id"]==policy and x["dataset"]==dataset}; common=sorted(set(cur)&set(jina)); dif=[cur[k]["metric"]-jina[k]["metric"] for k in common]; ds=[cur[k]["domain"] for k in common]
        paired.append({"dataset":dataset,"policy_id":policy,"anchor":"jina_v3_5","common_n":len(common),"effect_mean":statistics.mean(dif) if dif else None,"ci_95":boot(dif,ds if dataset=="BRIGHT" else None) if dif else [None,None],"positive":sum(x>0 for x in dif),"negative":sum(x<0 for x in dif),"unchanged":sum(x==0 for x in dif),"source_group_sensitivity":"NOT_AVAILABLE" if dataset=="GaRAGe" else "domain_cluster_bootstrap"})
    (OUT/"E12O_PER_QUERY_METRICS_V2.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in per)); (OUT/"E12O_SCORECARD_V2.json").write_text(json.dumps(summary,indent=2)+"\n"); (OUT/"E12O_PAIRED_EFFECTS_AND_CIS_V2.json").write_text(json.dumps(paired,indent=2)+"\n")
    (OUT/"E12O_SCORER_PROVENANCE_V2.json").write_text(json.dumps({"labels_joined_after_raw_freeze":True,"bright_gold_source":"local cached BRIGHT examples Arrow; gold_ids joined offline","garage_label_source":str(GARAGE_LABELS.relative_to(ROOT)),"raw_files":{str(p.relative_to(ROOT)):sha_file(p) for p in RANK.glob("*_NATIVE_RAW_V2.jsonl")},"bootstrap":{"n":5000,"seed":731}},indent=2)+"\n")
    print(json.dumps({"summary_rows":len(summary),"paired_rows":len(paired),"per_query_rows":len(per)},indent=2))
if __name__=="__main__": main()
