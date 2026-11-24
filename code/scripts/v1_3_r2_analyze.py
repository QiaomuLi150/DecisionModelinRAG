#!/usr/bin/env python3
"""Offline R2 label join, known-positive nDCG scoring and order summaries."""
from __future__ import annotations
import csv, hashlib, json, math, itertools
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
V13=ROOT/"results/v1_3/r2_order_sensitivity"
E9=ROOT/"results/non_jev/v2/e9_bright_v1_0"
SEED=731; DRAWS=40000
def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def fsha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def ndcg(selected,gold):
    ideal_n=min(len(gold),5)
    if ideal_n==0:return 0.0
    vals=[1.0 if x in gold else 0.0 for x in selected[:5]]
    dcg=sum(v/math.log2(i+2) for i,v in enumerate(vals)); ideal=sum(1.0/math.log2(i+2) for i in range(ideal_n)); return dcg/ideal if ideal else 0.0
def load_rows(path): return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]
def historical(policy):
    if policy=="pplx_budget": p=E9.parent/"e11_v1_0/02_native/E11_A_PPLX_BUDGETED_RAW.jsonl"
    else: p=E9/"02_native/BRIGHT_NATIVE_RAW_OUTPUTS_jina_v3_5.jsonl"
    rows=[]
    for x in load_rows(p):
        cid=f"{x['domain']}:{x['query_id']}"; ids=x.get("candidate_ids",[]); selected=x.get("selected_ids",[])
        rows.append({"canonical_id":cid,"domain":x["domain"],"query_id":str(x["query_id"]),"policy":policy,"order_id":"original","candidate_ids":ids,"status":x.get("status"),"selected_ids":selected,"reused_from":str(p.relative_to(ROOT)),"historical_raw_sha256":fsha(p),"labels_joined":False})
    return rows
def main():
    gold_manifest=json.loads((E9/"00_admission/BRIGHT_EXAMPLE_ID_MANIFEST.json").read_text())
    gold={f"{x['domain']}:{x['id']}":set(map(str,x.get("gold_ids",[]))) for x in gold_manifest["items"]}
    policies={"pplx_budget":historical("pplx_budget"),"jina_full":historical("jina_full"),"qwen3_reranker_0_6b":[]}
    qpath=V13/"raw/qwen3_reranker_0_6b_reused.jsonl"; policies["qwen3_reranker_0_6b"]=load_rows(qpath)
    for p in ("pplx_budget","jina_full"):
        for path in sorted((V13/"raw").glob(f"{p}_gpu*_orders_*.jsonl")):
            if "_perm_seed_" not in path.stem:  # one-row smoke files are kept as preflight evidence
                continue
            policies[p].extend(load_rows(path))
    all_scores=[]; perquery=[]; summaries=[]; effects={}
    for policy,rows in policies.items():
        unique={(x["canonical_id"],x["order_id"]) for x in rows}
        expected=1384*7
        if len(unique)!=len(rows): raise RuntimeError(f"duplicate rows for {policy}: {len(unique)} vs {len(rows)}")
        valid_by_key={}
        for x in rows:
            g=gold[x["canonical_id"]]; ok=x.get("status") in ("PASS","ok") and len(x.get("selected_ids",[]))==5 and len(set(x.get("selected_ids",[])))==5
            m=ndcg(x.get("selected_ids",[]) if ok else [],g); sh=sha(x["selected_ids"]) if ok else ""
            perquery.append({"canonical_id":x["canonical_id"],"domain":x["domain"],"policy":policy,"order_id":x["order_id"],"valid":str(bool(ok)).lower(),"selected_ids_hash":sh})
            all_scores.append({"canonical_id":x["canonical_id"],"domain":x["domain"],"policy":policy,"order_id":x["order_id"],"metric":m,"valid":str(bool(ok)).lower()})
            valid_by_key[(x["canonical_id"],x["order_id"])]=(m,ok,set(x.get("selected_ids",[])))
        if len(rows)!=expected: summaries.append({"policy":policy,"status":"PARTIAL","rows":len(rows),"expected":expected})
        order_means=[]
        for oid in ["original","reversed"]+[f"perm_seed_{i}" for i in range(1,6)]:
            vals=[valid_by_key[(cid,oid)][0] for cid in gold if (cid,oid) in valid_by_key]; order_means.append(float(np.mean(vals)) if vals else None)
        changes=[]
        for cid in gold:
            if (cid,"original") not in valid_by_key: continue
            base=valid_by_key[(cid,"original")][2]
            changes.extend(valid_by_key[(cid,oid)][2]!=base for oid in ["reversed"]+[f"perm_seed_{i}" for i in range(1,6)] if (cid,oid) in valid_by_key)
        summaries.append({"policy":policy,"status":"COMPLETE" if len(rows)==expected else "PARTIAL","rows":len(rows),"expected":expected,"mean_over_orders":float(np.mean(order_means)) if order_means else None,"sd_over_orders":float(np.std(order_means,ddof=1)) if len(order_means)>1 else None,"min":min(order_means) if order_means else None,"max":max(order_means) if order_means else None,"top5_set_change_rate_vs_original":float(np.mean(changes)) if changes else None,"valid_rows":sum(x["valid"]=="true" for x in all_scores if x["policy"]==policy),"failed_rows":sum(x["valid"]!="true" for x in all_scores if x["policy"]==policy)})
    write_csv(V13/"export/order_perquery.csv",perquery,["canonical_id","domain","policy","order_id","valid","selected_ids_hash"])
    write_csv(V13/"export/order_scores.csv",all_scores,["canonical_id","domain","policy","order_id","metric","valid"])
    write_csv(V13/"export/order_summary_detailed.csv",summaries,["policy","status","rows","expected","mean_over_orders","sd_over_orders","min","max","top5_set_change_rate_vs_original","valid_rows","failed_rows"])
    write_csv(V13/"export/order_summary.csv",[{k:x.get(k) for k in ["policy","mean_over_orders","sd_over_orders","min","max","top5_set_change_rate_vs_original"]} for x in summaries],["policy","mean_over_orders","sd_over_orders","min","max","top5_set_change_rate_vs_original"])
    # Borda over six non-original orders plus original: ranking each query by total rank score.
    borda=[]
    for policy in policies:
        if policy not in ("pplx_budget","jina_full","qwen3_reranker_0_6b"): continue
        raw=[x for x in (historical(policy) if policy in ("pplx_budget","jina_full") else load_rows(qpath))]
        for path in sorted((V13/"raw").glob(f"{policy}_gpu*_orders_*.jsonl")):
            if "_perm_seed_" in path.stem:
                raw.extend(load_rows(path))
        groups=defaultdict(list)
        for x in raw: groups[x["canonical_id"]].append(x)
        for cid,xs in groups.items():
            if len(xs)!=7: continue
            total=defaultdict(float)
            for x in xs:
                ids=x.get("candidate_ids",[]); scores=x.get("scores",[])
                if len(scores)!=len(ids): continue
                rank=sorted(range(len(ids)),key=lambda i:(-float(scores[i]),i,ids[i]))
                for r,i in enumerate(rank): total[ids[i]] += len(ids)-r
            ranked=[x for x,_ in sorted(total.items(),key=lambda kv:(-kv[1],kv[0]))[:5]]
            borda.append({"canonical_id":cid,"domain":cid.split(":",1)[0],"policy":policy,"aggregation":"borda_original_plus_six_permutations","valid":len(ranked)==5,"selected_ids_hash":sha(ranked),"selected_ids":ranked})
    write_csv(V13/"export/order_borda.csv",[{k:v for k,v in x.items() if k!="selected_ids"} for x in borda],["canonical_id","domain","policy","aggregation","valid","selected_ids_hash"])
    # Domain bootstrap for original-vs-Borda where both are available; no family-wise claims beyond this exploratory file.
    boot={}
    scoremap={(x["canonical_id"],x["policy"],x["order_id"]):float(x["metric"]) for x in all_scores}
    domains=sorted({x["domain"] for x in all_scores}); rng=np.random.Generator(np.random.PCG64(SEED))
    for policy in policies:
        if policy not in ("pplx_budget","jina_full","qwen3_reranker_0_6b"): continue
        vals=[]; grouped=defaultdict(list)
        for cid in gold:
            d=cid.split(":",1)[0]
            a=scoremap.get((cid,policy,"original")); b=scoremap.get((cid,"jina_full","original"))
            if a is not None and b is not None: grouped[d].append(a-b)
        domain_effects=[float(np.mean(grouped[d])) for d in domains if grouped.get(d)]
        present_domains=[d for d in domains if grouped.get(d)]
        obs=float(np.mean([v for d in domains for v in grouped[d]])) if grouped else None
        if domain_effects:
            # Domain bootstrap: resample the 12 domain means with replacement.
            idx_rng=np.random.Generator(np.random.PCG64(SEED))
            idx=idx_rng.integers(0,len(domain_effects),size=(DRAWS,len(domain_effects)))
            vals=np.mean(np.asarray(domain_effects)[idx],axis=1).tolist()
            # Exact 2^12 sign-flip randomization over domain-level effects.
            sign_stats=[]
            for signs in itertools.product((-1.0,1.0),repeat=len(domain_effects)):
                sign_stats.append(float(np.mean(np.asarray(signs)*np.asarray(domain_effects))))
            abs_obs=abs(float(np.mean(domain_effects)))
            sign_p=float(np.mean(np.abs(sign_stats)>=abs_obs-1e-15))
            sign_ci=[float(np.quantile(sign_stats,.025)),float(np.quantile(sign_stats,.975))]
        else:
            sign_p=None; sign_ci=None
        boot[policy]={"n":sum(len(v) for v in grouped.values()),"observed_query_weighted":obs,"observed_domain_weighted":float(np.mean(domain_effects)) if domain_effects else None,"ci95_whole_domain_bootstrap":[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))] if vals else None,"draws":DRAWS,"seed":SEED,"exact_2_to_12_sign_flip_p_two_sided":sign_p,"exact_2_to_12_sign_flip_ci95":sign_ci,"domain_count":len(present_domains),"domains":{d:{"n":len(grouped[d]),"effect":float(np.mean(grouped[d]))} for d in domains if grouped.get(d)}}
    (V13/"export/order_bootstrap.json").write_text(json.dumps(boot,indent=2)+"\n")
    print(json.dumps({"policies":{k:{"rows":len(v),"valid":sum(x.get("status") in ("PASS","ok") for x in v)} for k,v in policies.items()},"summary_rows":len(summaries),"borda_rows":len(borda),"labels_joined_offline":True},indent=2))
if __name__=="__main__": main()
