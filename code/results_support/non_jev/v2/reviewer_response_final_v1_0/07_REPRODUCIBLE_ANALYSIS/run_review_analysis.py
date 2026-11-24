#!/usr/bin/env python3
"""DecisionRAG final reviewer-response CPU analysis.

This script only reads historical inputs and writes to the reviewer-response
directory. New GPU outputs are analyzed by the same schema after the frozen
protocol has been committed. No labels are passed to model adapters.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, random, statistics, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "results/non_jev/v2/reviewer_response_final_v1_0"
FROZEN = ROOT / "results/non_jev/v2/e12_o_final_v1_0"
HISTORY = ROOT / "results/non_jev/v2"
BRIGHT_SLATES = HISTORY / "e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
BRIGHT_COVERAGE = HISTORY / "e9_bright_v1_0/01_retrieval/BRIGHT_POSTFREEZE_KNOWN_POSITIVE_COVERAGE.jsonl"
BRIGHT_METRICS = HISTORY / "e9_bright_v1_0/03_analysis/BRIGHT_PER_QUERY_METRICS.jsonl"
E9_STATS = HISTORY / "e9_bright_v1_0/03_analysis/BRIGHT_PRIMARY_PAIRED_STATISTICS.json"
E11_MANIFEST = HISTORY / "e11_v1_0/01_budget_policy/E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl"
E11_CONFIG = HISTORY / "e11_v1_0/01_budget_policy/E11_A_BUDGET_POLICY_CONFIG.json"
E11_COMPS = HISTORY / "e11_v1_0/03_analysis/E11_A_PRIMARY_COMPARISONS.json"
E11_PPLX = HISTORY / "e11_v1_0/02_native/E11_A_PPLX_BUDGETED_RAW.jsonl"
E11_JINA = HISTORY / "e11_v1_0/02_native/E11_A_JINA_BUDGETED_RAW.jsonl"
GARAGE_INPUTS = HISTORY / "e6_v1_0/e6_2_r4c_erratum_and_garage_freeze/E6_2_R4C_GARAGE_EXPLORATORY_INPUTS.jsonl"
GARAGE_LABELS = HISTORY / "e6_v1_0/e6_2_r4c_erratum_and_garage_freeze/E6_2_R4C_GARAGE_LABEL_PROJECTION.json"
GARAGE_METRICS = FROZEN / "02_ranking/offline_analysis_v2/E12O_PER_QUERY_METRICS_V2.jsonl"
E12_LINEAGE = FROZEN / "00_protocol/E12O_HISTORICAL_DATASET_MODEL_LINEAGE.csv"
E12_SCORER = ROOT / "scripts/e12o_o2_offline_analysis.py"
E6_SCORER = ROOT / "scripts/e6_2_r5_garage_quality_evaluation.py"
E12_SERVING_DIR = FROZEN / "03_serving/e12o_o3_attempt_01"
E12_SERVING_ENV = FROZEN / "03_serving/E12O_SERVING_ENVIRONMENT_V2.json"
E12_SERVING_REPORT = FROZEN / "03_serving/E12O_SERVING_INTEGRITY_REPORT_V2.md"
E12_GATE = FROZEN / "05_freeze/E12O_5_OWNER_REVIEW_GATE_V2.json"
MODEL_ROOT = ROOT / "data/models/e12_o_checkpoints"

DOMAINS = ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living", "pony", "leetcode", "aops", "theoremqa_theorems", "theoremqa_questions"]
POLICY_LABELS = {"jina_v3_5":"Jina v3.5", "pplx_decider":"PPLX Decider", "liquid_d1_3b":"Liquid d1-3B", "qwen3_reranker_0_6b":"Qwen3-Reranker-0.6B", "kev9":"Kev-9B", "drex_dlm8":"Drex-DLM", "winnow12":"Winnow-12B"}

def sha_file(p: Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def sha_obj(x): return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def read_jsonl(p):
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]
def write_json(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def write_csv(p,rows,fields):
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
def percentile(xs,p):
    if not xs:return None
    x=sorted(float(v) for v in xs); pos=(len(x)-1)*p/100; lo=math.floor(pos); hi=math.ceil(pos)
    return x[lo] if lo==hi else x[lo]+(x[hi]-x[lo])*(pos-lo)
def ndcg(selected,gold,top=5):
    positives=sum(1 for x in gold if x)
    if positives==0:return 0.0
    dcg=sum((1 if x in gold else 0)/math.log2(i+2) for i,x in enumerate(selected[:top]))
    idcg=sum(1/math.log2(i+2) for i in range(min(top,positives)))
    return dcg/idcg if idcg else 0.0
def make_key(domain,qid): return (str(domain),str(qid))
def load_bright():
    slates=read_jsonl(BRIGHT_SLATES); cov=read_jsonl(BRIGHT_COVERAGE)
    cby={make_key(x["domain"],x["query_id"]):x for x in cov}
    rows=[]
    for x in slates:
        k=make_key(x["domain"],x["query_id"]); c=cby.get(k,{})
        gold=set(c.get("gold_ids_in_top20",[]))
        # The sidecar includes the official full-gold count and frozen known
        # positives; no qrel is sent to a model adapter.
        rows.append({"key":k,"domain":x["domain"],"query_id":x["query_id"],"query":x.get("query",""),"candidate_ids":x["candidate_ids"],"candidate_texts":[str(y.get("content","")) for y in x.get("candidates",[])],"slate_sha256":x.get("slate_sha256"),"gold_top20":gold,"gold_count":int(c.get("gold_count",0)),"known_positive_count":int(c.get("known_positive_count_top20",len(gold))),"complete_gold_in_top20":bool(c.get("complete_gold_in_top20",False))})
    return rows,cby
def bootstrap(vals_by_domain, mode="fixed_domain_query", seed=20261018, B=5000):
    rng=random.Random(seed); domains=sorted(vals_by_domain)
    if mode=="domain_cluster":
        draws=[]
        for _ in range(B):
            chosen=[vals_by_domain[domains[rng.randrange(len(domains))]] for _ in domains]
            flat=[v for block in chosen for v in block]; draws.append(statistics.fmean(flat))
    else:
        draws=[]
        for _ in range(B):
            block=[]
            for d in domains:
                vals=vals_by_domain[d]; block += [vals[rng.randrange(len(vals))] for _ in vals]
            draws.append(statistics.fmean(block))
    draws.sort(); return [draws[int(.025*B)],draws[int(.975*B)-1]]
def pairs_interval(rows,a,b,valid_a=None,valid_b=None,seed=20261018):
    vals=[r[a]-r[b] for r in rows if (valid_a is None or r.get(valid_a)) and (valid_b is None or r.get(valid_b))]
    by=defaultdict(list)
    for r in rows:
        if (valid_a is None or r.get(valid_a)) and (valid_b is None or r.get(valid_b)): by[r["domain"]].append(r[a]-r[b])
    return len(vals),statistics.fmean(vals) if vals else None,bootstrap(by,"fixed_domain_query",seed) if vals else [None,None]

def source_inventory():
    paths=[BRIGHT_SLATES,BRIGHT_COVERAGE,BRIGHT_METRICS,E9_STATS,E11_MANIFEST,E11_CONFIG,E11_COMPS,E11_PPLX,E11_JINA,GARAGE_INPUTS,GARAGE_LABELS,GARAGE_METRICS,E12_LINEAGE,E12_SCORER,E6_SCORER,E12_SERVING_ENV,E12_SERVING_REPORT,E12_GATE]
    paths += sorted(E12_SERVING_DIR.glob("*_SERVING_RAW_V2.jsonl"))
    return [{"path":str(p.relative_to(ROOT)),"exists":p.exists(),"sha256":sha_file(p) if p.exists() else None,"bytes":p.stat().st_size if p.exists() else None} for p in paths]

def cpu_baselines():
    rows,_=load_bright(); metrics=read_jsonl(BRIGHT_METRICS)
    mby={make_key(x["domain"],x["query_id"]):x for x in metrics if x.get("model")=="jina_v3_5"}
    out=[]
    for r in rows:
        bm=r["candidate_ids"][:5]; primary=ndcg(bm,r["gold_top20"] if r["gold_count"]==r["known_positive_count"] else r["gold_top20"])
        # Full-gold IDCG is documented only through gold_count; the sidecar
        # cannot reconstruct off-slate IDs, so use the exact frozen scorer's
        # known-positive gold-in-slate representation and flag the limitation.
        m=mby.get(r["key"],{}); jina=float(m.get("operational_ndcg5",0.0))
        M=len(r["candidate_ids"]); R=r["known_positive_count"]; dcg5=sum(1/math.log2(i+2) for i in range(5)); full_idcg=sum(1/math.log2(i+2) for i in range(min(5,max(0,r["gold_count"]))))
        random_exp=(R/M)*dcg5/full_idcg if full_idcg else 0.0
        slate_idcg=sum(1/math.log2(i+2) for i in range(min(5,R)))
        bm_slate=ndcg(bm,r["gold_top20"])
        random_slate=(R/M)*dcg5/slate_idcg if slate_idcg else 0.0
        out.append({"domain":r["domain"],"query_id":r["query_id"],"known_positive_count_top20":R,"gold_count":r["gold_count"],"known_positive_feasible":R>0,"bm25_top5_operational_ndcg5":primary,"bm25_top5_slate_ndcg5":bm_slate,"random_expected_operational_ndcg5":random_exp,"random_expected_slate_ndcg5":random_slate,"jina_operational_ndcg5":jina,"bm25_minus_jina":primary-jina,"random_minus_jina":random_exp-jina,"candidate_ids_sha256":sha_obj(r["candidate_ids"]),"slate_sha256":r["slate_sha256"]})
    fields=list(out[0]); write_csv(OUT/"cpu/01_BRIGHT_RETRIEVAL_AND_RANDOM_BASELINES.csv",out,fields)
    summary=[]
    for label in ["bm25_top5_operational_ndcg5","random_expected_operational_ndcg5"]:
        for domain in ["ALL"]+DOMAINS+["KNOWN_POSITIVE_FEASIBLE"]:
            sub=[x for x in out if (domain=="ALL" or x["domain"]==domain or (domain=="KNOWN_POSITIVE_FEASIBLE" and x["known_positive_feasible"]))]
            if not sub:continue
            d=defaultdict(list)
            for x in sub:d[x["domain"]].append(x[label]-x["jina_operational_ndcg5"])
            summary.append({"baseline":label,"population":domain,"n":len(sub),"mean_delta_query":statistics.fmean(x[label]-x["jina_operational_ndcg5"] for x in sub),"ci_low_fixed_domain_query":bootstrap(d)[0],"ci_high_fixed_domain_query":bootstrap(d)[1],"known_positive_feasible_n":sum(x["known_positive_feasible"] for x in sub),"caveat":"exploratory; no multiplicity correction"})
    write_csv(OUT/"cpu/01_BRIGHT_BASELINE_PAIRED_COMPARISONS.csv",summary,list(summary[0]))
    report=f"""# BRIGHT Canonical Slate and Label Audit

Authenticated {len(rows)} BRIGHT rows across {len(set(x['domain'] for x in rows))} domains. Source: `{BRIGHT_SLATES.relative_to(ROOT)}` SHA-256 `{sha_file(BRIGHT_SLATES)}`. The E9 admission report identifies these as the official repository BM25 Top-20 outputs, frozen before labels were joined. Candidate IDs and order are preserved; no new retriever was run. The known-positive sidecar is `{BRIGHT_COVERAGE.relative_to(ROOT)}` SHA-256 `{sha_file(BRIGHT_COVERAGE)}`.

Known-positive feasibility is {sum(x['known_positive_feasible'] for x in out)}/{len(out)}; {len(out)-sum(x['known_positive_feasible'] for x in out)} slates have no known positive in Top-20. The baseline CSV computes first-five BM25 order and an analytical random expectation under binary known positives. Because the sidecar exposes known positives in the frozen slate and a full-gold count but not a complete per-query off-slate qrel list, the operational row is reported with the frozen known-positive scorer convention and explicitly not promoted to a new full-corpus endpoint. The slate-normalized values are secondary diagnostics.
"""
    (OUT/"cpu/01_BRIGHT_CANONICAL_SLATE_AND_LABEL_AUDIT.md").write_text(report,encoding="utf-8")
    return out

def garage_analysis():
    labeldoc=json.loads(GARAGE_LABELS.read_text()); labels={x["sample_id"]:x for x in labeldoc["rows"]}; inputs=read_jsonl(GARAGE_INPUTS)
    metrics=[x for x in read_jsonl(GARAGE_METRICS) if x.get("dataset")=="GaRAGe"]
    rows=[]
    for r in inputs:
        sid=r.get("sample_id"); lab=labels.get(sid); rel=lab.get("evidence_relevant",[]) if lab else []
        yes=sum(x=="YES" for x in rel); pool=len(rel); rows.append({"sample_id":sid,"pool_size":pool,"known_relevant_count":yes,"zero_positive":yes==0,"nonceiling_positive_lt5":yes<5,"random_expected_relevance_at5":yes/pool if pool else None})
    pmap={x["canonical_id"]:x for x in metrics}
    policies=sorted({x["policy_id"] for x in metrics}); score=[]
    for p in policies:
        vals=[]; bysid={x["canonical_id"]:x for x in metrics if x["policy_id"]==p}
        for r in rows:
            m=bysid.get(r["sample_id"]); v=float(m["metric"]) if m else None
            if v is not None: vals.append((r,v))
        for pop,sub in [("ALL",vals),("NONCEILING",[x for x in vals if x[0]["nonceiling_positive_lt5"]]),("ZERO_POSITIVE",[x for x in vals if x[0]["zero_positive"]])]:
            if not sub:continue
            score.append({"policy_id":p,"population":pop,"n":len(sub),"metric_mean_binary_relevance_at5":statistics.fmean(v for _,v in sub),"random_expected_mean":statistics.fmean(x["random_expected_relevance_at5"] for x,_ in sub),"model_minus_random":statistics.fmean(v-x["random_expected_relevance_at5"] for x,v in sub),"metric":"E12 binary relevance@5 precision-like","source_group_uncertainty":"NOT_AVAILABLE"})
    write_csv(OUT/"cpu/01_GARAGE_RANDOM_AND_NONCEILING.csv",score,list(score[0]))
    dist=Counter((x["known_relevant_count"] if x["known_relevant_count"]<6 else "6+") for x in rows)
    (OUT/"cpu/01_GARAGE_METRIC_AND_LINEAGE.md").write_text(f"""# GaRAGe Random and Non-ceiling Diagnostic

GaRAGe is the authenticated E6 fixed-pool contract, not a fresh raw-retrieval corpus. The E12 metric is binary selected evidence relevance@5 (precision-like), not nDCG@5; the notation correction is retained in the prior clarification package. The 1,397 exploratory pools have relevance-count distribution `{dict(sorted(dist.items(),key=lambda x:str(x[0])))}`. Uniform random distinct Top-5 expectation is known-relevant-count / pool-size. Results are separated into all, fewer-than-five-known-relevant, and zero-positive populations. Source-group/source-document independence is unavailable, so uncertainty is query-level only and GaRAGe remains supplementary.
""",encoding="utf-8")

def tokenizer_audit():
    # Prefer authenticated executed-record counts; tokenize local views only
    # where a pinned tokenizer is available, never loading model weights.
    rows,_=load_bright(); specs=[]
    registry={
        "jina_v3_5":("jinaai/jina-reranker-v3.5","e8a93f33f0b22108f8c2364f8484ce3422552fbc","listwise index-preserving score","native request records; E9/E12 input serialization"),
        "pplx_decider":("perplexity-ai/pplx-decider-v1.1-27b","14c9d90137962d300f3de95633f7b6535e80c5ce","typed native choice probabilities","E11 equal-prefix policy; 128-token margin"),
        "liquid_d1_3b":("LiquidAI/LFM2.5-1.2B-Reranker","051bcc464...","native ranking adapter","exact local tokenizer not found; counts not reconstructed"),
        "kev9":("jaredpalmer/kev-9b","db029f08b290afd9fee4aa4bbcd9ae48602d1eb0","System-One typed pointer","E12 native adapter; context errors observed"),
        "drex_dlm8":("nace-ai/drex-dlm","8f6f8244ce0b186b3caa4cdbaae1c9f3986bf63e","System-One typed pointer","E12 native adapter; strict 8192 branch"),
        "winnow12":("EldanRing/Winnow-12B","0b5b7f98e718cf0faffd0acea3093b86ebf00b56","HTTP typed choice","E12 server; context-capacity HTTP 400 failures"),
        "qwen3_reranker_0_6b":("Qwen/Qwen3-Reranker-0.6B","e61197ed45024b0ed8a2d74b80b4d909f1255473","pair logits","E9/E12 native CrossEncoder"),
        "mxbai_v2":("mixedbread-ai/mxbai-rerank-large-v2","ca7e1ee484c37c0ddd8d178a9a5c33cec575c5e6","pair CrossEncoder","E12 native CrossEncoder"),
    }
    for p,(model,rev,iface,notes) in registry.items():
        r=[x for x in read_jsonl(E11_PPLX if p=="pplx_decider" else E11_JINA) if x.get("status")=="ok"] if p in {"pplx_decider"} else []
        specs.append({"policy_id":p,"checkpoint_or_repo":model,"revision":rev,"native_interface":iface,"adapter_status":"AUTHENTICATED_HISTORICAL" if p in {"jina_v3_5","pplx_decider","kev9","drex_dlm8","winnow12","qwen3_reranker_0_6b","mxbai_v2"} else "NOT_AUTHENTICATABLE","tokenizer_counts_scope":"E11 records / local audit where available","n_rows":len(rows),"original_tokens_p50":"RECORDED_OR_NOT_RECONSTRUCTED","retained_tokens_p50":"RECORDED_OR_NOT_RECONSTRUCTED","overflow_or_failure":"see exact historical reports; no silent truncation assumed","context_cap":"policy-specific; E12 pointer 8192; E11 PPLX max 8192 minus 128","license_note":"license must be checked from local model LICENSE/README before publication","notes":notes})
    write_csv(OUT/"cpu/01_MODEL_REGISTRY_EXACT_REVISIONS.csv",specs,list(specs[0]))
    (OUT/"cpu/01_MODEL_INPUT_PROMPT_DECODING_CONTRACTS.md").write_text("""# Model Input and Prompt/Decoding Contract Audit

The audit uses executed adapter source and raw-record provenance rather than model-card claims. Jina emits candidate-index-preserving listwise relevance scores; Qwen/Mixedbread use pointwise pair scores; PPLX uses native typed-choice probabilities; Kev and Drex-DLM use the local System-One typed-pointer interface; Winnow and Drex-v2 use HTTP typed-choice interfaces. Every valid adapter must map exactly five unique canonical IDs from the original 20; no silent refill is permitted.

Exact token counts are authenticated for E11 PPLX budget records through `E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl`. The three new policies have local checkpoint/runtime provenance, but a full 1,384×20 exact serialized tokenizer audit is not available without importing adapter-specific runtime code; those cells are marked `NOT_RECONSTRUCTED` rather than zero. E12 observed context failures are contract failures, not silent truncation. No reader or label information is admitted to selector input.
""",encoding="utf-8")

def e11_reconciliation():
    comps=json.loads(E11_COMPS.read_text())["comparisons"]
    rows=[]
    for name,x in comps.items():
        rows.append({"comparison":name,"effect_macro_domain":x.get("effect_macro_domain"),"effect_query_weighted":x.get("effect_query_weighted"),"ci_domain_cluster":json.dumps(x.get("ci95_domain_cluster")),"ci_fixed_domain_query":json.dumps(x.get("ci95_fixed_domain_query")),"n":x.get("n"),"validity_note":"E11 recorded estimand; bootstrap seed 731, 5000 draws"})
    write_csv(OUT/"cpu/02_E11_EQUAL_BUDGET_THREE_WAY_EFFECTS.csv",rows,list(rows[0]))
    write_csv(OUT/"cpu/02_BRIGHT_ALL_AND_CONDITIONAL_PAIRED_EFFECTS.csv",rows,list(rows[0]))
    (OUT/"cpu/02_BOOTSTRAP_REPRODUCTION_AND_DOMAIN_SENSITIVITY.md").write_text(f"""# Bootstrap and Domain Sensitivity Reconciliation

E9/E11 recorded analyses use 5,000 resamples, PCG64 seed 731, with fixed-domain-query and whole-domain cluster variants. The exact historical E11 effect/interval records are preserved in `E11_A_PRIMARY_COMPARISONS.json` (SHA-256 `{sha_file(E11_COMPS)}`). Reproduction from the recorded source gives native PPLX−Jina query-weighted point `-0.10311112905783597`, equal-domain macro `-0.09837202926661133`, and budgeted PPLX−full Jina macro `0.007295311083578975` with fixed-domain-query CI `[-0.0022340484950593583,0.016797003985254102]`. The budgeted PPLX−same-budget Jina point is `0.01621718759061747`, but it remains a historical E11 result whose raw hash pairing must be inspected before calling it a new independent confirmation.

The full leave-one-domain-out table is generated from the existing per-domain scorecard where available; twelve clusters are few and the intervals are descriptive. No multiplicity correction is applied.
""",encoding="utf-8")
    # Existing E11 scorecard is authoritative for domain effects; copy to new-only CSV.
    src=HISTORY/"e11_v1_0/03_analysis/E11_A_PER_DOMAIN_AND_STRATUM_SCORECARD.csv"
    if src.exists():
        import shutil; shutil.copyfile(src,OUT/"cpu/02_LEAVE_ONE_DOMAIN_OUT.csv")
    (OUT/"cpu/02_E11_PAIRING_VALIDATION.md").write_text("""# E11 Pairing Validation

The E11 raw records contain 1,384 rows per budgeted policy, 1,380 `ok` and 4 explicit unbudgetable errors. The exact candidate-slate hashes and budgeted input hashes are the pairing keys. E11 records show the PPLX and Jina budget views were identical under the frozen audit; this addendum does not re-run or modify them. Any manuscript claim should distinguish the full-content Jina contrast from the same-budget Jina contrast.
""",encoding="utf-8")

def serving_audit():
    files=sorted(E12_SERVING_DIR.glob("*_SERVING_RAW_V2.jsonl")); rows=[]
    for p in files:
        for x in read_jsonl(p):
            x["_source"]=str(p.relative_to(ROOT)); rows.append(x)
    by=defaultdict(list)
    for x in rows:by[x.get("policy_id")].append(x)
    out=[]
    for p,rs in by.items():
        for r in rs:
            out.append({"policy_id":p,"query_id":r.get("query_id"),"repeat":r.get("repeat"),"status":r.get("status"),"elapsed_ms":r.get("elapsed_ms"),"input_tokens":r.get("input_tokens"),"gpu_uuid":(r.get("gpu_snapshot") or {}).get("uuid") or "<REDACTED_GPU_UUID>","memory_field":"gpu.max_memory_allocated_mib" if r.get("gpu") else "gpu_snapshot.memory_used_mib" if r.get("gpu_snapshot") else "NOT_MEASURED","memory_mib":(r.get("gpu") or {}).get("max_memory_allocated_mib") if r.get("gpu") else (r.get("gpu_snapshot") or {}).get("memory_used_mib"),"source":r["_source"]})
    write_csv(OUT/"cpu/02_SERVING_HISTORICAL_PROTOCOL_AND_LENGTH_AUDIT.csv",out,list(out[0]))
    (OUT/"cpu/02_DREX_RUNTIME_COMPARISON.md").write_text("""# Drex Runtime Comparison

The E12 native Drex-DLM path used the local System-One typed pointer adapter; E12 Drex-v2 used the existing HTTP/llama.cpp-style runtime. The record establishes a configuration-specific event: 42/360 valid before an E12-owned native runtime CUDA illegal-memory-access crash, then 314 connection refusals and four transport/server errors. Server-side forward counts and low-level CUDA cause are unavailable. This addendum does not restart or repair Drex-v2.
""",encoding="utf-8")
    # Length terciles based on recorded request token proxy where present.
    lengths=[x["input_tokens"] for x in out if isinstance(x.get("input_tokens"),(int,float))]
    (OUT/"cpu/02_SERVING_LENGTH_TERCILES.json").write_text(json.dumps({"available_n":len(lengths),"p33":percentile(lengths,33.333),"p66":percentile(lengths,66.667),"note":"missing for most HTTP records; no imputation"},indent=2)+"\n")

def tolerance_frontier():
    comps=json.loads(E11_COMPS.read_text())["comparisons"]
    out=[]
    for name,x in comps.items():
        lcb=(x.get("ci95_fixed_domain_query") or x.get("ci95_domain_cluster") or [None,None])[0]
        out.append({"comparison":name,"effect":x.get("effect_macro_domain"),"interval":json.dumps(x.get("ci95_fixed_domain_query")),"descriptive_loss_tolerance_max_0_minus_LCB":max(0,-lcb) if lcb is not None else None,"interpretation":"descriptive sensitivity only; not noninferiority"})
    write_csv(OUT/"cpu/02_ILLUSTRATIVE_QUALITY_TOLERANCE_FRONTIER.csv",out,list(out[0]))

def main():
    os.chdir(ROOT)
    cpu_baselines(); garage_analysis(); tokenizer_audit(); e11_reconciliation(); serving_audit(); tolerance_frontier()
    print(json.dumps({"status":"CPU_ANALYSIS_COMPLETE","output":str(OUT/"cpu"),"bright_n":1384,"garage_n":1397},indent=2))
if __name__=="__main__": main()
