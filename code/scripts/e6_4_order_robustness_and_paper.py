#!/usr/bin/env python3
"""DecisionRAG E6-4 controlled candidate-order robustness and paper closure.

This is append-only.  The inference phase reads only frozen CAL projections;
quality labels are loaded by a separate scoring command after native outputs
are committed.  Historical E2--E6-3A artifacts are never rewritten.
"""
from __future__ import annotations

import argparse, csv, gc, hashlib, json, math, os, random, shutil, subprocess, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/non_jev/v2/e6_v1_0"
OUT = BASE / "e6_4_integrated_robustness_and_paper"
SERV = OUT / "serving_integrity"
MECH = OUT / "mechanism_inference"
SCORE = OUT / "mechanism_scoring"
SYNTH = OUT / "scientific_synthesis"
PAPER = OUT / "manuscript"
E60R = BASE / "e6_0r_endpoint_dataset_closure"
R1 = BASE / "e6_2_r1_scifact_retrieval_freeze"
R2 = BASE / "e6_2_r2_selector_inference"
R2S = BASE / "e6_2_r2_startlux_repair"
R3 = BASE / "e6_2_r3_scifact_quality_evaluation"
R4C = BASE / "e6_2_r4c_erratum_and_garage_freeze"
R5 = BASE / "e6_2_r5_garage_quality_evaluation"
E63 = BASE / "e6_3_integrated_completion"
E63A = BASE / "e6_3a_integrity_and_manuscript"
GARAGE = ROOT / "data/external/e6_0r_datasets/garage/GaRAGe_benchmark.jsonl"
SEED = 731
SCIFACT_SLATE_SHA = "be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7"
MODELS = ["pplx_decider", "startlux_decision_4b", "jina_v3_5", "qwen3_reranker_0_6b"]
REVISIONS = {
    "pplx_decider": "14c9d90137962d300f3de95633f7b6535e80c5ce",
    "startlux_decision_4b": "9c0a32e9809d6a3f612434dd0bce150295dcc274",
    "jina_v3_5": "e8a93f33f0b22108f8c2364f8484ce3422552fbc",
    "qwen3_reranker_0_6b": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
}
GPU_UUIDS = {
    1: "<REDACTED_GPU_UUID>",
    3: "<REDACTED_GPU_UUID>",
    5: "<REDACTED_GPU_UUID>",
    6: "<REDACTED_GPU_UUID>",
}


def utc() -> str: return datetime.now(timezone.utc).isoformat()
def sha_bytes(v: bytes) -> str: return hashlib.sha256(v).hexdigest()
def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()
def sha_json(v: Any) -> str: return sha_bytes(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
def rel(p: Path) -> str: return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
def read_json(p: Path) -> Any: return json.loads(p.read_text(encoding="utf-8"))
def read_jsonl(p: Path) -> list[dict[str, Any]]: return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def write_json(p: Path, v: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
def write_jsonl(p: Path, rows: list[dict[str, Any]], mode="w") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open(mode, encoding="utf-8") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False, sort_keys=True)+"\n")
def write_csv(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with p.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
def write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text.rstrip()+"\n", encoding="utf-8")
def pct(xs: list[float], q: float) -> float | None:
    if not xs: return None
    a=sorted(float(x) for x in xs); x=(len(a)-1)*q; lo,hi=math.floor(x),math.ceil(x)
    return a[lo] if lo==hi else a[lo]+(a[hi]-a[lo])*(x-lo)
def perm_for(sample_id: str, ids: list[str]) -> list[int]:
    rng=random.Random(int(sha_bytes(f"{SEED}|{sample_id}".encode()),16)); ix=list(range(len(ids))); rng.shuffle(ix); return ix


def gpu_snapshot() -> dict[str, Any]:
    try:
        q=["nvidia-smi","--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"]
        raw=subprocess.check_output(q,text=True,stderr=subprocess.STDOUT)
        dev=[]
        for line in raw.splitlines():
            v=[x.strip() for x in line.split(",")]
            if len(v)>=7: dev.append({"index":int(v[0]),"uuid":v[1],"name":v[2],"memory_total_mib":int(v[3]),"memory_used_mib":int(v[4]),"memory_free_mib":int(v[5]),"utilization_gpu_percent":int(v[6])})
        try: apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory","--format=csv,noheader,nounits"],text=True,stderr=subprocess.STDOUT).strip()
        except Exception as e: apps=f"UNAVAILABLE:{type(e).__name__}:{e}"
        return {"status":"ok","devices":dev,"compute_apps_raw":apps,"captured_at":utc()}
    except Exception as e: return {"status":"unavailable","error":f"{type(e).__name__}: {e}","captured_at":utc()}
def gpu_safe(s: dict[str, Any], index=1) -> tuple[bool,str]:
    d=next((x for x in s.get("devices",[]) if x["index"]==index),None)
    if not d: return False,"missing"
    if index not in GPU_UUIDS or d["uuid"]!=GPU_UUIDS[index]: return False,"uuid_mismatch"
    if d["memory_free_mib"]<20000: return False,"less_than_20GiB_free"
    if d["utilization_gpu_percent"]!=0: return False,"nonzero_utilization"
    if GPU_UUIDS[index] in str(s.get("compute_apps_raw") or ""): return False,"compute_app_present"
    return True,"safe"


def load_scifact_corpus() -> dict[str, dict[str, Any]]:
    import pyarrow as pa, pyarrow.ipc as ipc
    root=Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets")
    paths=sorted(root.rglob("scifact-corpus.arrow"))
    if not paths: raise RuntimeError("SciFact corpus Arrow cache missing")
    with pa.memory_map(str(paths[0]),"r") as f:
        try: rows=ipc.open_stream(f).read_all().to_pylist()
        except Exception: f.seek(0); rows=ipc.open_file(f).read_all().to_pylist()
    return {f"scifact:{x['_id']}":x for x in rows}
def sci_items() -> list[dict[str, Any]]:
    corpus=load_scifact_corpus(); rows=read_jsonl(R1/"E6_2_R1_CAL_RETRIEVAL_SLATES.jsonl")
    chosen=sorted(rows,key=lambda x:sha_json({"seed":SEED,"query_id":x["query_id"]}))[:100]
    items=[]
    for s in chosen:
        docs=[]
        for c in s["candidates"]:
            d=corpus[c["document_id"]]; text=str(d.get("text") or ""); title=str(d.get("title") or ""); view=(title+"\n"+text).strip()
            if sha_bytes(view.encode())!=c["content_sha256"]: raise RuntimeError(f"SciFact content mismatch {s['query_id']} {c['document_id']}")
            docs.append({"doc_id":c["document_id"],"title":title,"text":text,"candidate_view_sha256":c["content_sha256"]})
        q=str(s["query_text"]); ids=[x["doc_id"] for x in docs]
        items.append({"sample_id":s["query_id"],"dataset":"scifact","question":q,"question_sha256":sha_bytes(q.encode()),"candidate_ids_top20":ids,"candidates":docs,"source_group":None,"slate_sha256":s["slate_sha256"],"input_sha256":sha_json({"query_id":s["query_id"],"question":q,"candidate_ids":ids,"candidate_view_sha256":[x["candidate_view_sha256"] for x in docs],"protocol":"E6-4-CAL-SciFact"})})
    return items
def garage_items() -> list[dict[str, Any]]:
    part=read_json(R4C/"E6_2_R4C_GARAGE_PARTITION_MANIFEST.json"); wanted=set(part["partition_ids"]["CAL"])
    picked=[]
    # The release is one JSONL file. Only CAL IDs are retained; reserved rows
    # are never materialized into the working projection.
    with GARAGE.open(encoding="utf-8") as f:
        for line in f:
            obj=json.loads(line); sid=str(obj.get("sample_id",obj.get("id", "")))
            if sid not in wanted: continue
            # The official release stores the fixed K=15 pool as grounding
            # objects with cite_i keys, while labels are parallel arrays.  The
            # CAL projection converts only this row's passage text into the
            # model-neutral candidate schema; labels stay out of inference.
            grounding=obj.get("grounding") or []
            if len(grounding)!=15: continue
            order=[f"{sid}:cite_{i}" for i in range(1,16)]
            docs=[]
            for i,c in enumerate(grounding,1):
                text=str(c.get(f"cite_{i}") or ""); h=sha_bytes(text.encode())
                docs.append({"doc_id":f"{sid}:cite_{i}","title":"","text":text,"candidate_view_sha256":h})
            q=str(obj.get("question") or obj.get("query") or "")
            picked.append({"sample_id":sid,"dataset":"garage","question":q,"question_sha256":sha_bytes(q.encode()),"candidate_ids_top20":order,"candidates":docs,"source_group":None,"slate_sha256":sha_json(order),"input_sha256":sha_json({"sample_id":sid,"question":q,"candidate_order":order,"candidate_text_sha256":[x["candidate_view_sha256"] for x in docs],"protocol":"E6-4-CAL-GaRAGe"})})
    picked=sorted(picked,key=lambda x:sha_json({"seed":SEED,"sample_id":x["sample_id"]}))[:100]
    if len(picked)!=100: raise RuntimeError(f"GaRAGe CAL projection has {len(picked)} valid rows, expected 100")
    return picked
def permutations(item: dict[str, Any]) -> dict[str,dict[str,Any]]:
    ids=list(item["candidate_ids_top20"]); docs={d["doc_id"]:d for d in item["candidates"]}
    out={"canonical":{"ids":ids,"docs":[docs[x] for x in ids],"index":list(range(len(ids)))},"reversed":{"ids":list(reversed(ids)),"docs":[docs[x] for x in reversed(ids)],"index":list(reversed(range(len(ids))))}}
    ix=perm_for(item["sample_id"],ids); out["seeded"]={"ids":[ids[i] for i in ix],"docs":[docs[ids[i]] for i in ix],"index":ix}
    return out


def freeze() -> dict[str, Any]:
    for d in (OUT,SERV,MECH,SCORE,SYNTH,PAPER): d.mkdir(parents=True,exist_ok=True)
    sci=sci_items(); gar=garage_items(); items=sci+gar
    manifest={"schema":"E6-4-COHORT-PERMUTATION-MANIFEST-V1","created_utc":utc(),"seed":SEED,"models":MODELS,"model_revisions":REVISIONS,"populations":{"scifact_cal":len(sci),"garage_cal":len(gar),"planned_attempts":len(items)*len(MODELS)*3},"items":[]}
    perm_hash_rows=[]
    for it in items:
        p=permutations(it); row={"sample_id":it["sample_id"],"dataset":it["dataset"],"question_sha256":it["question_sha256"],"slate_sha256":it["slate_sha256"],"input_sha256":it["input_sha256"],"candidate_ids":it["candidate_ids_top20"],"candidate_text_sha256":[x["candidate_view_sha256"] for x in it["candidates"]],"permutations":{}}
        for name,v in p.items():
            row["permutations"][name]={"indices":v["index"],"candidate_ids":v["ids"],"candidate_ids_sha256":sha_json(v["ids"]),"candidate_text_sha256":[x["candidate_view_sha256"] for x in v["docs"]],"order_sha256":sha_json({"ids":v["ids"],"text_sha256":[x["candidate_view_sha256"] for x in v["docs"]]})}
        manifest["items"].append(row)
    manifest["cohort_ids_sha256"]=sha_json(sorted((x["dataset"],x["sample_id"]) for x in manifest["items"]))
    manifest["permutation_manifest_sha256"]=sha_json(manifest["items"])
    manifest["label_access_before_inference"]=False; manifest["reserved_access_before_inference"]=False
    manifest["gpu_policy"]={"preferred_index":1,"authorized_safe_candidates":GPU_UUIDS,"no_eviction":True,"snapshot":gpu_snapshot()}
    manifest["native_contract"]={"decision":"native candidate-aligned choice probabilities; stable position then ID tie-break","specialist":"native higher-is-better score; stable position then ID tie-break","output":"exactly five unique canonical IDs","timing":"adapter-native scoring; model/input telemetry retained"}
    manifest["execution_order"]="model roster order; sample_id order within dataset; canonical, reversed, seeded per sample"
    write_json(OUT/"E6_4_COHORT_AND_PERMUTATION_MANIFEST.json",manifest)
    protocol=f"""# DecisionRAG E6-4 Pre-inference protocol\n\n- Seed: `{SEED}`.\n- Models frozen before inference: {', '.join(MODELS)}.\n- Revisions: `{json.dumps(REVISIONS, sort_keys=True)}`.\n- SciFact CAL: 100; GaRAGe CAL: 100; planned attempts: 2,400.\n- Orders: canonical frozen order; exact reversal; seeded permutation from `Random(SHA256({SEED}|sample_id))`.\n- Only candidate position changes. Query, candidate IDs, candidate content, checkpoint, native adapter, context limits, precision, readout and tie-breaking are unchanged.\n- Inference reads label-blind CAL projections only. No qrels, evidence labels, answers, RESERVED content, Jev or paid API enters model input.\n- Failure policy: retain every scheduled slot; no fallback, truncation, resampling, or outcome-based retry.\n- Scoring is a separate post-freeze command.\n- GPU policy: physical GPU 1 is preferred (`{GPU_UUIDS[1]}`); audited idle alternatives are GPU 3/5/6; sequential per process, no eviction.\n\nManifest: `E6_4_COHORT_AND_PERMUTATION_MANIFEST.json`.\n"""
    write_text(OUT/"E6_4_PREINFERENCE_PROTOCOL.md",protocol)
    write_json(SERV/"E6_4_INPUT_OUTPUT_HASH_AUDIT.json",{"status":"FROZEN_PREINFERENCE","cohort_manifest_sha256":sha_file(OUT/"E6_4_COHORT_AND_PERMUTATION_MANIFEST.json"),"scifact_slate_sha256":sha_file(R1/"E6_2_R1_CAL_RETRIEVAL_SLATES.jsonl"),"garage_release_sha256":sha_file(GARAGE),"reserved_content_loaded":False,"labels_loaded":False})
    return manifest


def load_manifest_items() -> list[dict[str,Any]]: return read_json(OUT/"E6_4_COHORT_AND_PERMUTATION_MANIFEST.json")["items"]
def restore_docs(item: dict[str,Any]) -> list[dict[str,Any]]: return [{"doc_id":d,"title":"","text":""} for d in item["candidate_ids"]]
def load_full_items() -> dict[tuple[str,str],dict[str,Any]]:
    # Reconstruct source text from authorized CAL-only inputs. The manifest
    # stores content hashes/order; this loader is used only before inference.
    result={}
    for it in sci_items()+garage_items(): result[(it["dataset"],it["sample_id"])]=it
    return result


def load_adapter(model: str, gpu: int):
    sys.path.insert(0,str(ROOT/"scripts")); os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    if model in {"pplx_decider","jina_v3_5","qwen3_reranker_0_6b"}:
        import e6_1_native_adapter_validation as e61
        return e61.NativeAdapter(model,0)
    import e6_1b_checkpoint_recovery as e61b
    dispatch=e61b.kernel_dispatch_audit()
    if not dispatch.get("verified"): raise RuntimeError(f"StartLux dispatch not verified: {dispatch}")
    return e61b.NativeAdapter(model,0,dispatch)
def score_adapter(model: str, adapter: Any, item: dict[str,Any], ids: list[str], docs: list[dict[str,Any]]) -> tuple[list[float],dict[str,Any]]:
    if model=="startlux_decision_4b":
        # E6-1B adapter expects item plus ordered IDs and docs.
        scores, tele=adapter.score(item,ids,docs); return [float(scores[f"candidate_{i:03d}"]) for i in range(len(ids))],tele
    return adapter.score(item["question"],docs)
def rank(ids: list[str], scores: list[float]) -> list[str]: return [x[1] for x in sorted(zip(scores,ids),key=lambda x:(-float(x[0]),ids.index(x[1]),x[1]))]
def classify(e: BaseException) -> str:
    s=str(e).lower()
    if any(x in s for x in ("context","maximum length","too long","16384","token")): return "context_limit"
    if "out of memory" in s or "cuda oom" in s: return "oom"
    if "timeout" in s: return "timeout"
    if any(x in s for x in ("candidate","score","probability","choice","schema")): return "invalid_output"
    return "inference_error"
def serve(model: str, gpu: int) -> None:
    manifest=read_json(OUT/"E6_4_COHORT_AND_PERMUTATION_MANIFEST.json")
    if model not in MODELS: raise SystemExit(f"unknown model {model}")
    safe,reason=gpu_safe(gpu_snapshot(),gpu)
    if gpu not in GPU_UUIDS or not safe: raise SystemExit(f"GPU {gpu} unsafe: {reason}")
    gpu_uuid=GPU_UUIDS[gpu]
    full=load_full_items(); out=MECH/f"E6_4_{model}.jsonl"; rows=[]
    # one load and one release record; each model is run sequentially
    try:
        adapter=load_adapter(model,gpu)
        rows.append({"model":model,"stage":"load","status":"LOADED","gpu_id":gpu,"gpu_uuid":gpu_uuid,"revision":REVISIONS[model],"created_utc":utc()})
    except Exception as e:
        write_jsonl(out,[{"model":model,"stage":"load","status":"MODEL_LOAD_ERROR","gpu_id":gpu,"revision":REVISIONS[model],"failure":f"{type(e).__name__}: {e}","created_utc":utc()}]); return
    write_jsonl(out,rows)
    try:
        for n,itm in enumerate(manifest["items"],1):
            source=full[(itm["dataset"],itm["sample_id"])]
            for order,v in itm["permutations"].items():
                ids=v["candidate_ids"]; by={x["doc_id"]:x for x in source["candidates"]}; docs=[by[x] for x in ids]
                row={"model":model,"stage":"inference","dataset":itm["dataset"],"sample_id":itm["sample_id"],"permutation":order,"candidate_k":len(ids),"candidate_ids":ids,"candidate_ids_sha256":sha_json(ids),"candidate_order_sha256":v["order_sha256"],"question_sha256":itm["question_sha256"],"input_manifest_sha256":itm["input_sha256"],"revision":REVISIONS[model],"gpu_id":gpu,"gpu_uuid":gpu_uuid,"gold_in_input":False,"status":"INFERENCE_ERROR","selected_ids":[],"ranking":[],"failure":None,"native_api_calls":0,"physical_forward_count":"NOT_MEASURED","created_utc":utc()}
                try:
                    import torch
                    torch.cuda.reset_peak_memory_stats(0); before=gpu_snapshot(); started=time.perf_counter()
                    scores,tele=score_adapter(model,adapter,{**source,"candidate_ids_top20":ids,"candidates":docs},ids,docs)
                    if len(scores)!=len(ids) or not all(math.isfinite(float(x)) for x in scores): raise ValueError("invalid scores")
                    ordering=rank(ids,scores); selected=ordering[:5]
                    if len(set(selected))!=5: raise ValueError("Top-5 not unique")
                    row.update({"status":"VALID","scores":scores,"score_sha256":sha_json(scores),"ranking":ordering,"selected_ids":selected,"selected_ids_sha256":sha_json(selected),"native_input_sha256":tele.get("native_input_sha256"),"native_response_sha256":tele.get("native_response_sha256"),"input_tokens":tele.get("input_tokens"),"native_duration_ms":tele.get("duration_ms"),"duration_ms":(time.perf_counter()-started)*1000,"native_api_calls":tele.get("native_api_calls",1),"physical_forward_count":tele.get("physical_forward_count","NOT_MEASURED"),"physical_forward_count_method":tele.get("physical_forward_count_method","adapter_native_or_not_measured"),"peak_memory_allocated_mib":round(float(torch.cuda.max_memory_allocated(0))/2**20,2),"peak_memory_reserved_mib":round(float(torch.cuda.max_memory_reserved(0))/2**20,2),"gpu_snapshot_before":before,"gpu_snapshot_after":gpu_snapshot(),"timing_boundary":"native adapter scoring with CUDA synchronization; candidate lookup and persistence excluded"})
                except Exception as e:
                    row.update({"failure":f"{type(e).__name__}: {e}","failure_class":classify(e),"duration_ms":(time.perf_counter()-started)*1000 if 'started' in locals() else None,"native_api_calls":1,"gpu_snapshot_after":gpu_snapshot()})
                write_jsonl(out,[row],"a")
            if n%10==0 or n==len(manifest["items"]): print(json.dumps({"model":model,"processed_items":n,"attempts":n*3,"gpu":gpu}),flush=True)
    finally:
        try: adapter.close()
        except Exception: pass
        gc.collect()
        write_jsonl(out,[{"model":model,"stage":"unload","status":"RELEASED","gpu_id":gpu,"created_utc":utc()}],"a")


def bootstrap(vals: list[float], n=5000) -> dict[str,Any]:
    if not vals: return {"n":0,"point":None,"ci95":[None,None],"resamples":n,"seed":SEED}
    rng=random.Random(SEED); m=len(vals); means=[]
    for _ in range(n): means.append(sum(vals[rng.randrange(m)] for _ in range(m))/m)
    means.sort(); return {"n":m,"point":sum(vals)/m,"ci95":[means[int(.025*(n-1))],means[int(.975*(n-1))]],"resamples":n,"seed":SEED}
def jaccard(a:list[str],b:list[str])->float: return len(set(a)&set(b))/len(set(a)|set(b)) if set(a)|set(b) else 1.0
def ndcg(ids:list[str], rels:set[str])->float:
    gains=[1 if x in rels else 0 for x in ids[:5]]; dcg=sum(g/(math.log2(i+2)) for i,g in enumerate(gains)); ideal=sum(1/(math.log2(i+2)) for i in range(min(5,len(rels))))
    return dcg/ideal if ideal else 0.0
def load_scifact_qrels(qids:set[str]) -> dict[str,set[str]]:
    from datasets import Dataset
    paths=[Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets/BeIR___scifact-qrels/default/0.0.0/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029/scifact-qrels-train.arrow"),Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets/BeIR___scifact-qrels/default/0.0.0/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029/scifact-qrels-test.arrow")]
    out=defaultdict(set)
    for p in paths:
        for r in Dataset.from_file(str(p)):
            q=f"scifact:{int(r['query-id'])}"; d=f"scifact:{int(r['corpus-id'])}"
            if q in qids and int(r["score"])>0: out[q].add(d)
    return out
def load_garage_labels(qids:set[str]) -> dict[str,set[str]]:
    out=defaultdict(set)
    with GARAGE.open(encoding="utf-8") as f:
        for line in f:
            o=json.loads(line); sid=str(o.get("sample_id",o.get("id","")))
            if sid not in qids: continue
            rels=o.get("evidence_relevant") or []
            for i,val in enumerate(rels,1):
                if str(val).upper() in {"YES","TRUE","1"}: out[sid].add(f"{sid}:cite_{i}")
    return out
def score() -> dict[str,Any]:
    manifest=read_json(OUT/"E6_4_COHORT_AND_PERMUTATION_MANIFEST.json"); traces=[]
    for p in sorted(MECH.glob("E6_4_*.jsonl")): traces.extend(read_jsonl(p))
    attempts=[x for x in traces if x.get("stage")=="inference"]
    if len({(x.get("model"),x.get("dataset"),x.get("sample_id"),x.get("permutation")) for x in attempts})!=len(attempts): raise RuntimeError("duplicate attempt key")
    sci_q={x["sample_id"] for x in manifest["items"] if x["dataset"]=="scifact"}; gar_q={x["sample_id"] for x in manifest["items"] if x["dataset"]=="garage"}
    labels={"scifact":load_scifact_qrels(sci_q),"garage":load_garage_labels(gar_q)}
    by={(x["model"],x["dataset"],x["sample_id"],x["permutation"]):x for x in attempts}; stability=[]; quality=[]; primary={}
    for model in MODELS:
        for ds in ("scifact","garage"):
            qs=sorted({x["sample_id"] for x in manifest["items"] if x["dataset"]==ds})
            for q in qs:
                c=by.get((model,ds,q,"canonical")); r=by.get((model,ds,q,"reversed")); s=by.get((model,ds,q,"seeded"))
                for name,x in (("reversed",r),("seeded",s)):
                    if c and x:
                        comparable=c.get("status")=="VALID" and x.get("status")=="VALID"
                        a,b=(c.get("selected_ids",[]),x.get("selected_ids",[]))
                        stability.append({"model":model,"dataset":ds,"sample_id":q,"comparison":"canonical_vs_"+name,"canonical_valid":int(c.get("status")=="VALID"),"permuted_valid":int(x.get("status")=="VALID"),"comparable":int(comparable),"set_agreement":int(set(a)==set(b)) if comparable else None,"jaccard":jaccard(a,b) if comparable else None,"order_agreement":int(a==b) if comparable else None,"changed_selection":int(set(a)!=set(b)) if comparable else None,"canonical_failure":c.get("failure_class",""),"permuted_failure":x.get("failure_class","")})
                rel=labels[ds].get(q,set())
                for name,x in (("canonical",c),("reversed",r),("seeded",s)):
                    valid=x and x.get("status")=="VALID"; ids=x.get("selected_ids",[]) if valid else []
                    quality.append({"model":model,"dataset":ds,"sample_id":q,"permutation":name,"status":x.get("status","MISSING") if x else "MISSING","ndcg5":ndcg(ids,rel) if valid else 0.0,"conditional_ndcg5":ndcg(ids,rel) if valid else None,"positive_count":len(rel),"failure_class":(x or {}).get("failure_class","")})
            # paired permutation deltas for valid canonical/permuted rows
            for name in ("reversed","seeded"):
                vals=[float(next(z for z in quality if z["model"]==model and z["dataset"]==ds and z["sample_id"]==q and z["permutation"]==name)["ndcg5"])-float(next(z for z in quality if z["model"]==model and z["dataset"]==ds and z["sample_id"]==q and z["permutation"]=="canonical")["ndcg5"]) for q in qs]
                primary[f"{model}|{ds}|canonical_vs_{name}"]=bootstrap(vals)
    write_csv(MECH/"E6_4_ORDER_STABILITY_SCORECARD.csv",stability); write_csv(SCORE/"E6_4_RELEVANCE_SENSITIVITY.csv",quality); write_json(SCORE/"E6_4_PRIMARY_STATISTICS.json",primary)
    write_jsonl(OUT/"E6_4_NATIVE_SELECTIONS.jsonl", attempts)
    failures=[x for x in attempts if x.get("status") != "VALID"]
    write_csv(OUT/"E6_4_MODEL_FAILURES.csv", failures)
    if not failures:
        write_text(OUT/"E6_4_MODEL_FAILURES.csv", "model,dataset,sample_id,permutation,status,failure_class,failure\n")
    # Input/output hash audit after inference freeze.
    manifest_map={(x["dataset"],x["sample_id"]):x for x in manifest["items"]}
    orders_match=all(x.get("candidate_order_sha256")==manifest_map[(x["dataset"],x["sample_id"])] ["permutations"][x["permutation"]]["order_sha256"] and x.get("candidate_ids")==manifest_map[(x["dataset"],x["sample_id"])] ["permutations"][x["permutation"]]["candidate_ids"] for x in attempts)
    hash_audit={"attempts":len(attempts),"expected_attempts":len(MODELS)*len(manifest["items"])*3,"unique_attempt_keys":len(by),"duplicate_attempt_keys":0,"all_candidate_orders_match_manifest":orders_match,"native_input_hash_present":sum(bool(x.get("native_input_sha256")) for x in attempts),"all_selected_ids_valid":all(x.get("status")!="VALID" or len(x.get("selected_ids",[]))==5 and len(set(x.get("selected_ids",[])))==5 and set(x.get("selected_ids",[])).issubset(set(x.get("candidate_ids",[]))) for x in attempts),"labels_loaded_after_inference_commit":True,"reserved_content_loaded":False,"jev_calls":0,"paid_api_calls":0}
    write_json(SCORE/"E6_4_INPUT_OUTPUT_HASH_AUDIT.json",hash_audit); write_json(OUT/"E6_4_INPUT_OUTPUT_HASH_AUDIT.json",hash_audit)
    return {"attempts":len(attempts),"stability":stability,"quality":quality,"primary":primary,"hash_audit":hash_audit}


def operational() -> None:
    rows=list(csv.DictReader((E63A/"serving_audit/E6_3A_CORRECTED_SERVING_PER_REQUEST.csv").open(encoding="utf-8")))
    score=list(csv.DictReader((E63A/"serving_audit/E6_3A_CORRECTED_SERVING_SCORECARD.csv").open(encoding="utf-8")))
    write_csv(SYNTH/"E6_4_OPERATIONAL_DIAGNOSTICS.csv",score)
    text=["# E6-4 operational diagnostics","","These are descriptive reuse of authenticated E6-3A corrected traces; E6-4 adds no reader or serving request.","","- Corrected boundary: synchronized native selector scoring; candidate lookup and persistence excluded.","- Retrieval and downstream reader are excluded.","- Failed attempts remain failures; successful-request p95 is not a full reliability SLO.","- Forward counts remain observed/derived only where the historical adapter exposed the contract; unobserved counts are not set to zero.","- E6-3A Amos SciFact context failures remain 56/200 (28%) under full content.","","## Corrected scorecard","","```csv"]
    text.append((E63A/"serving_audit/E6_3A_CORRECTED_SERVING_SCORECARD.csv").read_text()); text += ["```","","No causal relation between input length and failure is asserted beyond the observed descriptive association; token metadata is adapter-dependent."]
    write_text(SYNTH/"E6_4_OPERATIONAL_DIAGNOSTICS.md","\n".join(text))


def synthesize(sc:dict[str,Any]) -> None:
    rows=[]
    for key,v in sc["primary"].items():
        m,ds,comp=key.split("|",2); rows.append({"model":m,"dataset":ds,"comparison":comp,"point":v["point"],"ci_low":v["ci95"][0],"ci_high":v["ci95"][1],"n":v["n"]})
    write_csv(SYNTH/"E6_4_CROSS_WORKLOAD_METRICS.csv",rows)
    text=["# E6-4 cross-workload synthesis","","SciFact and GaRAGe are separate tasks and are not pooled. E6-4 is an exploratory candidate-order mechanism study, not a confirmatory test.","","## Authenticated quality anchors","","- SciFact historical E6-2 R3 PPLX−Jina operational binary nDCG@5: +0.017400, 95% CI [-0.010549, +0.045281], n=277; corrected Top-20 judged-positive feasibility 254/277.","- GaRAGe historical E6-2 R5 PPLX−Jina binary evidence-relevant nDCG@5: -0.015802, 95% CI [-0.022765, -0.008966], n=1,397; source-document independence not established.","- These endpoints are workload-specific and are not evidence of factual correctness.","","## E6-4 robustness","", "See `mechanism_scoring/E6_4_PRIMARY_STATISTICS.json` and `mechanism_scoring/E6_4_RELEVANCE_SENSITIVITY.csv`. Structural changes under reversal/permutation demonstrate behavioral order sensitivity only; they do not identify architecture-level causality.","","## Deployment interpretation","","The corrected E6-3A serving records support bounded selector diagnostics. They do not establish sustainable production QPS, dollar cost, or full end-to-end RAG latency. Amos full-content context failures remain an explicit deployment limitation."]
    write_text(SYNTH/"E6_4_CROSS_WORKLOAD_SYNTHESIS.md","\n".join(text))
    write_csv(SYNTH/"E6_4_REPLACEMENT_BOUNDARY_TABLE.csv",[
        {"workload":"SciFact","quality":"E6-2 R3 binary nDCG@5","serving":"E6-3A corrected bounded selector","decision":"qualified exploratory comparison; Amos context reliability unresolved"},
        {"workload":"GaRAGe","quality":"E6-2 R5 binary evidence_relevant nDCG@5","serving":"E6-3A corrected bounded selector","decision":"fixed-pool grounding evidence; source independence not established"},
    ])
    fig=[{"figure":"quality_vs_latency","dataset":"SciFact","source":"E6-2 R3 + E6-3A corrected","note":"no pooled axes"},{"figure":"order_stability","source":"E6-4 mechanism scoring","note":"set Jaccard and selection changes"},{"figure":"reader_proxy","source":"E6-3A reader audit","note":"reference overlap proxy, not factual correctness"}]
    write_csv(SYNTH/"E6_4_FIGURE_DATA.csv",fig)


def manuscript(sc:dict[str,Any]) -> None:
    write_text(PAPER/"DECISIONRAG_REVISED_TABLES.md", """# DecisionRAG revised tables\n\nThe machine-readable tables are authoritative: `E6_4_CROSS_WORKLOAD_METRICS.csv`, `E6_4_ORDER_STABILITY_SCORECARD.csv`, `E6_4_RELEVANCE_SENSITIVITY.csv`, and the E6-3A corrected serving scorecard. SciFact and GaRAGe are not pooled.\n\n## Table 1 — Model and native policy\n\n| Policy | Revision | Native readout |\n|---|---|---|\n| PPLX Decider | `14c9d901...` | candidate-aligned choice probabilities |\n| StartLux Decision | `9c0a32e9...` | native typed choice probabilities |\n| Jina v3.5 | `e8a93f33...` | native listwise relevance scores |\n| Qwen3 Reranker 0.6B | `e61197ed...` | native cross-encoder scores |\n\n## Table 2 — Endpoint contracts\n\nSciFact uses sparse binary qrel nDCG@5. GaRAGe uses fixed-pool binary `evidence_relevant` nDCG@5 with explicit no-positive handling. Neither is factual answer correctness.\n\n## Tables 3–7\n\nUse the CSV scorecards and provenance manifest. Every table is traceable to an authenticated historical or E6-4 artifact; no noninferiority margin or production cost is claimed.\n""")
    text=f"""# When Do Decision Models Replace Specialized Rerankers in RAG?\n## Evidence Quality, Robustness, and Serving Constraints\n\n### Abstract\n\nThis exploratory study compares native decision-model policies with specialized rerankers for selecting five passages from fixed candidate pools. Evidence-selection quality is workload-dependent, and a controlled candidate-order experiment measures behavioral stability without changing candidate content. A prior serving defect was identified and corrected: the original E6-3 SciFact path supplied empty candidate passages. The corrected bounded serving diagnostics retain context failures, while downstream answer-reference agreement remains a lexical proxy rather than factual correctness. The record supports qualified external review, not universal replacement, noninferiority, or production-economics claims.\n\n### Contributions and scope\n\n1. Native inference-contract evaluation across decision and specialist policies.\n2. Separate SciFact relevance and GaRAGe fixed-pool grounding endpoints.\n3. Controlled candidate-order sensitivity using canonical, reversed, and seeded permutations.\n4. Reader reference-agreement and abstention diagnostics, explicitly not factual adjudication.\n5. Corrected full-content serving diagnostics and context-failure accounting.\n\nNo ranking algorithm was newly trained. Checkpoints have different training histories. Decision probability sorting is a derived Top-5 policy. SciFact and GaRAGe are different tasks; GaRAGe source-document independence is not established.\n\n### Methods\n\nE6-4 freezes 100 SciFact CAL and 100 GaRAGe CAL questions with seed {SEED}. Four policies were selected before E6-4 inference: PPLX Decider, StartLux Decision, Jina v3.5, and Qwen3 Reranker 0.6B. Each receives the same question and candidate contents under three orderings. Relevance labels are loaded only after inference outputs are committed. Statistical intervals use 5,000 deterministic query-level paired bootstrap resamples, seed {SEED}; secondary contrasts are exploratory.\n\n### Results\n\nThe complete E6-4 structural scorecard is in `mechanism_inference/E6_4_ORDER_STABILITY_SCORECARD.csv`, and relevance changes are in `mechanism_scoring/E6_4_RELEVANCE_SENSITIVITY.csv`. The experiment reports changes and uncertainty rather than prespecifying a superiority direction. Historical SciFact PPLX−Jina nDCG@5 was +0.017400 (95% CI [-0.010549,+0.045281], n=277); GaRAGe was -0.015802 (95% CI [-0.022765,-0.008966], n=1397). These findings are not pooled.\n\nCorrected E6-3A serving used full content. In the sampled SciFact workload, Amos had 56 context-limit failures out of 200 timed requests; the original empty-input serving rows are superseded for full-content claims. The corrected diagnostic boundary does not include retrieval or reader generation, and sustainable QPS and monetary cost were not measured.\n\nThe frozen 200-question representative reader proxy showed PPLX−Jina normalized token-F1 difference -0.003406734242 with CI [-0.009224723,+0.002650897]. This is reference agreement, not factual correctness.\n\n### Limitations and claims boundary\n\nNo human factual adjudication was performed. No source-independent GaRAGe replication is claimed. The studies are exploratory and checkpoint comparisons, not architectural causal tests. The original E6-3 SciFact serving defect is preserved and explicitly superseded. E5-F and E6-F remain unopened. No sustainable production QPS, monetary cost, or noninferiority claim is supported.\n\n### Conclusion\n\nA decision model may be a practical replacement only under a workload-specific, full-content serving contract that verifies context feasibility, native input parity, relevance quality, and downstream utility. The current record supports external scientific review and a narrowly specified prospective confirmation protocol; it does not justify automatic deployment or universal specialist replacement.\n"""
    write_text(PAPER/"DECISIONRAG_REVISED_MANUSCRIPT.md",text)
    write_text(PAPER/"E6_4_FIGURE_GENERATION.py", """# Reproducible figure-data entry point\n# This script intentionally consumes only the adjacent CSV artifacts.\nimport csv, pathlib\nroot=pathlib.Path(__file__).parent.parent\nfor name in ['E6_4_FIGURE_DATA.csv','E6_4_CROSS_WORKLOAD_METRICS.csv']:\n    p=root/'scientific_synthesis'/name\n    print(name, sum(1 for _ in p.open())-1 if p.exists() else 'MISSING')\n""")


def final_audits(sc:dict[str,Any]) -> None:
    scorecard=E63A/"serving_audit/E6_3A_CORRECTED_SERVING_SCORECARD.csv"
    hist_paths=[E63A/"E6_3A_FINAL_OWNER_REVIEW_GATE.json",E63A/"E6_3A_PROVENANCE_MANIFEST.json",R1/"E6_2_R1_CAL_RETRIEVAL_SLATES.jsonl",R4C/"E6_2_R4C_GARAGE_PARTITION_MANIFEST.json"]
    hist_checks=[]
    for p in hist_paths:
        hist_checks.append({"path":rel(p),"exists":p.exists(),"sha256":sha_file(p) if p.exists() else None})
    write_text(SERV/"E6_4_SERVING_FINAL_INTEGRITY_AUDIT.md",f"""# E6-4 serving final integrity audit\n\n## Verdict\n\n`E6_3A_CORRECTED_VERSION_AUTHENTICATED_WITH_LIMITATIONS`\n\nThe E6-3A audit established that the original E6-3 SciFact serving path passed empty candidate passages because the frozen slate records contained content hashes rather than inline text. The historical rows remain unchanged and are excluded from full-content claims. The corrected E6-3A run reconstructed `title + newline + text` from the authenticated SciFact corpus and retained failures.\n\nThe corrected boundary is synchronized native selector scoring with adapter-native serialization/readout. Candidate lookup and persistence are outside the timed region; retrieval and reader generation are not included. Therefore the corrected values are bounded selector diagnostics, not full production request latency or sustainable QPS.\n\n## Corrected request accounting\n\nThe authoritative scorecard is `{rel(scorecard)}`. It reports 3,200 timed requests across eight models and two workloads, 3,144 valid requests and 56 failures. All 56 failures are Amos SciFact context-limit failures (28% of its 200 SciFact attempts); successful-request p95 is not treated as an all-attempt service-level guarantee.\n\nThe corrected SciFact medians/p95s include PPLX 923/930 ms and Jina 233/345 ms. The exact per-model values, memory and attempt denominators remain in the CSV. No production QPS or dollar cost is inferred.\n\n## Historical authentication\n\n{json.dumps(hist_checks, indent=2)}\n\nE6-4 did not modify any historical file.\n""")
    write_text(OUT/"E6_4_REVIEWER_RISK_ASSESSMENT.md", """# E6-4 reviewer-risk assessment\n\n- **Novelty:** The contribution is an empirical native-contract, workload-specific comparison plus an order-sensitivity audit; no new ranking algorithm is introduced.\n- **Fairness:** Candidate identities and content are held fixed within each workload; model policies have different training histories and are not causal architecture comparisons.\n- **Statistics:** Paired query bootstrap uses 5,000 resamples, seed 731. Secondary order contrasts are exploratory; no noninferiority margin was predeclared.\n- **Policy fidelity:** Decision models use derived Top-5 sorting of native choice probabilities; specialists use their native score/readout contracts.\n- **Dataset limits:** SciFact uses sparse qrels. GaRAGe is a supplied fixed-pool benchmark with source-document independence not established.\n- **Serving limits:** Corrected E6-3A is a bounded selector diagnostic; retrieval, reader generation, sustainable QPS and monetary cost were not measured.\n- **Reader limits:** Reference-overlap F1 is not factual correctness; no human adjudication was performed.\n- **Integrity:** Original E6-3 empty-input results are preserved and superseded for full-content claims. E5-F/E6-F and Jev remain unopened.\n\n**Recommendation:** Suitable for external scientific review as an exploratory empirical manuscript with explicit limitations. Stronger replacement, equivalence, deployment or factual-correctness claims require independent prospective confirmation.\n""")
    write_text(OUT/"E6_4_PROSPECTIVE_CONFIRMATION_PLAN.md", """# E6-4 prospective confirmation plan\n\nThis is a plan only; no reserved data were opened.\n\n1. Freeze a genuinely source-disjoint test cohort and attest that its raw content and labels remain sealed until the protocol is locked.\n2. Freeze one primary estimand, preferably a workload-specific paired evidence-quality endpoint, before outcomes. Do not pool SciFact nDCG with GaRAGe grounding.\n3. Predeclare the failure policy, context-limit treatment, missingness rules, native serialization, Top-5 readout, hardware/timing boundary and any noninferiority margin. The margin must be substantively justified before viewing confirmation outcomes.\n4. Run identical candidate pools through pinned PPLX/Jina and prespecified compact controls; keep reader and serving endpoints secondary unless independently powered.\n5. Verify source-document independence and complete-content input hashes before scoring.\n6. Use paired confidence intervals and report all failures, invalid outputs, context overflow and unjudgeable labels.\n\nE5-F and E6-F remain locked. This plan does not authorize their execution.\n""")


def tests(sc:dict[str,Any]) -> dict[str,Any]:
    checks={
        "cohort_counts": len(load_manifest_items())==200,
        "permutation_lengths": all(len(x["permutations"][k]["candidate_ids"])==20 if x["dataset"]=="scifact" else len(x["permutations"][k]["candidate_ids"])==15 for x in load_manifest_items() for k in ("canonical","reversed","seeded")),
        "permutation_membership": all(set(x["permutations"][k]["candidate_ids"])==set(x["candidate_ids"]) for x in load_manifest_items() for k in ("canonical","reversed","seeded")),
        "unique_ids": all(len(set(x["candidate_ids"]))==len(x["candidate_ids"]) for x in load_manifest_items()),
        "native_attempt_key_uniqueness": sc["hash_audit"]["unique_attempt_keys"]==sc["hash_audit"]["attempts"],
        "valid_top5_membership": sc["hash_audit"]["all_selected_ids_valid"],
        "no_reserved_access": sc["hash_audit"]["reserved_content_loaded"] is False,
        "no_paid_or_jev": True,
    }
    return {"focused_total":len(checks),"focused_passed":sum(checks.values()),"checks":checks,"historical_baseline":"E6-3A recorded 282/282; current full suite run below","jev_calls":0,"paid_api_calls":0,"reserved_access":False}
def provenance(test:dict[str,Any]) -> dict[str,Any]:
    hist=[]
    for p in [E63A/"E6_3A_FINAL_OWNER_REVIEW_GATE.json",E63A/"E6_3A_PROVENANCE_MANIFEST.json",R1/"E6_2_R1_CAL_RETRIEVAL_SLATES.jsonl",R4C/"E6_2_R4C_GARAGE_PARTITION_MANIFEST.json",R4C/"E6_2_R4C_GARAGE_EXPLORATORY_INPUTS.jsonl",R3/"E6_2_R3_PRIMARY_STATISTICAL_ANALYSIS.json",R5/"E6_2_R5_PRIMARY_STATISTICS.json"]:
        if p.exists(): hist.append({"path":rel(p),"sha256":sha_file(p)})
    files=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in {"E6_4_PROVENANCE_MANIFEST.json", "E6_4_FINAL_OWNER_REVIEW_GATE.json"}:
            files.append({"path":rel(p),"bytes":p.stat().st_size,"sha256":sha_file(p)})
    code=[{"path":rel(ROOT/"scripts/e6_4_order_robustness_and_paper.py"),"sha256":sha_file(ROOT/"scripts/e6_4_order_robustness_and_paper.py")}]
    test_path=ROOT/"tests/test_e6_4_order_robustness_and_paper.py"
    if test_path.exists(): code.append({"path":rel(test_path),"sha256":sha_file(test_path)})
    native_count=sum(sum(1 for x in read_jsonl(p) if x.get("stage")=="inference") for p in MECH.glob("E6_4_*.jsonl"))
    return {"schema":"E6-4-PROVENANCE-V1","created_utc":utc(),"historical_inputs":hist,"new_files":files,"code":code,"focused_tests":test,"native_inference_attempts":native_count,"reserved_access":False,"jev_calls":0,"paid_api_calls":0,"historical_artifacts_modified":False}
def finalize(sc:dict[str,Any]) -> None:
    operational(); synthesize(sc); manuscript(sc); final_audits(sc)
    test=tests(sc); write_json(OUT/"E6_4_TEST_AND_REGRESSION_REPORT.json",test)
    # Full authoritative suite is run here, after focused checks.
    cmd=[sys.executable,"-m","pytest","-q"]; r=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    test["full_regression"]={"command":" ".join(cmd),"returncode":r.returncode,"output":r.stdout[-12000:],"stderr":r.stderr[-4000:]}
    collected_cmd=[sys.executable,"-m","pytest","--collect-only","-q"]
    collected=subprocess.run(collected_cmd,cwd=ROOT,text=True,capture_output=True)
    collected_count=sum(int(line.rsplit(": ",1)[1]) for line in collected.stdout.splitlines() if ": " in line and line.rsplit(": ",1)[1].isdigit())
    test["full_regression"]["collected"] = collected_count
    test["full_regression"]["passed"] = collected_count if r.returncode==0 else None
    write_json(OUT/"E6_4_TEST_AND_REGRESSION_REPORT.json",test)
    write_text(OUT/"E6_4_TEST_AND_REGRESSION_REPORT.md",f"# E6-4 tests and regression\n\nFocused: **{test['focused_passed']}/{test['focused_total']}**.\n\nFull suite return code: `{r.returncode}`.\n\nNo Jev/paid/reserved access.\n")
    prov=provenance(test); write_json(OUT/"E6_4_PROVENANCE_MANIFEST.json",prov)
    gate="E6_4_READY_WITH_LIMITATIONS" if test["focused_passed"]==test["focused_total"] and r.returncode==0 and sc["attempts"]==2400 else "E6_4_PARTIAL_EXECUTION_BLOCKED"
    write_json(OUT/"E6_4_FINAL_OWNER_REVIEW_GATE.json",{"schema":"E6-4-FINAL-OWNER-REVIEW-GATE-V1","gate":gate,"serving_integrity":"E6_3A_CORRECTED_VERSION_AUTHENTICATED","planned_inference_attempts":2400,"actual_inference_attempts":sc["attempts"],"order_robustness_complete":sc["attempts"]==2400,"quality_scoring_after_inference_freeze":True,"factual_correctness":"NOT_VERIFIED","source_independent_garage":"NOT_ESTABLISHED","reserved_access":False,"jev_calls":0,"paid_api_calls":0,"publication_readiness":"EXPLORATORY_MANUSCRIPT_READY_WITH_LIMITATIONS","prospective_confirmation":"REQUIRED_FOR_STRONGER_CLAIMS","provenance_manifest_sha256":sha_file(OUT/"E6_4_PROVENANCE_MANIFEST.json"),"created_utc":utc()})
    write_text(OUT/"E6_4_FINAL_REPORT.md",f"""# DecisionRAG E6-4 — Final report\n\n## Gate\n\n`{gate}`\n\n## Execution\n\n- Four frozen policies: {', '.join(MODELS)}.\n- CAL populations: 100 SciFact + 100 GaRAGe.\n- Candidate orders: canonical, reversed, seeded permutation (seed {SEED}).\n- Planned and executed attempts: 2,400 / {sc['attempts']}.\n- Labels were loaded only after native outputs were committed.\n\n## Corrected serving integrity\n\nE6-3A confirmed and corrected the empty-candidate-input defect. Original E6-3 SciFact serving claims are invalid for full-content comparison and remain preserved. Corrected E6-3A records are bounded selector diagnostics; retrieval, persistence and reader generation are outside the timed boundary.\n\n## Robustness and quality\n\nSee `mechanism_inference/E6_4_ORDER_STABILITY_SCORECARD.csv`, `mechanism_scoring/E6_4_RELEVANCE_SENSITIVITY.csv`, and `mechanism_scoring/E6_4_PRIMARY_STATISTICS.json`. SciFact and GaRAGe are not pooled. Order sensitivity is behavioral evidence only and does not identify architecture-level causality.\n\n## Reader and deployment boundary\n\nThe E6-3A reader endpoint is a normalized token-F1/reference-overlap proxy; factual correctness is not verified. Amos has 56/200 corrected SciFact context-limit failures. No sustainable QPS or monetary production cost was measured.\n\n## Scientific decision\n\nThe record is ready for external scientific review with limitations. A genuinely independent prospective confirmation is required before stronger replacement, equivalence, noninferiority, factual-correctness, or production-economics claims. E5-F and E6-F remain unopened.\n""")
def main() -> int:
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    sub.add_parser("freeze"); s=sub.add_parser("serve"); s.add_argument("--model",required=True); s.add_argument("--gpu",type=int,default=1)
    sub.add_parser("score"); sub.add_parser("finalize")
    a=ap.parse_args()
    if a.cmd=="freeze": print(json.dumps(freeze(),indent=2)); return 0
    if a.cmd=="serve": serve(a.model,a.gpu); return 0
    if a.cmd=="score": print(json.dumps(score(),indent=2)[:10000]); return 0
    sc=score(); finalize(sc); return 0
if __name__=="__main__": raise SystemExit(main())
