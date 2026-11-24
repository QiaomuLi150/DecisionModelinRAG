#!/usr/bin/env python3
"""Physical two-stage PPLX inference for frozen non-original candidate orders."""
from __future__ import annotations
import argparse, gc, hashlib, json, math, os, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v1_3/r3_multipass_ordered/raw"
E9 = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
PPLX_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")

sys.path.insert(0, str(ROOT / "scripts"))
import v1_3_r3_multipass as base

def sha(x): return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def fsha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def order_ids(ids, oid):
    if oid == "reversed": return list(reversed(ids))
    p = list(ids); random.Random(int(oid.rsplit("_", 1)[1])).shuffle(p); return p

def gpu_snapshot():
    return base.gpu_snapshot()

def load_slates(start, limit):
    rows = [json.loads(x) for x in E9.open(encoding="utf-8") if x.strip()]
    rows = sorted(rows, key=lambda x: (x["domain"], str(x["query_id"])))
    return rows[start:start+limit] if limit else rows[start:]

def reordered(item, oid):
    ids = order_ids([str(x["candidate_id"]) for x in item["candidates"]], oid)
    by = {str(x["candidate_id"]): {"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in item["candidates"]}
    return [{"candidate_id": x, "content": by[x]["content"]} for x in ids]

def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    sys.path.insert(0, str(PPLX_PATH / "source" / "src"))
    from autojev.model import DecisionModel, decision_messages
    from transformers import AutoProcessor
    slates = load_slates(args.start, args.limit)
    load_t = time.perf_counter(); proc = AutoProcessor.from_pretrained(str(PPLX_PATH), local_files_only=True); tok = proc.tokenizer; model = DecisionModel(PPLX_PATH, device="cuda:0", max_image_pixels=512*512); torch.cuda.synchronize(); load_ms = round((time.perf_counter()-load_t)*1000, 3)
    out = []
    for n, item in enumerate(slates, 1):
        for oid in args.orders.split(","):
            docs20 = reordered(item, oid)
            row = {"canonical_id": f"{item['domain']}:{item['query_id']}", "domain": item["domain"], "query_id": str(item["query_id"]), "policy": "pplx_two_stage", "order_id": oid, "candidate_ids": [x["candidate_id"] for x in docs20], "candidate_ids_sha256": sha([x["candidate_id"] for x in docs20]), "source_slate_sha256": item["slate_sha256"], "status": "FAIL", "selected_ids": [], "stage_selected_ids": [], "n_calls": 0, "total_ms": None, "errors": [], "labels_joined": False, "gpu": args.gpu, "physical_forward_count": "NOT_EXPOSED_BY_RUNTIME", "model_revision": PPLX_REV}
            started = time.perf_counter(); stage_top = []
            try:
                for stage, docs0 in [("stage1_window1", docs20[:10]), ("stage1_window2", docs20[10:20])]:
                    docs, info = base.fit_docs(item, docs0, proc, decision_messages, tok, [f"candidate_{i:03d}" for i in range(10)]); row["n_calls"] += 1
                    if docs is None: raise RuntimeError(info.get("error_class", "UNBUDGETABLE_QUERY_OR_OVERHEAD"))
                    req = base.make_row(item, docs, [f"candidate_{i:03d}" for i in range(10)]); scores = [float(x) for x in model.predict([req], batch_size=1)[0]]; ids = [x["candidate_id"] for x in docs]
                    if len(scores) != 10 or not all(math.isfinite(x) for x in scores): raise ValueError("invalid stage-1 score vector")
                    idx = sorted(range(10), key=lambda i: (-scores[i], i, ids[i])); top = [docs[i] for i in idx[:5]]; stage_top.extend(top)
                    row.setdefault("stage_views", []).append({"stage": stage, "candidate_ids": ids, "candidate_ids_sha256": sha(ids), "input_sha256": sha(req), "scores_sha256": sha(scores), "budget": info})
                docs, info = base.fit_docs(item, stage_top, proc, decision_messages, tok, [f"candidate_{i:03d}" for i in range(10)]); row["n_calls"] += 1
                if docs is None: raise RuntimeError(info.get("error_class", "UNBUDGETABLE_QUERY_OR_OVERHEAD"))
                req = base.make_row(item, docs, [f"candidate_{i:03d}" for i in range(10)]); scores = [float(x) for x in model.predict([req], batch_size=1)[0]]; ids = [x["candidate_id"] for x in docs]
                if len(scores) != 10 or not all(math.isfinite(x) for x in scores): raise ValueError("invalid stage-2 score vector")
                idx = sorted(range(10), key=lambda i: (-scores[i], i, ids[i]))
                row.update({"status": "PASS", "selected_ids": [ids[i] for i in idx[:5]], "stage_selected_ids": [x["candidate_id"] for x in stage_top], "stage2_view": {"candidate_ids": ids, "candidate_ids_sha256": sha(ids), "input_sha256": sha(req), "scores_sha256": sha(scores), "budget": info}})
            except Exception as exc:
                row["errors"].append({"error_class": type(exc).__name__, "message": str(exc)}); row["error"] = f"{type(exc).__name__}: {exc}"
            torch.cuda.synchronize(); row["max_memory_allocated_mib"] = round(torch.cuda.max_memory_allocated()/2**20, 3); row["total_ms"] = round((time.perf_counter()-started)*1000, 3); out.append(row)
        if n % 25 == 0: print(json.dumps({"gpu": args.gpu, "completed_queries": n, "total_queries": len(slates), "orders": args.orders}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True); suffix=f"gpu{args.gpu}_start{args.start}_n{len(slates)}_orders_{args.orders.replace(',', '_')}"; path=OUT/f"multipass_{suffix}.jsonl"; path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True)+"\n" for x in out), encoding="utf-8"); meta={"policy":"pplx_two_stage","gpu":args.gpu,"orders":args.orders.split(","),"start":args.start,"n_queries":len(slates),"n_rows":len(out),"valid":sum(x["status"]=="PASS" for x in out),"failed":sum(x["status"]!="PASS" for x in out),"load_ms":load_ms,"raw_sha256":fsha(path),"labels_joined":False,"model_revision":PPLX_REV}; (OUT/f"multipass_{suffix}.meta.json").write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n"); print(json.dumps(meta, indent=2), flush=True); del model; gc.collect(); torch.cuda.empty_cache()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--gpu",type=int,required=True); ap.add_argument("--orders",required=True); ap.add_argument("--start",type=int,default=0); ap.add_argument("--limit",type=int,default=0); run(ap.parse_args())

if __name__ == "__main__": main()
