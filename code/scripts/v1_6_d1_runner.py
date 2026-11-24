#!/usr/bin/env python3
"""Label-blind D1: Jina two-stage selection on the authenticated R3 PPLX views."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
MANIFEST = ROOT / "results/v1_6/d1_perwindow_jina/D1_INPUT_MANIFEST.jsonl"
OUT = ROOT / "results/v1_6/d1_perwindow_jina"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
JINA_PATH = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots") / JINA_REV
ORDERS = ["original", "reversed", "perm_seed_1", "perm_seed_2", "perm_seed_3", "perm_seed_4", "perm_seed_5"]


def canon(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(x: Any) -> str:
    return hashlib.sha256(canon(x).encode()).hexdigest()


def fsha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prefix(tok, text: str, cap: int) -> str:
    return tok.decode(tok.encode(str(text), add_special_tokens=False)[:cap], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def prefix_many(tok, texts: list[str], caps: list[int]) -> list[str]:
    """Batch the expensive source tokenization while preserving prefix semantics."""
    if not texts:
        return []
    encoded = tok(texts, add_special_tokens=False, padding=False, return_attention_mask=False)["input_ids"]
    return [tok.decode(ids[:cap], skip_special_tokens=False, clean_up_tokenization_spaces=False) for ids, cap in zip(encoded, caps)]


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]


def stable_top(ids: list[str], scores: list[float], k: int = 5) -> list[int]:
    return sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i, ids[i]))[:k]


def make_row(item: dict[str, Any], docs: list[dict[str, str]], codes: list[str]) -> dict[str, Any]:
    return {
        "state": f"Query: {item['query']}\nEvaluate the supplied candidate passages using the option descriptions.",
        "question": {
            "type": "choice",
            "instructions": f"Select the candidate passage that best answers the query: {item['query']}",
            "criteria": {codes[i]: f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(docs)},
        },
    }


def make_plan(item: dict[str, Any], m: dict[str, Any], source_by_id: dict[str, str], tok, codes: list[str]) -> dict[str, Any]:
    if m["pplx_status"] != "PASS":
        return {"manifest": m, "item": item, "failed_source": True}
    stage1 = []
    stage1_by_id: dict[str, dict[str, str]] = {}
    for s in m["stages"]:
        if s["stage"].startswith("stage1_"):
            docs = [{"candidate_id": str(i), "content": prefix(tok, source_by_id[str(i)], int(s["final_cap"]))} for i in s["candidate_ids"]]
            if sha(make_row(item, docs, codes)) != s["input_sha256"]:
                raise RuntimeError(f"D1 source hash mismatch stage1 {m['canonical_id']} {m['order_id']}")
            stage1.append(docs)
            stage1_by_id.update({d["candidate_id"]: d for d in docs})
    stage2_record = next(s for s in m["stages"] if s["stage"] == "stage2_survivors")
    pplx_stage2 = [{"candidate_id": d["candidate_id"], "content": prefix(tok, stage1_by_id[d["candidate_id"]]["content"], int(stage2_record["final_cap"]))} for d in [stage1_by_id[i] for i in m["pplx_stage_selected_ids"]]]
    if sha(make_row(item, pplx_stage2, codes)) != stage2_record["input_sha256"]:
        raise RuntimeError(f"D1 source hash mismatch stage2 {m['canonical_id']} {m['order_id']}")
    return {"manifest": m, "item": item, "stage1": stage1, "stage1_by_id": stage1_by_id, "pplx_stage2": pplx_stage2, "failed_source": False}


def make_plans_batch(ms: list[dict[str, Any]], slates: dict[str, dict[str, Any]], tok, codes: list[str]) -> list[dict[str, Any]]:
    """Build only one execution batch, using batched tokenization for exact views."""
    plans = [{"manifest": m, "item": slates[m["canonical_id"]], "failed_source": m["pplx_status"] != "PASS"} for m in ms]
    work = []
    stage_slots = []
    for pi, p in enumerate(plans):
        if p["failed_source"]:
            continue
        item = p["item"]
        source = {str(d["candidate_id"]): str(d["content"]) for d in item["candidates"]}
        p["stage1"] = []
        p["stage1_by_id"] = {}
        for si, s in enumerate([s for s in p["manifest"]["stages"] if s["stage"].startswith("stage1_")]):
            for cid in s["candidate_ids"]:
                work.append((source[str(cid)], int(s["final_cap"])))
                stage_slots.append((pi, si, str(cid), s))
    texts = prefix_many(tok, [x[0] for x in work], [x[1] for x in work])
    cursors = [0] * len(plans)
    for pi, p in enumerate(plans):
        if p["failed_source"]:
            continue
        srecords = [s for s in p["manifest"]["stages"] if s["stage"].startswith("stage1_")]
        p["stage1"] = [[] for _ in srecords]
        for slot, content in zip(stage_slots, texts):
            qpi, si, cid, s = slot
            if qpi != pi:
                continue
            d = {"candidate_id": cid, "content": content}
            p["stage1"][si].append(d)
            p["stage1_by_id"][cid] = d
        for s, docs in zip(srecords, p["stage1"]):
            got = sha(make_row(p["item"], docs, codes))
            if got != s["input_sha256"]:
                raise RuntimeError(f"D1 source hash mismatch stage1 {p['manifest']['canonical_id']} {p['manifest']['order_id']}")
        s2 = next(s for s in p["manifest"]["stages"] if s["stage"] == "stage2_survivors")
        survivor_ids = [str(x) for x in p["manifest"]["pplx_stage_selected_ids"]]
        s2_work = [(p["stage1_by_id"][cid]["content"], int(s2["final_cap"])) for cid in survivor_ids]
        s2_texts = prefix_many(tok, [x[0] for x in s2_work], [x[1] for x in s2_work])
        p["pplx_stage2"] = [{"candidate_id": cid, "content": text} for cid, text in zip(survivor_ids, s2_texts)]
        got = sha(make_row(p["item"], p["pplx_stage2"], codes))
        if got != s2["input_sha256"]:
            raise RuntimeError(f"D1 source hash mismatch stage2 {p['manifest']['canonical_id']} {p['manifest']['order_id']}")
    return plans


def score_many(model, tokenizer, torch, requests: list[tuple[str, list[dict[str, str]]]]) -> list[tuple[list[float], dict[str, Any]]]:
    if not requests:
        return []
    remote = __import__(model.__class__.__module__, fromlist=["format_docs_prompts_func"])
    format_prompt = remote.format_docs_prompts_func
    model._ensure_tokenizer()
    prepared = []
    for query, docs in requests:
        q2, d2, lengths, qlen = model._truncate_texts(query, [d["content"] for d in docs], 1024, 8192)
        if 2 * qlen + sum(lengths) > tokenizer.model_max_length:
            raise RuntimeError("JINA_NATIVE_CONTEXT_TRUNCATION_REQUIRED")
        prompt = format_prompt(q2, d2, instruction=None, special_tokens=model.special_tokens, no_thinking=True)
        prepared.append(prompt)
    batch = tokenizer(text=prepared, padding=True, padding_side="left", return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model.forward(**batch)
    native = outputs.scores.detach().cpu().float().numpy()
    out = []
    for (query, docs), row_scores in zip(requests, native):
        scores = [float(x) for x in row_scores[: len(docs)]]
        if len(scores) != len(docs) or not all(math.isfinite(x) for x in scores):
            raise ValueError("invalid native score shape/value")
        req = {"query": query, "candidate_ids": [d["candidate_id"] for d in docs], "candidate_texts": [d["content"] for d in docs]}
        out.append((scores, {"request_sha256": sha(req), "candidate_ids_sha256": sha([d["candidate_id"] for d in docs]), "scores_sha256": sha(scores)}))
    return out


def process_batch(plans, model, tokenizer, torch, batch_size: int, gpu: int):
    rows = []
    valid_plans = [p for p in plans if not p.get("failed_source")]
    for p in plans:
        m = p["manifest"]
        base = {"canonical_id": m["canonical_id"], "domain": m["domain"], "query_id": m["query_id"], "order_id": m["order_id"], "policy": "jina_two_stage_pplxwindow", "status": "FAIL", "valid": False, "selected_ids": [], "stage_selected_ids": [], "errors": [], "labels_joined": False, "model_revision": JINA_REV, "gpu_index": gpu, "source_slate_sha256": m["source_slate_sha256"], "pplx_selected_ids": m["pplx_selected_ids"], "pplx_stage_selected_ids": m["pplx_stage_selected_ids"], "survivor_set_matches_pplx": False, "n_calls": 0}
        if p.get("failed_source"):
            base["errors"] = m.get("pplx_error_classes", []) + ["PPLX_SOURCE_FAILURE"]
        rows.append(base)
    if not valid_plans:
        return rows
    try:
        surv = [[] for _ in valid_plans]
        stage_records = [[] for _ in valid_plans]
        for w in (0, 1):
            reqs = [(p["item"]["query"], p["stage1"][w]) for p in valid_plans]
            scores_out = score_many(model, tokenizer, torch, reqs)
            for i, (scores, meta) in enumerate(scores_out):
                ids = [d["candidate_id"] for d in valid_plans[i]["stage1"][w]]
                keep = stable_top(ids, scores)
                surv[i].extend(ids[k] for k in keep)
                stage_records[i].append({"stage": f"stage1_window{w+1}", "candidate_ids": ids, "scores": scores, **meta})
        stage2_docs = []
        matches = []
        for i, p in enumerate(valid_plans):
            pplx_set = set(p["manifest"]["pplx_stage_selected_ids"])
            same = set(surv[i]) == pplx_set and len(surv[i]) == 10
            matches.append(same)
            if same:
                docs = p["pplx_stage2"]
            else:
                docs = [p["stage1_by_id"][cid] for cid in surv[i]]
            stage2_docs.append(docs)
        scores_out = score_many(model, tokenizer, torch, [(p["item"]["query"], docs) for p, docs in zip(valid_plans, stage2_docs)])
        for i, (scores, meta) in enumerate(scores_out):
            ids = [d["candidate_id"] for d in stage2_docs[i]]
            keep = stable_top(ids, scores)
            j = rows.index(next(x for x in rows if x["canonical_id"] == valid_plans[i]["manifest"]["canonical_id"] and x["order_id"] == valid_plans[i]["manifest"]["order_id"]))
            rows[j].update({"status": "PASS", "valid": True, "selected_ids": [ids[k] for k in keep], "stage_selected_ids": surv[i], "survivor_set_matches_pplx": matches[i], "n_calls": 3, "stage_records": stage_records[i] + [{"stage": "stage2_survivors", "candidate_ids": ids, "scores": scores, **meta}], "stage2_input_rule": "pplx_stage2_if_survivor_set_equal_else_pplx_stage1_strings"})
    except Exception as exc:
        for row in rows:
            if row["status"] != "PASS" and not row["errors"]:
                row["errors"] = [{"error_class": type(exc).__name__, "message": str(exc)}]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--determinism-sample", type=int, default=20)
    ap.add_argument("--orders", default=",".join(ORDERS))
    ap.add_argument("--skip-determinism", action="store_true")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if fsha(BRIGHT) != "c304b2c35f445153181098d3a899271db6fc653b44aea50186c6efaeec8e6ea2":
        raise RuntimeError("BRIGHT hash mismatch")
    slates = {f"{x['domain']}:{x['query_id']}": x for x in read_jsonl(BRIGHT)}
    all_manifest = read_jsonl(MANIFEST)
    requested_orders = [x for x in args.orders.split(",") if x]
    if any(x not in ORDERS for x in requested_orders):
        raise RuntimeError("invalid D1 order")
    manifest = [x for x in all_manifest if x["order_id"] in requested_orders]
    if len(all_manifest) != 9688 or len({(x["canonical_id"], x["order_id"]) for x in all_manifest}) != 9688:
        raise RuntimeError("D1 manifest coverage is not exactly 1384 x 7")
    if len(manifest) != 1384 * len(requested_orders):
        raise RuntimeError("requested D1 order coverage is incomplete")
    p=Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/14c9d90137962d300f3de95633f7b6535e80c5ce")
    sys.path.insert(0, str(p / "source/src"))
    from transformers import AutoProcessor, AutoTokenizer, AutoModel
    import torch
    proc = AutoProcessor.from_pretrained(str(p), local_files_only=True)
    tok = proc.tokenizer
    codes = [f"candidate_{i:03d}" for i in range(10)]
    load_t=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(str(JINA_PATH),local_files_only=True)
    model=AutoModel.from_pretrained(str(JINA_PATH),dtype="auto",trust_remote_code=True,local_files_only=True).to("cuda:0").eval()
    torch.cuda.synchronize()
    load_ms=round((time.perf_counter()-load_t)*1000,3)
    raw_dir=OUT/"raw"; raw_dir.mkdir(parents=True,exist_ok=True)
    order_tag = "_".join(requested_orders)
    raw=raw_dir/f"d1_jina_two_stage_pplxwindow__{order_tag}.jsonl"
    partial=raw.with_suffix(raw.suffix+".partial")
    out=[]
    for start in range(0,len(manifest),args.batch_size):
        plans = make_plans_batch(manifest[start:start+args.batch_size], slates, tok, codes)
        out.extend(process_batch(plans,model,tokenizer,torch,args.batch_size,args.gpu))
        partial.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in out),encoding="utf-8")
        print(json.dumps({"completed":min(start+args.batch_size,len(manifest)),"total":len(manifest),"valid":sum(x["valid"] for x in out),"failed":sum(not x["valid"] for x in out),"orders":requested_orders}),flush=True)
    partial.replace(raw)
    meta={"path":"results/v1_6/d1_perwindow_jina/raw/d1_jina_two_stage_pplxwindow.jsonl","records":len(out),"valid":sum(x["valid"] for x in out),"failed":sum(not x["valid"] for x in out),"sha256":fsha(raw),"labels_joined":False,"model_revision":JINA_REV,"gpu_index":args.gpu,"batch_size":args.batch_size,"model_load_ms":load_ms}
    (raw.with_suffix(".meta.json")).write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n")
    # Determinism recheck is performed after the raw output is closed and hashed.
    if args.skip_determinism:
        del model,tokenizer,proc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        print(json.dumps({"status":"COMPLETE","records":len(out),"valid":meta["valid"],"failed":meta["failed"],"raw_sha256":meta["sha256"],"orders":requested_orders},indent=2),flush=True)
        return
    rng=random.Random(731); sample_indices=rng.sample(range(len(manifest)), min(args.determinism_sample,len(manifest)))
    sample_ms=[manifest[i] for i in sample_indices]
    sample_plans=make_plans_batch(sample_ms, slates, tok, codes)
    repeat=process_batch(sample_plans,model,tokenizer,torch,max(1,min(args.batch_size,len(sample_plans))),args.gpu)
    original_by_key={(x["canonical_id"],x["order_id"]):x for x in out}
    det=[]
    for r in repeat:
        o=original_by_key.get((r["canonical_id"],r["order_id"]))
        det.append({"canonical_id":r["canonical_id"],"order_id":r["order_id"],"selected_ids_equal":bool(o and o.get("selected_ids")==r.get("selected_ids")),"scores_equal":bool(o and json.dumps(o.get("stage_records",[]),sort_keys=True)==json.dumps(r.get("stage_records",[]),sort_keys=True))})
    (OUT/"D1_DETERMINISM_RECHECK.json").write_text(json.dumps({"seed":731,"requested":args.determinism_sample,"evaluated":len(det),"selected_mismatches":sum(not x["selected_ids_equal"] for x in det),"score_or_trace_mismatches":sum(not x["scores_equal"] for x in det),"records":det},indent=2,sort_keys=True)+"\n")
    del model,tokenizer,proc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    print(json.dumps({"status":"COMPLETE","records":len(out),"valid":meta["valid"],"failed":meta["failed"],"raw_sha256":meta["sha256"],"determinism":len(det)},indent=2),flush=True)


if __name__ == "__main__":
    main()
