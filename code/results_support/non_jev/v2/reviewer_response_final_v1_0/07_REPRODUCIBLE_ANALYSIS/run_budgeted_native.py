#!/usr/bin/env python3
"""Run the frozen reviewer-response budgeted native ranking arms.

Only the frozen BRIGHT slates are read.  No qrels, known-positive sidecars,
prior selections, or reader answers are imported into the request.  Each model
gets its own tokenizer, an equal per-candidate prefix cap, and the frozen
128-token safety reserve.  Outputs are written only below the new reviewer
response directory.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "results/non_jev/v2/reviewer_response_final_v1_0/ranking"
SLATES = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
MAX_CONTEXT = 8192
MARGIN = 128


def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_obj(x):
    return hashlib.sha256(canon(x).encode()).hexdigest()


def sha_bytes(x):
    return hashlib.sha256(x).hexdigest()


def choose(ids, scores):
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i))[:5]]


def bright_views():
    rows = []
    with SLATES.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cs = r["candidates"]
            ids = [str(c["candidate_id"]) for c in cs]
            texts = [str(c.get("content", "")) for c in cs]
            query = str(r.get("query", r.get("question", "")))
            criteria = {i: t for i, t in zip(ids, texts)}
            req = {"state": {"query": query}, "model": "reviewer-response-budgeted", "questions": {"candidate": {"type": "choice", "instructions": "Which candidate passage is most relevant to the query?", "criteria": criteria}}}
            rows.append({"dataset": "BRIGHT", "domain": r["domain"], "query_id": str(r["query_id"]), "canonical_id": f"{r['domain']}:{r['query_id']}", "query": query, "candidate_ids": ids, "candidate_texts": texts, "full_request": req})
    return rows


def encode_ids(tok, text):
    try:
        return list(tok.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tok.encode(text))


def decode_ids(tok, ids):
    try:
        return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except TypeError:
        return tok.decode(ids)


def make_request(v, texts, model_name):
    criteria = {i: t for i, t in zip(v["candidate_ids"], texts)}
    return {"state": {"query": v["query"]}, "model": model_name, "questions": {"candidate": {"type": "choice", "instructions": "Which candidate passage is most relevant to the query?", "criteria": criteria}}}


def build_budget_view(v, tok, model, policy):
    token_lists = [encode_ids(tok, x) for x in v["candidate_texts"]]
    overhead_req = make_request(v, [""] * 20, f"reviewer-response-{policy}-budget")
    overhead = None
    try:
        rr, _ = model._request_to_record if False else (None, None)  # keep this branch side-effect free
    except Exception:
        pass
    # The native System-One adapter is the authoritative serialization path.
    # The caller supplies a helper below; this function only computes a first
    # cap from tokenizer-visible content and leaves exact rechecking to it.
    content_cap = max(1, (MAX_CONTEXT - MARGIN - len(encode_ids(tok, v["query"]))) // 20)
    retained = [min(len(x), content_cap) for x in token_lists]
    texts = [decode_ids(tok, ids[:content_cap]) for ids in token_lists]
    return texts, [len(x) for x in token_lists], retained, content_cap


def run_policy(policy):
    import torch
    if policy == "drex_dlm8":
        sys.path.insert(0, str(ROOT / "data/models/e12_o_checkpoints/drex_dlm8/code"))
        from kev.checkpoint import load
        from kev.api import SystemOneRequest, to_record
        tok, model = load(str(ROOT / "data/models/e12_o_checkpoints/drex_dlm8"), device="cuda", dtype=torch.bfloat16)
        interface = "Drex DLM official System-One typed pointer"
        forwards = 1
    elif policy == "kev9":
        sys.path.insert(0, str(ROOT / "data/models/e12_o_runtimes/kev"))
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.api import SystemOneRequest, to_record
        ck = Checkpoint(str(ROOT / "data/models/e12_o_checkpoints/kev9_adapter"))
        ck.meta.base = str(ROOT / "data/models/e12_o_checkpoints/kev9_base")
        ck.meta.base_revision = None
        tok, model = ck.load("cuda", LoadOptions(dtype=torch.bfloat16, merge=True, backend="torch", cuda_graphs=False, fused=False))
        interface = "Kev official System-One typed pointer"
        forwards = 2
    else:
        raise ValueError(policy)
    rows = []
    views = bright_views()
    budget_hash_rows = []
    for n, v in enumerate(views, 1):
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        status = "FAIL"
        error = None
        scores = []
        selected = []
        final_request = None
        original_tokens = []
        retained_tokens = []
        cap = None
        actual_tokens = None
        try:
            # Make the exact same native request type used by the historical
            # adapter, then reduce the shared cap until strict serialization fits.
            token_lists = [encode_ids(tok, x) for x in v["candidate_texts"]]
            cap = max(1, (MAX_CONTEXT - MARGIN - len(encode_ids(tok, v["query"]))) // 20)
            while cap >= 1:
                texts = [decode_ids(tok, ids[:cap]) for ids in token_lists]
                req = make_request(v, texts, f"reviewer-response-{policy}-budget")
                native_req = SystemOneRequest.model_validate(req)
                record, _ = to_record(native_req)
                try:
                    enc = model.encode(tok, record, max_state=MAX_CONTEXT, max_branch=MAX_CONTEXT, strict=True)
                    actual_tokens = len(enc["ids"])
                    final_request = req
                    original_tokens = [len(x) for x in token_lists]
                    retained_tokens = [min(len(x), cap) for x in token_lists]
                    break
                except Exception as e:
                    if cap == 1:
                        raise RuntimeError(f"INPUT_UNBUDGETABLE: {type(e).__name__}: {e}")
                    cap -= 1
            if final_request is None:
                raise RuntimeError("INPUT_UNBUDGETABLE: no valid shared prefix cap")
            with torch.inference_mode():
                probs = model.probs(enc)
            torch.cuda.synchronize()
            scores = [float(x) for x in probs[0].tolist()]
            if len(scores) != len(v["candidate_ids"]) or not all(math.isfinite(x) for x in scores):
                raise ValueError("invalid candidate-aligned score vector")
            selected = choose(v["candidate_ids"], scores)
            if len(set(selected)) != 5:
                raise ValueError("selection did not contain five unique IDs")
            status = "PASS"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            if "INPUT_UNBUDGETABLE" in error:
                status = "INPUT_UNBUDGETABLE"
        elapsed = (time.perf_counter() - started) * 1000
        source_text_hash = sha_obj(v["candidate_texts"])
        budget_texts = final_request["questions"]["candidate"]["criteria"] if final_request else {}
        budget_hash = sha_obj([budget_texts.get(i, "") for i in v["candidate_ids"]]) if final_request else None
        budget_hash_rows.append({"canonical_id": v["canonical_id"], "policy_id": policy, "content_hash_full": source_text_hash, "content_hash_budget": budget_hash, "budget_cap": cap, "original_tokens_total": sum(original_tokens) if original_tokens else None, "retained_tokens_total": sum(retained_tokens) if retained_tokens else None, "retained_tokens_each": json.dumps(retained_tokens), "serialized_tokens": actual_tokens, "status": status})
        rec = {"dataset": v["dataset"], "domain": v["domain"], "query_id": v["query_id"], "canonical_id": v["canonical_id"], "policy_id": policy, "status": status, "error": error, "candidate_ids": v["candidate_ids"], "scores": scores, "selected_ids": selected, "input_sha256": sha_obj(final_request) if final_request else None, "native_request_sha256": sha_obj(final_request) if final_request else None, "source_candidate_text_sha256": source_text_hash, "budget_candidate_text_sha256": budget_hash, "elapsed_ms": round(elapsed, 3), "labels_joined": False, "input_tokens": actual_tokens, "original_tokens": original_tokens, "retained_tokens": retained_tokens, "budget_cap": cap, "native_interface": interface, "score_direction": "higher_better", "physical_forward_calls": forwards if status == "PASS" else 0, "gpu": {"visible_device": int(torch.cuda.current_device()), "name": torch.cuda.get_device_name(0), "memory_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 3), "memory_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 3), "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 3)}, "raw_output_sha256": sha_obj({"candidate_ids": v["candidate_ids"], "scores": scores, "selected_ids": selected, "status": status})}
        rows.append(rec)
        if n % 25 == 0:
            print(json.dumps({"policy": policy, "completed": n, "total": len(views), "status": status}), flush=True)
    out = OUT / f"04_RR_{policy}_BUDGET_RAW.jsonl"
    out.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    (OUT / f"04_RR_{policy}_BUDGET_SUMMARY.json").write_text(json.dumps({"policy_id": policy, "n": len(rows), "pass": sum(x["status"] == "PASS" for x in rows), "fail": sum(x["status"] != "PASS" for x in rows), "raw_sha256": sha_bytes(out.read_bytes()), "gpu_visible_device": int(torch.cuda.current_device()), "labels_joined": False}, indent=2) + "\n", encoding="utf-8")
    with (OUT / "04_RR_BUDGET_VIEW_HASHES.csv").open("a", encoding="utf-8") as f:
        if f.tell() == 0:
            f.write("canonical_id,policy_id,content_hash_full,content_hash_budget,budget_cap,original_tokens_total,retained_tokens_total,retained_tokens_each,serialized_tokens,status\n")
        for x in budget_hash_rows:
            f.write(",".join(json.dumps(str(x.get(k, "")), ensure_ascii=False) for k in ["canonical_id", "policy_id", "content_hash_full", "content_hash_budget", "budget_cap", "original_tokens_total", "retained_tokens_total", "retained_tokens_each", "serialized_tokens", "status"]) + "\n")
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"policy": policy, "n": len(rows), "pass": sum(x["status"] == "PASS" for x in rows), "fail": sum(x["status"] != "PASS" for x in rows), "output": str(out)}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["kev9", "drex_dlm8"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run_policy(args.policy)


if __name__ == "__main__":
    main()
