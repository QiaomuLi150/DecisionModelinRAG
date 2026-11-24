#!/usr/bin/env python3
"""Score one frozen model-specific D_budget text view with Jina v3.5.

The budget view is reconstructed from the exact frozen candidate strings using
the D model's pinned tokenizer and the persisted per-query cap.  Jina receives
those exact Unicode strings, without an independent Jina truncation policy.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "results/non_jev/v2/reviewer_response_final_v1_0/ranking"
SLATES = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
JINA = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/e8a93f33f0b22108f8c2364f8484ce3422552fbc")


def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_obj(x):
    return hashlib.sha256(canon(x).encode()).hexdigest()


def sha_bytes(x):
    return hashlib.sha256(x).hexdigest()


def choose(ids, scores):
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i))[:5]]


def enc(tok, text):
    try:
        return list(tok.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tok.encode(text))


def dec(tok, ids):
    try:
        return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except TypeError:
        return tok.decode(ids)


def rows(policy):
    caps = {}
    with (OUT / "04_RR_BUDGET_VIEW_HASHES.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["policy_id"] == policy:
                caps[r["canonical_id"]] = int(r["budget_cap"])
    if len(caps) != 1384:
        raise RuntimeError(f"expected 1384 persisted {policy} caps, got {len(caps)}")
    sys.path.insert(0, str(ROOT / "data/models/e12_o_runtimes/kev"))
    from transformers import AutoTokenizer
    base = str(ROOT / "data/models/e12_o_checkpoints/kev9_base" if policy == "kev9" else ROOT / "data/models/e12_o_checkpoints/drex_dlm8")
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True, local_files_only=True)
    out = []
    with SLATES.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key = f"{r['domain']}:{r['query_id']}"
            cs = r["candidates"]
            ids = [str(c["candidate_id"]) for c in cs]
            full = [str(c.get("content", "")) for c in cs]
            cap = caps[key]
            budget = [dec(tok, enc(tok, t)[:cap]) for t in full]
            out.append({"canonical_id": key, "domain": r["domain"], "query_id": str(r["query_id"]), "query": str(r.get("query", r.get("question", ""))), "candidate_ids": ids, "full_texts": full, "budget_texts": budget, "budget_cap": cap, "full_hash": sha_obj(full), "budget_hash": sha_obj(budget)})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", choices=["kev9", "drex_dlm8"], required=True); args = ap.parse_args()
    import torch
    from transformers import AutoModel
    OUT.mkdir(parents=True, exist_ok=True)
    views = rows(args.policy)
    model = AutoModel.from_pretrained(str(JINA), dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda").eval()
    torch.cuda.synchronize()
    raw = []
    for n, v in enumerate(views, 1):
        t = time.perf_counter(); status = "FAIL"; error = None; scores = []; selected = []
        try:
            result = model.rerank(v["query"], v["budget_texts"])
            mapping = {int(x["index"]): float(x["relevance_score"]) for x in result}
            if set(mapping) != set(range(20)):
                raise RuntimeError("Jina output indices do not cover all 20 candidates")
            scores = [mapping[i] for i in range(20)]
            if not all(math.isfinite(x) for x in scores):
                raise ValueError("non-finite Jina score")
            selected = choose(v["candidate_ids"], scores)
            if len(set(selected)) != 5:
                raise ValueError("invalid Top-5 IDs")
            status = "PASS"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        torch.cuda.synchronize()
        raw.append({"dataset": "BRIGHT", "domain": v["domain"], "query_id": v["query_id"], "canonical_id": v["canonical_id"], "policy_id": f"jina_D_budget_{args.policy}", "status": status, "error": error, "candidate_ids": v["candidate_ids"], "scores": scores, "selected_ids": selected, "budget_source_policy": args.policy, "source_candidate_text_sha256": v["full_hash"], "budget_candidate_text_sha256": v["budget_hash"], "budget_cap": v["budget_cap"], "native_input_sha256": sha_obj({"query": v["query"], "candidate_ids": v["candidate_ids"], "texts": v["budget_texts"]}), "elapsed_ms": round((time.perf_counter() - t) * 1000, 3), "native_interface": "Jina v3.5 native listwise rerank, exact D-budget Unicode strings", "model_revision": "e8a93f33f0b22108f8c2364f8484ce3422552fbc", "physical_forward_count": "NOT_EXPOSED_BY_RUNTIME", "gpu": {"visible_device": int(torch.cuda.current_device()), "name": torch.cuda.get_device_name(0), "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 3)}, "labels_joined": False, "raw_output_sha256": sha_obj({"scores": scores, "selected_ids": selected, "status": status})})
        if n % 25 == 0:
            print(json.dumps({"policy": args.policy, "completed": n, "total": len(views), "status": status}), flush=True)
    path = OUT / f"04_RR_jina_D_budget_{args.policy}_RAW.jsonl"
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in raw), encoding="utf-8")
    (OUT / f"04_RR_jina_D_budget_{args.policy}_SUMMARY.json").write_text(json.dumps({"policy_id": f"jina_D_budget_{args.policy}", "n": len(raw), "pass": sum(x["status"] == "PASS" for x in raw), "fail": sum(x["status"] != "PASS" for x in raw), "raw_sha256": sha_bytes(path.read_bytes()), "gpu_visible_device": int(torch.cuda.current_device()), "labels_joined": False}, indent=2) + "\n", encoding="utf-8")
    del model; gc.collect(); torch.cuda.empty_cache()
    print(json.dumps({"policy": args.policy, "n": len(raw), "pass": sum(x["status"] == "PASS" for x in raw), "fail": sum(x["status"] != "PASS" for x in raw), "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
