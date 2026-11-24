#!/usr/bin/env python3
"""Label-blind R1 native inference runner.

One process owns one model and one visible GPU.  Failures are written as
records and are never retried or filled in.
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

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "v3_6_essential" / "followup_v3_repro_fixes"
PREP = BASE / "R1_transfer_budget_grid"
RAW = BASE / "raw"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
PPLX = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")
JINA = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/{JINA_REV}")


def canon(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(obj):
    return hashlib.sha256(canon(obj).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def req(query, docs):
    codes = [f"candidate_{i:03d}" for i in range(len(docs))]
    return {
        "state": f"Query: {query}\nEvaluate the supplied candidate passages using the option descriptions.",
        "question": {
            "type": "choice",
            "instructions": f"Select the candidate passage that best answers the query: {query}",
            "criteria": {codes[i]: f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(docs)},
        },
    }


def read_rows(input_path=None):
    if input_path is not None:
        return [json.loads(x) for x in Path(input_path).open(encoding="utf-8") if x.strip()]
    rows = []
    for L in (2048, 4096, 6144, 8192):
        p = PREP / f"prepared_L{L}.jsonl"
        rows.extend(json.loads(x) for x in p.open(encoding="utf-8") if x.strip())
    return rows


def top5(ids, scores):
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i, ids[i]))[:5]]


def classify(exc):
    s = str(exc).lower()
    if "out of memory" in s or "cuda oom" in s:
        return "CUDA_OOM"
    if "timeout" in s:
        return "TIMEOUT"
    if "invalid" in s or "length" in s or "token" in s:
        return "INPUT_OR_OUTPUT_VALIDATION"
    return type(exc).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("pplx", "jina"), required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--min-L", type=int, default=None)
    ap.add_argument("--max-L", type=int, default=None)
    ap.add_argument("--output-tag", default="")
    ap.add_argument("--input", default=None)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    RAW.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input)
    selected = [x for i, x in enumerate(rows) if i % args.shards == args.shard and (args.min_L is None or int(x["length_limit"]) >= args.min_L) and (args.max_L is None or int(x["length_limit"]) <= args.max_L)]
    tag = f"_{args.output_tag}" if args.output_tag else ""
    out = RAW / f"{args.model}{tag}_shard{args.shard}_of{args.shards}.jsonl"
    part = out.with_suffix(out.suffix + ".part")
    if out.exists() or part.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {out}")
    import torch
    if args.model == "pplx":
        sys.path.insert(0, str(PPLX / "source" / "src"))
        from transformers import AutoProcessor
        from autojev.model import DecisionModel
        processor = AutoProcessor.from_pretrained(str(PPLX), local_files_only=True)
        model = DecisionModel(PPLX, device="cuda:0", max_image_pixels=512 * 512)
        revision = PPLX_REV
    else:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(JINA), local_files_only=True)
        model = AutoModel.from_pretrained(str(JINA), dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda:0").eval()
        revision = JINA_REV
    torch.cuda.synchronize()
    counts = {}
    valid_n = 0
    t0 = time.perf_counter()
    with part.open("w", encoding="utf-8") as f:
        for ix, x in enumerate(selected, 1):
            docs = x.get("budgeted_candidates") or []
            ids = [str(d["candidate_id"]) for d in docs]
            request_obj = req(str(x["query"]), docs) if docs else None
            rec = {
                "dataset": x["dataset"], "canonical_id": x["canonical_id"], "query_id": x["query_id"],
                "group_id": x.get("group_id", ""), "budget_L": x["length_limit"],
                "policy": f"{args.model}_single_corrected_builder", "order_id": "original",
                "candidate_ids": ids, "source_slate_sha256": x["source_slate_sha256"],
                "input_sha256": sha(request_obj) if request_obj is not None else None,
                "serialized_tokens": x.get("serialized_tokens"), "builder_status": x["budget_info"].get("builder_status"),
                "labels_joined": False, "input_label_blind": True, "model_revision": revision,
                "gpu_index_requested": args.gpu, "gpu_visible_index": 0, "status": "FAIL", "valid": False,
                "selected_ids": [], "scores": [], "n_calls": 0, "physical_forward_count": 0,
                "error_class": None, "error": None, "latency_ms": None, "peak_memory_allocated_mib": None,
            }
            t = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            try:
                if not docs:
                    raise ValueError("BUILDER_FAIL")
                if args.model == "pplx":
                    scores = [float(v) for v in model.predict([request_obj], batch_size=1)[0]]
                else:
                    result = model.rerank(str(x["query"]), [d["content"] for d in docs])
                    by_index = {int(z["index"]): float(z["relevance_score"]) for z in result}
                    scores = [by_index[i] for i in range(len(docs))]
                if len(scores) != len(ids) or not all(math.isfinite(v) for v in scores):
                    raise ValueError("INVALID_SCORE_VECTOR")
                selected_ids = top5(ids, scores)
                if len(selected_ids) != 5 or len(set(selected_ids)) != 5:
                    raise ValueError("INVALID_TOP5")
                rec.update({"status": "PASS", "valid": True, "selected_ids": selected_ids, "scores": scores, "n_calls": 1, "physical_forward_count": 1})
                valid_n += 1
            except Exception as exc:
                rec.update({"error_class": classify(exc), "error": f"{type(exc).__name__}: {exc}"})
                counts[rec["error_class"]] = counts.get(rec["error_class"], 0) + 1
            torch.cuda.synchronize()
            rec["latency_ms"] = round((time.perf_counter() - t) * 1000.0, 3)
            rec["peak_memory_allocated_mib"] = round(float(torch.cuda.max_memory_allocated()) / 2**20, 3)
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            if ix % 50 == 0:
                print(json.dumps({"model": args.model, "gpu": args.gpu, "completed": ix, "total": len(selected), "valid": valid_n, "failures": counts}), flush=True)
    part.replace(out)
    summary = {"status": "PASS", "model": args.model, "gpu": args.gpu, "shard": args.shard, "shards": args.shards,
               "rows": len(selected), "valid": valid_n, "failures": len(selected) - valid_n, "failure_classes": counts,
               "elapsed_s": time.perf_counter() - t0, "raw_sha256": sha_file(out), "labels_joined": False,
               "input_builder": "R1 exact serialized decrement-to-fit", "no_retries": True,
               "length_filter": {"min_L": args.min_L, "max_L": args.max_L}}
    out.with_suffix(".SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
