#!/usr/bin/env python3
"""Native selector execution for the approved V2 hybrid-RRF amendment.

The input manifest is frozen before this script is allowed to run. Labels are
never loaded here. Each policy/order writes an append-only raw trace and keeps
all failures; only successful completion renames the partial trace.
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
BASE = ROOT / "results" / "v2_1_replication"
MANIFEST = BASE / "stage0_preflight_v2_hybrid" / "INPUT_MANIFEST_V2_HYBRID.jsonl"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
PPLX_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")
JINA_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/{JINA_REV}")
L = 8192
RHO = 128


def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(x):
    return hashlib.sha256(canon(x).encode()).hexdigest()


def sha_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_row(query, docs, codes):
    return {
        "state": f"Query: {query}\nEvaluate the supplied candidate passages using the option descriptions.",
        "question": {
            "type": "choice",
            "instructions": f"Select the candidate passage that best answers the query: {query}",
            "criteria": {
                codes[i]: f"Candidate ID {d['candidate_id']}. Passage: {d['content']}"
                for i, d in enumerate(docs)
            },
        },
    }


def classify_error(exc):
    msg = str(exc)
    low = msg.lower()
    if "out of memory" in low or "cuda oom" in low:
        return "CUDA_OOM"
    if "context" in low or "8192" in low or "length" in low:
        return "CONTEXT_LIMIT"
    if "timeout" in low:
        return "TIMEOUT"
    if "parse" in low or "invalid" in low:
        return "PARSE_OR_INVALID_OUTPUT"
    return type(exc).__name__


def order_rows(order_id, dataset=None, start=0, limit=0):
    rows = []
    for line in MANIFEST.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["order_id"] != order_id:
            continue
        if dataset and row["dataset"] != dataset:
            continue
        rows.append(row)
    rows.sort(key=lambda x: (x["dataset"], x["canonical_id"]))
    return rows[start:start + limit] if limit else rows[start:]


def prefix(tokenizer, text, cap):
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=cap, return_attention_mask=False)["input_ids"]
    return tokenizer.decode(ids[:cap], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def token_count(processor, decision_messages, query, docs, codes):
    messages = decision_messages(make_row(query, docs, codes), codes)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return len(processor.tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"])


def fit_docs(query, docs, processor, decision_messages):
    codes = [f"candidate_{i:03d}" for i in range(len(docs))]
    overhead = token_count(processor, decision_messages, query, [{"candidate_id": d["candidate_id"], "content": ""} for d in docs], codes)
    available = L - RHO - overhead
    if available < len(docs):
        raise RuntimeError(f"UNBUDGETABLE_QUERY_OR_OVERHEAD:{available}")
    cap = max(1, available // len(docs))
    reductions = 0
    while cap >= 1:
        budgeted = [{"candidate_id": d["candidate_id"], "content": prefix(processor.tokenizer, d["content"], cap)} for d in docs]
        total = token_count(processor, decision_messages, query, budgeted, codes)
        if total <= L - RHO:
            return budgeted, {"overhead_tokens": overhead, "effective_content_budget": available, "final_cap": cap, "reductions": reductions, "serialized_tokens": total}
        cap -= 1
        reductions += 1
    raise RuntimeError(f"UNBUDGETABLE_QUERY_OR_OVERHEAD:{available}")


def choose(ids, scores):
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i, ids[i]))[:5]]


def score_jina(adapter, query, docs):
    result = adapter.rerank(query, [d["content"] for d in docs])
    mapping = {int(x["index"]): float(x["relevance_score"]) for x in result}
    if set(mapping) != set(range(len(docs))):
        raise RuntimeError("JINA_INDEX_COVERAGE_INVALID")
    return [mapping[i] for i in range(len(docs))]


def score_pplx(adapter, query, docs, decision_messages):
    codes = [f"candidate_{i:03d}" for i in range(len(docs))]
    req = make_row(query, docs, codes)
    vals = adapter.predict([req], batch_size=1)[0]
    scores = [float(x) for x in vals]
    if len(scores) != len(docs) or not all(math.isfinite(x) for x in scores):
        raise RuntimeError("PPLX_SCORE_VECTOR_INVALID")
    return scores, req


def run_policy(policy, rows, gpu, out_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    adapter = None
    processor = None
    decision_messages = None
    budget_processor = None
    budget_decision_messages = None
    model_type = "pplx" if policy.startswith("pplx") else "jina"
    load_error = None
    load_started = time.perf_counter()
    try:
        if model_type == "pplx":
            sys.path.insert(0, str(PPLX_PATH / "source" / "src"))
            from autojev.model import DecisionModel, decision_messages as dm
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(str(PPLX_PATH), local_files_only=True)
            decision_messages = dm
            budget_processor = processor
            budget_decision_messages = dm
            adapter = DecisionModel(PPLX_PATH, device="cuda:0", max_image_pixels=512 * 512)
        else:
            from transformers import AutoModel, AutoTokenizer
            sys.path.insert(0, str(PPLX_PATH / "source" / "src"))
            from autojev.model import decision_messages as budget_decision_messages
            from transformers import AutoProcessor
            budget_processor = AutoProcessor.from_pretrained(str(PPLX_PATH), local_files_only=True)
            processor = AutoTokenizer.from_pretrained(str(JINA_PATH), local_files_only=True)
            adapter = AutoModel.from_pretrained(str(JINA_PATH), dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda:0").eval()
        torch.cuda.synchronize()
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
    load_ms = (time.perf_counter() - load_started) * 1000
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(out_path.suffix + ".part")
    if out_path.exists():
        raise RuntimeError(f"refusing to overwrite frozen raw trace: {out_path}")
    if partial.exists():
        raise RuntimeError(f"partial trace exists; inspect and resume manually: {partial}")
    valid = 0
    failures = {}
    started_all = time.perf_counter()
    with partial.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            ids = row["candidate_ids"]
            rec = {
                "dataset": row["dataset"], "canonical_id": row["canonical_id"], "query_id": row["query_id"], "group_id": row["group_id"],
                "policy": policy, "order_id": row["order_id"], "candidate_ids": ids, "candidate_ids_sha256": sha(ids),
                "source_slate_sha256": row["source_slate_sha256"], "status": "FAIL", "error_class": None, "error": None,
                "selected_ids": [], "selected_ids_sha256": sha([]), "scores": [], "n_calls": 0, "input_tokens": None,
                "labels_joined": False, "model_revision": PPLX_REV if model_type == "pplx" else JINA_REV,
                "load_ms": load_ms, "gpu_index_requested": gpu, "physical_forward_count": "NOT_EXPOSED_BY_RUNTIME",
            }
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            try:
                original = [{"candidate_id": d["candidate_id"], "content": d["content"]} for d in row["candidates"]]
                if load_error:
                    raise RuntimeError(f"MODEL_LOAD_ERROR:{load_error}")
                if policy.endswith("native"):
                    scores, req = score_pplx(adapter, row["query"], original, decision_messages)
                    rec["native_input_sha256"] = sha(req)
                    rec["input_tokens"] = row.get("native_input_tokens")
                    rec["n_calls"] = 1
                elif policy == "pplx_budget":
                    docs = row["budgeted_candidates"]
                    scores, req = score_pplx(adapter, row["query"], docs, decision_messages)
                    rec["native_input_sha256"] = sha(req)
                    rec["input_tokens"] = row.get("budgeted_input_tokens")
                    rec["n_calls"] = 1
                elif policy == "jina_full":
                    scores = score_jina(adapter, row["query"], original)
                    rec["native_input_sha256"] = sha({"query": row["query"], "candidate_ids": ids, "documents": [d["content"] for d in original]})
                    rec["input_tokens"] = row.get("native_input_tokens")
                    rec["n_calls"] = 1
                elif policy == "jina_budget":
                    docs = row["budgeted_candidates"]
                    scores = score_jina(adapter, row["query"], docs)
                    rec["native_input_sha256"] = sha({"query": row["query"], "candidate_ids": ids, "documents": [d["content"] for d in docs]})
                    rec["input_tokens"] = row.get("budgeted_input_tokens")
                    rec["n_calls"] = 1
                elif policy in {"pplx_two_stage", "jina_two_stage_matched"}:
                    stage_top = []
                    stage_records = []
                    for window_name, docs in (("A", row["window_a_candidates"]), ("B", row["window_b_candidates"])):
                        if not docs:
                            raise RuntimeError(f"UNBUDGETABLE_WINDOW_{window_name}")
                        if model_type == "pplx":
                            scores1, req1 = score_pplx(adapter, row["query"], docs, decision_messages)
                        else:
                            scores1 = score_jina(adapter, row["query"], docs)
                            req1 = {"query": row["query"], "candidate_ids": [d["candidate_id"] for d in docs], "documents": [d["content"] for d in docs]}
                        rec["n_calls"] += 1
                        idx = sorted(range(len(docs)), key=lambda i: (-scores1[i], i, docs[i]["candidate_id"]))[:5]
                        stage_top.extend([docs[i] for i in idx])
                        stage_records.append({"window": window_name, "candidate_ids": [d["candidate_id"] for d in docs], "selected_ids": [docs[i]["candidate_id"] for i in idx], "input_sha256": sha(req1), "scores_sha256": sha(scores1)})
                    stage2_docs, stage2_info = fit_docs(row["query"], stage_top, budget_processor, budget_decision_messages)
                    if model_type == "pplx":
                        scores2, req2 = score_pplx(adapter, row["query"], stage2_docs, decision_messages)
                    else:
                        scores2 = score_jina(adapter, row["query"], stage2_docs)
                        req2 = {"query": row["query"], "candidate_ids": [d["candidate_id"] for d in stage2_docs], "documents": [d["content"] for d in stage2_docs]}
                    rec["n_calls"] += 1
                    final_ids = choose([d["candidate_id"] for d in stage2_docs], scores2)
                    rec["stage1_selected_ids"] = [d["candidate_id"] for d in stage_top]
                    rec["stage_views"] = stage_records
                    rec["stage2_input_sha256"] = sha(req2)
                    rec["stage2_scores_sha256"] = sha(scores2)
                    rec["stage2_budget"] = stage2_info
                    scores = scores2
                else:
                    raise ValueError(f"unsupported policy {policy}")
                selected = choose(ids if len(scores) == 20 else [d["candidate_id"] for d in stage2_docs], scores)
                if len(selected) != 5 or len(set(selected)) != 5 or not set(selected).issubset(set(ids)):
                    raise RuntimeError("INVALID_TOP5_ID_SET")
                rec.update({"status": "PASS", "selected_ids": selected, "selected_ids_sha256": sha(selected), "scores": scores})
                valid += 1
            except Exception as exc:
                cls = classify_error(exc)
                rec.update({"error_class": cls, "error": f"{type(exc).__name__}: {exc}"})
                failures[cls] = failures.get(cls, 0) + 1
            torch.cuda.synchronize()
            rec["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            rec["peak_memory_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 3)
            handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if index % 25 == 0:
                print(json.dumps({"policy": policy, "order": rows[0]["order_id"] if rows else None, "completed": index, "total": len(rows), "valid": valid, "failures": failures}), flush=True)
    os.replace(partial, out_path)
    summary = {"policy": policy, "order_id": rows[0]["order_id"] if rows else None, "n_requests": len(rows), "valid": valid, "failures": len(rows) - valid, "failure_classes": failures, "load_ms": load_ms, "elapsed_s": time.perf_counter() - started_all, "raw_sha256": sha_file(out_path), "labels_joined": False, "gpu_index_requested": gpu, "model_revision": PPLX_REV if model_type == "pplx" else JINA_REV}
    out_path.with_suffix(".SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    del adapter, processor
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["pplx_native", "pplx_budget", "pplx_two_stage", "jina_full", "jina_budget", "jina_two_stage_matched"])
    ap.add_argument("--order", required=True, choices=["original", "reversed", "perm_seed_20261020", "perm_seed_20261021"])
    ap.add_argument("--gpu", required=True, type=int)
    ap.add_argument("--dataset", choices=["enterprise", "musique"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = order_rows(args.order, args.dataset, args.start, args.limit)
    if not rows:
        raise SystemExit("no manifest rows selected")
    stage = "stage1_native_v2_hybrid" if args.order == "original" else "stage2_orders_v2_hybrid"
    suffix = "__full"
    if args.start or args.limit:
        suffix = f"__start{args.start}__n{args.limit or len(rows)}"
    name = f"{args.policy}__{args.order}" + (f"__{args.dataset}" if args.dataset else "") + suffix + ".jsonl"
    run_policy(args.policy, rows, args.gpu, BASE / stage / name)


if __name__ == "__main__":
    main()
