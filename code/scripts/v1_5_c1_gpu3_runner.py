#!/usr/bin/env python3
"""Label-blind v1.5 C1 Jina continuation runner.

The runner writes only versioned v1.5 files. It never opens gold labels and
uses one native Jina model instance sequentially on the selected device.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v1_5/c1_two_stage_specialist_gpu3"
RAW = OUT / "raw"
BRIGHT = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
BUDGET = ROOT / "results/non_jev/v2/e11_v1_0/01_budget_policy/E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
JINA_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/{JINA_REV}")
ORDERS = ["original", "reversed", "perm_seed_1", "perm_seed_2", "perm_seed_3", "perm_seed_4", "perm_seed_5"]
POLICIES = ["jina_two_stage_full", "jina_two_stage_pplxtext", "jina_pplx_budget"]


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def order_ids(ids: list[str], order_id: str) -> list[str]:
    if order_id == "original":
        return list(ids)
    if order_id == "reversed":
        return list(reversed(ids))
    seed = int(order_id.rsplit("_", 1)[1])
    out = list(ids)
    random.Random(seed).shuffle(out)
    return out


def resource_snapshot() -> dict[str, Any]:
    try:
        rows = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=20).splitlines()
        for line in rows:
            vals = [x.strip() for x in line.split(",")]
            if vals and vals[0] == "3":
                return {"gpu_index": 3, "memory_used_mib": int(vals[1]), "memory_free_mib": int(vals[2]), "utilization_percent": int(vals[3])}
    except Exception as exc:
        return {"gpu_index": 3, "status": "UNOBSERVABLE", "error_class": type(exc).__name__}
    return {"gpu_index": 3, "status": "NOT_FOUND"}


def assert_gpu3_idle() -> dict[str, Any]:
    snapshot = resource_snapshot()
    if snapshot.get("status") or snapshot.get("memory_used_mib", 1) > 64 or snapshot.get("utilization_percent", 100) != 0:
        raise RuntimeError(f"GPU3 dedicated gate failed: {snapshot}")
    return snapshot


def load_inputs() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    slates = {f"{x['domain']}:{x['query_id']}": x for x in read_jsonl(BRIGHT)}
    budgets = {f"{x['domain']}:{x['query_id']}": x for x in read_jsonl(BUDGET)}
    if len(slates) != 1384 or set(slates) != set(budgets):
        raise RuntimeError(f"frozen input mismatch slates={len(slates)} budgets={len(budgets)}")
    if file_sha(BRIGHT) != "c304b2c35f445153181098d3a899271db6fc653b44aea50186c6efaeec8e6ea2":
        raise RuntimeError("BRIGHT slate hash mismatch")
    return slates, budgets


def peak(torch) -> float:
    torch.cuda.synchronize()
    return round(float(torch.cuda.max_memory_allocated()) / 2**20, 3)


def stable_top(ids: list[str], scores: list[float], k: int) -> list[int]:
    return sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i, ids[i]))[:k]


def score(model, query: str, docs: list[dict[str, str]]) -> tuple[list[float], dict[str, Any]]:
    ids = [d["candidate_id"] for d in docs]
    request = {"query": query, "candidate_ids": ids, "candidate_texts": [d["content"] for d in docs]}
    result = model.rerank(query, [d["content"] for d in docs])
    mapping = {int(item["index"]): float(item["relevance_score"]) for item in result}
    if set(mapping) != set(range(len(docs))):
        raise ValueError(f"Jina indices do not cover candidates: {sorted(mapping)}")
    scores = [mapping[i] for i in range(len(docs))]
    if not all(math.isfinite(x) for x in scores):
        raise ValueError("non-finite native score")
    return scores, {"request_sha256": sha(request), "candidate_ids_sha256": sha(ids), "scores_sha256": sha(scores)}


def score_many(model, tokenizer, torch, requests: list[tuple[str, list[dict[str, str]]]]) -> list[tuple[list[float], dict[str, Any]]]:
    """Run independent same-width requests in one native forward batch.

    Jina's native adapter emits one score per document token.  For the C1
    accelerated continuation every request in a batch has the same candidate
    width (10 or 20), so this preserves the native readout and index mapping.
    Requests whose complete prompt would exceed the tokenizer context are
    deliberately evaluated through the pinned serial adapter instead of being
    silently shortened.
    """
    if not requests:
        return []
    remote = importlib.import_module(model.__class__.__module__)
    format_prompt = remote.format_docs_prompts_func
    model._ensure_tokenizer()
    prepared = []
    for query, docs in requests:
        q2, d2, lengths, qlen = model._truncate_texts(query, [d["content"] for d in docs], 1024, 8192)
        total = 2 * qlen + sum(lengths)
        if total > tokenizer.model_max_length:
            prepared.append(None)
            continue
        prompt = format_prompt(q2, d2, instruction=None, special_tokens=model.special_tokens, no_thinking=True)
        prepared.append((prompt, q2, d2))
    if any(x is None for x in prepared):
        # Keep the exact remote adapter semantics for exceptional long inputs.
        out = []
        for query, docs in requests:
            out.append(score(model, query, docs))
        return out
    batch = tokenizer(
        text=[x[0] for x in prepared],
        padding=True,
        padding_side="left",
        return_tensors="pt",
    ).to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model.forward(**batch)
    native = outputs.scores.detach().cpu().float().numpy()
    out = []
    for (query, docs), row_scores in zip(requests, native):
        ids = [d["candidate_id"] for d in docs]
        scores = [float(x) for x in row_scores[: len(docs)]]
        if len(scores) != len(docs) or not all(math.isfinite(x) for x in scores):
            raise ValueError("invalid batched native score shape/value")
        request = {"query": query, "candidate_ids": ids, "candidate_texts": [d["content"] for d in docs]}
        out.append((scores, {"request_sha256": sha(request), "candidate_ids_sha256": sha(ids), "scores_sha256": sha(scores)}))
    return out


def base(cid: str, slate: dict[str, Any], policy: str, order_id: str, gpu: int) -> dict[str, Any]:
    return {
        "canonical_id": cid,
        "domain": slate["domain"],
        "query_id": slate["query_id"],
        "policy": policy,
        "order_id": order_id,
        "status": "FAIL",
        "valid": False,
        "error": None,
        "candidate_ids": [],
        "selected_ids": [],
        "scores": [],
        "slate_sha256": slate["slate_sha256"],
        "source_input_sha256": slate["input_sha256"],
        "model_revision": JINA_REV,
        "gpu_index": gpu,
        "labels_joined": False,
        "physical_forward_count": "NOT_EXPOSED_BY_RUNTIME",
    }


def run_one(model, torch, cid: str, slate: dict[str, Any], budget: dict[str, Any], policy: str, order_id: str, gpu: int) -> dict[str, Any]:
    row = base(cid, slate, policy, order_id, gpu)
    original = [{"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in slate["candidates"]]
    original_ids = [x["candidate_id"] for x in original]
    if policy == "jina_two_stage_pplxtext" or policy == "jina_pplx_budget":
        if str(budget.get("status", "")).lower() not in {"ok", "pass", "complete"}:
            row["error"] = f"BudgetUnavailable:{budget.get('status')}"
            return row
        budget_docs = [{"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in budget["budgeted_candidates"]]
        by_id = {x["candidate_id"]: x for x in budget_docs}
        docs_by_id = by_id
    else:
        docs_by_id = {x["candidate_id"]: x for x in original}
    ordered = order_ids(original_ids, order_id)
    if set(docs_by_id) != set(original_ids):
        row["error"] = "CandidateIdentityMismatch"
        return row
    ordered_docs = [docs_by_id[x] for x in ordered]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        if policy == "jina_pplx_budget":
            scores, meta = score(model, slate["query"], ordered_docs)
            selected_idx = stable_top(ordered, scores, 5)
            row.update({"candidate_ids": ordered, "scores": scores, "selected_ids": [ordered[i] for i in selected_idx], "n_calls": 1, "stage_records": [{"stage": "single", **meta}], "input_view": "E11-A frozen budgeted_candidates"})
        else:
            stage_records = []
            survivors = []
            for stage, start, end in (("stage1_window1", 0, 10), ("stage1_window2", 10, 20)):
                window_ids = ordered[start:end]
                window_docs = [docs_by_id[x] for x in window_ids]
                scores, meta = score(model, slate["query"], window_docs)
                keep = stable_top(window_ids, scores, 5)
                survivors.extend(window_ids[i] for i in keep)
                stage_records.append({"stage": stage, "candidate_ids": window_ids, "scores": scores, **meta})
            survivor_docs = [docs_by_id[x] for x in survivors]
            stage2_scores, meta = score(model, slate["query"], survivor_docs)
            keep2 = stable_top(survivors, stage2_scores, 5)
            stage_records.append({"stage": "stage2_survivors", "candidate_ids": survivors, "scores": stage2_scores, **meta})
            row.update({"candidate_ids": ordered, "selected_ids": [survivors[i] for i in keep2], "scores": stage2_scores, "stage_selected_ids": survivors, "stage_records": stage_records, "n_calls": 3, "input_view": "original_content" if policy == "jina_two_stage_full" else "E11-A frozen budgeted_candidates"})
        row["status"] = "PASS"
        row["valid"] = len(row["selected_ids"]) == 5 and len(set(row["selected_ids"])) == 5
        if not row["valid"]:
            raise ValueError("invalid Top-5")
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["status"] = "FAIL"
        row["valid"] = False
    torch.cuda.synchronize()
    row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    row["peak_memory_allocated_mib"] = peak(torch)
    row["timed_boundary"] = "view preparation + native scoring + score parsing + ID mapping + Top-5 packaging"
    return row


def run_batch(model, tokenizer, torch, entries: list[tuple[str, dict[str, Any], dict[str, Any]]], policy: str, order_id: str, gpu: int) -> list[dict[str, Any]]:
    """Batch independent queries while retaining canonical per-query outputs.

    This is a quality-throughput path.  Batch elapsed time is recorded, but
    it is not reported as isolated per-request latency.
    """
    rows = [base(cid, slate, policy, order_id, gpu) for cid, slate, _ in entries]
    plans = []
    for row, (cid, slate, budget) in zip(rows, entries):
        original = [{"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in slate["candidates"]]
        original_ids = [x["candidate_id"] for x in original]
        if policy in {"jina_two_stage_pplxtext", "jina_pplx_budget"}:
            if str(budget.get("status", "")).lower() not in {"ok", "pass", "complete"}:
                row["error"] = f"BudgetUnavailable:{budget.get('status')}"
                plans.append(None)
                continue
            docs = [{"candidate_id": str(x["candidate_id"]), "content": str(x["content"])} for x in budget["budgeted_candidates"]]
        else:
            docs = original
        docs_by_id = {x["candidate_id"]: x for x in docs}
        ordered = order_ids(original_ids, order_id)
        if set(docs_by_id) != set(original_ids):
            row["error"] = "CandidateIdentityMismatch"
            plans.append(None)
            continue
        plans.append((slate, docs_by_id, ordered))
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        if policy == "jina_pplx_budget":
            reqs = [(slate["query"], [docs_by_id[x] for x in ordered]) if plan is not None else ("", []) for plan in plans for slate, docs_by_id, ordered in [plan] if plan is not None]
            scored = score_many(model, tokenizer, torch, reqs)
            j = 0
            for row, plan in zip(rows, plans):
                if plan is None:
                    continue
                slate, docs_by_id, ordered = plan
                scores, meta = scored[j]; j += 1
                keep = stable_top(ordered, scores, 5)
                row.update({"candidate_ids": ordered, "scores": scores, "selected_ids": [ordered[i] for i in keep], "n_calls": 1, "stage_records": [{"stage": "single", **meta}], "input_view": "E11-A frozen budgeted_candidates"})
        else:
            all_stage_records = [[ ] for _ in rows]
            survivors_by_row = [[] for _ in rows]
            for stage, start, end in (("stage1_window1", 0, 10), ("stage1_window2", 10, 20)):
                reqs = [(plan[0]["query"], [plan[1][x] for x in plan[2][start:end]]) for plan in plans if plan is not None]
                scored = score_many(model, tokenizer, torch, reqs)
                j = 0
                for i, plan in enumerate(plans):
                    if plan is None:
                        continue
                    slate, docs_by_id, ordered = plan
                    ids = ordered[start:end]
                    scores, meta = scored[j]; j += 1
                    keep = stable_top(ids, scores, 5)
                    survivors_by_row[i].extend(ids[k] for k in keep)
                    all_stage_records[i].append({"stage": stage, "candidate_ids": ids, "scores": scores, **meta})
            reqs = [(plan[0]["query"], [plan[1][x] for x in survivors_by_row[i]]) for i, plan in enumerate(plans) if plan is not None]
            scored = score_many(model, tokenizer, torch, reqs)
            j = 0
            for i, plan in enumerate(plans):
                if plan is None:
                    continue
                slate, docs_by_id, ordered = plan
                survivors = survivors_by_row[i]
                scores, meta = scored[j]; j += 1
                keep = stable_top(survivors, scores, 5)
                rows[i].update({"candidate_ids": ordered, "selected_ids": [survivors[k] for k in keep], "scores": scores, "stage_selected_ids": survivors, "stage_records": all_stage_records[i] + [{"stage": "stage2_survivors", "candidate_ids": survivors, "scores": scores, **meta}], "n_calls": 3, "input_view": "original_content" if policy == "jina_two_stage_full" else "E11-A frozen budgeted_candidates"})
        for row in rows:
            row["status"] = "PASS"
            row["valid"] = len(row["selected_ids"]) == 5 and len(set(row["selected_ids"])) == 5
            if not row["valid"]:
                row["status"] = "FAIL"
                row["error"] = row.get("error") or "invalid Top-5"
    except Exception as exc:
        for row in rows:
            row["status"] = "FAIL"
            row["valid"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"
    torch.cuda.synchronize()
    elapsed = round((time.perf_counter() - started) * 1000, 3)
    peak_mib = peak(torch)
    for row in rows:
        row["elapsed_ms"] = None
        row["batch_elapsed_ms"] = elapsed
        row["batch_size"] = len(rows)
        row["peak_memory_allocated_mib"] = peak_mib
        row["timing_valid"] = False
        row["timed_boundary"] = "batched view preparation + native scoring + score parsing + ID mapping + Top-5 packaging; not isolated-request timing"
    return rows


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--orders", default=",".join(ORDERS))
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--receipt-name", default="EXECUTION_RECEIPT_V2.json")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    before = assert_gpu3_idle()
    slates, budgets = load_inputs()
    keys = sorted(slates)
    orders = [x for x in args.orders.split(",") if x]
    policies = [x for x in args.policies.split(",") if x]
    if args.smoke:
        keys = keys[:20]
        orders = ["original"]
        policies = ["jina_two_stage_full"]
    if any(x not in ORDERS for x in orders) or any(x not in POLICIES for x in policies):
        raise ValueError("invalid order or policy")
    import torch
    from transformers import AutoModel, AutoTokenizer
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(JINA_PATH), local_files_only=True)
    model = AutoModel.from_pretrained(str(JINA_PATH), dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda:0").eval()
    torch.cuda.synchronize()
    load_ms = round((time.perf_counter() - t0) * 1000, 3)
    RAW.mkdir(parents=True, exist_ok=True)
    receipt_name = "SMOKE_EXECUTION_RECEIPT.json" if args.smoke else args.receipt_name
    receipt = {"status": "RUNNING", "gpu_index": args.gpu, "before": before, "model_revision": JINA_REV, "model_load_ms": load_ms, "labels_joined": False, "policies": policies, "orders": orders, "n_queries": len(keys), "batch_size": args.batch_size, "timing_valid": args.batch_size == 1, "raw_files": []}
    (OUT / receipt_name).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    for policy in policies:
        for order_id in orders:
            rows = []
            path = RAW / f"{policy}__{order_id}.jsonl"
            if path.exists() and not args.smoke:
                meta_path = path.with_suffix(".meta.json")
                if meta_path.exists():
                    try:
                        if int(json.loads(meta_path.read_text()).get("records", 0)) == len(keys):
                            continue
                    except Exception:
                        pass
            if args.batch_size <= 1:
                for i, cid in enumerate(keys, 1):
                    rows.append(run_one(model, torch, cid, slates[cid], budgets[cid], policy, order_id, args.gpu))
                    if i % 25 == 0 or i == len(keys):
                        print(json.dumps({"policy": policy, "order_id": order_id, "completed": i, "total": len(keys), "batch_size": 1}), flush=True)
            else:
                for start in range(0, len(keys), args.batch_size):
                    batch_keys = keys[start:start + args.batch_size]
                    rows.extend(run_batch(model, tokenizer, torch, [(cid, slates[cid], budgets[cid]) for cid in batch_keys], policy, order_id, args.gpu))
                    print(json.dumps({"policy": policy, "order_id": order_id, "completed": min(start + args.batch_size, len(keys)), "total": len(keys), "batch_size": args.batch_size}), flush=True)
            save_rows(path, rows)
            meta = {"path": str(path.relative_to(ROOT)), "sha256": file_sha(path), "records": len(rows), "valid": sum(r["valid"] for r in rows), "failed": sum(not r["valid"] for r in rows), "labels_joined": False, "batch_size": args.batch_size, "timing_valid": args.batch_size == 1}
            (path.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            receipt["raw_files"].append(meta)
            (OUT / receipt_name).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    receipt["status"] = "COMPLETE"
    receipt["after_in_process"] = resource_snapshot()
    (OUT / receipt_name).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "COMPLETE", "smoke": args.smoke, "queries": len(keys), "policies": policies, "orders": orders, "load_ms": load_ms}, indent=2), flush=True)


if __name__ == "__main__":
    main()
