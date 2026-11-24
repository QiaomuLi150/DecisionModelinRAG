#!/usr/bin/env python3
"""Run the frozen local Qwen reader over HERB H1 policy outcomes."""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/herb_h1"
QUERY_MANIFEST = OUT / "HERB_H1_QUERY_MANIFEST.jsonl"
RETRIEVAL = OUT / "HERB_H1_RETRIEVAL_SLATES.jsonl"
SELECTORS = OUT / "HERB_H1_SELECTOR_RESULTS.jsonl"
DB = OUT / "HERB_H1_CORPUS.sqlite3"
POLICIES = ["raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5"]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def call_reader(task: dict, endpoint: str, legacy) -> dict:
    import requests
    payload = {"model": legacy.READER_MODEL, "messages": [{"role": "user", "content": task["prompt"]}], "temperature": 0.0, "top_p": 1.0, "max_tokens": 64, "seed": 0, "stream": False}
    last_error = None
    for attempt in range(1, 4):
        started = time.perf_counter_ns()
        try:
            response = requests.post(endpoint, json=payload, timeout=300)
            response.raise_for_status()
            body = response.json()
            answer = str(body["choices"][0]["message"].get("content") or "").strip()
            usage = body.get("usage") or {}
            return {"cache_key": task["cache_key"], "source_policy_id": task["policy_id"], "prompt_hash": task["prompt_hash"], "answer": answer, "reader_status": "ok", "real_inference": True, "duration_ms": (time.perf_counter_ns() - started) / 1_000_000.0, "attempts": attempt, "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"), "response_hash": digest(body), "reused_cached_answer_from": None, "error_type": None, "payload_hash": digest(payload), "reader_endpoint": endpoint}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return {"cache_key": task["cache_key"], "source_policy_id": task["policy_id"], "prompt_hash": task["prompt_hash"], "answer": "", "reader_status": "error", "real_inference": False, "duration_ms": None, "attempts": 3, "input_tokens": None, "output_tokens": None, "response_hash": None, "reused_cached_answer_from": None, "error_type": last_error, "payload_hash": digest(payload), "reader_endpoint": endpoint}


def legacy_reader():
    spec = importlib.util.spec_from_file_location("decisionrag_h1_reader_legacy", ROOT / "scripts/run_m4_reader.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen reader implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reader-url", action="append", default=[], help="reader completion endpoint; repeat for local replicas")
    args = parser.parse_args()
    legacy = legacy_reader()
    reader_urls = args.reader_url or [legacy.READER_URL]
    questions = {row["query_id"]: row for row in load(QUERY_MANIFEST)}
    selections = load(SELECTORS)
    expected = len(questions) * len(POLICIES)
    if len(questions) != 1514 or len(selections) != expected:
        raise RuntimeError(f"expected 1514 queries and {expected} selector rows, got {len(questions)} and {len(selections)}")
    if {row["policy_id"] for row in selections} != set(POLICIES):
        raise RuntimeError("H1 selector policy set mismatch")
    retrieval = {row["query_id"]: row for row in load(RETRIEVAL)}
    candidate_map = {}
    for slate in retrieval.values():
        for candidate in slate["candidates"]:
            candidate_map[candidate["doc_id"]] = candidate
    conn = sqlite3.connect(DB)
    for doc_id, candidate in candidate_map.items():
        row = conn.execute("SELECT title, text FROM chunks WHERE chunk_id = ?", (candidate["representative_chunk_id"],)).fetchone()
        if row is None or not row[1]:
            raise RuntimeError(f"missing H1 reader representative chunk {candidate['representative_chunk_id']}")
        candidate["title"], candidate["full_text"] = row
    conn.close()
    tokenizer = AutoTokenizer.from_pretrained(str(legacy.GENERATOR_PATH), trust_remote_code=True)
    view_cache, tasks = {}, []
    # The selector file is already in frozen retrieval order.  Preserve that
    # order and policy-major task order to make the output auditable.
    by_query_policy = {(row["query_id"], row["policy_id"]): row for row in selections}
    for policy in POLICIES:
        for query in questions.values():
            selection = by_query_policy[(query["query_id"], policy)]
            passages = [candidate_map[doc_id] for doc_id in selection["selected_doc_ids"]]
            pieces, view_info = [], []
            for index, passage in enumerate(passages):
                doc_id = passage["doc_id"]
                if doc_id not in view_cache:
                    view_cache[doc_id] = legacy.cap_reader_text(tokenizer, passage["title"], passage["full_text"])
                view, token_count, truncated = view_cache[doc_id]
                pieces.append(f"[P{index} | {passage['title']} | {doc_id}] {view}")
                view_info.append({"doc_id": doc_id, "reader_tokens": token_count, "truncated": truncated, "reader_view_sha256": digest(view)})
            prompt = (
                "You must answer using only the supplied passages.\n"
                "If the answer is not supported, output exactly: INSUFFICIENT_EVIDENCE.\n"
                "Provide one short answer, not reasoning. Do not use external knowledge.\n"
                f"Question: {query['question']}\nPassage<redacted-path>" + "\n".join(pieces) + "\nAnswer:"
            )
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
            if prompt_tokens > 8192:
                raise RuntimeError(f"H1 reader input exceeds service context for {query['query_id']}:{policy}: {prompt_tokens}")
            cache_key = digest({"query_id": query["query_id"], "ordered_doc_ids": selection["selected_doc_ids"], "reader_view": view_info, "prompt_version": legacy.PROMPT_VERSION, "reader_revision": legacy.READER_REVISION, "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64, "seed": 0}})
            tasks.append({"query_id": query["query_id"], "dataset": "herb", "product": query.get("product"), "answerability": query.get("answerability"), "policy_id": policy, "selected_doc_ids": selection["selected_doc_ids"], "top20_doc_ids": selection["top20_doc_ids"], "prompt": prompt, "prompt_hash": digest(prompt), "cache_key": cache_key, "view_info": {"passages": view_info, "prompt_tokens": prompt_tokens}})
            if len(tasks) % 1000 == 0:
                print(f"reader prompt preparation {len(tasks)}/{expected} docs_cached={len(view_cache)}", flush=True)
    cache_path = OUT / "HERB_H1_READER_CACHE.jsonl"
    output_path = OUT / "HERB_H1_READER_RESULTS.jsonl"
    existing = {row["cache_key"]: row for row in load(cache_path)} if cache_path.exists() else {}
    unique = {}
    for task in tasks:
        unique.setdefault(task["cache_key"], task)
    missing = [task for key, task in unique.items() if key not in existing]
    print(json.dumps({"tasks": len(tasks), "unique_cache_keys": len(unique), "cache_records": len(existing), "new_reader_requests": len(missing), "jev_calls": 0, "hosted_cloudflare_calls": 0}, indent=2), flush=True)
    new_results, lock = {}, threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(call_reader, task, reader_urls[index % len(reader_urls)], legacy): task for index, task in enumerate(missing)}
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            task = futures[future]
            result = future.result()
            new_results[result["cache_key"]] = result
            with lock:
                with cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
            if index % 100 == 0 or index == len(missing):
                print(f"reader requests {index}/{len(missing)}", flush=True)
    cache = {**existing, **new_results}
    output = []
    for task in tasks:
        cached = cache.get(task["cache_key"])
        if cached is None:
            raise RuntimeError(f"missing reader result for {task['query_id']}:{task['policy_id']}")
        row = {key: value for key, value in task.items() if key != "prompt"}
        row.update({key: value for key, value in cached.items() if key != "cache_key"})
        row.update({"cache_key": task["cache_key"], "reader_revision": legacy.READER_REVISION, "prompt_version": legacy.PROMPT_VERSION, "cache_enabled": True, "gold_accessed": False})
        if cached.get("reader_status") == "ok" and cached.get("source_policy_id") != task["policy_id"]:
            row.update({"real_inference": False, "duration_ms": 0.0, "reused_cached_answer_from": cached.get("source_policy_id") or "exact_context_cache"})
        output.append(row)
    write(output_path, output)
    status_counts = {status: sum(row.get("reader_status") == status for row in output) for status in sorted({row.get("reader_status") for row in output})}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "benchmark": "HERB", "track": "H1", "query_count": len(questions), "policy_count": len(POLICIES), "tasks": len(tasks), "unique_cache_keys": len(unique), "new_reader_requests": len(missing), "exact_context_cache_reuses": len(tasks) - len(unique), "status_counts": status_counts, "reader_endpoints": reader_urls, "selector_results_sha256": file_digest(SELECTORS), "reader_results_sha256": file_digest(output_path), "gold_accessed": False, "jev_calls": 0, "hosted_cloudflare_calls": 0}
    (OUT / "HERB_H1_READER.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0 if status_counts.get("error", 0) == 0 and status_counts.get("ok", 0) == len(tasks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
