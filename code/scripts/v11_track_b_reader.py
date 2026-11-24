#!/usr/bin/env python3
"""Run the frozen Qwen reader over Track B's 6,000 selector outcomes."""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/raw/vendor/enterprise_rag_bench/v1.0.0"
QUESTIONS = BASE / "questions.jsonl"
OUT = ROOT / "results/non_jev/v11/track_b"
SELECTORS = OUT / "V11_TRACK_B_SELECTOR_RESULTS.jsonl"
RETRIEVAL = OUT / "V11_TRACK_B_RETRIEVAL_SLATES.jsonl"
POLICIES = ["raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5"]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, values: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_reader():
    spec = importlib.util.spec_from_file_location("decisionrag_track_b_reader_legacy", ROOT / "scripts/run_m4_reader.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen reader implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_prompt(question: str, passages: list[dict], tokenizer, legacy, view_cache: dict, token_cache: dict) -> tuple[str, dict]:
    pieces, view_info, ordered_ids = [], [], []
    for index, passage in enumerate(passages):
        doc_id = passage["doc_id"]
        ordered_ids.append(doc_id)
        if doc_id not in view_cache:
            view_cache[doc_id] = legacy.cap_reader_text(tokenizer, passage["title"], passage["full_text"])
        view, token_count, truncated = view_cache[doc_id]
        pieces.append(f"[P{index} | {passage['title']} | {doc_id}] {view}")
        view_info.append({"doc_id": doc_id, "reader_tokens": token_count, "truncated": truncated, "reader_view_sha256": legacy.digest(view)})
    prompt = (
        "You must answer using only the supplied passages.\n"
        "If the answer is not supported, output exactly: INSUFFICIENT_EVIDENCE.\n"
        "Provide one short answer, not reasoning. Do not use external knowledge.\n"
        f"Question: {question}\nPassage<redacted-path>" + "\n".join(pieces) + "\nAnswer:"
    )
    key = (question, tuple(ordered_ids))
    if key not in token_cache:
        token_cache[key] = len(tokenizer.encode(prompt, add_special_tokens=True))
    return prompt, {"passages": view_info, "prompt_tokens": token_cache[key]}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    questions = {r["question_id"]: {"query_id": r["question_id"], "question": r["question"]} for r in load_rows(QUESTIONS)}
    retrieval = {r["query_id"]: r for r in load_rows(RETRIEVAL)}
    selections = load_rows(SELECTORS)
    if len(questions) != 500 or len(selections) != 6000:
        raise RuntimeError(f"expected 500 questions and 6000 selector outcomes, got {len(questions)} and {len(selections)}")
    candidate_map = {}
    for slate in retrieval.values():
        for candidate in slate["candidates"]:
            candidate_map[candidate["doc_id"]] = {
                "doc_id": candidate["doc_id"], "title": candidate["title"], "full_text": None,
                "representative_chunk_id": candidate["representative_chunk_id"], "archive_path": candidate["archive_path"],
                "original_doc_id": candidate["original_doc_id"],
            }
    # Load representative chunk text once, without touching enterprise gold.
    import sqlite3
    conn = sqlite3.connect(OUT / "V11_TRACK_B_ENTERPRISE_CORPUS.sqlite3")
    for doc_id, candidate in candidate_map.items():
        row = conn.execute("SELECT title, text FROM chunks WHERE chunk_id = ?", (candidate["representative_chunk_id"],)).fetchone()
        if row is None:
            raise RuntimeError(f"missing reader representative chunk {candidate['representative_chunk_id']}")
        candidate["title"], candidate["full_text"] = row
    conn.close()
    legacy = legacy_reader()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(legacy.GENERATOR_PATH), trust_remote_code=True)
    view_cache, token_cache, tasks = {}, {}, []
    for index, selection in enumerate(selections, start=1):
        passages = [candidate_map[doc_id] for doc_id in selection["selected_doc_ids"]]
        prompt, view_info = build_prompt(questions[selection["query_id"]]["question"], passages, tokenizer, legacy, view_cache, token_cache)
        if view_info["prompt_tokens"] > 8192:
            raise RuntimeError(f"reader context exceeds service limit: {selection['query_id']}:{selection['policy_id']}")
        cache_key = legacy.digest({"query_id": selection["query_id"], "ordered_doc_ids": selection["selected_doc_ids"], "reader_view": view_info["passages"], "prompt_version": legacy.PROMPT_VERSION, "reader_revision": legacy.READER_REVISION, "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64, "seed": 0}})
        tasks.append({"query_id": selection["query_id"], "dataset": "enterprise", "policy_id": selection["policy_id"], "selected_doc_ids": selection["selected_doc_ids"], "top20_doc_ids": selection["top20_doc_ids"], "prompt": prompt, "prompt_hash": legacy.digest(prompt), "cache_key": cache_key, "view_info": view_info})
        if index % 1000 == 0 or index == len(selections):
            print(f"reader prompt preparation {index}/{len(selections)} docs_cached={len(view_cache)}", flush=True)
    cache_path = OUT / "V11_TRACK_B_READER_CACHE.jsonl"
    output_path = OUT / "V11_TRACK_B_READER_RESULTS.jsonl"
    existing = {r["cache_key"]: r for r in load_rows(cache_path)} if cache_path.exists() else {}
    unique = {}
    for task in tasks:
        unique.setdefault(task["cache_key"], task)
    missing = [task for key, task in unique.items() if key not in existing]
    new_results, lock = {}, threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(legacy.call_reader, task): task for task in missing}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            result = future.result()
            result["query_id"] = task["query_id"]
            new_results[result["cache_key"]] = result
            with lock:
                with cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
    cache = {**existing, **new_results}
    output = []
    for task in tasks:
        cached = cache[task["cache_key"]]
        row = {key: value for key, value in task.items() if key != "prompt"}
        row.update({key: value for key, value in cached.items() if key != "cache_key"})
        row.update({"cache_key": task["cache_key"], "reader_revision": legacy.READER_REVISION, "prompt_version": legacy.PROMPT_VERSION, "cache_enabled": True, "gold_accessed": False})
        if cached.get("reader_status") == "ok" and cached.get("source_policy_id") != task["policy_id"]:
            row.update({"real_inference": False, "duration_ms": 0.0, "reused_cached_answer_from": cached.get("source_policy_id") or "exact_context_cache"})
        output.append(row)
    write_rows(output_path, output)
    status_counts = {s: sum(r.get("reader_status") == s for r in output) for s in sorted({r.get("reader_status") for r in output})}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "track": "B", "query_count": 500, "policy_count": len(POLICIES), "tasks": len(tasks),
        "unique_cache_keys": len(unique), "new_reader_requests": len(missing), "exact_context_cache_reuses": len(tasks) - len(unique), "status_counts": status_counts,
        "selector_results_sha256": sha256(SELECTORS), "reader_results_sha256": sha256(output_path), "gold_accessed": False, "jev_calls": 0, "hosted_cloudflare_calls": 0,
    }
    (OUT / "V11_TRACK_B_READER.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0 if status_counts.get("error", 0) == 0 and status_counts.get("ok", 0) == len(tasks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
