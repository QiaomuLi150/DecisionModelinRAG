#!/usr/bin/env python3
"""M4: fixed Qwen reader answer-quality evaluation for all non-Jev policies."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
READER_URL = "http://service.example.invalid/v1/chat/completions"
READER_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
READER_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
GENERATOR_PATH = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots") / READER_REVISION
POLICY_ORDER = ["raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5"]
PROMPT_VERSION = "decisionrag-reader-v1"
CACHE_PATH = ROOT / "data/results/m4/reader_cache.jsonl"
RESULT_PATH = ROOT / "data/results/m4/reader_results.jsonl"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def cap_reader_text(tokenizer, title: str, text: str) -> tuple[str, int, bool]:
    ids = tokenizer.encode(str(text), add_special_tokens=False)
    clipped = len(ids) > 256
    return f"{title}\n{tokenizer.decode(ids[:256], skip_special_tokens=True).strip()}", min(len(ids), 256), clipped


def build_prompt(question: str, passages: list[dict], tokenizer) -> tuple[str, dict]:
    pieces = []
    view_info = []
    for index, passage in enumerate(passages):
        view, token_count, truncated = cap_reader_text(tokenizer, passage["title"], passage["full_text"])
        pieces.append(f"[P{index} | {passage['title']} | {passage['doc_id']}] {view}")
        view_info.append({"doc_id": passage["doc_id"], "reader_tokens": token_count, "truncated": truncated, "reader_view_sha256": digest(view)})
    prompt = (
        "You must answer using only the supplied passages.\n"
        "If the answer is not supported, output exactly: INSUFFICIENT_EVIDENCE.\n"
        "Provide one short answer, not reasoning. Do not use external knowledge.\n"
        f"Question: {question}\nPassage<redacted-path>" + "\n".join(pieces) + "\nAnswer:"
    )
    return prompt, {"passages": view_info, "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=True))}


def call_reader(task: dict) -> dict:
    payload = {"model": READER_MODEL, "messages": [{"role": "user", "content": task["prompt"]}], "temperature": 0.0, "top_p": 1.0, "max_tokens": 64, "seed": 0, "stream": False}
    last_error = None
    for attempt in range(1, 4):
        started = time.perf_counter_ns()
        try:
            response = requests.post(READER_URL, json=payload, timeout=300)
            response.raise_for_status()
            body = response.json()
            answer = str(body["choices"][0]["message"].get("content") or "").strip()
            usage = body.get("usage") or {}
            return {"cache_key": task["cache_key"], "source_policy_id": task["policy_id"], "prompt_hash": task["prompt_hash"], "answer": answer, "reader_status": "ok", "real_inference": True, "duration_ms": (time.perf_counter_ns() - started) / 1_000_000.0, "attempts": attempt, "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"), "response_hash": digest(body), "reused_cached_answer_from": None, "error_type": None, "payload_hash": digest(payload)}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return {"cache_key": task["cache_key"], "source_policy_id": task["policy_id"], "prompt_hash": task["prompt_hash"], "answer": "", "reader_status": "error", "real_inference": False, "duration_ms": None, "attempts": 3, "input_tokens": None, "output_tokens": None, "response_hash": None, "reused_cached_answer_from": None, "error_type": last_error, "payload_hash": digest(payload)}


def main() -> int:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(GENERATOR_PATH), trust_remote_code=True)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    questions = {row["query_id"]: row for row in load_rows(ROOT / "data/manifests/questions.jsonl")}
    corpus = {row["doc_id"]: row for row in load_rows(ROOT / "data/corpus/passages.jsonl")}
    selections = load_rows(ROOT / "data/results/m3/selection_results.jsonl")
    tasks = []
    for policy in POLICY_ORDER:
        for selection in sorted((row for row in selections if row["policy_id"] == policy), key=lambda row: row["query_id"]):
            passages = [corpus[doc_id] for doc_id in selection["selected_doc_ids"]]
            prompt, view_info = build_prompt(questions[selection["query_id"]]["question"], passages, tokenizer)
            if view_info["prompt_tokens"] > 8192:
                raise RuntimeError(f"reader input exceeds service context for {selection['query_id']}:{policy}: {view_info['prompt_tokens']}")
            cache_key = digest({"query_id": selection["query_id"], "ordered_doc_ids": selection["selected_doc_ids"], "reader_view": view_info["passages"], "prompt_version": PROMPT_VERSION, "reader_revision": READER_REVISION, "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64, "seed": 0}})
            tasks.append({"query_id": selection["query_id"], "dataset": selection["dataset"], "policy_id": policy, "selected_doc_ids": selection["selected_doc_ids"], "top20_doc_ids": selection["top20_doc_ids"], "prompt": prompt, "prompt_hash": digest(prompt), "cache_key": cache_key, "view_info": view_info})
    existing = {}
    if CACHE_PATH.exists():
        for row in load_rows(CACHE_PATH):
            if row.get("reader_status") == "ok": existing[row["cache_key"]] = row
    unique_tasks = {}
    for task in tasks: unique_tasks.setdefault(task["cache_key"], task)
    missing = [task for key, task in unique_tasks.items() if key not in existing]
    print(json.dumps({"tasks": len(tasks), "unique_cache_keys": len(unique_tasks), "cache_hits": len(unique_tasks) - len(missing), "new_reader_requests": len(missing), "jev_calls": 0}, indent=2), flush=True)
    new_results = {}
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(call_reader, task): task for task in missing}
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            new_results[result["cache_key"]] = result
            with lock:
                with CACHE_PATH.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            if index % 100 == 0 or index == len(missing): print(f"reader requests {index}/{len(missing)}", flush=True)
    cache = {**existing, **new_results}
    output = []
    for task in tasks:
        cached = cache[task["cache_key"]]
        row = dict(task)
        row.pop("prompt", None)
        row.update({key: value for key, value in cached.items() if key != "cache_key"})
        row["cache_key"] = task["cache_key"]
        row["reader_revision"] = READER_REVISION
        row["prompt_version"] = PROMPT_VERSION
        row["cache_disabled"] = False
        source_policy = cached.get("source_policy_id")
        if cached["reader_status"] == "ok" and source_policy != task["policy_id"]:
            row["real_inference"] = False
            row["duration_ms"] = 0.0
            row["reused_cached_answer_from"] = source_policy or "previous_identical_offline_request"
        output.append(row)
    write_rows(RESULT_PATH, output)
    status_counts = {}
    for row in output: status_counts[row["reader_status"]] = status_counts.get(row["reader_status"], 0) + 1
    manifest = {"study_id": "decisionrag-frozen-A4-B3-C4-D3-E3-F1-G2", "created_utc": datetime.now(timezone.utc).isoformat(), "reader_model": READER_MODEL, "reader_revision": READER_REVISION, "endpoint": READER_URL, "prompt_version": PROMPT_VERSION, "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64, "seed": 0}, "tasks": len(tasks), "unique_cache_keys": len(unique_tasks), "new_reader_requests": len(missing), "cache_hits": len(unique_tasks) - len(missing), "in_run_cache_reuses": len(tasks) - len(unique_tasks), "status_counts": status_counts, "results_sha256": hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest(), "jev_calls": 0, "hosted_cloudflare_calls": 0}
    (ROOT / "data/manifests/m4_reader_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if status_counts.get("error", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
