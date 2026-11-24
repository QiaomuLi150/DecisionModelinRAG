#!/usr/bin/env python3
"""Run the frozen E2-S/P downstream reader through the existing local service.

The selection manifest is immutable and model-neutral.  This runner only
constructs the already-frozen reader prompt and sends it to the pinned local
Qwen3-VL-8B-Instruct service.  It has no gold fields in its prompt and no
application answer cache.  Every request keeps the exact serialized request
body and raw response hash so a later audit can distinguish real inference
from a resumed row.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
E2 = ROOT / "results/non_jev/v2/e2_sp"
GPU5_OUT = E2 / "gpu5"
SELECTION_PATH = GPU5_OUT / "E2_SP_S2_SELECTION_RESULTS.jsonl"
COHORT_PATH = E2 / "E2_SP_READER_COHORT_MANIFEST.json"
OUT = GPU5_OUT
RESULT_PATH = OUT / "E2_SP_P1_READER_OUTPUTS.jsonl"
SUMMARY_PATH = OUT / "E2_SP_P1_READER_SUMMARY.json"
READER_URL = "http://service.example.invalid/v1/chat/completions"
READER_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
READER_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
TOKENIZER_PATH = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface")
PROMPT_REVISION = "E2_SP_READER_CONTRACT_v1"
MAX_CONTEXT_TOKENS = 8192
MAX_OUTPUT_TOKENS = 64
TOKEN_SEED = 0
TIMEOUT_SECONDS = 300
COHORT_SEED = "E2-SP-READER-COHORT-20261013"
POLICY_ORDER = [
    "retrieval5", "bge_pointwise", "qwen06", "qwen4", "jina35", "gte",
    "minilm", "clef_joint", "nimble", "pplx_decider", "qwen_direct",
]
NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store",
    "Pragma": "no-cache",
    "X-DecisionRAG-Cache": "disabled",
}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def cap_reader_text(tokenizer, title: str, text: str) -> tuple[str, int, bool]:
    ids = tokenizer.encode(str(text), add_special_tokens=False)
    clipped = len(ids) > 256
    view = tokenizer.decode(ids[:256], skip_special_tokens=True).strip()
    return f"{title}\n{view}", min(len(ids), 256), clipped


def build_prompt(question: str, passages: list[dict[str, Any]], tokenizer) -> tuple[str, dict[str, Any]]:
    pieces = []
    views = []
    for index, passage in enumerate(passages):
        view, token_count, truncated = cap_reader_text(tokenizer, passage["title"], passage["full_text"])
        pieces.append(f"[P{index} | {passage['title']} | {passage['doc_id']}] {view}")
        views.append({
            "doc_id": passage["doc_id"],
            "reader_tokens": token_count,
            "truncated": truncated,
            "reader_view_sha256": sha_bytes(view.encode("utf-8")),
        })
    prompt = (
        "You must answer using only the supplied passages.\n"
        "If the answer is not supported, output exactly: INSUFFICIENT_EVIDENCE.\n"
        "Provide one short answer, not reasoning. Do not use external knowledge.\n"
        f"Question: {question}\nPassage<redacted-path>" + "\n".join(pieces) + "\nAnswer:"
    )
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
    return prompt, {
        "passages": views,
        "prompt_tokens": prompt_tokens,
        "prompt_sha256": sha_bytes(prompt.encode("utf-8")),
        "context_overflow": prompt_tokens > MAX_CONTEXT_TOKENS,
    }


def gpu_snapshot() -> dict[str, Any]:
    # The reader is intentionally sent through the pre-existing service; this
    # records the experiment's declared topology without pretending that the
    # service exposes per-request physical attribution.
    return {"reader_service_gpu": "resident_service_unattributed", "physical_gpu5_used_by_reader": False}


def make_tasks() -> list[dict[str, Any]]:
    sys_path = str(ROOT / "scripts")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from e2_sp_gpu5_runner import load_cohort_items

    items = {item["query_id"]: item for item in load_cohort_items()}
    selection_rows = load_jsonl(SELECTION_PATH)
    if len(selection_rows) != 3300:
        raise RuntimeError(f"selection manifest has {len(selection_rows)} rows, expected 3300")
    tasks = []
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        revision=READER_REVISION,
        cache_dir=str(TOKENIZER_PATH),
        trust_remote_code=True,
        local_files_only=True,
    )
    for row in sorted(selection_rows, key=lambda value: (POLICY_ORDER.index(value["policy"]), value["query_id"])):
        if row["policy"] not in POLICY_ORDER or row.get("status") != "ok":
            raise RuntimeError("invalid policy/status in selection manifest")
        item = items.get(row["query_id"])
        if item is None:
            raise RuntimeError(f"selection query missing from frozen cohort: {row['query_id']}")
        if row["candidate_ids"] != [candidate["doc_id"] for candidate in item["candidates"]]:
            raise RuntimeError(f"candidate identity mismatch: {row['query_id']}:{row['policy']}")
        selected_ids = row["selected_ids"]
        if len(selected_ids) != 5 or len(set(selected_ids)) != 5 or not set(selected_ids).issubset(set(row["candidate_ids"])):
            raise RuntimeError(f"invalid selected IDs: {row['query_id']}:{row['policy']}")
        by_id = {candidate["doc_id"]: candidate for candidate in item["candidates"]}
        passages = [by_id[doc_id] for doc_id in selected_ids]
        prompt, view = build_prompt(item["question"], passages, tokenizer)
        if view["context_overflow"]:
            raise RuntimeError(f"reader context overflow: {row['query_id']}:{row['policy']}:{view['prompt_tokens']}")
        generation = {"temperature": 0.0, "top_p": 1.0, "max_tokens": MAX_OUTPUT_TOKENS, "seed": TOKEN_SEED, "stream": False}
        payload = {
            "model": READER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            **generation,
        }
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        tasks.append({
            "task_id": f"{row['policy']}::{row['query_id']}",
            "policy": row["policy"],
            "query_id": row["query_id"],
            "dataset": row["dataset"],
            "cohort": row["cohort"],
            "selected_ids": selected_ids,
            "candidate_ids": row["candidate_ids"],
            "selection_row_sha256": sha_json(row),
            "prompt": prompt,
            "prompt_metadata": view,
            "payload": payload,
            "request_body": body,
            "request_body_sha256": sha_bytes(body),
        })
    if len(tasks) != 3300 or len({task["task_id"] for task in tasks}) != 3300:
        raise RuntimeError("reader task manifest is not 3300 unique policy/query tasks")
    return tasks


def call_one(task: dict[str, Any], worker: int) -> dict[str, Any]:
    started = time.perf_counter_ns()
    request_body = task["request_body"]
    request = requests.Request(
        "POST",
        READER_URL,
        data=request_body,
        headers={**NO_CACHE_HEADERS, "Content-Type": "application/json"},
    ).prepare()
    wire_body = request.body if isinstance(request.body, bytes) else str(request.body).encode("utf-8")
    result = {
        "task_id": task["task_id"],
        "policy": task["policy"],
        "query_id": task["query_id"],
        "dataset": task["dataset"],
        "cohort": task["cohort"],
        "selected_ids": task["selected_ids"],
        "candidate_ids": task["candidate_ids"],
        "selection_row_sha256": task["selection_row_sha256"],
        "reader_model": READER_MODEL,
        "reader_revision": READER_REVISION,
        "prompt_revision": PROMPT_REVISION,
        "prompt_sha256": task["prompt_metadata"]["prompt_sha256"],
        "actual_model_input_sha256": sha_bytes(wire_body),
        "request_body_sha256": sha_bytes(wire_body),
        "prompt_metadata": task["prompt_metadata"],
        "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": MAX_OUTPUT_TOKENS, "seed": TOKEN_SEED, "stream": False},
        "cache_disabled": True,
        "cache_reuse": False,
        "cache_provenance": "no_application_cache; fresh_http_request",
        "gold_in_reader_input": False,
        "reader_service_url": READER_URL,
        "worker": worker,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "real_inference": False,
        "answer": "",
        "raw_response_sha256": None,
        "raw_response_text_b64": None,
        "response_json": None,
        "input_tokens": None,
        "output_tokens": None,
        "duration_ms": None,
        "error_type": None,
        "http_status": None,
        "retry_count": 0,
    }
    session = requests.Session()
    last_error = None
    for attempt in range(1, 4):
        result["retry_count"] = attempt - 1
        try:
            response = session.send(request, timeout=TIMEOUT_SECONDS)
            raw = response.content
            result["http_status"] = response.status_code
            result["raw_response_sha256"] = sha_bytes(raw)
            result["raw_response_text_b64"] = base64.b64encode(raw).decode("ascii")
            parsed = response.json()
            result["response_json"] = parsed
            response.raise_for_status()
            choices = parsed.get("choices") or []
            if not choices:
                raise RuntimeError("reader response has no choices")
            message = choices[0].get("message") or {}
            result["answer"] = str(message.get("content") or "").strip()
            usage = parsed.get("usage") or {}
            result["input_tokens"] = usage.get("prompt_tokens")
            result["output_tokens"] = usage.get("completion_tokens")
            result["status"] = "ok"
            result["real_inference"] = True
            result["duration_ms"] = (time.perf_counter_ns() - started) / 1_000_000.0
            return result
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 3:
                time.sleep(0.5 * attempt)
    result["duration_ms"] = (time.perf_counter_ns() - started) / 1_000_000.0
    result["error_type"] = last_error
    return result


def run(tasks: list[dict[str, Any]], workers: int, smoke_only: bool, resume: bool) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    selected = tasks
    if smoke_only:
        smoke_queries = set()
        for dataset in ("hotpot", "musique", "wiki2", "enterprise"):
            smoke_queries.update([task["query_id"] for task in tasks if task["dataset"] == dataset][:2])
        selected = [task for task in tasks if task["query_id"] in smoke_queries]
    existing = {}
    if resume and RESULT_PATH.exists():
        for row in load_jsonl(RESULT_PATH):
            if row.get("status") == "ok":
                existing[row["task_id"]] = row
    pending = [task for task in selected if task["task_id"] not in existing]
    print(json.dumps({"stage": "P1_READER", "tasks": len(selected), "pending": len(pending), "resume_ok": len(existing), "workers": workers, "smoke_only": smoke_only, "cache_disabled": True, "jev_calls": 0, "paid_api_calls": 0}, sort_keys=True), flush=True)
    completed = dict(existing)
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call_one, task, index % workers): task for index, task in enumerate(pending)}
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            row = future.result()
            completed[row["task_id"]] = row
            if index % 25 == 0 or index == len(pending):
                with lock:
                    write_jsonl(RESULT_PATH, [completed[key] for key in sorted(completed)])
                print(json.dumps({"completed": index, "pending": len(pending), "ok": sum(row0.get("status") == "ok" for row0 in completed.values()), "errors": sum(row0.get("status") != "ok" for row0 in completed.values())}, sort_keys=True), flush=True)
    with lock:
        write_jsonl(RESULT_PATH, [completed[key] for key in sorted(completed)])
    rows = [completed[key] for key in sorted(completed)]
    status = {value: sum(row.get("status") == value for row in rows) for value in {row.get("status") for row in rows}}
    summary = {
        "stage": "P1_READER",
        "status": "PASS" if len(rows) == len(selected) and status.get("error", 0) == 0 else "PARTIAL_BLOCKED",
        "reader_model": READER_MODEL,
        "reader_revision": READER_REVISION,
        "prompt_revision": PROMPT_REVISION,
        "cohort_manifest_sha256": sha_bytes(COHORT_PATH.read_bytes()),
        "selection_manifest_sha256": sha_bytes(SELECTION_PATH.read_bytes()),
        "requested_tasks": len(selected),
        "output_rows": len(rows),
        "status_counts": status,
        "real_inference_rows": sum(bool(row.get("real_inference")) for row in rows),
        "cache_reuse_rows": sum(bool(row.get("cache_reuse")) for row in rows),
        "gold_in_reader_input_rows": sum(bool(row.get("gold_in_reader_input")) for row in rows),
        "actual_model_input_hash_rows": sum(bool(row.get("actual_model_input_sha256")) for row in rows),
        "raw_response_hash_rows": sum(bool(row.get("raw_response_sha256")) for row in rows),
        "prompt_context_overflow_rows": sum(bool(row.get("prompt_metadata", {}).get("context_overflow")) for row in rows),
        "dataset_counts": {dataset: sum(row.get("dataset") == dataset for row in rows) for dataset in ("hotpot", "musique", "wiki2", "enterprise")},
        "policy_counts": {policy: sum(row.get("policy") == policy for row in rows) for policy in POLICY_ORDER},
        "cache_contract": "application cache disabled; no exact-context reuse in this run; retries only on errors",
        "reader_service_topology": "existing resident Qwen service; physical GPU attribution unavailable at request boundary",
        "gpu5_used_by_reader": False,
        "gpu_snapshot": gpu_snapshot(),
        "jev_calls": 0,
        "paid_api_calls": 0,
        "result_sha256": sha_bytes(RESULT_PATH.read_bytes()),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        raise SystemExit("workers must be 1..8")
    tasks = make_tasks()
    summary = run(tasks, args.workers, args.smoke_only, args.resume)
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
