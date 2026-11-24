#!/usr/bin/env python3
"""Run the frozen Qwen reader for versioned Track A selector results."""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/non_jev/v11/track_a"
COHORT_PATH = ROOT / "results/preflight/v11/NEW_TEST_IDS.jsonl"
CORPUS_PATH = ROOT / "data/corpus/passages.jsonl"
POLICIES = ["raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5"]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_prompt_memoized(question: str, passages: list[dict], tokenizer, legacy, view_cache: dict[str, tuple[str, int, bool]], prompt_token_cache: dict[tuple[str, tuple[str, ...]], int]) -> tuple[str, dict]:
    """Byte-for-byte equivalent prompt construction with reusable doc views.

    The frozen reader prompt is unchanged.  Only the repeated tokenization of
    the same corpus document across policies is memoized.
    """
    pieces = []
    view_info = []
    ordered_ids = []
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
    token_key = (question, tuple(ordered_ids))
    if token_key not in prompt_token_cache:
        prompt_token_cache[token_key] = len(tokenizer.encode(prompt, add_special_tokens=True))
    return prompt, {"passages": view_info, "prompt_tokens": prompt_token_cache[token_key]}


def legacy_reader_module():
    spec = importlib.util.spec_from_file_location("decisionrag_v11_reader_legacy", ROOT / "scripts/run_m4_reader.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen reader implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_smoke_ids(cohort: list[dict]) -> set[str]:
    by_dataset = {}
    for row in cohort:
        by_dataset.setdefault(row["dataset"], []).append(row)
    quotas = {"hotpot": 7, "musique": 7, "wiki2": 6}
    return {row["query_id"] for dataset in ("hotpot", "musique", "wiki2") for row in by_dataset[dataset][:quotas[dataset]]}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    suffix = args.mode.upper()
    cohort = load_rows(COHORT_PATH)
    allowed = fixed_smoke_ids(cohort) if args.mode == "smoke" else {row["query_id"] for row in cohort}
    questions = {row["query_id"]: row for row in cohort if row["query_id"] in allowed}
    corpus = {row["doc_id"]: row for row in load_rows(CORPUS_PATH)}
    selection_rows = load_rows(OUT_DIR / f"V11_TRACK_A_SELECTOR_RESULTS_{suffix}.jsonl")
    if len(selection_rows) != len(questions) * len(POLICIES):
        raise RuntimeError(f"expected {len(questions) * len(POLICIES)} selection rows, got {len(selection_rows)}")
    legacy = legacy_reader_module()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(legacy.GENERATOR_PATH), trust_remote_code=True)
    view_cache = {}
    prompt_token_cache = {}
    tasks = []
    for index, selection in enumerate(selection_rows, start=1):
        passages = [corpus[doc_id] for doc_id in selection["selected_doc_ids"]]
        prompt, view_info = build_prompt_memoized(questions[selection["query_id"]]["question"], passages, tokenizer, legacy, view_cache, prompt_token_cache)
        if view_info["prompt_tokens"] > 8192:
            raise RuntimeError(f"reader context exceeds service limit: {selection['query_id']}:{selection['policy_id']}")
        cache_key = legacy.digest({"query_id": selection["query_id"], "ordered_doc_ids": selection["selected_doc_ids"], "reader_view": view_info["passages"], "prompt_version": legacy.PROMPT_VERSION, "reader_revision": legacy.READER_REVISION, "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64, "seed": 0}})
        tasks.append({"query_id": selection["query_id"], "dataset": selection["dataset"], "policy_id": selection["policy_id"], "selected_doc_ids": selection["selected_doc_ids"], "top20_doc_ids": selection["top20_doc_ids"], "prompt": prompt, "prompt_hash": legacy.digest(prompt), "cache_key": cache_key, "view_info": view_info})
        if index % 1000 == 0 or index == len(selection_rows):
            print(f"reader prompt preparation {index}/{len(selection_rows)} docs_cached={len(view_cache)}", flush=True)
    cache_path = OUT_DIR / f"V11_TRACK_A_READER_CACHE_{suffix}.jsonl"
    output_path = OUT_DIR / f"V11_TRACK_A_READER_RESULTS_{suffix}.jsonl"
    existing = {row["cache_key"]: row for row in load_rows(cache_path)} if cache_path.exists() else {}
    unique_tasks = {}
    for task in tasks:
        unique_tasks.setdefault(task["cache_key"], task)
    missing = [task for key, task in unique_tasks.items() if key not in existing]
    new_results = {}
    lock = threading.Lock()
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
        row["cache_key"] = task["cache_key"]
        row["reader_revision"] = legacy.READER_REVISION
        row["prompt_version"] = legacy.PROMPT_VERSION
        row["cache_enabled"] = True
        row["gold_accessed"] = False
        source_policy = cached.get("source_policy_id")
        if cached.get("reader_status") == "ok" and source_policy != task["policy_id"]:
            row["real_inference"] = False
            row["duration_ms"] = 0.0
            row["reused_cached_answer_from"] = source_policy or "exact_context_cache"
        output.append(row)
    write_rows(output_path, output)
    status_counts = {status: sum(row.get("reader_status") == status for row in output) for status in sorted({row.get("reader_status") for row in output})}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "query_count": len(questions),
        "policy_count": len(POLICIES),
        "tasks": len(tasks),
        "unique_cache_keys": len(unique_tasks),
        "new_reader_requests": len(missing),
        "exact_context_cache_reuses": len(tasks) - len(unique_tasks),
        "status_counts": status_counts,
        "selector_results_sha256": sha256(OUT_DIR / f"V11_TRACK_A_SELECTOR_RESULTS_{suffix}.jsonl"),
        "reader_results_sha256": sha256(output_path),
        "gold_accessed": False,
        "jev_calls": 0,
        "hosted_cloudflare_calls": 0,
    }
    (OUT_DIR / f"V11_TRACK_A_READER_{suffix}.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0 if status_counts.get("error", 0) == 0 and status_counts.get("ok", 0) == len(tasks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
