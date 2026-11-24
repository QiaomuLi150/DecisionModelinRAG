#!/usr/bin/env python3
"""Frozen Track B selector execution over the enterprise Top-20 slates.

This wrapper reuses the existing selector implementations and supplies only
the enterprise question/slate/chunk adapter.  It never reads enterprise gold
fields and writes only versioned Track B traces.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/raw/vendor/enterprise_rag_bench/v1.0.0"
QUESTIONS = BASE / "questions.jsonl"
OUT = ROOT / "results/non_jev/v11/track_b"
RETRIEVAL = OUT / "V11_TRACK_B_RETRIEVAL_SLATES.jsonl"
DB = OUT / "V11_TRACK_B_ENTERPRISE_CORPUS.sqlite3"
MODELS = ("bge", "qwen", "laya", "clef")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_legacy():
    path = ROOT / "scripts/run_m3_selectors.py"
    spec = importlib.util.spec_from_file_location("decisionrag_track_b_legacy_m3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen selector implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_items() -> list[dict]:
    questions = [{"query_id": r["question_id"], "question": r["question"]} for r in rows(QUESTIONS)]
    retrieval = rows(RETRIEVAL)
    if len(questions) != 500 or len({r["query_id"] for r in questions}) != 500:
        raise RuntimeError("enterprise question input is not exactly 500 unique questions")
    if len(retrieval) != 500 or {r["query_id"] for r in retrieval} != {r["query_id"] for r in questions}:
        raise RuntimeError("enterprise retrieval slates do not cover the 500 questions")
    retrieval_map = {r["query_id"]: r for r in retrieval}
    conn = sqlite3.connect(DB)
    items = []
    for query in questions:
        slate = retrieval_map[query["query_id"]]
        if slate.get("status") != "ok" or len(slate.get("candidate_ids", [])) != 20:
            raise RuntimeError(f"invalid frozen slate: {query['query_id']}")
        if [c["doc_id"] for c in slate["candidates"]] != slate["candidate_ids"]:
            raise RuntimeError(f"candidate ordering mismatch: {query['query_id']}")
        candidates = []
        for rank, candidate in enumerate(slate["candidates"], start=1):
            chunk_id = candidate.get("representative_chunk_id")
            source = conn.execute(
                "SELECT chunk_id, title, text, source_type FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
            if source is None:
                raise RuntimeError(f"representative chunk is missing: {query['query_id']} {chunk_id}")
            if source[0] != chunk_id or not source[2]:
                raise RuntimeError(f"invalid representative chunk: {query['query_id']} {chunk_id}")
            candidates.append({
                "doc_id": candidate["doc_id"],
                "dataset": source[3],
                "title": source[1],
                "full_text": source[2],
                "original_rank": rank,
                "representative_chunk_id": chunk_id,
                "archive_path": candidate["archive_path"],
                "original_doc_id": candidate["original_doc_id"],
            })
        if len({c["doc_id"] for c in candidates}) != 20:
            raise RuntimeError(f"duplicate candidate IDs: {query['query_id']}")
        items.append({"query_id": query["query_id"], "dataset": "enterprise", "question": query["question"], "candidates": candidates})
    conn.close()
    return items


def write_rows(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.shard_index is not None and (args.shard_count < 2 or not 0 <= args.shard_index < args.shard_count):
        raise SystemExit("shard index/count is invalid")
    all_items = build_items()
    tag = ""
    items = all_items
    if args.shard_index is not None:
        tag = f"_SHARD_{args.shard_index}_OF_{args.shard_count}"
        items = [item for position, item in enumerate(all_items) if position % args.shard_count == args.shard_index]
    legacy = load_legacy()
    revision = legacy.LAYER_PATHS[args.model][1]
    point_path = OUT / f"V11_TRACK_B_{args.model.upper()}_POINTWISE_FULL{tag}.jsonl"
    group_path = OUT / f"V11_TRACK_B_{args.model.upper()}_GROUPWISE_FULL{tag}.jsonl"
    existing = {r["query_id"]: r for r in rows(point_path)} if point_path.exists() else {}
    pending = [item for item in items if item["query_id"] not in existing or existing[item["query_id"]].get("status") != "ok"]
    if args.model == "bge":
        pointwise, groups = legacy.run_bge(pending, revision), []
    elif args.model == "qwen":
        pointwise, groups = legacy.run_qwen(pending, revision), []
    elif args.model == "laya":
        pointwise, groups = legacy.run_laya(pending, revision)
    else:
        pointwise, groups = legacy.run_clef(pending, revision)
    existing.update({r["query_id"]: r for r in pointwise})
    ordered = [existing[item["query_id"]] for item in items if item["query_id"] in existing]
    write_rows(point_path, ordered)
    if args.model in ("laya", "clef"):
        prior = {r["query_id"]: r for r in rows(group_path)} if group_path.exists() else {}
        prior.update({r["query_id"]: r for r in groups})
        write_rows(group_path, [prior[item["query_id"]] for item in items if item["query_id"] in prior])
    group_rows = rows(group_path) if group_path.exists() else []
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": "B",
        "model_key": args.model,
        "model_id": legacy.LAYER_PATHS[args.model][0],
        "revision": revision,
        "query_count": len(items),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "pointwise_rows": len(ordered),
        "groupwise_rows": len(group_rows),
        "pointwise_status_counts": {s: sum(r.get("status") == s for r in ordered) for s in sorted({r.get("status") for r in ordered})},
        "groupwise_status_counts": {s: sum(r.get("status") == s for r in group_rows) for s in sorted({r.get("status") for r in group_rows})},
        "retrieval_sha256": sha256(RETRIEVAL),
        "pointwise_sha256": sha256(point_path),
        "groupwise_sha256": sha256(group_path) if group_path.exists() else None,
        "gold_accessed": False,
        "jev_calls": 0,
        "hosted_cloudflare_calls": 0,
    }
    summary_path = OUT / f"V11_TRACK_B_{args.model.upper()}_SELECTOR_FULL{tag}_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    complete = summary["pointwise_status_counts"].get("ok", 0) == len(items) and (not group_rows or summary["groupwise_status_counts"].get("ok", 0) == len(items))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
