#!/usr/bin/env python3
"""Run the frozen DecisionRAG selector adapters over HERB H1 slates.

The model functions are imported from the existing Track A/B selector module;
this wrapper changes only the query/candidate adapter and output location.
It does not read HERB gold labels and does not call Jev or hosted Cloudflare.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/herb_h1"
RETRIEVAL = OUT / "HERB_H1_RETRIEVAL_SLATES.jsonl"
DB = OUT / "HERB_H1_CORPUS.sqlite3"
MODELS = ("bge", "qwen", "laya", "clef")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_legacy():
    path = ROOT / "scripts/run_m3_selectors.py"
    spec = importlib.util.spec_from_file_location("decisionrag_h1_legacy_selectors", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen selector implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_items(query_ids: set[str] | None = None) -> list[dict]:
    retrieval = rows(RETRIEVAL)
    conn = sqlite3.connect(DB)
    items = []
    for slate in retrieval:
        if query_ids is not None and slate["query_id"] not in query_ids:
            continue
        if slate.get("status") != "ok" or len(slate.get("candidate_ids", [])) != 20:
            raise RuntimeError(f"invalid H1 slate: {slate.get('query_id')}")
        candidates = []
        for rank, candidate in enumerate(slate["candidates"], start=1):
            chunk_id = candidate["representative_chunk_id"]
            source = conn.execute("SELECT title, text, text_sha256, source_type, product FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
            if source is None or not source[1]:
                raise RuntimeError(f"missing H1 representative chunk {slate['query_id']}:{chunk_id}")
            candidates.append({
                "doc_id": candidate["doc_id"], "dataset": "herb", "title": source[0],
                "full_text": source[1], "original_rank": rank,
                "original_source_id": candidate["original_source_id"],
                "product": source[4], "source_type": source[3],
                "representative_chunk_id": chunk_id, "representative_text_sha256": source[2],
            })
        if len({c["doc_id"] for c in candidates}) != 20:
            raise RuntimeError(f"duplicate H1 candidate IDs: {slate['query_id']}")
        items.append({"query_id": slate["query_id"], "dataset": "herb", "question": slate["question"], "candidates": candidates})
    conn.close()
    if query_ids is None and len(items) != 1514:
        raise RuntimeError(f"expected 1514 H1 slates, got {len(items)}")
    if any(not item["question"].strip() for item in items):
        raise RuntimeError("blank H1 question passed to selector")
    return items


def write_rows(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def run(model: str, items: list[dict], legacy):
    revision = legacy.LAYER_PATHS[model][1]
    if model == "bge":
        return legacy.run_bge(items, revision), []
    if model == "qwen":
        return legacy.run_qwen(items, revision), []
    if model == "laya":
        return legacy.run_laya(items, revision)
    return legacy.run_clef(items, revision)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--query-id", action="append", default=[])
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.smoke and args.query_id:
        raise SystemExit("use --smoke or --query-id, not both")
    retrieval = rows(RETRIEVAL)
    if args.smoke:
        # Four deterministic positions cover both answerability cohorts without
        # reading their labels: first, early-middle, late-middle, last.
        positions = sorted({0, len(retrieval) // 3, (2 * len(retrieval)) // 3, len(retrieval) - 1})
        query_ids = {retrieval[i]["query_id"] for i in positions}
    elif args.query_id:
        query_ids = set(args.query_id)
    else:
        query_ids = None
    if args.shard_index is not None and (args.shard_count < 2 or not 0 <= args.shard_index < args.shard_count):
        raise SystemExit("invalid shard index/count")
    items = build_items(query_ids)
    if args.shard_index is not None:
        items = [item for position, item in enumerate(items) if position % args.shard_count == args.shard_index]
    legacy = load_legacy()
    pointwise, groupwise = run(args.model, items, legacy)
    tag = "SMOKE" if args.smoke else "FULL"
    if args.shard_index is not None:
        tag += f"_SHARD_{args.shard_index}_OF_{args.shard_count}"
    point_path = OUT / f"HERB_H1_{args.model.upper()}_POINTWISE_{tag}.jsonl"
    group_path = OUT / f"HERB_H1_{args.model.upper()}_GROUPWISE_{tag}.jsonl"
    write_rows(point_path, pointwise)
    if groupwise:
        write_rows(group_path, groupwise)
    summary = {
        "status": "PASS" if sum(row.get("status") == "ok" for row in pointwise) == len(items) and (not groupwise or sum(row.get("status") == "ok" for row in groupwise) == len(items)) else "BLOCKED",
        "model_key": args.model, "model_id": legacy.LAYER_PATHS[args.model][0], "revision": legacy.LAYER_PATHS[args.model][1],
        "population": tag, "queries": len(items), "pointwise_ok": sum(row.get("status") == "ok" for row in pointwise),
        "groupwise_ok": sum(row.get("status") == "ok" for row in groupwise),
        "pointwise_sha256": sha256(point_path), "groupwise_sha256": sha256(group_path) if groupwise else None,
        "retrieval_sha256": sha256(RETRIEVAL), "gold_accessed": False, "jev_calls": 0, "hosted_cloudflare_calls": 0,
        "query_ids": [item["query_id"] for item in items],
    }
    summary_path = OUT / f"HERB_H1_{args.model.upper()}_SELECTOR_{tag}_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
