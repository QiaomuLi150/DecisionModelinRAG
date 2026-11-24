#!/usr/bin/env python3
"""Track A selector adapter using the existing frozen selector implementations.

The historical M3 files are never overwritten.  This wrapper supplies the
versioned Track A question/slate inputs and writes separate traces per model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/non_jev/v11/track_a"
COHORT_PATH = ROOT / "results/preflight/v11/NEW_TEST_IDS.jsonl"
CORPUS_PATH = ROOT / "data/corpus/passages.jsonl"
RETRIEVAL_PATH = OUT_DIR / "V11_TRACK_A_RETRIEVAL_SLATES.jsonl"
MODELS = ("bge", "qwen", "laya", "clef")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_legacy_module():
    path = ROOT / "scripts/run_m3_selectors.py"
    spec = importlib.util.spec_from_file_location("decisionrag_v11_legacy_m3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen M3 selector implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_smoke_ids(cohort: list[dict]) -> set[str]:
    by_dataset = {}
    for row in cohort:
        by_dataset.setdefault(row["dataset"], []).append(row)
    quotas = {"hotpot": 7, "musique": 7, "wiki2": 6}
    return {row["query_id"] for dataset in ("hotpot", "musique", "wiki2") for row in by_dataset[dataset][:quotas[dataset]]}


def build_items(cohort: list[dict], retrieval: list[dict], mode: str) -> list[dict]:
    corpus = {row["doc_id"]: row for row in load_rows(CORPUS_PATH)}
    retrieval_map = {row["query_id"]: row for row in retrieval}
    allowed = fixed_smoke_ids(cohort) if mode == "smoke" else {row["query_id"] for row in cohort}
    items = []
    for query in cohort:
        if query["query_id"] not in allowed:
            continue
        slate = retrieval_map.get(query["query_id"])
        if not slate or slate.get("status") != "ok":
            raise RuntimeError(f"missing successful retrieval slate: {query['query_id']}")
        candidates = []
        for rank, doc_id in enumerate(slate["candidate_ids"], start=1):
            source = corpus[doc_id]
            candidates.append({"doc_id": doc_id, "dataset": source["dataset"], "title": source["title"], "full_text": source["full_text"], "original_rank": rank})
        items.append({"query_id": query["query_id"], "dataset": query["dataset"], "question": query["question"], "candidates": candidates})
    return items


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.shard_index is not None and (args.mode != "full" or args.shard_count < 2 or not 0 <= args.shard_index < args.shard_count):
        raise SystemExit("sharding is only supported for full mode with a valid shard index/count")
    cohort = load_rows(COHORT_PATH)
    retrieval_name = "V11_TRACK_A_RETRIEVAL_SMOKE.jsonl" if args.mode == "smoke" else "V11_TRACK_A_RETRIEVAL_SLATES.jsonl"
    retrieval = load_rows(OUT_DIR / retrieval_name)
    items = build_items(cohort, retrieval, args.mode)
    shard_tag = ""
    if args.shard_index is not None:
        shard_tag = f"_SHARD_{args.shard_index}_OF_{args.shard_count}"
        items = [item for position, item in enumerate(items) if position % args.shard_count == args.shard_index]
    legacy = load_legacy_module()
    revision = legacy.LAYER_PATHS[args.model][1]
    point_path = OUT_DIR / f"V11_TRACK_A_{args.model.upper()}_POINTWISE_{args.mode.upper()}{shard_tag}.jsonl"
    group_path = OUT_DIR / f"V11_TRACK_A_{args.model.upper()}_GROUPWISE_{args.mode.upper()}{shard_tag}.jsonl"
    existing = {row["query_id"]: row for row in load_rows(point_path)} if point_path.exists() else {}
    pending = [item for item in items if item["query_id"] not in existing or existing[item["query_id"]].get("status") != "ok"]
    if args.model == "bge":
        pointwise = legacy.run_bge(pending, revision)
        groups = []
    elif args.model == "qwen":
        pointwise = legacy.run_qwen(pending, revision)
        groups = []
    elif args.model == "laya":
        pointwise, groups = legacy.run_laya(pending, revision)
    else:
        pointwise, groups = legacy.run_clef(pending, revision)
    merged = {**existing, **{row["query_id"]: row for row in pointwise}}
    ordered = [merged[item["query_id"]] for item in items if item["query_id"] in merged]
    write_rows(point_path, ordered)
    if args.model in ("laya", "clef"):
        prior_groups = {row["query_id"]: row for row in load_rows(group_path)} if group_path.exists() else {}
        prior_groups.update({row["query_id"]: row for row in groups})
        write_rows(group_path, [prior_groups[item["query_id"]] for item in items if item["query_id"] in prior_groups])
    status_counts = {status: sum(row.get("status") == status for row in ordered) for status in sorted({row.get("status") for row in ordered})}
    group_rows = load_rows(group_path) if group_path.exists() else []
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "model": legacy.LAYER_PATHS[args.model][0],
        "model_key": args.model,
        "revision": revision,
        "query_count": len(items),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "pointwise_status_counts": status_counts,
        "groupwise_status_counts": {status: sum(row.get("status") == status for row in group_rows) for status in sorted({row.get("status") for row in group_rows})} if group_rows else {},
        "pointwise_rows": len(ordered),
        "groupwise_rows": len(group_rows),
        "retrieval_sha256": sha256(OUT_DIR / retrieval_name),
        "pointwise_sha256": sha256(point_path),
        "groupwise_sha256": sha256(group_path) if group_path.exists() else None,
        "gold_accessed": False,
        "jev_calls": 0,
        "hosted_cloudflare_calls": 0,
    }
    (OUT_DIR / f"V11_TRACK_A_{args.model.upper()}_SELECTOR_{args.mode.upper()}{shard_tag}_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if all(value == len(items) for key, value in status_counts.items() if key == "ok") and status_counts.get("ok", 0) == len(items) and (not group_rows or all(row.get("status") == "ok" for row in group_rows)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
