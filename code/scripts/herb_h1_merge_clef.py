#!/usr/bin/env python3
"""Merge deterministic Clef selector shards into canonical H1 traces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/herb_h1"
RETRIEVAL = OUT / "HERB_H1_RETRIEVAL_SLATES.jsonl"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def merge(kind: str, shards: int) -> dict:
    retrieval = load(RETRIEVAL)
    order = [row["query_id"] for row in retrieval]
    seen: dict[str, dict] = {}
    shard_counts = {}
    for index in range(shards):
        path = OUT / f"HERB_H1_CLEF_{kind.upper()}_FULL_SHARD_{index}_OF_{shards}.jsonl"
        if not path.exists():
            raise RuntimeError(f"missing shard: {path.name}")
        rows = load(path)
        shard_counts[str(index)] = len(rows)
        for row in rows:
            qid = row.get("query_id")
            if qid in seen:
                raise RuntimeError(f"duplicate shard query: {qid}")
            if row.get("status") != "ok":
                raise RuntimeError(f"non-ok Clef shard row: {qid}")
            seen[qid] = row
    if set(seen) != set(order):
        missing = sorted(set(order) - set(seen))
        extra = sorted(set(seen) - set(order))
        raise RuntimeError(f"Clef coverage mismatch missing={missing[:3]} extra={extra[:3]}")
    rows = [seen[qid] for qid in order]
    target = OUT / f"HERB_H1_CLEF_{kind.upper()}_FULL.jsonl"
    write(target, rows)
    return {"rows": len(rows), "shard_counts": shard_counts, "path": target.name, "sha256": digest(target)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=3)
    args = parser.parse_args()
    pointwise = merge("pointwise", args.shards)
    groupwise = merge("groupwise", args.shards)
    summary = {
        "status": "PASS",
        "model_key": "clef",
        "population": "FULL",
        "queries": pointwise["rows"],
        "pointwise_ok": pointwise["rows"],
        "groupwise_ok": groupwise["rows"],
        "pointwise": pointwise,
        "groupwise": groupwise,
        "retrieval_sha256": digest(RETRIEVAL),
        "gold_accessed": False,
        "jev_calls": 0,
        "hosted_cloudflare_calls": 0,
    }
    (OUT / "HERB_H1_CLEF_SELECTOR_FULL_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
