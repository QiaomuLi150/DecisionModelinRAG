#!/usr/bin/env python3
"""Merge deterministic Clef Track A selector shards without reordering rows."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v11/track_a"
COHORT = ROOT / "results/preflight/v11/NEW_TEST_IDS.jsonl"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, values: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    cohort = rows(COHORT)
    order = {value["query_id"]: index for index, value in enumerate(cohort)}
    merged: dict[str, dict] = {}
    group_merged: dict[str, dict] = {}
    shard_files = []
    for shard_index in range(2):
        suffix = f"_SHARD_{shard_index}_OF_2"
        point_path = OUT / f"V11_TRACK_A_CLEF_POINTWISE_FULL{suffix}.jsonl"
        group_path = OUT / f"V11_TRACK_A_CLEF_GROUPWISE_FULL{suffix}.jsonl"
        point_rows = rows(point_path)
        group_rows = rows(group_path)
        if len(point_rows) != len(set(row["query_id"] for row in point_rows)):
            raise RuntimeError(f"duplicate pointwise query IDs in {point_path}")
        if len(group_rows) != len(set(row["query_id"] for row in group_rows)):
            raise RuntimeError(f"duplicate groupwise query IDs in {group_path}")
        for value in point_rows:
            if value["query_id"] in merged:
                raise RuntimeError(f"overlapping pointwise query ID: {value['query_id']}")
            merged[value["query_id"]] = value
        for value in group_rows:
            if value["query_id"] in group_merged:
                raise RuntimeError(f"overlapping groupwise query ID: {value['query_id']}")
            group_merged[value["query_id"]] = value
        shard_files.extend([str(point_path), str(group_path)])

    expected = {value["query_id"] for value in cohort}
    if set(merged) != expected or set(group_merged) != expected:
        raise RuntimeError(f"shard coverage mismatch: pointwise={len(merged)} groupwise={len(group_merged)} expected={len(expected)}")
    point_out = OUT / "V11_TRACK_A_CLEF_POINTWISE_FULL.jsonl"
    group_out = OUT / "V11_TRACK_A_CLEF_GROUPWISE_FULL.jsonl"
    write(point_out, [merged[value["query_id"]] for value in cohort])
    write(group_out, [group_merged[value["query_id"]] for value in cohort])
    point_status = {}
    for value in merged.values():
        point_status[value.get("status")] = point_status.get(value.get("status"), 0) + 1
    group_status = {}
    for value in group_merged.values():
        group_status[value.get("status")] = group_status.get(value.get("status"), 0) + 1
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "full",
        "model": "Cloudflare/clef-flash",
        "model_key": "clef",
        "revision": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
        "query_count": len(cohort),
        "shard_count": 2,
        "shard_files": shard_files,
        "pointwise_rows": len(merged),
        "groupwise_rows": len(group_merged),
        "pointwise_status_counts": dict(sorted(point_status.items())),
        "groupwise_status_counts": dict(sorted(group_status.items())),
        "pointwise_sha256": digest(point_out),
        "groupwise_sha256": digest(group_out),
        "retrieval_sha256": "1891f41e2b8614b871aa42cdbe61af3d0c193410a85041e5426c5e7eb3905f0f",
        "gold_accessed": False,
        "jev_calls": 0,
        "hosted_cloudflare_calls": 0,
    }
    (OUT / "V11_TRACK_A_CLEF_SELECTOR_FULL_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
