#!/usr/bin/env python3
"""Merge a preserved prefix and relocated suffix without rerunning inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v2_1_replication/stage2_orders_v2_hybrid"
PREFIX = BASE / "pplx_two_stage__perm_seed_20261020__full.jsonl.part"
SUFFIX = BASE / "pplx_two_stage__perm_seed_20261020__start733__n1934.jsonl"
OUT = BASE / "pplx_two_stage__perm_seed_20261020__full.jsonl"


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    prefix, suffix = read(PREFIX), read(SUFFIX)
    if len(prefix) != 733 or len(suffix) != 1934:
        raise RuntimeError(f"unexpected_parts:{len(prefix)}:{len(suffix)}")
    rows = prefix + suffix
    keys = [(r.get("canonical_id"), r.get("order_id"), r.get("policy")) for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate_query_records")
    if any(r.get("order_id") != "perm_seed_20261020" or r.get("policy") != "pplx_two_stage" for r in rows):
        raise RuntimeError("wrong_policy_or_order")
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {"status": "MERGED_OFFLINE", "prefix": str(PREFIX.name), "prefix_rows": len(prefix),
               "suffix": str(SUFFIX.name), "suffix_rows": len(suffix), "rows": len(rows),
               "raw_sha256": sha(OUT), "labels_joined": False, "model_calls_added": 0}
    (BASE / "pplx_two_stage__perm_seed_20261020__MERGE.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
