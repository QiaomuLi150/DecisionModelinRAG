#!/usr/bin/env python3
"""Offline merge of complete Enterprise and MuSiQue split traces."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v2_1_replication/stage2_orders_v2_hybrid"
ORDERS = ["perm_seed_20261020", "perm_seed_20261021"]


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def fsha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    outputs = []
    for order in ORDERS:
        parts = [BASE / f"pplx_budget__{order}__enterprise__full.jsonl", BASE / f"pplx_budget__{order}__musique__full.jsonl"]
        if any(not p.exists() for p in parts):
            raise RuntimeError(f"missing_split:{order}")
        rows = read(parts[0]) + read(parts[1])
        if len(rows) != 2667 or len({(r.get("dataset"), r.get("canonical_id"), r.get("order_id"), r.get("policy")) for r in rows}) != 2667:
            raise RuntimeError(f"invalid_split_merge:{order}:{len(rows)}")
        out = BASE / f"pplx_budget__{order}__full.jsonl"
        if out.exists():
            raise RuntimeError(f"refusing_to_overwrite:{out}")
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        summary = {"status": "MERGED_OFFLINE", "order_id": order, "parts": [p.name for p in parts], "rows": len(rows), "raw_sha256": fsha(out), "model_calls_added": 0, "labels_joined": False}
        meta = BASE / f"pplx_budget__{order}__MERGE.json"
        meta.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        outputs.append(summary)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
