#!/usr/bin/env python3
"""Create a path-free schema inventory for every released CSV and JSONL file."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/DATA_DICTIONARY.md"


def columns_for(path: Path) -> tuple[list[str], int]:
    if path.name.endswith(".csv"):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            return list(reader.fieldnames or []), sum(1 for _ in reader)
    opener = gzip.open if path.name.endswith(".gz") else open
    keys: set[str] = set()
    count = 0
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                keys.update(value)
            count += 1
    return sorted(keys), count


def unit(key: str) -> str:
    lower = key.lower()
    if "ratio" in lower:
        return "unitless ratio"
    if lower.endswith("_ms") or lower in {"elapsed_ms", "latency_ms", "duration_ms"}:
        return "milliseconds"
    if "mib" in lower:
        return "MiB"
    if "gib" in lower:
        return "GiB"
    if "percent" in lower or lower.endswith("_pct"):
        return "percent"
    if "ndcg" in lower or "fs5" in lower or "rate" in lower or "share" in lower or "prob" in lower:
        return "unit interval unless field name says percent"
    if lower.endswith("_sha256") or "hash" in lower:
        return "SHA-256 or digest text"
    if "token" in lower:
        return "token count"
    return "identifier / count / categorical / source-defined"


def main() -> None:
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and (p.name.endswith(".csv") or p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz")))
    lines = [
        "# Released tabular-data dictionary",
        "",
        "This file inventories every released CSV and JSONL/JSONL.GZ path, its row count, and the columns/keys present. It is generated from the released headers/records; it does not infer semantic meanings beyond conventional field suffixes.",
        "",
        "All query/document text columns were omitted from public projections. `valid` is a strict valid Top-5 selection indicator; failed requests contribute zero to all-request quality metrics. IDs retain canonical values and order where supplied, except path-like/private-looking E2-N IDs replaced by deterministic SHA-256 pseudonyms (see SANITIZATION_LOG.md). Ranking score ties follow the recorded stable original ordering; where an adapter does not expose a tied score order, do not assume a cross-model tie rule.",
        "",
    ]
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        cols, count = columns_for(path)
        lines.extend([f"## `{rel}`", "", f"Rows: {count}.", "", "| Field | Unit/type |", "|---|---|"])
        lines.extend(f"| `{key}` | {unit(key)} |" for key in cols)
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"DATA_DICTIONARY entries={len(files)} output=data/DATA_DICTIONARY.md")


if __name__ == "__main__":
    main()
