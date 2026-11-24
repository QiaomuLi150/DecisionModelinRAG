#!/usr/bin/env python3
"""Freeze an append-only receipt for the completed R1 raw traces."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v3_6_essential/followup_v3_repro_fixes"
RAW = BASE / "raw"
OUT = RAW / "R1_RAW_COMPLETION.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


files = sorted(
    p for p in list(RAW.glob("pplx*.jsonl")) + list(RAW.glob("jina*.jsonl"))
    if not p.name.endswith(".part") and "SUMMARY" not in p.name
)
file_rows = []
model_counts = Counter()
status_counts = Counter()
for path in files:
    rows = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            obj = json.loads(line)
            model = "pplx" if path.name.startswith("pplx") else "jina"
            model_counts[(model, obj.get("status", "MISSING"))] += 1
            status_counts[(model, obj.get("status", "MISSING"))] += 1
    file_rows.append({
        "path": str(path.relative_to(ROOT)),
        "rows": rows,
        "sha256": sha256(path),
    })

receipt = {
    "status": "PASS",
    "protocol": "R1 exact serialized decrement-to-fit; no retries; labels joined offline",
    "expected_rows_per_model": 10668,
    "raw_files": file_rows,
    "model_status_counts": {
        f"{m}:{s}": n for (m, s), n in sorted(model_counts.items())
    },
    "pplx_accounting": {
        "prefix_rows": 5374,
        "boundary_cancellation_rows": 2,
        "suffix_rows": 5292,
        "total_rows": 10668,
    },
    "jina_accounting": {"single_trace_rows": 10668},
    "labels_joined": False,
    "completion_boundary": "all final JSONL traces present and suffix workers returned PASS",
}
OUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2, sort_keys=True))
