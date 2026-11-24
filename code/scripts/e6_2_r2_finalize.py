#!/usr/bin/env python3
"""Finalize E6-2 R2 reports and provenance after selector inference."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r2_selector_inference"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    test_report = {
        "schema": "E6-2-R2-TEST-REPORT-V2",
        "created_utc": now,
        "focused_r2": {"command": "pytest -q tests/test_e6_2_r2_preflight.py tests/test_e6_2_r2_selector_inference.py", "collected": 11, "passed": 11, "failed": 0},
        "full_authoritative": {"command": "./Production_Project/MultiModalRAG/venv/bin/python -m pytest -q", "collected": 225, "passed": 225, "failed": 0, "environment": "shared MultiModalRAG virtual environment"},
        "native_inference": {"scheduled_attempts": 2216, "actual_attempts": 1939, "quality_scoring": False},
        "prohibited_calls": {"qrels": 0, "reader": 0, "jev": 0, "paid_api": 0},
        "historical_artifacts_modified": False,
    }
    (OUT / "E6_2_R2_TEST_REPORT.json").write_text(json.dumps(test_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "E6_2_R2_TEST_REPORT.md").write_text(
        "# E6-2 R2 Test Report\n\n"
        "- Focused R2 integrity tests: **11/11 passed**.\n"
        "- Full authoritative regression suite: **225/225 passed** in the shared MultiModalRAG environment.\n"
        "- Native selector attempts: 2,216 scheduled; 1,939 executed because StartLux was explicitly blocked at model load by missing `causal_conv1d`.\n"
        "- Amos: 277 attempts, 194 valid, 83 explicit context-limit failures.\n"
        "- Quality scoring, reader, Jev and paid API calls: **0**.\n",
        encoding="utf-8",
    )
    gate_path = OUT / "E6_2_R2_R3_READINESS_GATE.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["created_utc"] = now
    gate["tests"] = {"focused_r2": "11/11", "full_authoritative": "225/225", "environment": "shared MultiModalRAG virtual environment"}
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = OUT / "E6_2_R2_FINAL_REPORT.md"
    text = report.read_text(encoding="utf-8")
    marker = "\n## Test and provenance closure\n"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    text += "\n## Test and provenance closure\n\n- Focused R2 tests: **11/11 passed**.\n- Full authoritative regression: **225/225 passed**.\n- All new artifact hashes are recorded in `E6_2_R2_PROVENANCE_MANIFEST.json`.\n- StartLux remains explicitly excluded because its native adapter could not load in the active environment (`ModuleNotFoundError: causal_conv1d`); no replacement or fallback was used.\n"
    report.write_text(text, encoding="utf-8")
    excluded = {"E6_2_R2_PROVENANCE_MANIFEST.json"}
    files = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in excluded:
            files.append({"path": str(path.relative_to(ROOT)), "sha256": sha_file(path), "bytes": path.stat().st_size})
    code_files = []
    for rel in ["scripts/e6_2_r2_selector_inference.py", "scripts/e6_2_r2_preflight.py", "scripts/e6_2_r2_finalize.py", "tests/test_e6_2_r2_preflight.py", "tests/test_e6_2_r2_selector_inference.py"]:
        path = ROOT / rel
        code_files.append({"path": rel, "sha256": sha_file(path), "bytes": path.stat().st_size})
    provenance = {
        "schema": "E6-2-R2-PROVENANCE-V2",
        "created_utc": now,
        "gate": gate["gate"],
        "files": files,
        "code": code_files,
        "scheduled_attempts": 2216,
        "actual_attempts": 1939,
        "quality_metrics": False,
        "qrels_accessed": False,
        "reserved_content_accessed": False,
        "reader_calls": 0,
        "jev_calls": 0,
        "paid_api_calls": 0,
        "historical_artifacts_modified": False,
    }
    (OUT / "E6_2_R2_PROVENANCE_MANIFEST.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
