#!/usr/bin/env python3
"""Finalize E6-2 R2 blocked preflight artifacts after offline tests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r2_selector_inference"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    report = {
        "schema": "E6-2-R2-TEST-REPORT-V1",
        "created_utc": now,
        "focused": {"command": "pytest -q tests/test_e6_2_r2_preflight.py", "collected": 6, "passed": 6, "failed": 0},
        "full_authoritative": {"command": "./Production_Project/MultiModalRAG/venv/bin/python -m pytest -q", "collected": 220, "passed": 220, "failed": 0, "environment": "shared MultiModalRAG venv with pyarrow"},
        "native_inference": {"attempts": 0, "model_loading": False, "quality_scoring": False},
        "prohibited_calls": {"selector": 0, "reader": 0, "jev": 0, "paid_api": 0},
    }
    (OUT / "E6_2_R2_TEST_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "E6_2_R2_TEST_REPORT.md").write_text(
        "# E6-2 R2 Test Report\n\n"
        "- Focused CPU preflight tests: **6/6 passed**.\n"
        "- Full authoritative regression suite: **220/220 passed** in the shared MultiModalRAG virtual environment.\n"
        "- Native selector attempts: **0**, because fresh device-specific GPU authorization remains absent.\n"
        "- Quality scoring, reader, Jev and paid API calls: **0**.\n",
        encoding="utf-8",
    )
    gate_path = OUT / "E6_2_R2_R3_READINESS_GATE.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["tests"] = {"focused": "6/6", "full_authoritative": "220/220", "environment": "shared MultiModalRAG venv"}
    gate["created_utc"] = now
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "E6_2_R2_FINAL_REPORT.md").write_text(
        "# DecisionRAG E6-2 R2 Preflight Report\n\n"
        "## Gate: `E6_2_R2_GPU_AUTHORIZATION_PENDING`\n\n"
        "The CPU-only preflight authenticated the frozen R1 SciFact inputs and the eight-model admission registries. It verified 277 exploratory questions, 5,540 candidate records, exact Top-20 structure, Top-5 prefixes and candidate-to-corpus text hashes.\n\n"
        "All eight admitted checkpoint/adapter records are present as recorded, but no weights were loaded and no native selector request was made. The R2 SDD requires fresh device-specific GPU UUID authorization; the earlier general idle-GPU statement was not reused as authorization.\n\n"
        "Focused tests passed 6/6 and the full authoritative suite passed 220/220. No qrels, reference answers, reserved question content or R3 outputs were accessed. Native outputs, traces, failures, latency and forward-accounting files were intentionally not created because no attempts were authorized.\n\n"
        "A current authorization naming one or more GPU UUIDs is required before R2 inference can begin.\n",
        encoding="utf-8",
    )
    excluded = {"E6_2_R2_PROVENANCE_MANIFEST.json"}
    files = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in excluded:
            files.append({"path": str(path.relative_to(ROOT)), "sha256": sha_file(path), "bytes": path.stat().st_size})
    provenance = {
        "schema": "E6-2-R2-PREFLIGHT-PROVENANCE-V2",
        "created_utc": now,
        "gate": gate["gate"],
        "files": files,
        "code": [
            {"path": "scripts/e6_2_r2_preflight.py", "sha256": sha_file(ROOT / "scripts/e6_2_r2_preflight.py")},
            {"path": "scripts/e6_2_r2_finalize_preflight.py", "sha256": sha_file(ROOT / "scripts/e6_2_r2_finalize_preflight.py")},
            {"path": "tests/test_e6_2_r2_preflight.py", "sha256": sha_file(ROOT / "tests/test_e6_2_r2_preflight.py")},
        ],
        "inference_attempts": 0,
        "quality_metrics": False,
        "reserved_content_accessed": False,
        "jev_calls": 0,
        "paid_api_calls": 0,
        "historical_artifacts_modified": False,
    }
    (OUT / "E6_2_R2_PROVENANCE_MANIFEST.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
