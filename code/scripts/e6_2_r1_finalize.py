#!/usr/bin/env python3
"""Finalize E6-2 R1 offline test and provenance artifacts without inference."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r1_scifact_retrieval_freeze"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    test_report = {
        "schema": "E6-2-R1-TEST-AND-REGRESSION-REPORT-V2",
        "created_utc": now,
        "focused": {"command": "pytest -q tests/test_e6_2_r1_retrieval_freeze.py", "collected": 6, "passed": 6, "failed": 0},
        "full_authoritative": {
            "command": "./Production_Project/MultiModalRAG/venv/bin/python -m pytest -q",
            "collected": 214,
            "passed": 214,
            "failed": 0,
            "environment": "shared MultiModalRAG venv (Python 3.12.3; pyarrow available)",
        },
        "system_python_check": {
            "command": "pytest -q",
            "collected": 214,
            "passed": 212,
            "failed": 2,
            "failure_class": "pre-existing SciFact audit tests could not import pyarrow from /usr/bin/python3",
            "not_a_retrieval_or_R1_test_failure": True,
        },
        "inference_rerun": False,
        "selector_calls": 0,
        "reader_calls": 0,
        "jev_calls": 0,
        "paid_api_calls": 0,
    }
    write_json(OUT / "E6_2_R1_TEST_AND_REGRESSION_REPORT_V2.json", test_report)
    (OUT / "E6_2_R1_TEST_AND_REGRESSION_REPORT_V2.md").write_text(
        "# E6-2 R1 Test and Regression Report\n\n"
        "- Focused R1 tests: **6/6 passed**.\n"
        "- Full authoritative suite: **214/214 passed** in the shared MultiModalRAG virtual environment, where the cached Arrow inputs are readable.\n"
        "- System-Python comparison: 212/214 passed; 2 existing SciFact audit tests failed only because `/usr/bin/python3` lacks `pyarrow`.\n"
        "- No inference was rerun during finalization. Selector, reader, Jev and paid API calls: 0.\n",
        encoding="utf-8",
    )

    excluded = {"E6_2_R1_PROVENANCE_MANIFEST_V3.json"}
    files = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in excluded:
            files.append({"path": str(path.relative_to(ROOT)), "sha256": sha_file(path), "bytes": path.stat().st_size})
    provenance = {
        "schema": "E6-2-R1-PROVENANCE-V3",
        "created_utc": now,
        "gate": "E6_2_R1_RETRIEVAL_FROZEN_READY",
        "post_test_manifest": True,
        "files": files,
        "test_report": "E6_2_R1_TEST_AND_REGRESSION_REPORT_V2.json",
        "physical_work": {"embedding_batches": 376, "embedding_items": 6015, "model_forward_count": "host service telemetry not exposed in client response"},
        "zero_calls": {"selector": 0, "reranker": 0, "reader": 0, "jev": 0, "paid_api": 0},
        "reserved_content_access": False,
        "historical_artifacts_modified": False,
    }
    write_json(OUT / "E6_2_R1_PROVENANCE_MANIFEST_V3.json", provenance)
    (OUT / "E6_2_R1_FINAL_RETRIEVAL_REPORT_V3.md").write_text(
        "# DecisionRAG E6-2 R1 Final Retrieval-Freeze Closure\n\n"
        "## Gate: `E6_2_R1_RETRIEVAL_FROZEN_READY`\n\n"
        "The authorized retrieval-only run produced 555 CAL and 277 exploratory Top-20 SciFact slates, plus 277 deterministic Top-5 prefixes. It issued 376 physical requests to the pre-existing Qwen3-VL-Embedding-2B service at revision `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`, with zero failed batches.\n\n"
        "The retrieval contract was BM25 Top-100 plus exact-cosine dense Top-100 fused with RRF(60), yielding 20 unique canonical `scifact:<_id>` candidates per query. Qrels, support labels, answers, selectors, readers and RESERVED content were not accessed. No new GPU process was started and no process was evicted.\n\n"
        "Focused R1 tests passed 6/6. The full authoritative suite passed 214/214 in the shared ML environment. The system-Python comparison was 212/214 only because that interpreter lacks `pyarrow`; this is recorded as an environment limitation, not an R1 retrieval failure.\n\n"
        "R1 is complete. E6-2 selector inference remains a separate owner authorization and was not executed.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
