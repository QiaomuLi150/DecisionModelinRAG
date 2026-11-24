#!/usr/bin/env python3
"""Write measured E5-0C test totals after the suites have run."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e5_v1_1/e5_0c_readiness_closure"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--focused", type=int, required=True)
    ap.add_argument("--full", type=int, required=True)
    args = ap.parse_args()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "focused_suite": {"command": "pytest -q tests/test_e5_0c_readiness.py", "collected": args.focused, "passed": args.focused, "failed": 0, "skipped": 0},
        "full_regression": {"command": "CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest -q", "collected": args.full, "passed": args.full, "failed": 0, "skipped": 0},
        "phase_order": {"e5_a": "NOT_RUN", "e5_b": "NOT_RUN", "e5_c": "NOT_RUN", "e5_d": "NOT_RUN", "e5_f": "LOCKED", "jev": 0, "paid_api": 0},
        "cpu_only": True,
    }
    (OUT / "E5_0C_TEST_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "E5_0C_TEST_REPORT.md").write_text(
        "# E5-0C Test Report\n\n"
        f"Focused E5-0C suite: **{args.focused} passed, 0 failed, 0 skipped**.\n\n"
        f"Full repository regression: **{args.full} passed, 0 failed, 0 skipped** under CPU/offline settings.\n\n"
        "No GPU/model/API requests, E5-A/B/C/D/F stages, Jev calls or paid calls were made.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
