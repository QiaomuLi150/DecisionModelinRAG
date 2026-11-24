"""Repair E4-1 execution telemetry from the immutable raw trace.

The first telemetry writer counted only scheduler jobs that remained in the
current invocation, so exact historical A/B reuses that were correctly skipped
were reported as zero.  This offline repair preserves the prior telemetry and
derives accounting from the raw trace instead; it makes no model or service
request.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e4_v1_1/e4_1_interventions"
TELEMETRY = OUT / "E4_1_EXECUTION_TELEMETRY.json"
RAW = OUT / "E4_1_NATIVE_READER_RAW.jsonl"
BACKUP = OUT / "E4_1_EXECUTION_TELEMETRY_PRE_REPAIR.json"


def main() -> None:
    prior = json.loads(TELEMETRY.read_text())
    if not BACKUP.exists():
        BACKUP.write_text(json.dumps(prior, indent=2, sort_keys=True) + "\n")
    rows = [json.loads(line) for line in RAW.read_text().splitlines() if line.strip()]
    accepted = [row for row in rows if row.get("accepted_status") == "accepted"]
    corrected = dict(prior)
    corrected.update(
        {
            "telemetry_source": "E4_1_NATIVE_READER_RAW.jsonl",
            "telemetry_repaired_offline": True,
            "raw_rows": len(rows),
            "accepted_rows": len(accepted),
            "raw_cal_rows": sum(row.get("phase") == "cal" for row in rows),
            "raw_study_rows": sum(row.get("phase") == "study" for row in rows),
            "accepted_cal_rows": sum(row.get("phase") == "cal" for row in accepted),
            "accepted_study_rows": sum(row.get("phase") == "study" for row in accepted),
            "raw_infeasible_rows": sum(row.get("status") == "ARM_INFEASIBLE" for row in rows),
            "new_physical_calls": sum(int(row.get("physical_calls") or 0) for row in rows),
            "new_physical_cal_calls": sum(int(row.get("physical_calls") or 0) for row in rows if row.get("phase") == "cal"),
            "new_physical_study_calls": sum(int(row.get("physical_calls") or 0) for row in rows if row.get("phase") == "study"),
            "reused_historical_exact": sum(bool(row.get("reused_historical")) for row in rows),
            "reused_historical_exact_study": sum(bool(row.get("reused_historical")) for row in rows if row.get("phase") == "study"),
            "cache_accounting_note": "Application cache was disabled. Exact A/B reuse is historical provenance, not a serving-cache hit. Duplicate ARM_INFEASIBLE rows are retained from the interrupted pre-CAL attempt and excluded from accepted analysis.",
        }
    )
    TELEMETRY.write_text(json.dumps(corrected, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: corrected[key] for key in ["raw_rows", "accepted_rows", "new_physical_calls", "reused_historical_exact", "raw_infeasible_rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()
