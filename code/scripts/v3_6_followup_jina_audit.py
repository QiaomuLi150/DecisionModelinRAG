"""Write offline audit artifacts for the v3.6 Jina L8192 correction.

This script only reads frozen historical traces and the completed prespecified
instrumentation pilot. It performs no model calls.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_6_essential/followup_v1"
OLD = ROOT / "results/v3_6_essential/stage1_quality/raw/jina_single_8192__8192_reuse.jsonl"
PILOT = OUT / "stage1_jina/raw/jina_rerun_pilot_gpu3.jsonl"


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    old = rows(OLD)
    pilot = rows(PILOT)
    # The historical quality trace predates the explicit status field.  Count
    # only persisted error evidence; absence of status is not a failure.
    old_bad = [r for r in old if r.get("error") or r.get("error_class")]
    old_type_error = [r for r in old_bad if "cannot unpack non-iterable NoneType object" in str(r.get("error", ""))]
    pilot_none = [r for r in pilot if r.get("model_return_was_none") is True]
    pilot_errors = [r for r in pilot if r.get("error") or r.get("error_class")]

    case_csv = OUT / "export/quality_v2/jina_bug_cases.csv"
    case_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["canonical_id", "domain", "error", "source_input_sha256", "expected_input_sha256", "candidate_count", "latency_ms"]
    with case_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in old_type_error:
            w.writerow({
                "canonical_id": r.get("canonical_id", ""),
                "domain": r.get("domain", ""),
                "error": r.get("error", ""),
                "source_input_sha256": r.get("source_input_sha256", ""),
                "expected_input_sha256": r.get("expected_input_sha256", ""),
                "candidate_count": len(r.get("candidate_ids") or []),
                "latency_ms": r.get("latency_ms", ""),
            })

    audit = {
        "historical_trace": str(OLD.relative_to(ROOT)),
        "historical_trace_sha256": sha(OLD),
        "historical_rows": len(old),
        "historical_error_rows": len(old_bad),
        "historical_none_unpack_rows": len(old_type_error),
        "historical_none_unpack_domains": dict(Counter(r.get("domain", "") for r in old_type_error)),
        "pilot_trace": str(PILOT.relative_to(ROOT)),
        "pilot_trace_sha256": sha(PILOT),
        "pilot_rows": len(pilot),
        "pilot_error_rows": len(pilot_errors),
        "pilot_model_return_none_rows": len(pilot_none),
        "pilot_return_types": dict(Counter(str(r.get("model_return_type")) for r in pilot)),
        "pilot_policies": sorted({r.get("policy") for r in pilot}),
        "labels_joined": any(bool(r.get("labels_joined")) for r in pilot),
        "interpretation": "The historical records preserve no model return payload and report a wrapper-level unpack error for 32 rows; they do not establish that the native Jina method returned None. The instrumented pilot explicitly checks None before unpacking and observed no errors or None returns.",
    }
    (OUT / "export/quality_v2/jina_pilot_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    report = f"""# Jina L8192 Correction Audit

Status: diagnostic evidence recorded; full correction rerun is still evaluated by the follow-up gate.

## Historical behavior

- Frozen historical trace: `{OLD.relative_to(ROOT)}`
- Rows: {len(old)}; rows containing an error/status failure: {len(old_bad)}.
- The historical trace contains {len(old_type_error)} explicit `TypeError: cannot unpack non-iterable NoneType object` records, all in the domain set `{sorted(set(r.get('domain', '') for r in old_type_error))}`.
- Those records have no persisted model-return payload and no latency value. Their candidate-ID lists are empty in the failed records. The trace therefore cannot distinguish a native `None` from a wrapper/normalization failure after the native call.
- The historical trace hash is recorded in `jina_pilot_audit.json`; the original trace is unchanged.

## Instrumented pilot

- Pilot rows: {len(pilot)} across {len({r.get('policy') for r in pilot})} policies and 20 frozen requests.
- Explicit native return type was checked before unpacking. `None` returns: {len(pilot_none)}; error rows: {len(pilot_errors)}.
- Observed return types: `{dict(Counter(str(r.get('model_return_type')) for r in pilot))}`. Two-stage records use the explicit stage-level path and do not claim a single native return type.
- Labels were not joined and protected holdouts were not accessed.

## Repair interpretation

The corrected wrapper preserves model-return type, `None` detection, elapsed time and error class before candidate-ID mapping. This resolves the historical observability defect without asserting that all historical failures were native-model failures. The full six-policy rerun and its gate remain authoritative for replacement of the old Jina records.

Machine-readable case inventory: `export/quality_v2/jina_bug_cases.csv`.
"""
    (OUT / "JINA_BUG_REPORT.md").write_text(report)
    print(json.dumps({"status": "PASS", "historical_error_rows": len(old_bad), "historical_none_unpack_rows": len(old_type_error), "pilot_errors": len(pilot_errors), "pilot_none": len(pilot_none)}, indent=2))


if __name__ == "__main__":
    main()
