#!/usr/bin/env python3
"""CPU-only integrity checks for the V2 hybrid replication closure.

This script never loads benchmark labels and never changes any raw trace.
It reports incomplete traces instead of treating them as valid evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "v2_1_replication"
MANIFEST = BASE / "stage0_preflight_v2_hybrid" / "INPUT_MANIFEST_V2_HYBRID.jsonl"
PROTOCOL = BASE / "PROTOCOL_V2.json"
PROTOCOL_RECEIPT = BASE / "PROTOCOL_V2_SHA256.json"
EXPECTED = 2667


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                yield line_no, json.loads(line)


def rows(path: Path):
    return sum(1 for _ in read_jsonl(path))


def main() -> None:
    checks = {}
    failures = []

    protocol_sha = sha_file(PROTOCOL)
    receipt = json.loads(PROTOCOL_RECEIPT.read_text(encoding="utf-8"))
    checks["protocol_sha256"] = {"computed": protocol_sha, "receipt": receipt.get("sha256"), "pass": protocol_sha == receipt.get("sha256")}
    if not checks["protocol_sha256"]["pass"]:
        failures.append("PROTOCOL_SHA256_MISMATCH")

    manifest_sha = sha_file(MANIFEST)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    expected_manifest_sha = protocol.get("input_manifest_sha256")
    checks["input_manifest_sha256"] = {"computed": manifest_sha, "protocol": expected_manifest_sha, "pass": manifest_sha == expected_manifest_sha}
    if not checks["input_manifest_sha256"]["pass"]:
        failures.append("INPUT_MANIFEST_SHA256_MISMATCH")

    manifest_rows = {}
    for _, record in read_jsonl(MANIFEST):
        key = (record["dataset"], record["canonical_id"], record["order_id"])
        manifest_rows[key] = record
    checks["manifest_rows"] = {"count": len(manifest_rows), "expected": 10668, "pass": len(manifest_rows) == 10668}
    if not checks["manifest_rows"]["pass"]:
        failures.append("MANIFEST_ROW_COUNT_OR_KEY_FAILURE")

    trace_results = {}
    trace_dirs = [BASE / "stage1_native_v2_hybrid", BASE / "stage2_orders_v2_hybrid"]
    for trace_dir in trace_dirs:
        if not trace_dir.exists():
            continue
        for path in sorted(trace_dir.glob("*.jsonl")):
            if path.name.endswith("__full.jsonl"):
                total = rows(path)
                valid = 0
                invalid_ids = 0
                label_leaks = 0
                for _, record in read_jsonl(path):
                    if record.get("labels_joined") is not False:
                        label_leaks += 1
                    if record.get("status") == "PASS":
                        valid += 1
                        ids = record.get("candidate_ids", [])
                        selected = record.get("selected_ids", [])
                        if len(selected) != 5 or len(set(selected)) != 5 or not set(selected).issubset(set(ids)):
                            invalid_ids += 1
                key = path.stem.replace("__full", "")
                trace_results[key] = {
                    "path": str(path.relative_to(ROOT)),
                    "records": total,
                    "valid": valid,
                    "sha256": sha_file(path),
                    "complete": total == EXPECTED,
                    "invalid_pass_ids": invalid_ids,
                    "label_leak_records": label_leaks,
                }
                if label_leaks:
                    failures.append(f"LABEL_LEAK:{path.name}")
                if invalid_ids:
                    failures.append(f"INVALID_TOP5:{path.name}")
    checks["full_traces"] = trace_results

    partials = []
    for trace_dir in trace_dirs:
        if trace_dir.exists():
            for path in sorted(trace_dir.glob("*.jsonl.part")):
                partials.append({"path": str(path.relative_to(ROOT)), "records": rows(path), "sha256": sha_file(path)})
    checks["partial_traces_preserved"] = {"count": len(partials), "traces": partials, "pass": True}

    analysis_state = BASE / "stage4_analysis_v2_hybrid" / "STATE.json"
    if analysis_state.exists():
        analysis = json.loads(analysis_state.read_text(encoding="utf-8"))
        checks["analysis_state"] = {"path": str(analysis_state.relative_to(ROOT)), "trace_files": analysis.get("trace_files"), "perquery_rows": analysis.get("perquery_rows"), "pass": True}
    else:
        checks["analysis_state"] = {"pass": False, "reason": "NOT_YET_GENERATED"}

    result = {
        "test": "v2_1_replication_final_integrity",
        "mode": "CPU_ONLY",
        "labels_loaded": False,
        "checks": checks,
        "failures": failures,
        "status": "PASS" if not failures else "PARTIAL_OR_BLOCKED",
    }
    out = BASE / "FINAL_INTEGRITY_TEST_REPORT_V2.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
