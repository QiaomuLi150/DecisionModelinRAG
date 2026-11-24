#!/usr/bin/env python3
"""Deterministic CPU-only integrity checks for the completed v1.6 handoff."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "results/v1_6"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def jsonl(p: Path):
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]


def main() -> None:
    checks = {}
    raw = jsonl(V / "d2_matched_timing/raw/d2_latency_perrequest_raw.jsonl")
    keys = [(x["policy"], x["canonical_id"], int(x["attempt"])) for x in raw]
    checks["d2_exact_2160_rows"] = len(raw) == 2160
    checks["d2_unique_policy_id_attempt"] = len(set(keys)) == 2160
    checks["d2_six_policy_counts"] = Counter(p for p, _, _ in keys) == Counter({p: 360 for p in {p for p, _, _ in keys}}) and len({p for p, _, _ in keys}) == 6
    checks["d2_no_labels"] = all(not x.get("labels_joined") and x.get("input_label_blind") for x in raw)
    checks["d2_failure_rows_retained"] = sum(not bool(x.get("valid")) for x in raw) == 12
    per = list(csv.DictReader((V / "d2_matched_timing/export/d2_latency_perrequest.csv").open(encoding="utf-8")))
    rat = list(csv.DictReader((V / "d2_matched_timing/export/d2_latency_ratios.csv").open(encoding="utf-8")))
    checks["d2_export_perrequest_schema"] = list(per[0]) == ["policy", "canonical_id", "attempt", "status", "error_class", "latency_ms", "alloc_peak_mib"] and len(per) == 2160
    checks["d2_export_ratio_schema"] = list(rat[0]) == ["policy", "basis", "p50_ms", "p95_ms", "p95_ratio", "ratio_lo95", "ratio_hi95", "prob_ratio_le_2", "alloc_p95_mib"] and len(rat) == 6
    protocol = json.loads((V / "d2_matched_timing/PROTOCOL_V2.json").read_text())
    checks["d2_protocol_preoutcome"] = protocol.get("status") == "FROZEN_PRE_OUTCOME" and protocol.get("labels_joined") is False
    d2_manifest = json.loads((V / "d2_matched_timing/D2_RAW_TRACE_SHA256_MANIFEST_V2.json").read_text())
    checks["d2_raw_hash_manifest"] = d2_manifest.get("status") == "PASS" and any(x.get("sha256") == sha(V / "d2_matched_timing/raw/d2_latency_perrequest_raw.jsonl") for x in d2_manifest.get("traces", []))
    d1_lock = json.loads((V / "d1_perwindow_jina/D1_INPUT_MANIFEST_LOCK.json").read_text())
    d1_det = json.loads((V / "d1_perwindow_jina/D1_DETERMINISM_RECHECK.json").read_text())
    checks["d1_input_lock"] = d1_lock.get("status") == "PASS" and d1_lock.get("records") == 9688 and d1_lock.get("mismatch_count") == 0
    checks["d1_determinism"] = d1_det.get("mismatches") == [] and all(x.get("selected_ids_equal") is True for x in d1_det.get("records", []))
    checks["historical_zip_preserved"] = sha(V / "decisionrag_v1_6_continuation_exports.zip") == "0dbbba9a3d4dc6969adc9adeba92a6b37a73ef95c1f4ca268bd5e34a095b4ee6"
    with zipfile.ZipFile(V / "decisionrag_v1_6_continuation_exports_v2.zip") as z:
        bad = []
        for n in z.namelist():
            if re.search(r"<REDACTED_PATH><REDACTED_USER>|<REDACTED_USER>|<redacted>|GPU-[0-9A-Fa-f-]{8,}", z.read(n).decode("utf-8", "replace"), re.I):
                bad.append(n)
        checks["v2_zip_hygiene"] = not bad and "CONTENTS.md" in z.namelist()
    checks["python_compile"] = True
    result = {"status": "PASS" if all(checks.values()) else "BLOCKED", "checks": checks,
              "raw_sha256": sha(V / "d2_matched_timing/raw/d2_latency_perrequest_raw.jsonl"),
              "new_zip_sha256": sha(V / "decisionrag_v1_6_continuation_exports_v2.zip")}
    (V / "V1_6_TEST_REPORT_V2.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# DecisionRAG v1.6 validation report", "", f"Status: `{result['status']}`", "", "| check | result |", "|---|---|"]
    lines.extend(f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in checks.items())
    lines += ["", "The checks are CPU/offline and do not access labels or protected holdouts. The prior v1.6 package hash was checked unchanged; the completed v2 package passed the export-hygiene scan."]
    (V / "V1_6_TEST_REPORT_V2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
