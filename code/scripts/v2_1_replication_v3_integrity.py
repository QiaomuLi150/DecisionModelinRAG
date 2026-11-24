#!/usr/bin/env python3
"""Focused integrity gates for the v3 derived exports."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v2_1_replication"
V3 = BASE / "stage4_analysis_v3"
EXPORT = V3 / "export"
MANIFEST = BASE / "stage0_preflight_v2_hybrid/INPUT_MANIFEST_V2_HYBRID.jsonl"
RAW_STAGE3 = BASE / "stage3_serving_v2_hybrid/raw/stage3_perrequest.jsonl"


def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = {}
    manifest = [json.loads(x) for x in MANIFEST.open(encoding="utf-8") if x.strip()]
    checks["manifest_rows"] = {"pass": len(manifest) == 10668, "count": len(manifest)}
    expected = {"enterprise": 12500, "musique": 54175}
    for ds, n in expected.items():
        p = EXPORT / ds / "perquery.csv"; rs = rows(p)
        checks[f"perquery_{ds}"] = {"pass": len(rs) == n and all(not (r["policy"] == "pplx_native" and r["order_id"] != "original") for r in rs), "rows": len(rs), "expected": n}
        s = rows(EXPORT / ds / "summary.csv")
        not_run = [r for r in s if r["policy"] == "pplx_native" and r["order_id"] != "original"]
        checks[f"summary_{ds}"] = {"pass": len(s) == 28 and all(r["status"] == "NOT_RUN" and r["n_valid"] == "" for r in not_run), "rows": len(s), "not_run_cells": len(not_run)}
        checks[f"serving_{ds}"] = {"pass": len(rows(EXPORT / ds / "serving_perrequest.csv")) == 1800 and len(rows(EXPORT / ds / "serving_ratios.csv")) == 10, "perrequest_rows": len(rows(EXPORT / ds / "serving_perrequest.csv")), "ratio_rows": len(rows(EXPORT / ds / "serving_ratios.csv"))}
        checks[f"survivor_{ds}"] = {"pass": all(r["survivor_sets_equal"] in {"True", "False"} and "NOT_EMITTED" not in str(r) for r in rows(EXPORT / ds / "survivors_audit.csv")), "rows": len(rows(EXPORT / ds / "survivors_audit.csv"))}
        checks[f"determinism_{ds}"] = {"pass": all(int(r["n_mismatch"]) == 0 for r in rows(EXPORT / ds / "determinism.csv")), "rows": len(rows(EXPORT / ds / "determinism.csv"))}
        checks[f"length_{ds}"] = {"pass": len(rows(EXPORT / ds / "length_scan.csv")) == 3, "rows": len(rows(EXPORT / ds / "length_scan.csv"))}
        checks[f"budget_{ds}"] = {"pass": len(rows(EXPORT / ds / "budget_sensitivity.csv")) == 14, "rows": len(rows(EXPORT / ds / "budget_sensitivity.csv"))}
        checks[f"nonempty_csv_{ds}"] = {"pass": all(p.stat().st_size > 0 and sum(1 for _ in p.open(encoding="utf-8")) > 1 for p in (EXPORT / ds).glob("*.csv"))}
    stage3 = [json.loads(x) for x in RAW_STAGE3.open(encoding="utf-8") if x.strip()]
    checks["stage3_raw"] = {"pass": len(stage3) == 3600 and all(r.get("labels_joined") is False for r in stage3), "rows": len(stage3), "label_leak": any(r.get("labels_joined") is not False for r in stage3)}
    checks["cohort_ids"] = {"pass": len(rows(EXPORT / "serving_cohort_ids.csv")) == 240}
    checks["reconciliation"] = {"pass": all(r["ok"] == "True" for r in rows(V3 / "RECONCILIATION.csv")), "rows": len(rows(V3 / "RECONCILIATION.csv"))}
    checks["order_application"] = {"pass": all(int(r["n_input_order_differs_from_original"]) == int(r["n_requests"]) for r in rows(V3 / "order_audit.csv") if r["order_id"] != "original" and r["status"] == "RUN"), "rows": len(rows(V3 / "order_audit.csv"))}
    checks["no_protocol_change"] = {"pass": not (V3 / "PROTOCOL_V3.json").exists()}
    package = BASE / "decisionrag_v2_1_replication_exports_v3.zip"
    pattern = re.compile(r"GPU-[0-9a-fA-F-]{20,}|/m\x6et/|/h\x6fme/|/r\x6fot/", re.I)
    with zipfile.ZipFile(package) as z:
        bad = [n for n in z.namelist() if pattern.search(z.read(n).decode("utf-8", "ignore"))]
        checks["package_hygiene"] = {"pass": not bad and len(z.namelist()) == len(set(z.namelist())), "members": len(z.namelist()), "matches": bad}
    failures = [k for k, v in checks.items() if not v.get("pass")]
    result = {"status": "PASS" if not failures else "FAIL", "test": "v2_1_replication_v3_integrity", "failures": failures, "checks": checks, "package_sha256": sha(package)}
    (V3 / "V3_INTEGRITY_TEST_REPORT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
