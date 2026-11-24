#!/usr/bin/env python3
"""Read-only integrity check for the completed E2-M1 result package."""
import hashlib
import json
import math
from pathlib import Path


def audit_rows(rows):
    return {
        "rows": len(rows),
        "unique_query_ids": len({r["query_id"] for r in rows}),
        "authentic_ok": sum(r.get("status") == "ok" and r.get("real_inference") for r in rows),
        "k20": sum(len(r.get("candidate_ids", [])) == 20 and len(r.get("scores", [])) == 20 for r in rows),
        "aligned_unique": sum(
            len(r.get("candidate_ids", [])) == len(r.get("scores", []))
            and len(set(r.get("candidate_ids", []))) == 20
            for r in rows
        ),
        "gold_excluded": sum(r.get("gold_in_inference_input") is False for r in rows),
        "one_physical_forward": sum(r.get("physical_forward_calls") == 1 for r in rows),
        "finite_probability_scores": sum(
            all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in r.get("scores", []))
            for r in rows
        ),
    }


def main():
    out = Path(__file__).resolve().parents[1] / "results/non_jev/v2/e2_m1"
    provenance = json.loads((out / "E2_M1_PROVENANCE_MANIFEST.json").read_text())
    mismatches = []
    for name, expected in provenance["outputs"].items():
        path = out / name
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if not path.is_file() or path.stat().st_size != expected["bytes"] or actual_hash != expected["sha256"]:
            mismatches.append(name)
    point = [json.loads(line) for line in (out / "E2_M1_POINTWISE_SCORES.jsonl").read_text().splitlines() if line]
    joint = [json.loads(line) for line in (out / "E2_M1_JOINT_BINARY_SCORES.jsonl").read_text().splitlines() if line]
    result = {
        "provenance_output_files": len(provenance["outputs"]),
        "provenance_hashes_all_match": not mismatches,
        "provenance_hash_mismatches": mismatches,
        "pointwise": audit_rows(point),
        "joint": audit_rows(joint),
        "jev_calls": provenance["scope"]["jev_calls"],
        "paid_api_calls": provenance["scope"]["paid_api_calls"],
        "status": "PASS" if not mismatches and len(point) == len(joint) == 4267 else "BLOCKED",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
