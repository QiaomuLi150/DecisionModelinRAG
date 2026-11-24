#!/usr/bin/env python3
"""CPU-only integrity checks for the blocked v2.1 replication preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v2_1_replication"
STAGE0 = OUT / "stage0_preflight"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_hash(path: Path) -> str:
    obj = json.loads(path.read_text())
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(name: str):
    return json.loads((STAGE0 / name).read_text())


def load_id_values(name: str, key: str) -> list[str]:
    """Accept the enterprise JSON array and MuSiQue JSONL manifest forms."""
    path = STAGE0 / name
    text = path.read_text()
    if text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [row[key] for row in rows]


def main() -> None:
    protocol_hash = canonical_json_hash(STAGE0 / "PROTOCOL.json")
    receipt_hash = load_json("PROTOCOL_SHA256.json")["sha256"]
    enterprise = load_json("ENTERPRISE_AUDIT.json")
    musique = load_json("MUSIQUE_AUDIT.json")
    gate = json.loads((OUT / "FINAL_GATE.json").read_text())
    provenance = json.loads((OUT / "PROVENANCE_MANIFEST.json").read_text())
    source_hashes_match = all(
        (ROOT / relative_path).is_file() and sha256_file(ROOT / relative_path) == expected_hash
        for relative_path, expected_hash in provenance.get("historical_inputs", {}).items()
    )

    enterprise_ids = load_id_values("ENTERPRISE_QUESTION_IDS.jsonl", "canonical_id")
    musique_ids = load_id_values("MUSIQUE_QUESTION_IDS.jsonl", "canonical_id")

    checks = {
        "protocol_canonical_hash_matches_receipt": protocol_hash == receipt_hash,
        "protocol_pre_outcome_flag": bool(load_json("PROTOCOL.json").get("labels_joined_after_raw_trace_freeze")),
        "enterprise_question_count_500": len(enterprise_ids) == 500,
        "enterprise_question_ids_unique": len(enterprise_ids) == len(set(enterprise_ids)),
        "enterprise_applicable_count_443": enterprise.get("potentially_labeled_questions") == 443,
        "musique_selected_count_2167": len(musique_ids) == 2167,
        "musique_source_ids_unique": len(musique_ids) == len(set(musique_ids)),
        "enterprise_exact_bm25_top20_available_false": enterprise.get("exact_required_bm25_top20_slate_available") is False,
        "musique_exact_bm25_top20_available_false": musique.get("exact_required_bm25_top20_slate_available") is False,
        "final_gate_blocked_under_r2": gate.get("status") == "BLOCKED" and gate.get("decision_rule") == "A3-R2",
        "zero_model_calls": gate.get("model_calls") == 0,
        "zero_gpu_workloads": gate.get("gpu_workloads") == 0,
        "zero_jev_calls": gate.get("jev_calls") == 0,
        "zero_paid_api_calls": gate.get("paid_api_calls") == 0,
        "no_new_raw_traces": provenance.get("new_raw_traces") == [],
        "historical_artifacts_unmodified": provenance.get("historical_artifacts_modified") is False,
        "new_test_ids_hash_recorded": provenance.get("historical_inputs", {}).get(
            "results/preflight/v11/NEW_TEST_IDS.jsonl"
        ) == "95bdeb7ccd76cb1aadd766ffe9384b728e63f48be9e14f97e3a4b3b4525f63fc",
        "all_historical_source_hashes_match": source_hashes_match,
    }

    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "CPU_ONLY_STAGE0_INTEGRITY",
        "protocol_sha256": protocol_hash,
        "checks": checks,
        "model_calls": 0,
        "gpu_workloads": 0,
        "api_calls": 0,
        "notes": [
            "No selector inference, retrieval, serving timing, or statistical model comparison was run.",
            "The blocked gate is preserved because the required exact BM25-only Top-20 inputs are absent for both benchmarks.",
        ],
    }
    (OUT / "CPU_INTEGRITY_TEST_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
