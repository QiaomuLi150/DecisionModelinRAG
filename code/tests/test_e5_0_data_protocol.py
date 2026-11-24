"""Focused E5-0 integrity tests.

These tests validate data/protocol invariants only.  They never load or call a
selector, reader, Jev, or paid endpoint.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e5_v1_1/e5_0_data_protocol"


def stable(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_required_artifacts_and_gate_exist():
    required = [
        "E5_0_FINAL_DATA_AND_PROTOCOL_REPORT.md",
        "E5_0_OWNER_REVIEW_GATE.md",
        "E5_0_INPUT_AND_HISTORICAL_EXCLUSION_INVENTORY.json",
        "E5_0_DATASET_FEASIBILITY_MATRIX.csv",
        "E5_0_GOLD_MAPPING_AND_ELIGIBILITY_AUDIT.csv",
        "E5_0_SOURCE_GROUP_COMPONENTS.csv",
        "E5_0_SPLIT_CONFIG_AND_HASHES.json",
        "E5_0_PROSPECTIVE_RESERVATION_RECEIPT.json",
        "E5_0_NATURAL_TOP20_MANIFEST.jsonl",
        "E5_0_FROZEN_MODEL_POLICY_CONTRACTS.json",
        "E5_0_PROVENANCE_MANIFEST.json",
    ]
    assert all((OUT / name).is_file() for name in required)
    assert "E5_0_CONDITIONAL_READY" in (OUT / "E5_0_OWNER_REVIEW_GATE.md").read_text()


def test_source_groups_are_partitioned_by_metadata_only():
    rows = list(csv.DictReader((OUT / "E5_0_SOURCE_GROUP_COMPONENTS.csv").open(encoding="utf-8")))
    assert len(rows) > 30
    assert {r["partition"] for r in rows} == {"CAL", "E5_EXPLORATORY", "E5_F_RESERVED"}
    assert len({r["source_group_pseudonym"] for r in rows}) == len(rows)
    receipt = json.loads((OUT / "E5_0_PROSPECTIVE_RESERVATION_RECEIPT.json").read_text())
    assert receipt["metadata_only"] is True
    assert receipt["E5_F_CONFIRMATION"] == "LOCKED"
    assert receipt["external_reservation_attestation"] == "UNAVAILABLE"
    assert receipt["reserved_access_isolation"] == "PARTIAL"


def test_natural_slates_are_exactly_top20_and_hash_stable():
    slates = read_jsonl(OUT / "E5_0_NATURAL_TOP20_MANIFEST.jsonl")
    assert len(slates) == 1502
    assert len({row["query_id"] for row in slates}) == len(slates)
    permitted = {row["passage_id"] for row in read_jsonl(OUT / "E5_0_PERMITTED_CORPUS.jsonl")}
    for row in slates:
        assert len(row["candidate_ids"]) == 20
        assert len(set(row["candidate_ids"])) == 20
        assert set(row["candidate_ids"]) <= permitted
        unsigned = dict(row)
        observed = unsigned.pop("slate_sha256")
        assert sha_text(stable(unsigned)) == observed
        assert row["qrels_accessed_during_generation"] is False
        assert row["gold_in_request"] is False
        assert row["retriever_only"] is True


def test_no_reserved_question_content_is_in_permitted_manifests():
    receipt = json.loads((OUT / "E5_0_PROSPECTIVE_RESERVATION_RECEIPT.json").read_text())
    permitted_queries = read_jsonl(OUT / "E5_0_PERMITTED_QUERIES.jsonl")
    slates = read_jsonl(OUT / "E5_0_NATURAL_TOP20_MANIFEST.jsonl")
    assert all(q["partition"] != "E5_F_RESERVED" for q in permitted_queries)
    exploratory_qids = {q["query_id"] for q in permitted_queries if q["partition"] == "E5_EXPLORATORY"}
    assert {q["query_id"] for q in slates} <= exploratory_qids
    forbidden = {"answer", "answers", "reference", "qrels", "gold", "support_ids"}
    for row in slates:
        assert not any(any(part in key.casefold() for part in forbidden) and row[key] not in (False, None, "") for key in row)
    assert receipt["reserved_question_count"] > 0


def test_cssr_u_and_cssr_c_toy_contract():
    gold = {"g1", "g2"}
    assert int(gold <= {"g1", "g2", "n1", "n2", "n3"}) == 1
    assert int(gold <= {"g1", "n1", "n2", "n3", "n4"}) == 0
    # Retrieval miss is an unconditional failure, not dropped from CSSR-U.
    assert int(gold <= set()) == 0


def test_provenance_declares_forbidden_work_absent():
    manifest = json.loads((OUT / "E5_0_PROVENANCE_MANIFEST.json").read_text())
    assert manifest["no_jev"] is True
    assert manifest["no_paid_api"] is True
    assert manifest["no_benchmark_selector_inference"] is True
    assert manifest["synthetic_selector_smoke"] is True
    assert manifest["no_reader_inference"] is True
    assert manifest["no_reserved_outcomes"] is True
    assert manifest["qdrant_modified"] is False


def test_synthetic_smoke_latest_rows_are_physical_and_quality_free():
    telemetry = OUT / "E5_0_PREFLIGHT_TELEMETRY.jsonl"
    assert telemetry.exists()
    latest = {}
    for line in telemetry.read_text().splitlines():
        row = json.loads(line)
        latest[row["policy"]] = row
    assert set(latest) >= {"jina35", "pplx_decider"}
    for row in (latest["jina35"], latest["pplx_decider"]):
        assert row["status"] == "PASS"
        assert row["synthetic_only"] is True
        assert row["quality_scored"] is False
        assert row["score_count"] == 20
        assert len(row["selected_ids"]) == 5
        assert row["telemetry"]["physical_forward_calls"] >= 1
