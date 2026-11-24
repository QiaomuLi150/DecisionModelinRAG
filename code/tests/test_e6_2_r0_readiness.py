"""Deterministic checks for the E6-2 R0 readiness package.

These tests inspect manifests/reports only. They never load a checkpoint or call
retrieval/model services.
"""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r0_scientific_readiness"


def test_e6_2_r0_gate_and_prohibitions():
    gate = json.loads((OUT / "E6_2_R0_OWNER_REVIEW_GATE.json").read_text())
    assert gate["gate"] == "E6_2_R0_RETRIEVAL_FREEZE_REQUIRED"
    assert gate["prohibitions_observed"] == {
        "gpu_calls": 0,
        "historical_artifacts_modified": False,
        "jev_calls": 0,
        "model_inference_calls": 0,
        "paid_api_calls": 0,
        "reserved_holdout_access": False,
        "retrieval_construction_calls": 0,
    }


def test_scifact_and_garage_audits_are_frozen():
    registry = json.loads((OUT / "E6_2_R0_DATASET_ENDPOINT_REGISTRY.json").read_text())
    assert registry["scifact_audit"]["rows"] == {
        "corpus": 5183,
        "positive_qrels": 1258,
        "queries": 1109,
        "test_qrels": 339,
        "total_qrels": 1258,
        "train_qrels": 919,
    }
    assert registry["scifact_audit"]["join_audit"] == {
        "qrel_docs_without_corpus": 0,
        "qrel_queries_without_query": 0,
        "queries_without_qrels": 0,
    }
    assert registry["scifact_audit"]["candidate_slate_status"] == "NOT_FOUND_AUTHENTIC_SCIFACT_TOP20"
    assert registry["garage_audit"]["rows"] == 2366
    assert registry["garage_audit"]["exact_k15_pools"] == 2329
    assert registry["garage_audit"]["unique_sample_ids"] == 2366


def test_eight_model_admission_matrix_has_unique_models():
    with (OUT / "E6_2_R0_EIGHT_MODEL_ADMISSION_MATRIX.csv").open() as f:
        rows = list(csv.DictReader(f))
    names = [row["model"] for row in rows]
    assert len(rows) == 8
    assert len(set(names)) == 8
    assert "amos_1_small_2b" in names
    amos = next(row for row in rows if row["model"] == "amos_1_small_2b")
    assert amos["failures"] == "6"
    assert amos["physical_forwards"] == "NOT_MEASURED"


def test_historical_e5_hash_and_manifest_checks():
    path = ROOT / "results/non_jev/v2/e5_v1_1/e5_0_data_protocol/E5_0_NATURAL_TOP20_MANIFEST.jsonl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == "be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7"
    report = json.loads((OUT / "E6_2_R0_TEST_AND_PROVENANCE_REPORT.md.json").read_text())
    assert all(item["status"] == "PASS" for item in report["historical_manifest_verification"])
    assert all(item["checked"] == item["passed"] for item in report["historical_manifest_verification"])


def test_protocol_keeps_tracks_separate_and_reservation_locked():
    registry = json.loads((OUT / "E6_2_R0_DATASET_ENDPOINT_REGISTRY.json").read_text())
    assert registry["tracks"]["A_QASPER"]["status"] == "HISTORICAL_ONLY_NO_REOPEN"
    assert registry["tracks"]["B_SCIFACT"]["primary_endpoint"] == "binary nDCG@5"
    assert registry["tracks"]["C_GARAGE"]["status"] == "AUTHENTIC_FIXED_POOL_CONDITIONAL"
    assert registry["reserved"]["metadata_only"] is True
    assert registry["reserved"]["raw_question_text_exported"] is False
    assert registry["reserved"]["gold_qrels_exported"] is False
