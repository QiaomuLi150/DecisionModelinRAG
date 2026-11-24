"""CPU-only E5-0C readiness and no-look integrity tests."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e5_v1_1/e5_0c_readiness_closure"
E5 = ROOT / "results/non_jev/v2/e5_v1_1/e5_0_data_protocol"


def read_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_e5_0c_gate_and_scope_are_cpu_only():
    gate = (OUT / "E5_0C_E5_A_OWNER_DECISION_GATE.md").read_text(encoding="utf-8")
    manifest = read_json("E5_0C_PROVENANCE_MANIFEST.json")
    assert "E5_0C_PROTOCOL_AMENDMENT_REQUIRED" in gate
    assert "Do not execute E5-A" in gate
    assert manifest["e5_0c_cpu_only"] is True
    assert manifest["no_gpu"] is True
    assert manifest["no_model_calls"] is True
    assert manifest["no_reserved_outcomes"] is True
    assert manifest["e5_f_confirmation"] == "LOCKED"


def test_prior_e5_hashes_and_frozen_slate_are_exact():
    integrity = read_json("E5_0C_INTEGRITY_AND_PROVENANCE.json")
    audit = integrity["e5_0_hash_audit"]
    assert audit["e5_0_missing"] == []
    assert audit["e5_0_mismatches"] == []
    assert audit["e5_0_historical_audit"]["clean"] is True
    assert audit["slate_hash_exact_match"] is True
    rows = [json.loads(x) for x in (E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl").read_text().splitlines() if x.strip()]
    assert len(rows) == 1502
    assert all(len(r["candidate_ids"]) == 20 and len(set(r["candidate_ids"])) == 20 for r in rows)


def test_support_mapping_reproduces_frozen_first_wrapper():
    summary = read_json("E5_0C_SUPPORT_SEMANTICS_SUMMARY.json")
    assert summary["mapping_hash_mismatches"] == []
    assert summary["raw_questions_missing"]
    classes = summary["cal_class_counts"]
    assert sum(classes.values()) == 519
    assert classes["SINGLE_DESIGNATED_SET"] > 0
    assert classes["MULTIPLE_ALTERNATIVE_SETS"] > 0
    assert classes["PARTIAL_OR_NONEXHAUSTIVE"] > 0


def test_retrieval_scope_is_global_and_gold_blind():
    scope = read_json("E5_0C_RETRIEVAL_SCOPE_SUMMARY.json")
    assert scope["global_search"] is True
    assert scope["query_text_only"] is True
    assert scope["qrels_accessed_during_generation"] is False
    assert scope["gold_in_request"] is False
    assert scope["slates"] == 1502
    assert scope["same_paper_candidate_fraction"] < 0.5


def test_cal_ceiling_is_not_fabricated_and_exploratory_ceiling_is_structural():
    f = read_json("E5_0C_FEASIBILITY_SUMMARY.json")
    assert f["cal_exact_ceiling"] == "CAL_EXACT_CEILING_NOT_MEASURED"
    assert f["exploratory_population"] == 1502
    assert 0 <= f["exploratory_complete_support_top20"] <= 1502
    assert f["exploratory_complete_support_top20"] == 343


def test_lineage_and_gpu_governance_are_not_overclaimed():
    lineage = read_json("E5_0C_SOURCE_LINEAGE_SUMMARY.json")
    text = (OUT / "E5_0C_GPU_AUTHORIZATION_AND_SMOKE_RECONCILIATION.md").read_text(encoding="utf-8")
    assert lineage["reserved_near_duplicate_disjointness"] == "RESERVED_NEAR_DUPLICATE_DISJOINTNESS_NOT_INDEPENDENTLY_VERIFIED"
    assert "GPU_5_6_AUTHORIZATION_UNVERIFIED" in text
    assert "E5-0C itself used no GPU" in text


def test_manifest_covers_all_e5_0c_outputs():
    manifest = read_json("E5_0C_PROVENANCE_MANIFEST.json")
    bad = []
    for item in manifest["files"]:
        p = ROOT / item["path"]
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
            bad.append(item["path"])
    assert not bad
