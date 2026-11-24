"""Regression checks for the append-only E6-2 R4C offline release."""
import json
from pathlib import Path

from scripts.e6_2_r4c_erratum_garage_freeze import DSU, obj_sha


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r4c_erratum_and_garage_freeze"


def read_json(name):
    return json.loads((OUT / name).read_text())


def test_scifact_erratum_and_primary_reproduction_are_frozen():
    doc = read_json("E6_2_R4C_SCIFACT_PRIMARY_REPRODUCTION.json")
    assert doc["corrected_top20_feasibility"]["n"] == 254
    assert doc["corrected_top20_feasibility"]["denominator"] == 277
    assert doc["primary_pplx_minus_jina_ndcg5"]["matches_r3_primary"] is True


def test_garage_structural_population_and_partition_counts():
    manifest = read_json("E6_2_R4C_GARAGE_PARTITION_MANIFEST.json")
    assert manifest["full_release_rows"] == 2366
    assert manifest["eligible_count"] == 2329
    assert manifest["actual_counts"] == {"CAL": 466, "EXPLORATORY": 1397, "RESERVED": 466}
    assert manifest["labels_used_for_allocation"] is False
    assert manifest["outcome_used_for_allocation"] is False


def test_partition_ids_are_unique_and_disjoint():
    ids = read_json("E6_2_R4C_GARAGE_PARTITION_MANIFEST.json")["partition_ids"]
    all_ids = [x for values in ids.values() for x in values]
    assert len(all_ids) == len(set(all_ids)) == 2329
    assert not (set(ids["CAL"]) & set(ids["EXPLORATORY"]))
    assert not (set(ids["CAL"]) & set(ids["RESERVED"]))
    assert not (set(ids["EXPLORATORY"]) & set(ids["RESERVED"]))


def test_label_blind_exploratory_projection():
    forbidden = {"evidence_relevant", "evidence_correct", "evidence_cited", "answer_generate", "answer_validate"}
    path = OUT / "E6_2_R4C_GARAGE_EXPLORATORY_INPUTS.jsonl"
    rows = [json.loads(line) for line in path.open() if line.strip()]
    assert len(rows) == 1397
    assert all(not forbidden.intersection(row) for row in rows)
    assert all(not forbidden.intersection(candidate) for row in rows for candidate in row["candidates"])


def test_release_gate_blocks_inference_and_reserves_source_claims():
    gate = read_json("E6_2_R4C_R4_INFERENCE_RELEASE_GATE.json")
    assert gate["native_inference_calls"] == 0
    assert gate["quality_metrics_computed"] is False
    assert gate["gpu_used"] is False
    assert gate["garage"]["source_identity_established"] is False
    assert gate["garage"]["reservation_sealed"] is False


def test_transitive_component_union():
    graph = DSU(["a", "b", "c"])
    graph.union("a", "b")
    graph.union("b", "c")
    assert graph.groups() == [["a", "b", "c"]]


def test_partition_manifest_hash_is_reproducible():
    manifest = read_json("E6_2_R4C_GARAGE_PARTITION_MANIFEST.json")
    assert manifest["partition_id_sha256"]["CAL"] == obj_sha(manifest["partition_ids"]["CAL"])
    assert manifest["partition_id_sha256"]["EXPLORATORY"] == obj_sha(manifest["partition_ids"]["EXPLORATORY"])
    assert manifest["partition_id_sha256"]["RESERVED"] == obj_sha(manifest["partition_ids"]["RESERVED"])


def test_provenance_declares_zero_inference():
    provenance = read_json("E6_2_R4C_PROVENANCE_MANIFEST.json")
    assert provenance["native_inference_calls"] == 0
    assert provenance["physical_forwards"] == 0
    assert provenance["quality_metrics_computed"] is False
    assert provenance["gpu_used"] is False
    assert provenance["jev_calls"] == 0
    assert provenance["paid_api_calls"] == 0
