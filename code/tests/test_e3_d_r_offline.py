import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e3_v1/e3_d_r_reference_audit"
E3D = ROOT / "results/non_jev/v2/e3_v1/e3_d_external"
GOV = ROOT / "results/non_jev/v2/e3_v1/e3_0_governance"


def _jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0_exact_reader_and_selector_cardinality():
    readers = _jsonl(E3D / "E3_D_FIXED_READER_OUTPUTS.jsonl")
    selected = _jsonl(E3D / "E3_D_SELECTED_TOP5.jsonl")
    assert len(readers) == 2154
    assert len(selected) == 6054
    assert len({(r["policy"], r["query_id"]) for r in readers}) == 2154
    assert all(r["real_inference"] and not r["cache_reuse"] and not r["gold_in_reader_input"] for r in readers)


def test_finance_legacy_reproduction_scorecard():
    rows = list(csv.DictReader((OUT / "E3_D_R_FINANCE_LEGACY_VS_CANONICAL_SCORECARD.csv").open()))
    expected = {"retrieval5": 8, "bge_pointwise": 10, "jina35": 12, "pplx_decider": 10, "liquid_joint20_binary": 12, "zerank2_pointwise": 12}
    assert {r["policy"]: int(r["legacy_success_n"]) for r in rows} == expected
    assert all(int(r["n"]) == 109 for r in rows)


def test_multihop_matched_denominator_and_four_cells():
    rows = list(csv.DictReader((OUT / "E3_D_R_MULTIHOP_PAIRED_ANSWER_TRANSITIONS.csv").open()))
    assert len(rows) == 250
    cells = list(csv.DictReader((OUT / "E3_D_R_MULTIHOP_EVIDENCE_ANSWER_4X4.csv").open()))
    assert sum(int(r["n"]) for r in cells) == 250
    assert {r["evidence_cell"] for r in cells} == {"BOTH_COMPLETE", "PPLX_ONLY", "JINA_ONLY", "NEITHER_COMPLETE"}


def test_holdout_metadata_only_and_no_reserved_use():
    audit = json.loads((OUT / "E3_D_R_HOLDOUT_AUDIT.json").read_text())
    manifest = json.loads((GOV / "E3_0_RESERVED_PARTITIONS_MANIFEST.json").read_text())
    assert audit["status"] == "E3_0_QUERY_HELD_OUT_ONLY"
    assert audit["reserved_answers_opened"] is False
    assert audit["reserved_model_outcomes_opened"] is False
    assert audit["reserved_used_by_e3_d"] == 0
    assert len(manifest["reserved_query_ids"]) == 195


def test_r_provenance_and_no_paid_work():
    manifest = json.loads((OUT / "E3_D_R_PROVENANCE_MANIFEST.json").read_text())
    assert manifest["status"] == "E3_D_R_OWNER_REVIEW_READY"
    assert manifest["new_model_inference"] is False
    assert manifest["gpu_use"] is False
    assert manifest["jev_calls"] == 0
    assert manifest["paid_api_calls"] == 0
    for name, digest in manifest["artifact_sha256"].items():
        path = OUT / name
        assert path.is_file(), name
        assert _sha(path) == digest, name


def test_normalizer_fixture_artifact_passes():
    report = (OUT / "E3_D_R_FULL_REGRESSION_REPORT.md").read_text()
    assert "6/6 PASS" in report
