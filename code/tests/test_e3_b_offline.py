import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e3_v1/e3_b_mechanism"


def _jsonl(name):
    with (OUT / name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _csv(name):
    with (OUT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_e3_b_owner_gate_and_provenance():
    status = json.loads((OUT / "E3_B_PROVENANCE_MANIFEST.json").read_text())
    gate = (OUT / "E3_B_OWNER_REVIEW_GATE.md").read_text()
    assert status["status"] == "E3_B_PASS_WITH_LIMITATIONS"
    assert status["owner_review_gate"] == "E3_B_OWNER_REVIEW_READY"
    assert "E3_B_OWNER_REVIEW_READY" in gate
    assert status["jev_calls"] == 0
    assert status["paid_api_calls"] == 0
    assert status["training"] is False
    assert status["b1_counts"]["native_errors"] == 0
    assert status["b1_counts"]["new_physical_forward_calls"] == 2102
    assert status["b1_counts"]["gpu_ids_used"] == [0, 1]


def test_e3_b_sampling_and_panel_counts():
    rows = _jsonl("E3_B_SAMPLE_AND_PANEL_MANIFEST.jsonl")
    assert len(rows) == 2400
    assert Counter(row["result_type"] for row in rows) == {
        "authenticated_reuse": 400,
        "new_native_inference": 2000,
    }
    full = [row for row in rows if row["arm"] != "B1_BASE_JOINT"]
    assert len(full) == 2000
    assert Counter(row["arm"] for row in full) == {
        "B1_ORDER_P1": 400,
        "B1_ORDER_P2": 400,
        "B1_ORDER_P3": 400,
        "B1_CONTEXT_MATCHED": 400,
        "B1_CONTEXT_IRRELEVANT": 400,
    }


def test_e3_b_native_detail_alignment_and_counts():
    rows = _jsonl("E3_B_CONTEXT_INTERVENTION_NATIVE_DETAIL.jsonl")
    assert len(rows) == 2102
    assert Counter(row["stage"] for row in rows) == {"smoke": 72, "cal": 30, "full": 2000}
    assert Counter(row["status"] for row in rows) == {"ok": 2102}
    assert Counter(int(row["gpu_id"]) for row in rows) == {0: 1950, 1: 152}
    full = [(row["query_id"], row["arm"]) for row in rows if row["stage"] == "full"]
    assert len(full) == len(set(full)) == 2000
    for row in rows:
        assert row["score_alignment"] == "field_id_candidate_order"
        assert row["score_direction"] == "higher_is_more_relevant"
        assert row["gold_in_inference_input"] is False
        assert len(row["canonical_candidate_ids"]) == len(set(row["canonical_candidate_ids"])) == 20
        assert len(row["scores"]) == 20


def test_e3_b_offline_result_cardinalities():
    quota = _csv("E3_B_SAME_SCORE_QUOTA_RESULTS.csv")
    selected = _jsonl("E3_B_SETAWARE_SELECTED_IDS.jsonl")
    stats = _csv("E3_B_SAME_SCORE_QUOTA_PAIRED_STATS.csv")
    assert len(quota) == 8534
    assert len(selected) == 4267 * 7
    assert len(stats) > 0
    for row in selected:
        assert len(row["selected_ids"]) == 5
        assert len(row["selected_ids"]) == len(set(row["selected_ids"]))
        assert set(row["selected_ids"]).issubset(set(row["candidate_ids"]))
        assert row["gold_in_input"] is False


def test_e3_b_setr_label_and_cal_freeze():
    report = (OUT / "E3_B_SETR_FEASIBILITY_AUDIT.md").read_text()
    config = json.loads((OUT / "E3_B_SETAWARE_CAL_CONFIG.json").read_text())
    assert "SETR_NOT_REPRODUCIBLE" in report
    assert config["selected"]["mmr"]["lambda"] == 1.0
    assert config["selected"]["coverage"]["lambda"] == 1.0
    assert {row["cal_n"] for row in config["grid"]} == {22}


def test_e3_b_artifact_hashes():
    status = json.loads((OUT / "E3_B_PROVENANCE_MANIFEST.json").read_text())
    bad = []
    for relative, expected in status["artifact_sha256"].items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            bad.append(relative)
    assert bad == []
