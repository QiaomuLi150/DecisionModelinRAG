import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e8_v1_0"
PRE = OUT / "00_preflight"
RAW = OUT / "02_rank_metadata_inference"
ANA = OUT / "03_offline_analysis"


def _rows():
    return [json.loads(x) for x in (RAW / "E8_RANK_ANNOTATED_RAW_OUTPUTS.jsonl").read_text().splitlines() if x.strip()]


def test_e8_freeze_is_preoutcome_and_exact():
    protocol = json.loads((PRE / "E8_PREINFERENCE_PROTOCOL.json").read_text())
    freeze = json.loads((PRE / "E8_FREEZE_MANIFEST.json").read_text())
    assert protocol["planned_logical_attempts"] == 640
    assert protocol["labels_joined_after_raw_freeze"] is True
    assert freeze["outcomes_observed_at_freeze"] is False
    assert len(freeze["cohort_query_keys"]) == 80


def test_e8_raw_attempts_complete_and_label_blind():
    rows = [x for x in _rows() if x.get("stage") == "inference"]
    assert len(rows) == 640
    keys = [(x["model"], x["dataset"], x["sample_id"], x["condition"]) for x in rows]
    assert len(keys) == len(set(keys))
    assert {x["condition"] for x in rows} == {"C", "D"}
    assert all(x.get("gold_in_input") is False for x in rows)
    assert all(x.get("reference_answer_in_input") is False for x in rows)


def test_e8_rank_identity_survives_reversal():
    rows = [x for x in _rows() if x.get("stage") == "inference"]
    groups = {}
    for row in rows:
        groups.setdefault((row["model"], row["dataset"], row["sample_id"]), {})[row["condition"]] = row
    assert len(groups) == 320
    for pair in groups.values():
        c, d = pair["C"], pair["D"]
        assert c["original_rank_by_id"] == d["original_rank_by_id"]
        assert list(reversed(c["candidate_ids"])) == d["candidate_ids"]
        assert c["original_candidate_ids"] == d["original_candidate_ids"]
        assert len(set(c["candidate_ids"])) == len(c["candidate_ids"])
        assert len(set(d["candidate_ids"])) == len(d["candidate_ids"])


def test_e8_valid_top5_or_explicit_failure():
    for row in [x for x in _rows() if x.get("stage") == "inference"]:
        if row.get("status") == "VALID":
            assert len(row["selected_ids"]) == 5
            assert len(set(row["selected_ids"])) == 5
            assert set(row["selected_ids"]).issubset(set(row["candidate_ids"]))
        else:
            assert row.get("failure")
            assert row.get("failure_class")


def test_e8_offline_scorecard_has_separate_workloads_and_common_denominators():
    rows = list(__import__("csv").DictReader((ANA / "E8_ORDER_AND_RANK_METADATA_SCORECARD.csv").open()))
    assert len(rows) == 8
    assert {x["dataset"] for x in rows} == {"scifact", "garage"}
    assert all(int(x["scheduled"]) == 40 for x in rows)
    assert all(int(x["common_complete_n"]) <= min(int(x["ar_valid_pairs"]), int(x["cd_valid_pairs"])) for x in rows)


def test_e8_no_protected_or_paid_calls():
    receipt = json.loads((RAW / "E8_OUTPUT_FREEZE_AND_HASHES.json").read_text())
    assert receipt["jev_calls"] == 0
    assert receipt["paid_api_calls"] == 0
    assert receipt["reserved_access"] is False
    assert receipt["labels_joined"] is False
