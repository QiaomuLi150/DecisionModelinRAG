import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e7_v1_0"
ROB = OUT / "02_repeatability_and_order"


def test_e7_protocol_and_cohort_are_label_blind_and_exact():
    protocol = json.loads((ROB / "E7_B_PROTOCOL.json").read_text())
    manifest = json.loads((ROB / "E7_B_COHORT_MANIFEST.json").read_text())
    assert protocol["planned_physical_attempts"] == 1600
    assert protocol["holdout_access"] is False
    assert len(manifest["items"]) == 80
    assert len({(x["dataset"], x["sample_id"]) for x in manifest["items"]}) == 80
    assert sum(x["dataset"] == "scifact" for x in manifest["items"]) == 40
    assert sum(x["dataset"] == "garage" for x in manifest["items"]) == 40
    for item in manifest["items"]:
        assert item["gold_in_input"] is False
        assert len(item["candidate_ids"]) in (15, 20)
        assert len(item["candidate_ids"]) == len(set(item["candidate_ids"]))
        assert set(item["permutations"]["reversed"]["candidate_ids"]) == set(item["candidate_ids"])
        assert set(item["permutations"]["seeded"]["candidate_ids"]) == set(item["candidate_ids"])


def test_e7_raw_attempts_are_unique_and_complete():
    active = ROB / "E7_B_NATIVE_OUTPUTS_CORRECTED_V2.jsonl"
    if not active.exists():
        active = ROB / "E7_B_NATIVE_OUTPUTS.jsonl"
    rows = [json.loads(x) for x in active.read_text().splitlines() if x.strip()]
    attempts = [x for x in rows if x.get("stage") == "inference"]
    assert len(attempts) == 1600
    keys = [(x["model"], x["dataset"], x["sample_id"], x["condition"]) for x in attempts]
    assert len(keys) == len(set(keys))
    assert all(x.get("gold_in_input") is False for x in attempts)


def test_e7_valid_top5_and_explicit_failures():
    active = ROB / "E7_B_NATIVE_OUTPUTS_CORRECTED_V2.jsonl"
    if not active.exists():
        active = ROB / "E7_B_NATIVE_OUTPUTS.jsonl"
    rows = [json.loads(x) for x in active.read_text().splitlines() if x.strip()]
    attempts = [x for x in rows if x.get("stage") == "inference"]
    for row in attempts:
        if row["status"] == "VALID":
            assert len(row["selected_ids"]) == 5
            assert len(set(row["selected_ids"])) == 5
            assert set(row["selected_ids"]).issubset(set(row["candidate_ids"]))
            assert row.get("native_input_sha256")
        else:
            assert row.get("failure") or row.get("failure_class")


def test_e7_protected_access_and_no_paid_calls():
    receipt = json.loads((OUT / "02_repeatability_and_order/E7_B_CORRECTED_RAW_FREEZE_RECEIPT.json").read_text())
    assert receipt["jev_calls"] == 0
    assert receipt["paid_api_calls"] == 0
    assert receipt["labels_joined"] is False
    assert receipt["attempts"] == 1600


def test_e7_ledger_keeps_metrics_separate():
    rows = list(csv.DictReader((OUT / "03_evidence_ledger/E7_AUTHORITATIVE_RESULTS_LEDGER.csv").open()))
    assert rows
    assert any(r["status"] == "INVALID_SUPERSEDED" for r in rows)
    assert any(r["status"] == "PROXY_ONLY" for r in rows)
