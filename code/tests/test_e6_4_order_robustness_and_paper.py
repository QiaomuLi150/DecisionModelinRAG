import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_4_integrated_robustness_and_paper"


def manifest():
    return json.loads((OUT / "E6_4_COHORT_AND_PERMUTATION_MANIFEST.json").read_text())


def attempts():
    rows = []
    for p in sorted((OUT / "mechanism_inference").glob("E6_4_*.jsonl")):
        rows.extend(json.loads(x) for x in p.read_text().splitlines() if x.strip())
    return [x for x in rows if x.get("stage") == "inference"]


def test_cohort_and_attempt_counts():
    assert len(manifest()["items"]) == 200
    assert len(attempts()) == 2400


def test_permutations_preserve_candidate_membership():
    for item in manifest()["items"]:
        base = set(item["candidate_ids"])
        for value in item["permutations"].values():
            assert set(value["candidate_ids"]) == base
            assert len(value["candidate_ids"]) == len(base)


def test_permutations_are_deterministically_indexed():
    for item in manifest()["items"]:
        n = len(item["candidate_ids"])
        assert item["permutations"]["canonical"]["indices"] == list(range(n))
        assert item["permutations"]["reversed"]["indices"] == list(reversed(range(n)))


def test_attempt_keys_are_unique():
    keys = [(x["model"], x["dataset"], x["sample_id"], x["permutation"]) for x in attempts()]
    assert len(keys) == len(set(keys))


def test_all_outputs_have_valid_top5():
    for row in attempts():
        if row["status"] == "VALID":
            selected = row["selected_ids"]
            assert len(selected) == 5
            assert len(set(selected)) == 5
            assert set(selected).issubset(set(row["candidate_ids"]))


def test_failure_accounting_is_explicit():
    audit = json.loads((OUT / "E6_4_INPUT_OUTPUT_HASH_AUDIT.json").read_text())
    assert audit["attempts"] == audit["expected_attempts"]
    assert audit["reserved_content_loaded"] is False


def test_scorecard_rows_are_separate_by_workload():
    rows = list(csv.DictReader((OUT / "mechanism_scoring/E6_4_RELEVANCE_SENSITIVITY.csv").open()))
    assert {x["dataset"] for x in rows} == {"scifact", "garage"}
    assert len(rows) == 2400


def test_protected_services_and_paid_calls_are_not_claimed():
    gate = json.loads((OUT / "E6_4_FINAL_OWNER_REVIEW_GATE.json").read_text())
    assert gate["reserved_access"] is False
    assert gate["jev_calls"] == 0
    assert gate["paid_api_calls"] == 0
