import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e3_a_offline as e3


def read_csv(name):
    with (ROOT / "results/non_jev/v2/e3_v1/e3_a_paired" / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_frozen_population_and_unique_ids():
    items = e3.load_items()
    assert len(items) == 4267
    assert len({x["query_id"] for x in items}) == len(items)


def test_qst_0413_is_retained_but_not_identity_eligible():
    states = read_csv("E3_A_QUERY_LEVEL_EVIDENCE_STATES.csv")
    row = next(x for x in states if x["query_id"] == "qst_0413")
    assert row["identity_eligible"] == "False"
    assert row["status"] == "INELIGIBLE_IDENTITY"


def test_top5_containment_and_four_cell_partition():
    items = {x["query_id"]: x for x in e3.load_items()}
    selections, _hashes, errors = e3.load_selection(list(items.values()))
    assert errors == []
    for (policy, qid), row in selections.items():
        if policy == "retrieval5" or row.get("status") == "ok":
            assert len(row["selected_ids"]) == 5
            assert len(set(row["selected_ids"])) == 5
            assert set(row["selected_ids"]).issubset(set(row["candidate_ids"]))
    cells = read_csv("E3_A_FOUR_CELL_COUNTS.csv")
    for key in [("track_a", "hotpot", "unconditional_eligible"),
                ("track_a", "musique", "retrieval_conditional"),
                ("track_b", "enterprise", "retrieval_conditional")]:
        subset = [r for r in cells if (r["cohort"], r["dataset"], r["population"]) == key]
        assert sum(int(r["count"]) for r in subset) == int(subset[0]["denominator"])


def test_historical_anchor_counts_are_reproduced():
    rows = read_csv("E3_A_PAIRED_EVIDENCE_STATS.csv")
    a = next(r for r in rows if r["cohort"] == "track_a" and r["dataset"] == "__pooled__" and r["population"] == "retrieval_conditional")
    b = next(r for r in rows if r["cohort"] == "track_b" and r["population"] == "retrieval_conditional")
    assert (int(a["n"]), int(a["pplx_only_wins"]), int(a["jina_only_wins"]), int(a["ties"])) == (1490, 86, 101, 1303)
    assert (int(b["n"]), int(b["pplx_only_wins"]), int(b["jina_only_wins"]), int(b["ties"])) == (330, 47, 6, 277)


def test_reader_cohort_and_nested_judge_are_separate():
    audit = read_csv("E3_A_READER_ALIGNMENT_AND_REFERENCE_AUDIT.csv")
    assert len(audit) == 3300
    assert len({x["query_id"] for x in audit}) == 300
    secondary = read_csv("E3_A_NO_REFERENCE_JUDGE_SECONDARY.csv")
    assert len(secondary) == 864
    assert all(x["assessment_provenance"] == "LLM_ADJUDICATED_LOCAL_BLINDED" for x in secondary)
    assert all(x["reference_answer_in_prompt"] == "False" for x in secondary)


def test_answer_proxy_and_oracle_do_not_claim_factual_gold():
    answers = read_csv("E3_A_PAIRED_ANSWER_STATS.csv")
    assert all(x["metric"] == "normalized_EM_proxy" for x in answers)
    curves = read_csv("E3_A_ESCALATION_ORACLE_CURVES.csv")
    assert {x["policy"] for x in curves} == {"oracle", "random"}
    assert {float(x["rate"]) for x in curves} == {0.0, 0.05, 0.1, 0.2, 0.5, 1.0}
    assert all(x["latency_status"] == "TAIL_LATENCY_NOT_MEASURED" for x in curves)


def test_governance_remains_pending_without_metadata():
    blocker = json.loads((ROOT / "results/non_jev/v2/e3_v1/e3_0_governance/E3_0_RESERVATION_BLOCKER.json").read_text())
    assert blocker["status"] == "E3_0_HOLDOUT_RESERVATION_PENDING_METADATA"
    assert not (ROOT / "results/non_jev/v2/e3_v1/e3_0_governance/E3_0_RESERVATION_MANIFEST.json").exists()


def test_output_gate_and_zero_paid_calls():
    manifest = json.loads((ROOT / "results/non_jev/v2/e3_v1/e3_a_paired/E3_A_EXECUTION_MANIFEST.json").read_text())
    assert manifest["status"] == "E3_A_OWNER_REVIEW_READY"
    assert manifest["jev_calls"] == 0
    assert manifest["paid_api_calls"] == 0
    assert manifest["new_model_inference_requests"] == 0
