import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e3_v1/e3_f0_protocol_review"


def _json(path):
    return json.loads(path.read_text())


def _csv(path):
    return list(csv.DictReader(path.open()))


def test_f0_protocol_is_offline_and_protected_paths_denied():
    x = _json(OUT / "E3_F0_EXECUTION_PROTOCOL.json")
    assert x["new_model_inference"] is False
    assert x["gpu_use"] is False
    assert x["reserved_outcomes_accessed"] is False
    assert x["e3f_started"] is False
    assert any("reserved answers" in p for p in x["protected_denylist"])


def test_f0_denominators_and_nonpooling_dictionary():
    text = (OUT / "E3_F0_METRIC_AND_COHORT_DICTIONARY.md").read_text()
    assert "4,267" in text and "1,490" in text and "330" in text
    assert "250 MultiHop" in text and "109 FinanceBench" in text
    assert "must not be substituted" in text


def test_f0_evidence_tables_keep_cssr_u_c_separate():
    rows = _csv(OUT / "E3_F0_PAPER_PRIMARY_TABLES.md") if False else None
    text = (OUT / "E3_F0_PAPER_PRIMARY_TABLES.md").read_text()
    assert "CSSR-U" in text and "CSSR-C" in text
    assert "conditional" in text.lower()
    assert "FinanceBench deterministic proxies" in text


def test_f0_finance_incomplete_qrels_guard_and_proxy_status():
    text = (OUT / "E3_F0_METRIC_AND_COHORT_DICTIONARY.md").read_text()
    ledger = _csv(OUT / "E3_F0_CLAIM_EVIDENCE_LEDGER.csv")
    assert "not exhaustive" in text
    assert any(r["claim_id"] == "C10" and r["label"] == "NOT_ESTABLISHED" for r in ledger)
    assert all(r["label"] != "MANUAL_VERIFIED" for r in ledger)


def test_f0_external_denominators_and_selection_counts():
    text = (OUT / "E3_F0_INTEGRITY_AND_PROVENANCE_AUDIT.md").read_text()
    assert "6,054" in text and "2,154" in text and "900 MultiHop" in text


def test_f0_holdout_timestamp_ambiguity_is_not_promoted():
    text = (OUT / "E3_F0_CONFIRMATORY_DATA_FEASIBILITY_AUDIT.md").read_text()
    assert "CONFIRMATORY_DATA_NOT_READY" in text
    assert "QUERY_ONLY_OR_PROSPECTIVITY_UNVERIFIED" in text
    assert "reservation-before-analysis" in text
    assert "E3-F remains locked" in text


def test_f0_power_has_all_effects_targets_and_edge_policy():
    x = _json(OUT / "E3_F0_POWER_DESIGN_INPUTS.json")
    rows = _csv(OUT / "E3_F0_POWER_SENSITIVITY_RESULTS.csv")
    assert x["effects_pp"] == [2, 5, 10]
    assert x["targets"] == [0.8, 0.9]
    assert len(rows) == 3 * 50
    assert set(int(r["effect_pp"]) for r in rows) == {2, 5, 10}
    assert x["minimum_n"]["2"]["0.8"] is None or int(x["minimum_n"]["2"]["0.8"]) >= 100


def test_f0_options_are_proposals_not_approval():
    text = (OUT / "E3_F0_CONFIRMATORY_PROTOCOL_OPTIONS.md").read_text()
    assert "DRAFT, NOT APPROVED" in text
    assert "Option A" in text and "Option B" in text and "Option C" in text
    assert "NOT_FEASIBLE_NOW" in text


def test_f0_source_hash_index_and_new_artifact_scope():
    x = _json(OUT / "E3_F0_SOURCE_ARTIFACT_INDEX.json")
    assert x["historical_hash_mismatches"] == []
    assert x["protected_source_content_opened"] is False
    assert all("source/" not in r["path"] for r in x["inputs"])


def test_f0_reviewer_packet_is_critical_and_has_questions():
    text = (OUT / "E3_F0_REVIEWER_CONSULTATION_PACKET.md").read_text()
    assert len(text.split()) >= 1800
    assert text.count("?") >= 10
    assert "no new experiments" in text.lower()
    assert "not independently verified" in text.lower()


def test_f0_no_verified_accuracy_or_fabricated_reviewers():
    for name in ("E3_F0_REVIEWER_CONSULTATION_PACKET.md", "E3_F0_REVIEWER_EXECUTIVE_BRIEF.md"):
        text = (OUT / name).read_text().lower()
        assert "factual correctness" in text
        assert "reviewer" in text
    assert "endorsement" not in (OUT / "E3_F0_REVIEWER_CONSULTATION_PACKET.md").read_text().lower()
