import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results/non_jev/v2/e4_v1_1/e4_2_facet_pilot"
POLICIES = {
    "jina_native_top5",
    "jina_facet_coverage",
    "liquid_native_top5",
    "liquid_pointwise_facet_coverage",
    "liquid_joint_facet_coverage",
}


def jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_e4_2_frozen_cohorts_and_question_only_facets():
    manifest = json.loads((PILOT / "E4_2_INPUT_AND_COHORT_MANIFEST.json").read_text())
    facets = jsonl(PILOT / "E4_2_FACETS.jsonl")
    assert len(manifest["representative_qids"]) == 200
    assert len(manifest["diagnostic_enriched_qids"]) == 50
    assert len(manifest["cal_qids"]) == 30
    assert set(manifest["representative_qids"]).isdisjoint(manifest["diagnostic_enriched_qids"])
    assert set(manifest["cal_qids"]).isdisjoint(manifest["representative_qids"] + manifest["diagnostic_enriched_qids"])
    assert len(facets) == 280
    assert all(x["facet_status"] == "VALID" and 2 <= len(x["facets"]) <= 4 for x in facets)
    assert all(x["gold_in_facet_input"] is False and x["reference_in_facet_input"] is False and x["support_ids_in_facet_input"] is False for x in facets)


def test_e4_2_native_rows_have_complete_policy_alignment():
    raw = jsonl(PILOT / "E4_2_NATIVE_SCORING_RAW.jsonl")
    assert len(raw) == 1400
    assert not [x for x in raw if x["status"] not in {"VALID", "REUSED_HISTORICAL_EXACT"}]
    assert {x["policy"] for x in raw} == POLICIES
    assert len({(x["qid"], x["policy"]) for x in raw}) == len(raw)
    for x in raw:
        assert len(x["candidate_ids"]) == 20 == len(set(x["candidate_ids"]))
        assert len(x["selected_ids"]) == 5 == len(set(x["selected_ids"]))
        assert set(x["selected_ids"]).issubset(set(x["candidate_ids"]))
        assert x.get("gold_in_policy_input") is False
        assert x.get("reference_in_policy_input") is False


def test_e4_2_common_candidate_views_and_facets_are_identical_across_policies():
    raw = jsonl(PILOT / "E4_2_NATIVE_SCORING_RAW.jsonl")
    by_qid = {}
    for row in raw:
        by_qid.setdefault(row["qid"], []).append(row)
    for qid, rows in by_qid.items():
        assert {tuple(x["candidate_ids"]) for x in rows}.__len__() == 1
        assert {tuple(x["candidate_view_hashes"]) for x in rows}.__len__() == 1
        if any(x.get("facet_set_sha256") for x in rows):
            assert len({x.get("facet_set_sha256") for x in rows if x.get("facet_set_sha256")}) == 1


def test_e4_2_liquid_mechanism_uses_one_revision_and_explicit_work():
    raw = jsonl(PILOT / "E4_2_NATIVE_SCORING_RAW.jsonl")
    liquid = [x for x in raw if x["policy"].startswith("liquid_")]
    assert {x["revision"] for x in liquid} == {"051bcc464b01b9f92942b364d9586b0ef5912432"}
    assert all(int(x.get("physical_forward_calls") or 0) >= 1 for x in liquid)
    point = [x for x in liquid if x["policy"] == "liquid_pointwise_facet_coverage" and x["phase"] == "full"]
    joint = [x for x in liquid if x["policy"] == "liquid_joint_facet_coverage" and x["phase"] == "full"]
    assert sum(x["physical_forward_calls"] for x in point) > sum(x["physical_forward_calls"] for x in joint)


def test_e4_2_quality_artifact_has_typed_records_and_separate_cohorts():
    rows = list(csv.DictReader((PILOT / "E4_2_QUALITY_AND_PAIRED_STATS.csv").open()))
    types = {x["record_type"] for x in rows}
    assert {"query_outcome", "paired_stat", "policy_scorecard"}.issubset(types)
    scorecard = [x for x in rows if x["record_type"] == "policy_scorecard"]
    assert len(scorecard) == 10
    assert {x["cohort"] for x in scorecard} == {"representative", "diagnostic_enriched"}
    assert all(int(x["n_queries"]) in {50, 200} for x in scorecard)


def test_e4_2_work_reports_phase_and_shared_gpu_limitation():
    rows = list(csv.DictReader((PILOT / "E4_2_INFERENCE_WORK_AND_LATENCY.csv").open()))
    assert len(rows) == 15
    assert {x["phase"] for x in rows} == {"full", "cal", "smoke"}
    assert all(x["latency_status"] == "NOT_COMPARABLE_SHARED_GPU" for x in rows)


def test_e4_2_protocol_prohibits_holdout_paid_and_future_phases():
    protocol = json.loads((PILOT / "E4_2_FROZEN_EXPLORATORY_PROTOCOL.json").read_text())
    assert protocol["gold_in_inference_input"] is False
    assert protocol["reference_in_facet_input"] is False
    assert protocol["support_ids_in_policy_input"] is False
    assert protocol["candidate_text_contract"]["name"] == "common_word_capped_256_v1"
    manifest = json.loads((PILOT / "E4_2_PROVENANCE_MANIFEST.json").read_text())
    assert manifest["jev_calls"] == 0 and manifest["paid_api_calls"] == 0
    assert manifest["reserved_outcomes_accessed"] is False
    assert manifest["future_phases"] == {"E4_3": "NOT_RUN", "E4_5": "NOT_RUN"}


def test_e4_2_provenance_manifest_input_hashes_exist():
    manifest = json.loads((PILOT / "E4_2_PROVENANCE_MANIFEST.json").read_text())
    for rel, digest in manifest["input_sha256"].items():
        path = ROOT / rel
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_e4_2_cal_smoke_and_failure_events_are_explicit():
    report = (PILOT / "E4_2_CAL_AND_SMOKE_REPORT.md").read_text()
    events = (PILOT / "E4_2_EXECUTION_EVENTS.md").read_text()
    assert "Status: `PASS`" in report
    assert "unbounded-view Jina OOM" in report
    assert "no answer or selection from that attempt is used" in events


def test_e4_2_owner_gate_and_no_automatic_expansion():
    gate = (PILOT / "E4_2_OWNER_REVIEW_GATE.md").read_text()
    assert "E4_2_PILOT_OWNER_REVIEW_READY" in gate
    assert "E4-3/E4-5: `NOT_RUN`" in gate
    assert "do not expand" in gate.lower()
