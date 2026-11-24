import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOUNDATION = ROOT / "results/non_jev/v2/e4_v1_1/e4_0_foundation"
INTERVENTIONS = ROOT / "results/non_jev/v2/e4_v1_1/e4_1_interventions"
ARMS = ["A_JINA_TOP5", "B_PPLX_TOP5", "C_ANNOTATED_ORACLE", "D_SUPPORT_REMOVAL", "D_CONTROL_NEUTRAL_SWAP", "E_NO_EVIDENCE"]


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_e4_foundation_gate_and_cohort_freeze():
    gate = (FOUNDATION / "E4_0_VALIDITY_GATE.md").read_text()
    manifest = json.loads((FOUNDATION / "E4_0_COHORT_AND_ARMS_MANIFEST.json").read_text())
    assert "E4_0_EXPLORATORY_READY`: PASS" in gate
    assert manifest["status"] == "FROZEN_BEFORE_NEW_READER_CALLS"
    assert manifest["representative_n"] == 200
    assert manifest["diagnostic_enriched_n"] == 50
    assert set(manifest["representative_qids"]).isdisjoint(manifest["diagnostic_enriched_qids"])


def test_e4_six_arms_and_unique_ids():
    data = rows(INTERVENTIONS / "E4_1_ARM_CONSTRUCTION.jsonl")
    by_qid = {}
    for row in data:
        by_qid.setdefault(row["qid"], {})[row["arm"]] = row
        if row["arm_valid"]:
            assert len(row["candidate_ids_in_order"]) == len(set(row["candidate_ids_in_order"]))
    assert by_qid
    assert all(set(x) == set(ARMS) for x in by_qid.values())


def test_e4_no_gold_fields_in_reader_inputs():
    for row in rows(INTERVENTIONS / "E4_1_ARM_CONSTRUCTION.jsonl"):
        assert row["gold_in_reader_input"] is False
        assert row["answer_in_reader_input"] is False
        assert "reference_answer" in row["reference_only_fields_not_sent"]
        assert "support_doc_ids" in row["reference_only_fields_not_sent"]


def test_e4_ab_archived_top5_alignment():
    selected = {}
    for line in (ROOT / "results/non_jev/v2/e3_v1/e3_d_external/E3_D_SELECTED_TOP5.jsonl").read_text().splitlines():
        row = json.loads(line)
        selected[(row["policy"], row["query_id"])] = row["selected_ids"]
    for row in rows(INTERVENTIONS / "E4_1_ARM_CONSTRUCTION.jsonl"):
        if row["arm"] == "A_JINA_TOP5":
            assert row["candidate_ids_in_order"] == selected[("jina35", row["qid"])]
        if row["arm"] == "B_PPLX_TOP5":
            assert row["candidate_ids_in_order"] == selected[("pplx_decider", row["qid"])]


def test_e4_support_removal_and_placebo_change_only_intended_context():
    data = rows(INTERVENTIONS / "E4_1_ARM_CONSTRUCTION.jsonl")
    by = {(r["qid"], r["arm"]): r for r in data}
    for qid in sorted({r["qid"] for r in data}):
        c = by[(qid, "C_ANNOTATED_ORACLE")]
        d = by[(qid, "D_SUPPORT_REMOVAL")]
        dc = by[(qid, "D_CONTROL_NEUTRAL_SWAP")]
        if d["arm_valid"]:
            assert len(set(c["candidate_ids_in_order"]) - set(d["candidate_ids_in_order"])) == 1
            assert d["removed_support_id"] in set(c["candidate_ids_in_order"]) - set(d["candidate_ids_in_order"])
        if dc["arm_valid"]:
            assert len(set(c["candidate_ids_in_order"]) - set(dc["candidate_ids_in_order"])) == 1
            assert len(set(dc["candidate_ids_in_order"]) - set(c["candidate_ids_in_order"])) == 1


def test_e4_confirmation_and_future_phases_locked():
    inv = json.loads((FOUNDATION / "E4_0_INPUT_INVENTORY.json").read_text())
    manifest = json.loads((FOUNDATION / "E4_0_COHORT_AND_ARMS_MANIFEST.json").read_text())
    assert inv["reserved_outcomes_accessed"] is False
    assert inv["new_selector_inference"] is False
    assert "CONFIRMATORY_DATA_NOT_READY" in manifest["confirmation_status"]


def test_e4_native_call_and_reuse_accounting():
    raw = rows(INTERVENTIONS / "E4_1_NATIVE_READER_RAW.jsonl")
    assert sum(r.get("phase") == "cal" and r.get("status") == "ok" for r in raw) == 60
    assert sum(r.get("phase") == "study" and r.get("status") == "ok" for r in raw) == 1337
    assert sum(r.get("phase") == "study" and r.get("status") == "REUSED_HISTORICAL_EXACT" for r in raw) == 156
    assert sum(int(r.get("physical_calls") or 0) for r in raw) == 1397
    assert not [r for r in raw if r.get("status") == "error"]
    assert all(r.get("gold_in_reader_input") is False for r in raw if r.get("accepted_status") == "accepted")


def test_e4_paired_statistics_are_frozen_and_separate():
    import csv

    stats = list(csv.DictReader((INTERVENTIONS / "E4_1_PAIRED_INTERVENTION_STATS.csv").open()))
    assert len(stats) == 12
    assert {r["cohort"] for r in stats} == {"representative", "diagnostic_enriched"}
    assert all(int(r["repetitions"]) == 5000 and int(r["seed"]) == 731 for r in stats)
    assert (INTERVENTIONS / "E4_1_ADJUDICATION_QC.csv").read_text().find("NOT_RUN") >= 0
