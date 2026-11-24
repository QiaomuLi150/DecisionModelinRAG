import csv
import json
from pathlib import Path

from scripts.e5_qa_final_audit import (
    E5,
    E5A,
    E5DC,
    OUT,
    SLATE_SHA,
    load_state,
    norm_alias,
    norm_frozen,
    score_variant,
    sha_file,
)


def test_frozen_slate_and_e5a_population():
    assert sha_file(E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl") == SLATE_SHA
    outcomes = [json.loads(x) for x in (E5A / "E5_A_SELECTED_IDS_AND_SET_AWARE_OUTCOMES.jsonl").read_text().splitlines()]
    assert len(outcomes) == 1502
    assert len({x["query_id"] for x in outcomes}) == 1502


def test_reader_accounting_exact_overlap():
    state = load_state()
    rep, diag = state["rep"], state["diag"]
    rows = state["reader"]
    assert len(rep) == 200
    assert len(diag) == 55
    assert len(rep & diag) == 7
    assert len(rows) == 496
    assert len({(x["query_id"], x["policy"]) for x in rows}) == 496
    assert all(x["status"] == "OK" for x in rows)


def test_no_cache_reuse_and_gold_in_reader_inputs():
    state = load_state()
    assert all(not x.get("cache_reuse") for x in state["reader"])
    assert all(not x.get("gold_in_reader_input") for x in state["reader"])


def test_reader_pair_integrity_and_selected_cardinality():
    state = load_state()
    by = {(x["query_id"], x["policy"]): x for x in state["reader"]}
    for qid in state["rep"] | state["diag"]:
        for policy in ("pplx_decider", "jina35"):
            row = by[(qid, policy)]
            assert len(row["selected_ids"]) == 5
            assert len(set(row["selected_ids"])) == 5
            assert row["actual_model_input_sha256"] == row["request_body_sha256"]


def test_scorers_are_symmetric_and_deterministic():
    forms = ["10 percent", "yes"]
    assert score_variant("10 percent", forms, "frozen_normalized")
    assert score_variant("10%", forms, "numeric_unit_symmetric")
    assert score_variant("true", ["yes"], "alias_symmetric")
    assert norm_frozen("A, B") == norm_frozen("a b")
    assert norm_alias("TRUE") == norm_alias("yes")


def test_e5a_raw_counts_and_ceiling():
    state = load_state()
    outcomes = list(state["outcomes"].values())
    assert sum(int(x["pplx_y"]) for x in outcomes) == 306
    assert sum(int(x["jina_y"]) for x in outcomes) == 299
    assert sum(int(x["retrieval_feasible"]) for x in outcomes) == 355
    assert sum(int(x["pplx_y"]) != int(x["jina_y"]) for x in outcomes) == 55


def test_serving_trace_contract():
    rows = [json.loads(x) for x in (E5DC / "E5_D_NATIVE_SERVING_ATTEMPTS.jsonl").read_text().splitlines()]
    for policy in ("pplx_decider", "jina35"):
        measured = [x for x in rows if x["policy"] == policy and x["stage"] == "D_MEASURED"]
        assert len(measured) == 200
        assert all(x["status"] == "VALID" for x in measured)
        assert {x["gpu_uuid"] for x in measured} == {"<REDACTED_GPU_UUID>"}


def test_required_outputs_and_provenance():
    required = [
        "E5_QA_FINAL_SCIENTIFIC_VALIDITY_REPORT.md",
        "E5_QA_INPUT_PROVENANCE_INDEX.json",
        "E5_QA_READER_REQUEST_RECONCILIATION.md",
        "E5_QA_READER_SLOT_LEDGER.csv",
        "E5_QA_PAIRED_SAMPLE_INTEGRITY.json",
        "E5_QA_ANSWER_CONTRACT_AUDIT.md",
        "E5_QA_ANSWER_TYPE_AND_FORMAT_SCORECARD.csv",
        "E5_QA_DETERMINISTIC_SCORER_SENSITIVITY.csv",
        "E5_QA_ANSWER_CASEBOOK.md",
        "E5_QA_BLINDED_ADJUDICATION_PACKET.md",
        "E5_QA_READER_VALIDITY_VERDICT.md",
        "E5_QA_PRIMARY_EVIDENCE_REPRODUCTION.csv",
        "E5_QA_SERVING_COMPARABILITY_AUDIT.md",
        "E5_QA_REPLACEMENT_DECISION_MATRIX.md",
        "E5_QA_LIMITATIONS_AND_OPEN_ISSUES.md",
        "E5_QA_MASTER_CLAIMS_EVIDENCE_LEDGER.csv",
        "E5_QA_PAPER_READY_TABLES.md",
        "E5_QA_MANUSCRIPT_DRAFT.md",
        "E5_QA_EXTERNAL_REVIEWER_PACKET.md",
        "E5_QA_REVIEWER_EXECUTIVE_BRIEF.md",
        "E5_QA_NEXT_RESEARCH_DECISION.md",
        "E5_QA_TARGETED_AND_REGRESSION_TEST_REPORT.md",
        "E5_QA_PROVENANCE_MANIFEST.json",
        "E5_QA_OWNER_REVIEW_GATE.md",
    ]
    assert all((OUT / x).is_file() and (OUT / x).stat().st_size > 0 for x in required)
    verification = json.loads((OUT / "E5_QA_PROVENANCE_VERIFICATION.json").read_text())
    assert verification["all_artifacts_match"] is True


def test_casebook_and_blinded_packet_bounded():
    casebook = (OUT / "E5_QA_ANSWER_CASEBOOK.md").read_text()
    blinded = (OUT / "E5_QA_BLINDED_ADJUDICATION_PACKET.md").read_text()
    assert "Selected unique qids: **100**" in casebook
    assert "No adjudication was performed here" in blinded
    assert "pplx_decider" not in blinded
    assert "jina35" not in blinded


def test_reserved_and_paid_gates():
    manifest = json.loads((OUT / "E5_QA_PROVENANCE_MANIFEST.json").read_text())
    assert manifest["reserved_e5f_accessed"] is False
    assert manifest["jev_calls"] == 0
    assert manifest["paid_api_calls"] == 0
