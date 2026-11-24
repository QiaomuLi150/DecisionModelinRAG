"""Offline integrity checks for the E6-3 R1 frozen reader package."""

import json
from pathlib import Path

from scripts.e6_3_r1_paired_reader_inference import OUT, parse_response


def read_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_jsonl(name):
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_cohort_freeze_is_deterministic_and_deduplicated():
    freeze = read_json("E6_3_R1_COHORT_FREEZE_MANIFEST.json")
    assert freeze["seed"] == 731
    assert freeze["counts"] == {
        "disagreement": 60,
        "no_positive": 64,
        "nominal_cohort_slots": 1620,
        "representative": 200,
        "unique_logical_policy_question_slots": 1565,
        "unique_union": 313,
    }
    assert len(freeze["membership_flags"]) == 313
    assert len(set(freeze["cohorts"]["representative_primary"])) == 200


def test_reader_inputs_have_five_frozen_passages_and_no_gold_fields():
    rows = read_jsonl("E6_3_R1_NATIVE_EVIDENCE_INPUTS.jsonl")
    assert len(rows) == 1565
    assert len({row["request_id"] for row in rows}) == 1565
    forbidden = {"evidence_relevant", "evidence_correct", "evidence_cited", "answer_generate", "answer_validate", "reference_answer"}
    for row in rows:
        assert len(row["selected_ids"]) == 5
        assert len(set(row["selected_ids"])) == 5
        assert not forbidden.intersection(row)
        assert not any(field in passage for passage in row["selected_passages"] for field in forbidden)
        for field in forbidden:
            assert field not in row["prompt"]


def test_policy_roster_and_order_are_frozen():
    freeze = read_json("E6_3_R1_COHORT_FREEZE_MANIFEST.json")
    assert freeze["policy_order"] == ["pplx_decider", "jina_v3_5", "startlux_decision_4b", "qwen3_reranker_0_6b", "original_fixed_pool_prefix5"]
    rows = read_jsonl("E6_3_R1_NATIVE_EVIDENCE_INPUTS.jsonl")
    assert {row["policy"] for row in rows} == set(freeze["policy_order"])


def test_parser_accepts_frozen_json_and_tagged_forms_without_inventing_citations():
    mapping = {"C1": "passage-1", "C2": "passage-2", "C3": "passage-3", "C4": "passage-4", "C5": "passage-5"}
    assert parse_response('{"answer":"x","citations":["C1"],"abstain":false}', mapping)["citation_ids"] == ["passage-1"]
    assert parse_response("ANSWER: x\nCITATIONS: C2,C3\nABSTAIN: false", mapping)["citation_ids"] == ["passage-2", "passage-3"]
    assert parse_response("ANSWER: x\nCITATIONS: C9\nABSTAIN: false", mapping)["citation_ids"] == []


def test_parser_preserves_explicit_abstention():
    parsed = parse_response("INSUFFICIENT_EVIDENCE", {"C1": "p1"})
    assert parsed["answer"] == "INSUFFICIENT_EVIDENCE"
    assert parsed["abstain"] is True
    assert parsed["citation_ids"] == []


def test_reader_contract_freezes_authenticated_model_and_no_scoring():
    auth = read_json("E6_3_R1_HISTORICAL_INPUT_AUTHENTICATION.json")
    assert auth["reader_model"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert auth["reader_revision"] == "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    assert auth["no_answer_scoring"] is True
    assert auth["reserved_content_loaded_for_inference"] is False


def test_frozen_outputs_preserve_one_logical_row_per_policy_question():
    raw = read_jsonl("E6_3_R1_READER_RAW_OUTPUTS.jsonl")
    parsed = read_jsonl("E6_3_R1_READER_PARSED_OUTPUTS.jsonl")
    assert len(raw) == len(parsed) == 1565
    assert len({row["request_id"] for row in raw}) == 1565
    assert all(row["cache_reuse"] is False for row in raw)
    assert all(row["gold_in_reader_input"] is False for row in raw)
    assert all(row["reference_answer_in_reader_input"] is False for row in raw)


def test_r2_gate_has_no_answer_quality_or_reserved_access():
    gate = read_json("E6_3_R1_R2_RELEASE_GATE.json")
    assert gate["reader_outputs_frozen"] is True
    assert gate["answer_scoring_started"] is False
    assert gate["reserved_accessed"] is False
    assert gate["jev_calls"] == gate["paid_api_calls"] == 0
