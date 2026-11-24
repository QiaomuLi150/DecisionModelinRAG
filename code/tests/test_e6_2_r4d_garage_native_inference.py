"""Offline integrity checks for the completed E6-2 R4D native run."""

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r4d_garage_native_inference"
INPUTS = OUT.parent / "e6_2_r4c_erratum_and_garage_freeze" / "E6_2_R4C_GARAGE_EXPLORATORY_INPUTS.jsonl"

MODELS = {
    "pplx_decider",
    "liquid_d1_3b",
    "startlux_decision_4b",
    "amos_1_small_2b",
    "jina_v3_5",
    "qwen3_reranker_4b",
    "bge_reranker_v2_m3",
    "qwen3_reranker_0_6b",
}


def read_json(name):
    return json.loads((OUT / name).read_text())


def selection_rows():
    with (OUT / "E6_2_R4D_NATIVE_SELECTIONS.jsonl").open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_premanifest_freezes_exact_r4d_population_and_label_blind_contract():
    manifest = read_json("E6_2_R4D_PREINFERENCE_MANIFEST.json")
    population = manifest["input_population"]
    assert population["scheduled_questions"] == 1397
    assert population["models"] == 8
    assert population["scheduled_attempts"] == 11176
    assert population["candidate_k"] == 15
    assert population["selected_k"] == 5
    isolation = manifest["label_isolation"]
    assert isolation["annotation_arrays_loaded"] is False
    assert isolation["reserved_content_loaded"] is False
    assert isolation["quality_metrics"] is False
    assert isolation["jev_calls"] == 0
    assert isolation["paid_api_calls"] == 0


def test_all_eight_model_attempts_are_complete_and_quality_blind():
    with (OUT / "E6_2_R4D_MODEL_ATTEMPT_ACCOUNTING.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["model"] for row in rows} == MODELS
    assert all(int(row["scheduled"]) == 1397 for row in rows)
    assert all(int(row["attempted"]) == int(row["valid"]) == 1397 for row in rows)
    assert all(int(row["failed"]) == int(row["unexecuted"]) == 0 for row in rows)
    assert all(row["quality_scoring"] == "NOT_RUN" for row in rows)


def test_registry_has_one_valid_attempt_per_model_and_sample():
    rows = selection_rows()
    assert len(rows) == 11176
    keys = [(row["model"], row["sample_id"]) for row in rows]
    assert len(keys) == len(set(keys))
    assert Counter(row["model"] for row in rows) == {model: 1397 for model in MODELS}
    assert all(row["status"] == "ATTEMPTED_VALID" for row in rows)


def test_every_selection_is_exactly_five_unique_ids_from_its_frozen_fifteen():
    rows = selection_rows()
    assert all(len(row["candidate_ids"]) == 15 for row in rows)
    assert all(len(set(row["candidate_ids"])) == 15 for row in rows)
    assert all(len(row["selected_ids"]) == 5 for row in rows)
    assert all(len(set(row["selected_ids"])) == 5 for row in rows)
    assert all(set(row["selected_ids"]).issubset(set(row["candidate_ids"])) for row in rows)
    assert all(len(row["scores"]) == 15 for row in rows)


def test_frozen_input_projection_contains_no_annotation_or_reserved_fields():
    forbidden = {
        "evidence_relevant",
        "evidence_correct",
        "evidence_cited",
        "answer_generate",
        "answer_validate",
        "gold",
        "label",
        "labels",
        "reference_answer",
    }
    rows = [json.loads(line) for line in INPUTS.open() if line.strip()]
    assert len(rows) == 1397
    assert all(not forbidden.intersection(row) for row in rows)
    assert all(not forbidden.intersection(candidate) for row in rows for candidate in row["candidates"])


def test_gate_and_forward_accounting_preserve_unmeasured_values():
    gate = read_json("E6_2_R4D_R5_READINESS_GATE.json")
    assert gate["gate"] == "E6_2_R4D_SELECTIONS_FROZEN_READY"
    assert gate["native_selections_frozen"] is True
    assert gate["quality_scoring_started"] is False
    assert gate["reserved_accessed"] is False
    assert gate["jev_calls"] == gate["paid_api_calls"] == 0
    with (OUT / "E6_2_R4D_FORWARD_ACCOUNTING.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["model"] for row in rows} == MODELS
    assert all(int(row["valid_forward_count_not_measured"]) >= 0 for row in rows)
    assert all(row["failed_forward_count"] == "NOT_MEASURED" for row in rows)
    assert all("NOT_MEASURED" in row["method"] for row in rows)
    by_model = {row["model"]: row for row in rows}
    assert int(by_model["startlux_decision_4b"]["valid_forward_count_not_measured"]) == 375
    assert int(by_model["amos_1_small_2b"]["valid_forward_count_not_measured"]) == 1397
    assert int(by_model["jina_v3_5"]["observed_physical_forwards"]) == 1397
    assert int(by_model["qwen3_reranker_4b"]["observed_physical_forwards"]) == 2794

