import json
from pathlib import Path

from scripts.e6_0_readiness import (
    E5,
    E5A,
    E5DC,
    OUT,
    SLATE_SHA,
    official_normalize_answer,
    official_token_f1,
    sha_file,
)


def test_official_qasper_normalization_and_f1():
    assert official_normalize_answer("The 10%,") == "10"
    assert official_token_f1("the quick fox", "quick fox") == 1.0
    assert official_token_f1("yes", "no") == 0.0
    assert official_token_f1("", "Unanswerable") == 0.0


def test_frozen_e5_population_and_slate():
    assert sha_file(E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl") == SLATE_SHA
    rows = [json.loads(x) for x in (E5A / "E5_A_SELECTED_IDS_AND_SET_AWARE_OUTCOMES.jsonl").read_text().splitlines()]
    assert len(rows) == 1502
    assert len({x["query_id"] for x in rows}) == 1502


def test_existing_reader_accounting_is_preserved():
    manifest = json.loads((E5DC / "E5_DC_EXECUTION_MANIFEST_FROZEN.json").read_text())
    rows = [json.loads(x) for x in (E5DC / "E5_C_READER_INPUTS_AND_OUTPUTS.jsonl").read_text().splitlines()]
    assert len(manifest["cohorts"]["representative_qids"]) == 200
    assert len(manifest["cohorts"]["diagnostic_qids"]) == 55
    assert len(set(manifest["cohorts"]["representative_qids"]) & set(manifest["cohorts"]["diagnostic_qids"])) == 7
    assert len(rows) == 496
    assert all(not x.get("gold_in_reader_input") for x in rows)


def test_required_e6_outputs_exist_after_runner():
    required = [
        "E6_0_QASPER_ANSWER_F1_PER_QID.csv",
        "E6_0_QASPER_ANSWER_F1_SCORECARD.csv",
        "E6_0_DATASET_ADMISSION_DECISION.json",
        "E6_0_SOURCE_DISJOINT_SPLIT_MANIFEST.json",
        "E6_0_EIGHT_MODEL_VERSION_LICENSE_REGISTRY.csv",
        "E6_0_NATIVE_TOP5_CONTRACTS.json",
        "E6_0_EVALUATION_PROTOCOL_V1_FROZEN.md",
        "E6_0_PROVENANCE_MANIFEST.json",
        "E6_0_OWNER_REVIEW_GATE.md",
    ]
    assert all((OUT / name).is_file() and (OUT / name).stat().st_size > 0 for name in required)


def test_e6_gate_and_locked_holdout():
    decision = json.loads((OUT / "E6_0_DATASET_ADMISSION_DECISION.json").read_text())
    assert decision["overall"] == "E6_0_ENDPOINT_AMENDMENT_REQUIRED"
    assert (OUT / "E6_F_STATUS_LOCKED.txt").read_text().startswith("E6-F STATUS: LOCKED")


def test_e6_no_inference_or_paid_calls():
    manifest = json.loads((OUT / "E6_0_PROVENANCE_MANIFEST.json").read_text())
    assert manifest["no_model_inference"] is True
    assert manifest["gpu_calls"] == 0
    assert manifest["jev_calls"] == 0
    assert manifest["paid_api_calls"] == 0
    assert manifest["sealed_holdout_accessed"] is False


def test_registry_has_exactly_eight_primary_models():
    import csv
    rows = list(csv.DictReader((OUT / "E6_0_EIGHT_MODEL_VERSION_LICENSE_REGISTRY.csv").open()))
    assert len(rows) == 8
    assert {row["policy"] for row in rows} == {"pplx_decider", "liquid_d1_3b", "startlux_decision_4b", "amos_1_small_2b", "jina35", "qwen3_reranker_4b", "bge_reranker_v2_m3", "qwen3_reranker_0_6b"}


def test_provenance_self_excluded_and_matches():
    verification = json.loads((OUT / "E6_0_PROVENANCE_VERIFICATION.json").read_text())
    assert verification["all_new_artifacts_match"] is True
    assert all("E6_0_PROVENANCE_MANIFEST.json" not in row["path"] for row in verification["checks"])
