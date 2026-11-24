import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v11_track_c_r2_repair import (  # noqa: E402
    COHORT,
    INPUT_CONTRACT_VERSION,
    INPUT_MANIFEST,
    SCHEDULE,
    load_jsonl,
    resolve_questions,
    validate_outputs,
)
from v11_track_c_serving import (  # noqa: E402
    build_tasks,
    validate_serving_input_manifest,
)


def test_r2_manifest_has_exact_nonempty_canonical_questions():
    cohort = load_jsonl(COHORT)
    input_rows = load_jsonl(INPUT_MANIFEST)
    by_id = validate_serving_input_manifest(input_rows, cohort)
    assert len(by_id) == 100
    assert all(row["question"].strip() for row in input_rows)
    assert all(row["input_contract_version"] == INPUT_CONTRACT_VERSION for row in input_rows)
    assert len({row["question_sha256"] for row in input_rows}) == 100


def test_r2_schedule_attaches_the_right_question_hash_to_all_tasks():
    cohort = load_jsonl(COHORT)
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    input_rows = load_jsonl(INPUT_MANIFEST)
    inputs = validate_serving_input_manifest(input_rows, cohort)
    tasks = build_tasks(cohort, schedule, inputs)
    assert len(tasks) == 1200
    assert len({(task["query_id"], task["policy_id"]) for task in tasks}) == 1200
    assert all(task["question"].strip() for task in tasks)
    assert all(task["question_sha256"] for task in tasks)


def test_missing_question_fails_closed():
    cohort = load_jsonl(COHORT)
    input_rows = load_jsonl(INPUT_MANIFEST)
    broken = copy.deepcopy(input_rows)
    broken[0].pop("question")
    with pytest.raises(ValueError):
        validate_serving_input_manifest(broken, cohort)


def test_whitespace_question_fails_closed():
    cohort = load_jsonl(COHORT)
    input_rows = load_jsonl(INPUT_MANIFEST)
    broken = copy.deepcopy(input_rows)
    broken[0]["question"] = "   \n"
    with pytest.raises(ValueError):
        validate_serving_input_manifest(broken, cohort)


def test_wrong_query_mapping_fails_closed():
    cohort = load_jsonl(COHORT)
    input_rows = load_jsonl(INPUT_MANIFEST)
    broken = copy.deepcopy(input_rows)
    broken[0]["query_id"] = "not-a-frozen-query-id"
    with pytest.raises(ValueError):
        validate_serving_input_manifest(broken, cohort)


def test_wrong_dataset_association_fails_closed():
    cohort = load_jsonl(COHORT)
    input_rows = load_jsonl(INPUT_MANIFEST)
    broken = copy.deepcopy(input_rows)
    broken[0]["dataset"] = "wrong-dataset"
    with pytest.raises(ValueError):
        validate_serving_input_manifest(broken, cohort)


def test_question_hash_mismatch_fails_closed():
    cohort = load_jsonl(COHORT)
    input_rows = load_jsonl(INPUT_MANIFEST)
    broken = copy.deepcopy(input_rows)
    broken[0]["question_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_serving_input_manifest(broken, cohort)


def test_repair_resolution_is_deterministic_and_gold_free():
    cohort = load_jsonl(COHORT)
    first, first_sources = resolve_questions(cohort)
    second, second_sources = resolve_questions(cohort)
    assert first == second
    assert first_sources == second_sources
    forbidden = {"expected_doc_ids", "expected_internal_doc_ids", "reference_answer", "gold"}
    assert not forbidden.intersection(first[0])


def test_repair_output_contract_and_schedule_are_unchanged():
    cohort = load_jsonl(COHORT)
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    input_rows, _ = resolve_questions(cohort)
    checks = validate_outputs(input_rows, cohort, schedule)
    assert checks["cohort_count"] == 100
    assert checks["scheduled_task_count"] == 1200
    assert checks["unique_query_policy_pairs"] == 1200
    assert checks["schedule_policy_set_unchanged"]
