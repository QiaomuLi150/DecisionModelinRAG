"""Offline integrity tests for the completed E6-2 R2 selector bundle."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r2_selector_inference"


def rows(name):
    return [json.loads(line) for line in (OUT / name).read_text().splitlines() if line.strip()]


def test_r2_gate_and_attempt_accounting():
    gate = json.loads((OUT / "E6_2_R2_R3_READINESS_GATE.json").read_text())
    assert gate["gate"] == "E6_2_R2_SELECTIONS_FROZEN_WITH_EXCLUSIONS"
    assert gate["scheduled_attempts"] == 2216
    assert gate["actual_attempts"] == 1939
    assert gate["exclusions"] == ["startlux_decision_4b", "amos_1_small_2b"]
    assert gate["qrels_accessed"] is False
    assert gate["reserved_content_accessed"] is False
    assert gate["quality_metrics"] is False


def test_r2_each_attempt_uses_frozen_slate_and_valid_top5():
    attempts = rows("E6_2_R2_NATIVE_SELECTIONS.jsonl")
    by_model = Counter(row["model"] for row in attempts)
    assert by_model == {model: 277 for model in by_model}
    assert len(attempts) == 1939
    assert all(row["status"] in {"VALID", "INFERENCE_ERROR"} for row in attempts)
    for row in attempts:
        assert row["candidate_slate_sha256"]
        if row["status"] == "VALID":
            assert len(row["selected_ids"]) == 5
            assert len(set(row["selected_ids"])) == 5
            assert set(row["selected_ids"]).issubset(set(row["candidate_ids"]))
            assert len(row["scores"]) == 20


def test_r2_candidate_views_are_identical_across_models():
    attempts = rows("E6_2_R2_NATIVE_SELECTIONS.jsonl")
    views = defaultdict(set)
    for row in attempts:
        views[row["query_id"]].add((row["candidate_slate_sha256"], tuple(row["candidate_ids"])))
    assert len(views) == 277
    assert all(len(value) == 1 for value in views.values())


def test_r2_failures_preserve_amos_context_limit_and_no_fallback():
    with (OUT / "E6_2_R2_FAILURES.csv").open(newline="") as handle:
        failures = list(csv.DictReader(handle))
    assert len(failures) == 83
    assert {row["model"] for row in failures} == {"amos_1_small_2b"}
    assert {row["failure_class"] for row in failures} == {"context_limit"}
    assert all(row["physical_forward_count"] == "NOT_MEASURED" for row in failures)


def test_r2_no_quality_or_label_access_in_bundle():
    validation = json.loads((OUT / "E6_2_R2_INPUT_OUTPUT_VALIDATION.json").read_text())
    assert validation["qrels_accessed"] is False
    assert validation["reserved_content_accessed"] is False
    assert validation["quality_metrics"] is False
    assert validation["all_model_attempts_or_exclusions_accounted"] is True
    assert validation["all_valid_outputs_exactly_five_unique_ids"] is True
