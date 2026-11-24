"""Offline integrity and arithmetic checks for E6-2 R5 GaRAGe scoring."""

import csv
import json
import math
from pathlib import Path

from scripts.e6_2_r5_garage_quality_evaluation import MODELS, OUT, score_one


def read_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_csv(name):
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_r5_gate_and_population_are_finalized_without_new_calls():
    gate = read_json("E6_2_R5_E6_3_HANDOFF_GATE.json")
    assert gate["gate"] == "E6_2_R5_SCIENTIFIC_RESULTS_READY_WITH_LIMITATIONS"
    assert gate["r5_quality_results_ready"] is True
    assert gate["e6_3_serving_authorized"] is False
    assert gate["new_inference_in_r5"] == 0
    assert gate["reserved_accessed"] is False
    assert gate["jev_calls"] == gate["paid_api_calls"] == 0


def test_primary_statistics_use_the_complete_exploratory_denominator():
    stats = read_json("E6_2_R5_PRIMARY_STATISTICS.json")
    assert stats["n"] == 1397
    assert stats["resamples"] == 5000
    assert stats["seed"] == 731
    assert stats["unresolved_annotation_rows"] == 1
    assert stats["no_positive_n"] >= 0
    assert stats["pplx_only_wins"] + stats["jina_only_wins"] + stats["ties"] == 1397


def test_scorecard_contains_all_eight_models_and_prefix_control():
    rows = read_csv("E6_2_R5_MODEL_QUALITY_SCORECARD.csv")
    assert len(rows) == 9
    assert {row["model"] for row in rows} == set(MODELS) | {"ORIGINAL_FIXED_POOL_PREFIX5"}
    assert all(int(row["n_operational"]) == 1397 for row in rows)
    assert all(int(row["valid_selection_n"]) == 1397 for row in rows)


def test_per_sample_results_preserve_frozen_candidate_identity_and_counts():
    rows = read_csv("E6_2_R5_PER_SAMPLE_RESULTS.csv")
    assert len(rows) == 1397 * 9
    for row in rows:
        candidates = row["candidate_ids"].split("|")
        selected = row["selected_ids"].split("|")
        assert len(candidates) == 15 and len(set(candidates)) == 15
        assert len(selected) == 5 and len(set(selected)) == 5
        assert set(selected).issubset(set(candidates))


def test_exploratory_projection_excludes_reserved_rows():
    projection = read_json("E6_2_R5_EXPLORATORY_LABEL_PROJECTION.json")
    assert projection["rows_exported"] == 1397
    assert projection["reserved_rows_not_exported"] == 466
    assert projection["cal_rows_not_exported"] == 466
    assert all(row["partition"] == "EXPLORATORY" for row in projection["rows"])


def test_frozen_metric_arithmetic_covers_all_required_toy_cases():
    candidates = [f"c{i}" for i in range(1, 16)]
    all_yes = ["YES"] * 15
    all_no = ["NO"] * 15
    empty = [""] + ["NO"] * 14

    # all-positive and rank-1 cases
    assert score_one(candidates[:5], candidates, all_yes, all_no, all_no)["ndcg5"] == 1.0
    assert score_one(["c1", "c6", "c7", "c8", "c9"], candidates, all_yes, all_no, all_no)["ndcg5"] == 1.0
    # rank-5 singleton: 1 / log2(6)
    singleton = ["YES"] + ["NO"] * 14
    rank5 = score_one(["c2", "c3", "c4", "c5", "c1"], candidates, singleton, all_no, all_no)
    assert math.isclose(rank5["ndcg5"], 1.0 / math.log2(6), rel_tol=1e-12)
    # zero-positive and ties/no-gain cases
    zero = score_one(candidates[:5], candidates, all_no, all_no, all_no)
    assert zero["ndcg5"] == 0.0 and zero["no_positive"] == 1
    assert score_one(candidates[5:10], candidates, singleton, all_no, all_no)["ndcg5"] == 0.0
    # unresolved labels remain flagged, not silently relabeled
    unresolved = score_one(candidates[:5], candidates, empty, all_no, all_no)
    assert unresolved["unresolved_relevance_count"] == 1
    assert unresolved["annotation_status"] == "UNRESOLVED_LABEL_PRESENT"


def test_selection_registry_has_no_invalid_or_duplicate_model_sample_records():
    registry = read_json("E6_2_R5_COMBINED_SELECTION_REGISTRY.json")
    records = registry["records"]
    keys = [(row["model_id"], row["sample_id"]) for row in records]
    assert len(records) == 11176
    assert len(keys) == len(set(keys))
    assert {row["model_id"] for row in records} == set(MODELS)
    assert all(row["selection_status"] == "ATTEMPTED_VALID" for row in records)


def test_provenance_registers_outputs_and_zeroes_side_effect_counters():
    provenance = read_json("E6_2_R5_PROVENANCE_MANIFEST.json")
    assert provenance["gate"] == "E6_2_R5_SCIENTIFIC_RESULTS_READY_WITH_LIMITATIONS"
    assert provenance["native_inference_calls"] == 0
    assert provenance["quality_scoring_calls"] == 0
    assert provenance["retrieval_calls"] == 0
    assert provenance["reader_calls"] == 0
    assert provenance["gpu_calls"] == 0
    assert provenance["reserved_content_accessed"] is False
    assert provenance["reserved_labels_exported"] is False
    assert provenance["jev_calls"] == provenance["paid_api_calls"] == 0
    output_paths = {entry["path"] for entry in provenance["outputs"]}
    assert "results/non_jev/v2/e6_v1_0/e6_2_r5_garage_quality_evaluation/E6_2_R5_FINAL_SCIENTIFIC_REPORT.md" in output_paths
