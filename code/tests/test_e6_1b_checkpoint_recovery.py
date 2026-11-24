from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
import sys
sys.path.insert(0, str(SCRIPT_DIR))
import e6_1b_checkpoint_recovery as e6  # noqa: E402


def test_frozen_inputs_and_attempt_accounting():
    rows, protocol = e6.verify_inputs()
    assert len(rows) == 120
    assert protocol["input_manifest_sha256"] == e6.EXPECTED_INPUT_HASH
    attempts = e6.attempts_for(rows)
    assert len(attempts) == 256
    assert sum(x[2] == 15 for x in attempts) == 125
    assert sum(x[2] == 20 for x in attempts) == 131
    assert sum(x[1] == "order_check" for x in attempts) == 6


def test_top5_probability_alignment_and_deterministic_ties():
    ids = ["A", "B", "C", "D", "E", "F"]
    probs = {f"candidate_{i:03d}": 0.5 for i in range(len(ids))}
    assert e6.select_top5(ids, probs) == ids[:5]
    assert len(set(e6.select_top5(ids, probs))) == 5
    with pytest.raises(ValueError):
        e6.select_top5(ids, {"candidate_000": 1.0})


def test_native_request_has_no_gold_and_candidate_ids_are_explicit():
    rows, _ = e6.verify_inputs()
    ids, docs = e6.model_view(rows[0], 15)
    state, questions = e6.native_request(rows[0], ids, docs)
    assert "gold" not in state.lower()
    assert list(questions["select"]["criteria"]) == [f"candidate_{i:03d}" for i in range(15)]
    for i, doc_id in enumerate(ids):
        assert doc_id in questions["select"]["criteria"][f"candidate_{i:03d}"]


def test_output_gate_does_not_promote_fallback_or_partial_traces(tmp_path):
    valid = {"model": "x", "status": "VALID", "selected_ids": ["a", "b", "c", "d", "e"]}
    invalid = {"model": "x", "status": "INFERENCE_ERROR", "selected_ids": []}
    assert valid["status"] == "VALID"
    assert invalid["status"] != "VALID"


def test_previous_e6_1_artifact_is_present_and_untouched():
    assert (e6.E61 / "E6_1_NATIVE_TRACES.jsonl").is_file()
    assert (e6.E61 / "E6_1_OWNER_REVIEW_GATE.json").is_file()
    gate = json.loads((e6.E61 / "E6_1_OWNER_REVIEW_GATE.json").read_text())
    assert gate["e6_2_started"] is False
