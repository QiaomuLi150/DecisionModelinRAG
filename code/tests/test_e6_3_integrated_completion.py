import csv
import json
import ast
from pathlib import Path

import pytest

from scripts.e6_3_integrated_completion import (
    EXPECTED_SLATE_SHA,
    GPU_UUIDS,
    MODELS,
    R1_COHORT,
    READER,
    SERVE,
    SEED,
    bootstrap,
    exact_match,
    normalize_text,
    token_f1,
)


ROOT = Path(__file__).resolve().parents[1]


def _csv(name):
    with (SERVE / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_reader_union_and_primary_pair_are_frozen():
    cohort = json.loads(R1_COHORT.read_text())
    union = set().union(*[set(x) for x in cohort["cohorts"].values()])
    assert {k: len(v) for k, v in cohort["cohorts"].items()} == {
        "representative_primary": 200,
        "disagreement_diagnostic": 60,
        "no_positive_diagnostic": 64,
    }
    assert len(union) == 313
    rows = list(csv.DictReader((READER / "E6_3_READER_PER_CASE_RESULTS.csv").open()))
    assert len(rows) == 1565
    assert sum(r["primary_representative"] == "1" and r["policy"] == "pplx_decider" for r in rows) == 200
    assert sum(r["primary_representative"] == "1" and r["policy"] == "jina_v3_5" for r in rows) == 200


def test_reference_metric_toy_cases():
    assert normalize_text("A [cite_1], B!") == "a b"
    assert exact_match("The answer [cite_2]", "the answer") == 1.0
    assert token_f1("red blue blue", "blue red") == pytest.approx(0.8)
    assert token_f1("unseen", "reference") == 0.0
    assert token_f1("anything", "") is None


def test_bootstrap_is_deterministic_and_paired():
    a = bootstrap([0.0, 0.5, 1.0], seed=SEED, n=5000)
    b = bootstrap([0.0, 0.5, 1.0], seed=SEED, n=5000)
    assert a == b
    assert a["n"] == 3
    assert len(a["ci95"]) == 2


def test_serving_has_complete_timed_accounting():
    rows = _csv("E6_3_SERVING_PER_REQUEST.csv")
    timed = [r for r in rows if r["warmup"] == "0" and r["phase"].startswith("timed")]
    warm = [r for r in rows if r["warmup"] == "1"]
    assert len(rows) == 3216
    assert len(timed) == 3200
    assert len(warm) == 16
    assert all(r["status"] == "VALID" for r in rows)
    assert set(r["gpu_uuid"] for r in rows) == {GPU_UUIDS[1]}
    for model in MODELS:
        for workload in ("scifact", "garage"):
            subset = [r for r in timed if r["model"] == model and r["workload"] == workload]
            assert len(subset) == 200
            assert all(len(ast.literal_eval(r["selected_ids"])) == 5 for r in subset)


def test_serving_scorecard_has_two_workloads_per_model():
    rows = _csv("E6_3_SERVING_MODEL_SCORECARD.csv")
    assert len(rows) == 16
    assert {(r["model"], r["workload"]) for r in rows} == {(m, w) for m in MODELS for w in ("scifact", "garage")}
    assert all(r["attempts"] == "200" and r["successful"] == "200" and r["failures"] == "0" for r in rows)
    assert all(r["sustainable_qps"] == "NOT_MEASURED" for r in rows)


def test_primary_result_and_gate_are_qualified():
    stats = json.loads((READER / "E6_3_READER_PRIMARY_STATISTICS.json").read_text())
    assert stats["representative"]["paired_n"] == 200
    assert stats["representative"]["pplx_minus_jina"]["seed"] == SEED
    gate = json.loads((SERVE.parent / "scientific_closure/E6_3_FINAL_OWNER_REVIEW_GATE.json").read_text())
    assert gate["gate"] == "E6_3_INTEGRATED_RESULTS_READY_WITH_LIMITATIONS"
    assert gate["reader_factual_correctness"] == "NOT_VERIFIED"
    assert gate["sustainable_qps"] == "NOT_MEASURED"
    assert gate["jev_calls"] == 0
    assert gate["paid_api_calls"] == 0


def test_frozen_contract_and_no_holdout_claims():
    receipt = json.loads((SERVE.parent / "E6_3_PREEXECUTION_FREEZE_RECEIPT.json").read_text())
    assert receipt["historical_authentication"]["expected_original_frozen_slate_contract_sha256"] == EXPECTED_SLATE_SHA
    assert receipt["historical_authentication"]["reserved_access"] == "not performed by this script"
    assert receipt["reader_calls_authorized"] is False
    assert receipt["holdout_access"] is False
