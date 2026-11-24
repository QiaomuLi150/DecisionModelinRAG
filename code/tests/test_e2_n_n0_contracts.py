from __future__ import annotations

import math

import pytest

from decisionrag.e2_n_contracts import (
    ADAPTER_REGISTRY,
    CONTRACT_VERSION,
    canonical_hash,
    make_trace_template,
    registry_snapshot,
    select_top5,
    validate_scores,
    validate_slate,
)


IDS = [f"d{i:02d}" for i in range(20)]


def test_registry_is_cpu_only_and_versioned():
    snap = registry_snapshot()
    assert snap["contract_version"] == CONTRACT_VERSION
    assert snap["inference_enabled"] is False
    assert len(ADAPTER_REGISTRY) == 4


def test_slate_rejects_empty_questions_and_duplicate_ids():
    with pytest.raises(ValueError):
        validate_slate("   ", IDS)
    with pytest.raises(ValueError):
        validate_slate("q", IDS[:-1] + [IDS[-2]])


def test_slate_requires_exactly_twenty_unique_nonempty_ids():
    with pytest.raises(ValueError):
        validate_slate("q", IDS[:19])
    with pytest.raises(ValueError):
        validate_slate("q", IDS[:-1] + [""])


def test_scores_must_align_exactly_and_be_finite():
    with pytest.raises(ValueError):
        validate_scores(IDS, {x: 1.0 for x in IDS[:-1]})
    bad = {x: 1.0 for x in IDS}
    bad[IDS[0]] = math.nan
    with pytest.raises(ValueError):
        validate_scores(IDS, bad)


def test_score_direction_and_stable_top5():
    scores = {x: float(20 - i) for i, x in enumerate(IDS)}
    assert select_top5(IDS, scores) == IDS[:5]
    ties = {x: 1.0 for x in IDS}
    assert select_top5(IDS, ties) == IDS[:5]


def test_lower_is_more_relevant_is_explicit():
    scores = {x: float(i) for i, x in enumerate(IDS)}
    assert select_top5(IDS, scores, score_direction="lower_is_more_relevant") == IDS[:5]
    with pytest.raises(ValueError):
        validate_scores(IDS, scores, score_direction="unknown")


def test_trace_template_requires_physical_fields_without_fabricating_them():
    row = make_trace_template(model_id="x", revision="r", query_id="q", candidate_ids=IDS, actual_input={"q": "x"}, score_contract="INDEPENDENT_RELEVANCE")
    assert row["actual_model_input_sha256"] == canonical_hash({"q": "x"})
    assert row["physical_forward_calls"] is None
    assert row["raw_response_sha256"] is None
    assert row["status"] == "NOT_EXECUTED"
    assert row["gold_in_inference_input"] is False
