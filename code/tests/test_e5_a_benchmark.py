from __future__ import annotations

import numpy as np

from scripts.e5_a_benchmark import outcome, percentile_ci, sha_json, valid_selection


def test_set_aware_alternative_supports_do_not_union():
    assert outcome(["a", "b", "x", "y", "z"], [["a", "b"], ["c", "d"]]) == 1
    assert outcome(["a", "c", "x", "y", "z"], [["a", "b"], ["c", "d"]]) == 0


def test_retrieval_miss_is_failure():
    assert outcome(["x", "y"], [["a"], ["b"]]) == 0


def test_output_requires_exact_unique_top5_inside_slate():
    row = {"status": "VALID", "selected_ids": ["a", "b", "c", "d", "e"]}
    assert valid_selection(row, {"a", "b", "c", "d", "e", "f"})
    assert not valid_selection({**row, "selected_ids": ["a", "b", "c", "d", "a"]}, {"a", "b", "c", "d", "e", "f"})
    assert not valid_selection({**row, "selected_ids": ["a", "b", "c", "d", "x"]}, {"a", "b", "c", "d", "e", "f"})


def test_group_bootstrap_ci_is_deterministic_shape():
    values = np.array([-0.2, 0.0, 0.4])
    rng = np.random.default_rng(731)
    boot = np.array([np.mean(values[rng.integers(0, len(values), size=len(values))]) for _ in range(5000)])
    ci = percentile_ci(boot)
    assert len(ci) == 2 and ci[0] <= ci[1]


def test_canonical_hash_is_stable():
    assert sha_json({"b": 2, "a": 1}) == sha_json({"a": 1, "b": 2})
