import importlib.util
import math
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/e6_2_r3_scifact_quality.py"
spec = importlib.util.spec_from_file_location("e6_2_r3_quality", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_binary_ndcg_and_recall():
    n, ideal, top, total = mod.ndcg5(["a", "b", "c", "d", "e"], {"e": 1, "a": 1})
    assert top == 2 and total == 2 and ideal > 0
    assert round(n, 8) == round((1.0 + 1.0 / math.log2(6)) / ideal, 8)


def test_zero_idcg_is_explicit_zero():
    assert mod.ndcg5(["a", "b"], {}) == (0.0, 0, 0, 0)


def test_unjudged_is_not_positive():
    row = mod.metric_row(["unjudged", "gold"], {"gold": 1}, "VALID")
    assert row["judged_relevant_top5"] == 1
    assert row["total_judged_positive"] == 1
    assert row["top20_feasible"] == 1


def test_bootstrap_is_deterministic():
    assert mod.bootstrap([0.0, 0.5, 1.0], seed=731, n=100) == mod.bootstrap([0.0, 0.5, 1.0], seed=731, n=100)


def test_failed_selection_has_zero_operational_utility():
    row = mod.metric_row([], {"gold": 1}, "INFERENCE_ERROR", "context_limit")
    assert row["ndcg5"] == 0.0


def test_pairwise_outcome_direction():
    rows = [
        {"model": "a", "query_id": "q", "source_group_id": "g", "operational_ndcg5": 1.0, "status": "VALID", "hit5": 1},
        {"model": "b", "query_id": "q", "source_group_id": "g", "operational_ndcg5": 0.0, "status": "VALID", "hit5": 0},
    ]
    out, vals = mod.pairwise(rows, ["q"], "a", "b")
    assert vals == [1.0] and out[0]["paired_outcome"] == "a_only"
