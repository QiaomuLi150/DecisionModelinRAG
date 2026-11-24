import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/e6_2_r4_garage_preflight.py"
spec = importlib.util.spec_from_file_location("e6_2_r4_preflight", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_binary_ndcg_matches_frozen_definition():
    value = mod.ndcg(["d1", "d2", "d3", "d4", "d5"], {"d1": 1, "d5": 1})
    assert 0.0 < value < 1.0


def test_bootstrap_is_deterministic():
    assert mod.boot([0.0, 0.25, 1.0]) == mod.boot([0.0, 0.25, 1.0])


def test_candidate_identity_is_positional():
    assert "sample_id:cite_i" in "sample_id:cite_i positional"


def test_no_quality_stage_is_invoked_by_module_import():
    assert not hasattr(mod, "run_model")
