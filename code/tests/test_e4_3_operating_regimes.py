from __future__ import annotations

import json
from pathlib import Path

from scripts import e4_3_operating_regimes as e4


OUT = e4.OUT


def test_frozen_cohort_is_unique_and_nonoverlapping_with_e4_2():
    rows = e4.read_jsonl(OUT / "E4_3_COHORT_AND_STRATA_MANIFEST.jsonl")
    previous = e4.read_json(OUT / "E4_3_FROZEN_EXPLORATORY_PROTOCOL.json").get("prior_qids", [])
    qids = [row["query_id"] for row in rows]
    assert len(qids) == len(set(qids)) == 380
    assert not set(qids) & set(previous)
    assert {row["cohort"] for row in rows} == {"representative", "diagnostic"}


def test_k10_is_exact_k20_prefix_and_top5_helper_is_stable():
    for row in e4.read_jsonl(OUT / "E4_3_COHORT_AND_STRATA_MANIFEST.jsonl"):
        assert row["k10_candidate_ids"] == row["k20_candidate_ids"][:10]
        assert len(row["k20_candidate_ids"]) == 20
        assert len(set(row["k20_candidate_ids"])) == 20
    ids = ["a", "b", "c", "d", "e", "f"]
    assert e4.stable_top5(ids, [1, 1, 0, 0, 0, -1]) == ["a", "b", "c", "d", "e"]


def test_native_protocol_has_no_label_inputs_and_has_all_six_policies():
    protocol = e4.read_json(OUT / "E4_3_FROZEN_EXPLORATORY_PROTOCOL.json")
    assert set(protocol["policies"]) == set(e4.SPECS)
    assert protocol["input_leakage_policy"] == {
        "gold_in_policy_input": False,
        "support_ids_in_policy_input": False,
        "reference_in_policy_input": False,
    }
    assert protocol["jev_calls"] == 0
    assert protocol["paid_api_calls"] == 0


def test_bootstrap_is_deterministic_and_handles_empty():
    assert e4.bootstrap_mean([]) == (None, None, None)
    assert e4.bootstrap_mean([0, 1, 1, 0], seed=731) == e4.bootstrap_mean([0, 1, 1, 0], seed=731)
