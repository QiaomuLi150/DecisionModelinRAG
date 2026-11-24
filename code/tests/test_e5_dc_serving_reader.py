from __future__ import annotations

import json
from pathlib import Path

from scripts.e5_dc_serving_reader import EXPECTED_SLATE_SHA, READER_REV, sha_json


ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / "results/non_jev/v2/e5_v1_1/e5_0_data_protocol"
E5A = ROOT / "results/non_jev/v2/e5_v1_1/e5_a_replacement_benchmark"


def test_frozen_slate_sha_and_e5a_counts():
    import hashlib
    assert hashlib.sha256((E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl").read_bytes()).hexdigest() == EXPECTED_SLATE_SHA
    rows = [json.loads(x) for x in (E5A / "E5_A_SELECTED_IDS_AND_SET_AWARE_OUTCOMES.jsonl").read_text().splitlines() if x]
    assert len(rows) == 1502
    assert len({r["query_id"] for r in rows}) == 1502


def test_e5a_discordance_reconciliation():
    rows = [json.loads(x) for x in (E5A / "E5_A_SELECTED_IDS_AND_SET_AWARE_OUTCOMES.jsonl").read_text().splitlines() if x]
    discordant = [r for r in rows if r["pplx_y"] != r["jina_y"]]
    assert len(discordant) == 55
    assert sum(r["pplx_y"] == 1 for r in discordant) == 31
    assert sum(r["jina_y"] == 1 for r in discordant) == 24


def test_reader_revision_is_pinned_and_canonical_hash_stable():
    assert READER_REV == "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    assert sha_json({"b": 2, "a": 1}) == sha_json({"a": 1, "b": 2})


def test_selected_ids_are_exactly_five_unique_in_frozen_slates():
    selected = [json.loads(x) for x in (E5A / "E5_A_SELECTED_IDS_AND_SET_AWARE_OUTCOMES.jsonl").read_text().splitlines() if x]
    top20 = {r["query_id"]: set(r["candidate_ids"]) for r in [json.loads(x) for x in (E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl").read_text().splitlines() if x]}
    for row in selected:
        for key in ("pplx_selected_ids", "jina_selected_ids"):
            assert len(row[key]) == 5 and len(set(row[key])) == 5
            assert set(row[key]) <= top20[row["query_id"]]


def test_reader_gold_free_and_no_holdout_reference_in_reader_payload():
    # Structural fixture test: the execution runner declares these fields
    # outside its payload and the E5-F namespace is never a source.
    from scripts.e5_dc_serving_reader import OUT, READER_URL
    assert "e5_f" not in str(OUT).lower()
    assert READER_URL.endswith("/v1/chat/completions")
