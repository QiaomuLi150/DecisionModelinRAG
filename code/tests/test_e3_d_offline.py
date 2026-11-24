from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e3_v1/e3_d_external"
GOV = ROOT / "results/non_jev/v2/e3_v1/e3_0_governance"


def jl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_governance_partition_is_reserved_disjoint():
    m = json.loads((GOV / "E3_0_RESERVED_PARTITIONS_MANIFEST.json").read_text())
    assert m["integrity"]["status"] == "PASS"
    assert m["integrity"]["reserved_exploratory_overlap_count"] == 0


def test_frozen_candidate_contract_is_identical_top20():
    rows = jl(OUT / "E3_D_FIRST_STAGE_CANDIDATES.jsonl")
    assert len(rows) == 1009
    assert len({r["query_id"] for r in rows}) == 1009
    assert all(len(r["candidate_ids"]) == 20 and len(set(r["candidate_ids"])) == 20 for r in rows)
    assert all(r["gold_accessed"] is False for r in rows)


def test_selector_rows_have_aligned_scores_or_explicit_failure():
    rows = jl(OUT / "E3_D_NATIVE_SELECTION_TRACES.jsonl")
    full = [r for r in rows if r.get("stage") == "full"]
    assert len(full) >= 1009
    for row in full:
        if row.get("status") == "VALID":
            assert len(row["candidate_ids"]) == 20
            assert len(row["scores"]) == 20
            assert len(row["selected_ids"]) == 5
            assert len(set(row["selected_ids"])) == 5
            assert set(row["selected_ids"]).issubset(set(row["candidate_ids"]))
            assert row.get("gold_in_inference_input") is False


def test_exploratory_allowlist_has_no_reserved_assignment():
    rows = jl(OUT / "E3_D_EXPLORATORY_METADATA.jsonl")
    assert rows
    assert all(row.get("assignment") == "exploratory" for row in rows)
    assert len({row["query_id"] for row in rows}) == len(rows)


def test_reader_records_are_gold_free_and_hashable():
    path = OUT / "E3_D_FIXED_READER_OUTPUTS.jsonl"
    assert path.is_file()
    rows = jl(path)
    assert rows
    assert all(row.get("gold_in_reader_input") is False for row in rows)
    assert all(row.get("actual_model_input_sha256") for row in rows if row.get("status") == "ok")


def test_required_e3d_artifacts_exist():
    required = [
        "E3_D_RETRIEVAL_FEASIBILITY_AND_DENOMINATORS.csv",
        "E3_D_EVIDENCE_QUALITY_SCORECARD.csv",
        "E3_D_PAIRED_BOOTSTRAP_AND_FOUR_CELL.csv",
        "E3_D_REFERENCE_VALIDATION_RECORDS.jsonl",
        "E3_D_EVIDENCE_TO_ANSWER_TRANSITIONS.csv",
        "E3_D_INFERENCE_WORK_AND_LATENCY.csv",
        "E3_D_PROVENANCE_MANIFEST.json",
        "E3_D_FINAL_SCIENTIFIC_REPORT.md",
        "E3_D_OWNER_REVIEW_GATE.md",
    ]
    assert all((OUT / name).is_file() for name in required)
