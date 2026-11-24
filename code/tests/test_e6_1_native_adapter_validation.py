import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_1_native_adapter_validation"


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def test_frozen_cal_manifest_is_unique_and_exactly_two_candidate_sizes():
    rows = load_jsonl(OUT / "E6_1_CAL_INPUTS.jsonl")
    assert len(rows) == 120
    assert len({row["query_id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["gold_in_input"] is False
        assert len(row["candidate_ids_top15"]) == 15
        assert len(row["candidate_ids_top20"]) == 20
        assert row["candidate_ids_top15"] == row["candidate_ids_top20"][:15]
        assert len(set(row["candidate_ids_top20"])) == 20
        assert len(row["candidates"]) == 20
        assert {x["doc_id"] for x in row["candidates"]} == set(row["candidate_ids_top20"])


def test_all_valid_traces_have_canonical_top5():
    rows = load_jsonl(OUT / "E6_1_NATIVE_TRACES.jsonl")
    valid = [row for row in rows if row.get("status") == "VALID"]
    assert len(valid) == 1536
    for row in valid:
        assert row["gold_in_input"] is False
        assert len(row["selected_ids"]) == 5
        assert len(set(row["selected_ids"])) == 5
        assert set(row["selected_ids"]).issubset(set(row["candidate_ids"]))
        assert len(row["scores"]) == row["candidate_k"]


def test_registry_contains_eight_models_and_preserves_missing_contracts():
    with (OUT / "E6_1_MODEL_CONTRACT_REGISTRY.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    by_model = {row["model"]: row for row in rows}
    assert by_model["startlux_decision_4b"]["contract_status"] == "BLOCKED_MISSING_CHECKPOINT"
    assert by_model["amos_1_small_2b"]["contract_status"] == "BLOCKED_MISSING_CHECKPOINT"
    for name in ("pplx_decider", "liquid_d1_3b", "jina_v3_5", "qwen3_reranker_4b", "bge_reranker_v2_m3", "qwen3_reranker_0_6b"):
        assert by_model[name]["snapshot_present"] == "True"


def test_order_diagnostic_is_explicit_and_not_quality_evidence():
    report = (OUT / "E6_1_FINAL_READINESS_REPORT.md").read_text(encoding="utf-8")
    assert "Candidate-order diagnostic" in report
    assert "not a ranking-quality benchmark" in report
    assert "0/3 set-level shuffle checks passed" in report
    assert "3/3 set-level shuffle checks passed" in report


def test_provenance_manifest_verifies_and_forbidden_calls_are_zero():
    manifest = json.loads((OUT / "E6_1_PROVENANCE_MANIFEST.json").read_text(encoding="utf-8"))
    verification = json.loads((OUT / "E6_1_PROVENANCE_VERIFICATION.json").read_text(encoding="utf-8"))
    assert verification["failed"] == 0
    assert verification["passed"] == verification["checked"]
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert path.is_file()
        assert sha_file(path) == entry["sha256"]
    gate = json.loads((OUT / "E6_1_OWNER_REVIEW_GATE.json").read_text(encoding="utf-8"))
    assert gate["jev_calls"] == 0
    assert gate["paid_api_calls"] == 0
    assert gate["e6_2_started"] is False
