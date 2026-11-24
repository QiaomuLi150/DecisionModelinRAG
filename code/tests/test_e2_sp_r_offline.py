import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_sp_r"


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_r0_frozen_population_and_selection_reader_counts():
    audit = json.loads((OUT / "E2_SP_R_SOURCE_AUDIT.json").read_text())
    assert audit["status"] == "PASS"
    assert audit["query_count"] == 300
    assert audit["unique_query_count"] == 300
    assert audit["selection_rows"] == 3300
    assert audit["reader_rows"] == 3300
    assert audit["gold_in_selection_input_rows"] == 0
    assert audit["gold_in_reader_input_rows"] == 0


def test_r4_package_is_deterministic_complete_and_unadjudicated():
    rows = load_jsonl(OUT / "E2_SP_R_ADJUDICATION_CASES.jsonl")
    assert len(rows) == 720
    assert len({row["case_id"] for row in rows}) == 720
    assert len({row["query_id"] for row in rows}) == 120
    assert {row["policy_blind_id"] for row in rows} == {f"policy_{i:02d}" for i in range(1, 7)}
    assert all(row["assessment_status"] == "NOT_INDEPENDENTLY_VERIFIED" for row in rows)
    assert all(row["human_judgment"] is None and row["llm_judgment"] is None for row in rows)


def test_r3_shared_execution_remains_non_isolated():
    manifest = json.loads((OUT / "E2_SP_R_PROVENANCE_MANIFEST.json").read_text())
    assert manifest["status"] == "E2_SP_R_PARTIAL_BLOCKED"
    assert manifest["stage_status"]["R3"] in {
        "BLOCKED_SHARED_CONTENTION",
        "EXECUTED_SHARED_CONTENTION_LIMITED_ISOLATED_GATE_BLOCKED",
    }
    assert manifest["isolated_capacity_measurement"] is False
    assert manifest["counts"]["new_model_requests"] in {190, 990}
    assert manifest["jev_calls"] == 0
    assert manifest["paid_api_calls"] == 0
    rows = list(csv.DictReader((OUT / "E2_SP_R_GPU_SECONDS_AND_COST.csv").open()))
    assert rows
    assert all(row["status"] == "BLOCKED_NO_SERVING_RUN" for row in rows)
    assert all(row["gpu_hour_rate_usd"] == "NOT_ASSUMED" for row in rows)
