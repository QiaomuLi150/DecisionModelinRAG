import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_sp_r2"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_r2_manifest_and_zero_paid_calls():
    manifest = json.loads((OUT / "E2_SP_R2_PROVENANCE_MANIFEST.json").read_text())
    assert manifest["status"] == "PASS_WITH_QUALIFICATIONS"
    assert manifest["approved_gpu"] == 1
    assert manifest["r3_request_count"] == 800
    assert manifest["r4_primary_cases"] == 720
    assert manifest["r4_repeat_cases"] == 144
    assert manifest["jev_calls"] == 0
    assert manifest["paid_api_calls"] == 0


def test_isolated_r3_has_five_models_and_800_successes():
    rows = list(csv.DictReader((OUT / "E2_SP_R2_ISOLATED_R3_SUMMARY.csv").open()))
    assert len(rows) == 20
    assert {r["model"] for r in rows} == {"bge_pointwise", "jina35", "nimble", "gte", "pplx_decider"}
    assert sum(int(r["completed"]) for r in rows) == 800
    assert sum(int(r["errors"]) for r in rows) == 0
    assert all(r["stable_queue_observable"] == "QUEUE_GROWTH_PROXY_ONLY" for r in rows)


def test_local_adjudication_is_blinded_and_complete():
    rows = load_jsonl(OUT / "E2_SP_R2_LOCAL_ADJUDICATION.jsonl")
    assert len(rows) == 864
    assert sum(int(r["repeat_index"]) == 1 for r in rows) == 720
    assert sum(int(r["repeat_index"]) == 2 for r in rows) == 144
    assert all(r["status"] == "ok" for r in rows)
    assert all(r["gold_in_prompt"] is False and r["reference_answer_in_prompt"] is False for r in rows)
    assert all(r["policy_identity_in_prompt"] is False and r["model_scores_in_prompt"] is False for r in rows)


def test_cal_configuration_is_locked_and_all_models_passed():
    lock = json.loads((OUT / "E2_SP_R2_CAL_CONFIG_LOCK.json").read_text())
    assert lock["status"] == "FROZEN_BEFORE_MEASUREMENT"
    assert len(lock["cal_query_ids"]) == 30
    assert lock["selected_configuration"]["batch_size"] == 1
    assert lock["selected_configuration"]["request_concurrency"] == 1
    rows = list(csv.DictReader((OUT / "E2_SP_R2_CAL_OPTIMIZATION.csv").open()))
    assert {r["model"] for r in rows} == {"bge_pointwise", "jina35", "nimble", "gte", "pplx_decider"}
    assert all(r["status"] == "PASS" and int(r["errors"]) == 0 for r in rows)


def test_historical_shared_artifacts_still_exist():
    for name in ("E2_SP_R_FROZEN_PROTOCOL.json", "E2_SP_R_PROVENANCE_MANIFEST.json", "E2_SP_R_FINAL_REPORT.md", "E2_SP_R_R3_SHARED_SUMMARY.csv"):
        assert (ROOT / "results/non_jev/v2/e2_sp_r" / name).is_file()
