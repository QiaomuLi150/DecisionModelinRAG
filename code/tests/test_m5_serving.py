import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICIES = {"raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5"}


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_m5_all_policies_share_frozen_uncached_cohort():
    cohort = {row["query_id"] for row in rows(ROOT / "data/manifests/m5_serving_query_ids.jsonl")}
    assert len(cohort) == 100
    for policy in POLICIES:
        trace = rows(ROOT / "data/results/m5" / f"{policy}_serving_traces.jsonl")
        assert len(trace) == 100
        assert {row["query_id"] for row in trace} == cohort
        assert all(row["concurrency"] == 10 and row["cache_disabled"] for row in trace)
        assert all(row["status"] == "ok" for row in trace)
        assert all(row["paid_api_usage"] is False and row["provider_fee_usd"] == 0.0 for row in trace)
        assert all(row["model_topology_hash"] for row in trace)


def test_m5_stage_boundaries_and_summary_integrity():
    metrics = json.loads((ROOT / "data/results/m5/m5_metrics.json").read_text())
    assert metrics["m5_integrity"]["same_cohort"]
    assert metrics["m5_integrity"]["all_cache_disabled"]
    assert metrics["m5_integrity"]["all_status_ok"]
    assert metrics["m5_integrity"]["all_jev_calls_zero"]
    assert metrics["m5_integrity"]["all_hosted_cloudflare_calls_zero"]
    assert len(metrics["summary"]) == 12
    for item in metrics["summary"]:
        assert item["n_attempted"] == item["n_completed"] == 100
        assert item["error_count"] == item["retry_count"] == 0
        assert item["request_p95_ms"] >= item["request_p50_ms"]
        assert item["gpu_seconds_per_1k"] == "NOT_MEASURED"


def test_m5_required_analysis_artifacts_exist():
    required = [
        ROOT / "results/non_jev/analysis/summary.csv",
        ROOT / "results/non_jev/analysis/per_dataset.csv",
        ROOT / "results/non_jev/analysis/paired_ci.csv",
        ROOT / "results/non_jev/analysis/truncation_audit.csv",
        ROOT / "results/non_jev/analysis/failure_cases.jsonl",
        ROOT / "results/non_jev/analysis/operational_costs.csv",
        ROOT / "paper/tables/model_inventory.csv",
        ROOT / "paper/tables/policy_quality.csv",
        ROOT / "paper/tables/serving_cost.csv",
        ROOT / "paper/figures/support_recall_vs_answer_f1.png",
        ROOT / "paper/figures/quality_vs_cost.png",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in required)
    with (ROOT / "results/non_jev/analysis/summary.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 12
