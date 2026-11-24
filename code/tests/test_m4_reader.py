import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_m4_reader_outputs_and_cache_contract():
    manifest = json.loads((ROOT / "data/manifests/m4_reader_manifest.json").read_text())
    assert manifest["tasks"] == 9000
    # The corrected BGE and Qwen selections create the measured repaired
    # reader-context population rather than the archived defective one.
    assert manifest["unique_cache_keys"] == 7709
    assert manifest["in_run_cache_reuses"] == 1291
    assert manifest["status_counts"] == {"ok": 9000}
    assert manifest["jev_calls"] == manifest["hosted_cloudflare_calls"] == 0
    results = rows(ROOT / "data/results/m4/reader_results.jsonl")
    assert len(results) == 9000
    assert all(row["reader_status"] == "ok" for row in results)
    assert all(row["cache_disabled"] is False for row in results)


def test_m4_metrics_and_bootstrap_contract():
    metrics = json.loads((ROOT / "data/results/m4/m4_metrics.json").read_text())
    assert metrics["metric_rows"] == 9000
    assert metrics["best_local_baseline"] in {"bge_p5", "qwen_p5"}
    assert metrics["statistics"]["bootstrap_replicates"] == 5000
    assert metrics["statistics"]["bootstrap_seed"] == 731
    assert metrics["reader_status_counts"] == {"ok": 9000}
    assert metrics["jev_calls"] == metrics["hosted_cloudflare_calls"] == 0
    assert set(metrics["metrics"]) == {"raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5"}
