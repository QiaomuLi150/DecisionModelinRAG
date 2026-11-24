import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e3_v1/e3_c_models"


def _jsonl(name):
    with (OUT / name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _csv(name):
    with (OUT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _last_native_rows():
    rows = {}
    for row in _jsonl("E3_C_RAW_NATIVE_SCORES.jsonl"):
        rows[(row["model"], row["stage"], row["query_id"])] = row
    return rows


def test_e3_c_owner_gate_and_paid_holdout_provenance():
    status = json.loads((OUT / "E3_C_PROVENANCE_MANIFEST.json").read_text())
    gate = (OUT / "E3_C_OWNER_REVIEW_GATE.md").read_text()
    assert status["status"] == "E3_C_PASS_WITH_LIMITATIONS"
    assert status["owner_review_gate"] == "E3_C_OWNER_REVIEW_READY"
    assert "E3_C_OWNER_REVIEW_READY" in gate
    assert status["jev_calls"] == 0
    assert status["paid_api_calls"] == 0
    assert status["holdout_access"] is False
    assert status["training"] is False
    assert status["multimodal"] is False
    assert status["c5"] == "C5_PASS_GPU0_ISOLATED"


def test_e3_c_native_final_cardinalities_and_forward_telemetry():
    last = _last_native_rows()
    for model in ("liquid_d1_3b", "zerank_2"):
        for stage, expected in (("smoke", 12), ("cal", 30), ("full", 4267)):
            rows = [r for (m, s, _), r in last.items() if m == model and s == stage]
            assert len(rows) == expected
            assert Counter(r["status"] for r in rows) == {"VALID": expected}
            assert all(len(r["candidate_ids"]) == len(set(r["candidate_ids"])) == 20 for r in rows)
            assert all(len(r["scores"]) == 20 for r in rows)
            assert all(r["gold_in_inference_input"] is False for r in rows)
    liquid = [r for (m, s, _), r in last.items() if m == "liquid_d1_3b" and s == "full"]
    zerank = [r for (m, s, _), r in last.items() if m == "zerank_2" and s == "full"]
    assert Counter(r["physical_forward_count"] for r in liquid) == {1: 4267}
    assert Counter(r["physical_forward_count"] for r in zerank) == {3: 4267}
    assert Counter(r["batch_size"] for r in zerank) == {8: 4267}


def test_e3_c_repeated_native_attempts_are_score_stable():
    groups = defaultdict(list)
    for row in _jsonl("E3_C_RAW_NATIVE_SCORES.jsonl"):
        if row["status"] == "VALID":
            score_hash = hashlib.sha256(json.dumps(row["scores"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            groups[(row["model"], row["stage"], row["query_id"])].append((score_hash, tuple(row["selected_ids"]), row["input_sha256"]))
    assert len(groups) == 2 * (12 + 30 + 4267)
    assert all(len({x[0] for x in values}) == 1 for values in groups.values())
    assert all(len({x[1] for x in values}) == 1 for values in groups.values())
    assert all(len({x[2] for x in values}) == 1 for values in groups.values())


def test_e3_c_selected_top5_and_quality_denominators():
    selected = _jsonl("E3_C_SELECTED_TOP5.jsonl")
    assert len(selected) == 2 * 4267
    for row in selected:
        assert len(row["selected_ids"]) == len(set(row["selected_ids"])) == 5
        assert set(row["selected_ids"]).issubset(set(row["candidate_ids"]))
        assert row["status"] == "VALID"
        assert "gold" not in row and "qrel" not in row
    quality = _csv("E3_C_DATASET_QUALITY_SUMMARY.csv")
    assert len(quality) == 36
    for row in quality:
        if row["dataset"] == "__pooled__":
            assert row["identity_eligible_n"] == "3767"
            assert row["retrieval_conditional_n"] == "1490"
        if row["dataset"] == "enterprise":
            assert row["identity_eligible_n"] == "443"
            assert row["retrieval_conditional_n"] == "330"


def test_e3_c_paired_statistics_are_separated_and_pinned():
    rows = _csv("E3_C_PRIMARY_PAIRED_STATISTICS.csv")
    assert len(rows) == 20
    assert {row["bootstrap_reps"] for row in rows} == {"5000"}
    assert {row["bootstrap_seed"] for row in rows} == {"731"}
    assert {row["cohort"] for row in rows} == {"track_a", "track_b"}
    for row in rows:
        assert int(row["n"]) in ({1490, 3767} if row["cohort"] == "track_a" else {330, 443})


def test_e3_c_timing_is_measured_on_isolated_gpu0():
    latency = _jsonl("E3_C_SELECTOR_LATENCY_RAW.jsonl")
    assert len(latency) == 600
    assert {row["model"] for row in latency} == {"liquid_d1_3b", "zerank_2"}
    assert all(row["status"] == "VALID" for row in latency)
    assert all(row["gpu_id"] == 0 for row in latency)
    assert all(row["gold_in_inference_input"] is False for row in latency)
    summary = _csv("E3_C_SELECTOR_LATENCY_SUMMARY.csv")
    assert len(summary) == 2
    assert {row["n"] for row in summary} == {"300"}
    assert all(float(row["p50_ms"]) > 0 and float(row["p95_ms"]) >= float(row["p50_ms"]) for row in summary)
    report = (OUT / "E3_C_SELECTOR_LATENCY_AND_RESOURCE_REPORT.md").read_text()
    assert "C5_PASS_GPU0_ISOLATED" in report
    assert "single-client" in report


def test_e3_c_artifact_hashes():
    status = json.loads((OUT / "E3_C_PROVENANCE_MANIFEST.json").read_text())
    bad = []
    for relative, expected in status["artifact_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            bad.append(relative)
    assert bad == []
