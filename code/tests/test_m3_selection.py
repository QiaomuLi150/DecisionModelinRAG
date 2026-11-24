import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_m3_all_models_and_policies_are_real_and_complete():
    manifest = json.loads((ROOT / "data/manifests/m3_selection_manifest.json").read_text())
    expected = {"raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5"}
    assert set(manifest["policies"]) == expected
    assert manifest["rows"] == 9000
    assert manifest["jev_calls"] == manifest["hosted_cloudflare_calls"] == 0
    selection = rows(ROOT / "data/results/m3/selection_results.jsonl")
    assert len(selection) == 9000
    for row in selection:
        expected_size = 20 if row["policy_id"] == "raw20" else 5
        assert row["status"] == "ok"
        assert len(row["selected_doc_ids"]) == expected_size
        assert len(row["selected_doc_ids"]) == len(set(row["selected_doc_ids"]))
        assert set(row["selected_doc_ids"]).issubset(set(row["top20_doc_ids"]))


def test_m3_real_selector_traces_and_corrected_groupwise_limits():
    for model in ("bge", "qwen", "laya", "clef"):
        pointwise = rows(ROOT / "data/results/m3" / f"{model}_all_pointwise.jsonl")
        assert len(pointwise) == 750
        assert all(row["status"] == "ok" and row["real_inference"] for row in pointwise)
        assert all(len(row["pointwise"]) == 20 for row in pointwise)
    for model in ("laya", "clef"):
        groupwise = rows(ROOT / "data/results/m3" / f"{model}_all_groupwise.jsonl")
        assert len(groupwise) == 750
        assert all(row["status"] == "ok" and len(row["groups"]) == 5 for row in groupwise)
        assert all(not group.get("truncated", False) for row in groupwise for group in row["groups"])
        assert all(len({group["winning_doc_id"] for group in row["groups"]}) == 5 for row in groupwise)


def test_m3_shared_view_hashes_match_across_models():
    traces = {}
    for model in ("bge", "qwen", "laya", "clef"):
        traces[model] = {row["query_id"]: row for row in rows(ROOT / "data/results/m3" / f"{model}_all_pointwise.jsonl")}
    for query_id in traces["bge"]:
        reference = [x["selection_text_sha256"] for x in traces["bge"][query_id]["pointwise"]]
        assert all([x["selection_text_sha256"] for x in traces[model][query_id]["pointwise"]] == reference for model in ("qwen", "laya", "clef"))
