from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/non_jev/v2/e6_v1_0"
R1 = BASE / "e6_2_r1_scifact_retrieval_freeze"
R2 = BASE / "e6_2_r2_selector_inference"
E63 = BASE / "e6_3_integrated_completion"
OUT = BASE / "e6_3a_integrity_and_manuscript"


def jl(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_frozen_serving_sample_is_unique_and_reproducible():
    sample = json.loads((E63 / "matched_serving/E6_3_SERVING_SAMPLE_MANIFEST.json").read_text())
    slates = jl(R1 / "E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl")
    ids = [x["query_id"] for x in slates]
    assert len(ids) == 277 and len(set(ids)) == 277
    expected = sorted(ids, key=lambda q: __import__("hashlib").sha256(json.dumps({"query_id": q, "seed": 731}, sort_keys=True, separators=(",", ":")).encode()).hexdigest())[:100]
    assert sample["scifact_ids"] == expected
    assert len(sample["scifact_ids"]) == 100


def test_original_slate_candidates_require_corpus_reconstruction():
    sample = json.loads((E63 / "matched_serving/E6_3_SERVING_SAMPLE_MANIFEST.json").read_text())
    wanted = set(sample["scifact_ids"])
    rows = [x for x in jl(R1 / "E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl") if x["query_id"] in wanted]
    assert len(rows) == 100
    assert all("text" not in c for row in rows for c in row["candidates"])
    assert all("content_sha256" in c for row in rows for c in row["candidates"])


def test_amos_failure_overlap_is_retained():
    sample = json.loads((E63 / "matched_serving/E6_3_SERVING_SAMPLE_MANIFEST.json").read_text())
    failures = list(csv.DictReader((R2 / "E6_2_R2_FAILURES.csv").open()))
    failed = {x["query_id"] for x in failures if x["model"] == "amos_1_small_2b"}
    assert len(failed) == 83
    assert len(set(sample["scifact_ids"]) & failed) == 28


def test_original_candidate_identity_hashes_match_e62():
    sample = json.loads((E63 / "matched_serving/E6_3_SERVING_SAMPLE_MANIFEST.json").read_text())
    wanted = set(sample["scifact_ids"])
    old = {(x["model"], x["query_id"]): x for x in jl(R2 / "E6_2_R2_NATIVE_SELECTIONS.jsonl")}
    new = {(x["model"], x["query_id"]): x for x in csv.DictReader((E63 / "matched_serving/E6_3_SERVING_PER_REQUEST.csv").open()) if x["workload"] == "scifact" and x["warmup"] == "0" and x["repetition"] == "1"}
    for model in ["pplx_decider", "liquid_d1_3b", "jina_v3_5", "qwen3_reranker_4b", "bge_reranker_v2_m3", "qwen3_reranker_0_6b", "amos_1_small_2b"]:
        for q in wanted:
            assert old[(model, q)]["candidate_slate_sha256"] == new[(model, q)]["candidate_ids_sha256"]


def test_reader_primary_metric_reproduces_stored_rows():
    rows = list(csv.DictReader((E63 / "reader_quality/E6_3_READER_PER_CASE_RESULTS.csv").open()))
    assert len(rows) == 1565
    assert all(x["reader_status"] == "ok" for x in rows)
    assert all(x["parse_mode"] == "TAGGED" for x in rows)


def test_e63a_outputs_are_append_only_and_not_reserved():
    if not OUT.exists():
        return
    gate = OUT / "E6_3A_FINAL_OWNER_REVIEW_GATE.json"
    if gate.exists():
        data = json.loads(gate.read_text())
        assert data["reserved_access"] is False
        assert data["jev_calls"] == 0
        assert data["paid_api_calls"] == 0


def test_correction_scorecard_has_two_workloads_when_present():
    score = OUT / "serving_audit/E6_3A_CORRECTED_SERVING_SCORECARD.csv"
    if not score.exists():
        return
    rows = list(csv.DictReader(score.open()))
    assert len(rows) == 16
    assert {x["workload"] for x in rows} == {"scifact", "garage"}


def test_provenance_entries_hash_existing_files_when_present():
    manifest = OUT / "E6_3A_PROVENANCE_MANIFEST.json"
    if not manifest.exists():
        return
    data = json.loads(manifest.read_text())
    for entry in data["new_files"]:
        path = ROOT / entry["path"]
        assert path.exists()
        assert __import__("hashlib").sha256(path.read_bytes()).hexdigest() == entry["sha256"]

