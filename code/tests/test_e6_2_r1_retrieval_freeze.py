"""CPU-only integrity checks for the E6-2 R1 retrieval authorization gate."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r1_scifact_retrieval_freeze"


def test_r1_stops_before_dense_retrieval():
    authorized = OUT / "E6_2_R1_E6_2_RELEASE_GATE_V2.json"
    if authorized.exists():
        gate = json.loads(authorized.read_text())
        assert gate["gate"] == "E6_2_R1_RETRIEVAL_FROZEN_READY"
        assert gate["status"] == "READY_FOR_SEPARATE_E6_2_SELECTOR_AUTHORIZATION"
        assert gate["embedding"]["failed_batches"] == 0
        assert gate["frozen_slates"] == {"cal": 555, "exploratory_top20": 277, "exploratory_top5": 277}
        assert gate["contract"]["qrels_accessed_during_generation"] is False
        assert gate["contract"]["reserved_content_accessed"] is False
        return
    gate = json.loads((OUT / "E6_2_R1_E6_2_RELEASE_GATE.json").read_text())
    assert gate["gate"] == "E6_2_R1_GPU_AUTHORIZATION_PENDING"
    assert gate["completed"]["dense_inference"] is False
    assert gate["completed"]["rrf_freeze"] is False
    assert gate["frozen_slates"] == {"cal": 0, "exploratory_top20": 0, "exploratory_top5": 0}
    assert gate["prohibitions_observed"]["embedding_requests"] == 0
    assert gate["prohibitions_observed"]["reserved_content_access"] is False


def test_r1_does_not_create_empty_success_slate_files():
    if (OUT / "E6_2_R1_E6_2_RELEASE_GATE_V2.json").exists():
        expected = {
            "E6_2_R1_CAL_RETRIEVAL_SLATES.jsonl": 555,
            "E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl": 277,
            "E6_2_R1_EXPLORATORY_TOP5_BASELINE.jsonl": 277,
        }
        for name, count in expected.items():
            path = OUT / name
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            assert len(rows) == count
        return
    for name in (
        "E6_2_R1_CAL_RETRIEVAL_SLATES.jsonl",
        "E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl",
        "E6_2_R1_EXPLORATORY_TOP5_BASELINE.jsonl",
    ):
        assert not (OUT / name).exists()


def test_r1_public_population_and_cpu_bm25_validation():
    audit = json.loads((OUT / "E6_2_R1_CORPUS_AND_PARTITION_AUDIT.json").read_text())
    assert audit["status"] == "PASS"
    assert audit["corpus"]["rows"] == 5183
    assert audit["queries"]["rows"] == 1109
    assert audit["partition_files"]["cal"]["rows"] == 555
    assert audit["partition_files"]["exploratory"]["rows"] == 277
    assert audit["partition_files"]["reserved"]["rows"] == 277
    assert audit["reserved_receipt"]["content_loaded"] is False
    bm25 = json.loads((OUT / "E6_2_R1_CPU_BM25_CAL_VALIDATION.json").read_text())
    assert bm25["status"] == "PASS"
    assert bm25["queries"] == 555
    assert bm25["deterministic"] is True
    assert bm25["relevance_metrics_run"] is False


def test_r1_manifest_is_frozen_before_new_dense_work():
    authorized = OUT / "E6_2_R1_EXECUTION_MANIFEST_V2.json"
    if authorized.exists():
        manifest = json.loads(authorized.read_text())
        assert manifest["schema"] == "E6-2-R1-AUTHORIZED-EXECUTION-MANIFEST-V2"
        assert manifest["reserved_content_loaded"] is False
        assert manifest["retrieval_contract"]["rrf"]["top_k"] == 20
        assert manifest["retrieval_contract"]["dense"]["revision"] == "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
        return
    manifest = json.loads((OUT / "E6_2_R1_EXECUTION_MANIFEST.json").read_text())
    assert manifest["frozen_before_new_embedding_or_dense_retrieval"] is True
    assert manifest["retrieval_started"] is False
    assert manifest["retriever"]["dense"]["inference_status"] == "NOT_RUN_GPU_AUTH_PENDING"
    assert manifest["holdout"] == {
        "reserved_content_loaded": False,
        "reserved_qrels_loaded": False,
        "reserved_slates_loaded": False,
    }


def test_r1_authorized_slates_have_valid_identity_and_prefixes():
    top20 = [json.loads(line) for line in (OUT / "E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl").read_text().splitlines() if line.strip()]
    top5 = [json.loads(line) for line in (OUT / "E6_2_R1_EXPLORATORY_TOP5_BASELINE.jsonl").read_text().splitlines() if line.strip()]
    assert len(top20) == len(top5) == 277
    by_qid = {row["query_id"]: row for row in top20}
    assert len(by_qid) == 277
    for row in top20:
        ids = row["candidate_ids"]
        assert len(ids) == 20
        assert len(set(ids)) == 20
        assert all(item.startswith("scifact:") for item in ids)
        assert row["qrels_accessed_during_generation"] is False
        assert row["reserved_content_accessed"] is False
    for row in top5:
        assert row["candidate_ids"] == by_qid[row["query_id"]]["candidate_ids"][:5]


def test_r1_authorized_provenance_and_no_selector_calls():
    gate = json.loads((OUT / "E6_2_R1_E6_2_RELEASE_GATE_V2.json").read_text())
    provenance = json.loads((OUT / "E6_2_R1_PROVENANCE_MANIFEST_V2.json").read_text())
    assert gate["embedding"]["physical_requests"] == 376
    assert gate["embedding"]["failed_batches"] == 0
    assert provenance["zero_calls"] == {"selector": 0, "reranker": 0, "reader": 0, "jev": 0, "paid_api": 0}
    assert provenance["reserved_content_access"] is False
    assert provenance["historical_artifacts_modified"] is False
    listed = {item["path"] for item in provenance["new_outputs"]}
    assert "results/non_jev/v2/e6_v1_0/e6_2_r1_scifact_retrieval_freeze/E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl" in listed
