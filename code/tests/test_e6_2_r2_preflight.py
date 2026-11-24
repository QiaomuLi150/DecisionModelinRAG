"""CPU-only tests for the blocked-but-authenticated E6-2 R2 preflight."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r2_selector_inference"


def test_r2_stops_before_model_inference_without_fresh_gpu_uuid():
    gate = json.loads((OUT / "E6_2_R2_R3_READINESS_GATE.json").read_text())
    manifest = json.loads((OUT / "E6_2_R2_PREINFERENCE_MANIFEST.json").read_text())
    if gate["gate"] == "E6_2_R2_GPU_AUTHORIZATION_PENDING":
        assert gate["preflight"]["native_attempts"] == 0
        assert manifest["authorization"]["fresh_device_specific_authorization_present"] is False
        assert manifest["authorization"]["authorized_gpu_uuids"] == []
    else:
        assert gate["gate"] == "E6_2_R2_SELECTIONS_FROZEN_WITH_EXCLUSIONS"
        assert json.loads((OUT / "E6_2_R2_PREINFERENCE_MANIFEST_V2.json").read_text())["authorization"]["fresh_device_specific_authorization_present"] is True


def test_r2_frozen_input_counts_and_hashes_are_authenticated():
    validation = json.loads((OUT / "E6_2_R2_INPUT_OUTPUT_VALIDATION.json").read_text())
    assert validation["status"] == "PASS"
    if "frozen_input_authentication" in validation:
        audit = validation["frozen_input_authentication"]
        assert audit["checks_passed"] == audit["checks_total"]
        assert audit["counts"] == {"corpus_documents": 5183, "exploratory_queries": 277, "top20_candidate_records": 5540, "top5_slates": 277}
        assert audit["files"]["top20"]["sha256"] == "b9b2a3c3971da72f7e7aff6066c381c681f0ff6f7b4592d85b480e12cd449333"
    else:
        assert validation["actual_attempts"] == 1939
        assert 1 <= validation["candidate_slate_hash_count"] <= 277


def test_r2_registry_has_exactly_eight_unattempted_models():
    with (OUT / "E6_2_R2_MODEL_EXECUTION_REGISTRY.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    if "r2_attempts" in rows[0]:
        assert {row["r2_attempts"] for row in rows} == {"0"}
        assert {row["status"] for row in rows} == {"NOT_RUN_GPU_AUTH_PENDING"}
    else:
        assert {row["scheduled_attempts"] for row in rows} == {"277"}
        assert {row["model"] for row in rows if row["status"] == "VALIDATED"} == {"pplx_decider", "liquid_d1_3b", "jina_v3_5", "qwen3_reranker_4b", "bge_reranker_v2_m3", "qwen3_reranker_0_6b"}
    if "checkpoint_present" in rows[0]:
        assert all(row["checkpoint_present"] == "True" for row in rows)


def test_r2_does_not_create_unexecuted_native_result_files():
    if (OUT / "E6_2_R2_NATIVE_SELECTIONS.jsonl").exists():
        assert (OUT / "E6_2_R2_NATIVE_TRACES.jsonl").exists()
        assert (OUT / "E6_2_R2_FAILURES.csv").exists()
        return
    for name in ("E6_2_R2_NATIVE_SELECTIONS.jsonl", "E6_2_R2_NATIVE_TRACES.jsonl", "E6_2_R2_FAILURES.csv", "E6_2_R2_LATENCY_AND_RESOURCE_REPORT.csv", "E6_2_R2_FORWARD_ACCOUNTING.csv"):
        assert not (OUT / name).exists()


def test_r2_provenance_is_self_consistent_and_label_free():
    provenance = json.loads((OUT / "E6_2_R2_PROVENANCE_MANIFEST.json").read_text())
    assert provenance.get("inference_attempts", provenance.get("actual_attempts")) in (0, 1939)
    assert provenance["quality_metrics"] is False
    assert provenance["reserved_content_accessed"] is False
    assert provenance["jev_calls"] == 0
    assert provenance["paid_api_calls"] == 0
    for item in provenance["files"]:
        path = ROOT / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_r2_gpu_audit_does_not_reuse_prior_idle_observation_as_authorization():
    audit = json.loads((OUT / "E6_2_R2_GPU_AUDIT.json").read_text())
    if audit["status"] == "PENDING":
        assert audit["authorized_gpu_uuids"] == []
        assert audit["prior_observation_is_not_reused_as_authorization"] is True
    else:
        assert audit["status"] == "PASS"
        assert set(audit["approved_gpu_uuids"]) == {"1", "5", "6"}
    assert audit["new_process_started"] is False
