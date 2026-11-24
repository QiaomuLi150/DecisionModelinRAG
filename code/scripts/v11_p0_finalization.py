#!/usr/bin/env python3
"""Create deterministic v1.1 P0 finalization and execution-readiness artifacts.

This script is deliberately offline.  It reads already-frozen manifests and
configuration files, validates their population, and writes only reproducibility
metadata.  It never contacts a model, Qdrant, a serving endpoint, or Jev.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TODAY = "2026-10-11"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    ids_path = ROOT / "results/preflight/v11/NEW_TEST_IDS.jsonl"
    rows = load_jsonl(ids_path)
    assert len(rows) == 3767, len(rows)
    query_ids = [row["query_id"] for row in rows]
    source_keys = [f'{row["dataset"]}:{row["source_record_id"]}' for row in rows]
    assert len(query_ids) == len(set(query_ids))
    assert len(source_keys) == len(set(source_keys))
    assert all(row.get("provenance") == "LOCAL_PINNED_SOURCE_DRY_RUN" for row in rows)
    id_bytes = ("\n".join(query_ids) + "\n").encode()
    source_bytes = ("\n".join(source_keys) + "\n").encode()
    id_hash = hashlib.sha256(id_bytes).hexdigest()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    counts = dict(sorted(Counter(row["dataset"] for row in rows).items()))

    track_a = {
        "manifest_version": "decisionrag-v1.1-track-a-execution-r1",
        "freeze_date": TODAY,
        "status": "PREPARED_NOT_EXECUTED",
        "execution_authorization": "OWNER_APPROVAL_REQUIRED",
        "inference_executed": False,
        "jev_calls": 0,
        "cohort": {
            "query_count": len(query_ids),
            "dataset_counts": counts,
            "query_ids": query_ids,
            "query_ids_sha256": id_hash,
            "source_record_keys_sha256": source_hash,
            "source_manifest": rel(ids_path),
            "source_manifest_sha256": sha256(ids_path),
            "selection_seed": 20261011,
            "selection_hash_seed": "20261011-v1.1",
            "selection_rule": "first_800_by_sha256(dataset:query_id:20261011-v1.1), after source-record exclusion validation",
            "source_id_validation": rel(ROOT / "results/preflight/v11_remediation/MULTIHOP_SOURCE_ID_INTERSECTIONS_V11R1.jsonl"),
            "replay_validation": rel(ROOT / "results/preflight/v11_remediation/MULTIHOP_SELECTION_REPLAY_V11R1.jsonl"),
        },
        "separation": {
            "prior_cal_test_debug_overlap": 0,
            "gold_labels_in_inference_inputs": False,
            "outcome_based_resampling": False,
            "track_c_query_ids_reused": False,
            "enterprise_qst_0413_in_track_a": False,
        },
        "retrieval_contract": {
            "status": "FROZEN_PROTOCOL_TOP20_TO_BE_GENERATED_AFTER_APPROVAL",
            "corpus": "existing multi-hop canonical corpus and immutable document IDs",
            "bm25_top_k": 100,
            "dense_top_k": 100,
            "dense_model": "Qwen3-VL-Embedding-2B",
            "dense_endpoint": "http://service.example.invalid",
            "qdrant_collection": "decisionrag_m2_qwen_embedding_v1",
            "qdrant_endpoint": "http://service.example.invalid",
            "rrf_k": 60,
            "slate_size": 20,
            "gold_blind_generation": True,
            "slate_artifact": "NOT_YET_GENERATED",
        },
        "policies": [
            "raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5",
            "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5",
        ],
        "selector_contract": {
            "bge_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "qwen_revision": "4bd860ac4f15ad1897a214615cccc700f8f71818",
            "laya_contract": "native pinned local adapter, max sequence length 512",
            "clef_contract": "native pinned local adapter, max sequence length 512",
            "pointwise_candidate_count": 20,
            "groupwise_quartet_count": 5,
            "rescue_uses_cached_corresponding_pointwise_scores": True,
        },
        "reader_contract": {
            "model_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
            "endpoint": "http://service.example.invalid",
            "prompt_version": "decisionrag-reader-v1",
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 64,
            "seed": 0,
            "reader_view": "title plus first 256 tokens per selected full paragraph",
            "cache_key": "query_id + ordered doc IDs + reader views + prompt version + model revision + generation parameters",
        },
        "statistics": {
            "primary_metrics": ["conditional_complete_support_set_recall_at_5", "unconditional_cssr_at_5", "support_recall_at_5", "reader_em", "reader_f1"],
            "bootstrap_replicates": 5000,
            "bootstrap_seed": 731,
            "dataset_stratification": ["hotpot", "musique", "wiki2", "equal_dataset_macro"],
            "query_pairing": "exact same query IDs and frozen slate per policy",
            "old_pilot_population_separate": True,
        },
        "workload": {
            "logical_policy_outcomes": 45204,
            "pointwise_candidate_evaluations": 301360,
            "typed_groupwise_choices": 37670,
            "reader_result_slots_upper_bound": 45204,
            "estimated_window": "Track A retrieval 10-12 h serial; selectors 3-6 h; readers 4-8 h; not a guarantee",
        },
        "resume_and_provenance": {
            "required_per_record": ["study_id", "config_sha256", "slate_sha256", "model_revision", "view_hash", "prompt_version", "policy_id", "run_id"],
            "cache_and_resume": "immutable input hashes, append-only result JSONL, checkpoint manifest, failed-query ledger",
            "logs": "results/non_jev/v11/track_a/",
        },
    }
    track_a_path = ROOT / "V11_TRACK_A_EXECUTION_MANIFEST.json"
    write_json(track_a_path, track_a)

    snapshot = {
        "snapshot_date": TODAY,
        "read_only": True,
        "authorized_execution_window": "NOT_PROVIDED",
        "gpu_gate": "BLOCKED_GPU_WINDOW_NOT_AUTHORIZED",
        "gpu_snapshot": [
            {"index": 0, "name": "NVIDIA A100-SXM4-80GB", "memory_total_mib": 81920, "memory_used_mib": 16987, "utilization_percent": 3},
            {"index": 1, "name": "NVIDIA A100-SXM4-80GB", "memory_total_mib": 81920, "memory_used_mib": 44483, "utilization_percent": 100},
            {"index": 2, "name": "NVIDIA A100-SXM4-80GB", "memory_total_mib": 81920, "memory_used_mib": 80934, "utilization_percent": 100},
            {"index": 3, "name": "NVIDIA A100-SXM4-80GB", "memory_total_mib": 81920, "memory_used_mib": 78809, "utilization_percent": 0},
            {"index": 4, "name": "NVIDIA A100-SXM4-80GB", "memory_total_mib": 81920, "memory_used_mib": 49083, "utilization_percent": 33},
            {"index": 5, "name": "NVIDIA A100-SXM4-80GB", "memory_total_mib": 81920, "memory_used_mib": 2809, "utilization_percent": 28},
            {"index": 6, "name": "NVIDIA A100-SXM4-80GB", "memory_total_mib": 81920, "memory_used_mib": 74287, "utilization_percent": 0},
            {"index": 7, "name": "NVIDIA A100-SXM4-80GB", "memory_total_mib": 81920, "memory_used_mib": 52221, "utilization_percent": 0},
        ],
        "host_capacity_snapshot": {
            "memory": "2.0 TiB total; 813 GiB used; 26 GiB free; 1.2 TiB available; 8 GiB swap used",
            "project_filesystem": "45 TiB total; 35 TiB used; 8.2 TiB available; 81%",
        },
        "services": [
            {"name": "embedding", "endpoint": "<REDACTED_IP>:8001", "owner": "<REDACTED_USER>", "pid_observed": 880888, "model_revision": "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda", "health_probe": "SANDBOX_BLOCKED_OPERATION_NOT_PERMITTED"},
            {"name": "reranker", "endpoint": "<REDACTED_IP>:8002", "owner": "<REDACTED_USER>", "pid_observed": 2322116, "model_revision": "4bd860ac4f15ad1897a214615cccc700f8f71818", "health_probe": "SANDBOX_BLOCKED_OPERATION_NOT_PERMITTED"},
            {"name": "reader", "endpoint": "<REDACTED_IP>:8003", "owner": "<REDACTED_USER>", "pid_observed": 880348, "model_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "health_probe": "SANDBOX_BLOCKED_OPERATION_NOT_PERMITTED"},
            {"name": "qdrant", "endpoint": "<REDACTED_IP>:6333", "owner": "root", "pid_observed": 971009, "version": "1.19.1", "health_probe": "SANDBOX_BLOCKED_OPERATION_NOT_PERMITTED"},
        ],
        "policy": "No process was stopped, restarted, relocated, or claimed by this finalization.",
    }
    snapshot_path = ROOT / "V11_HARDWARE_SERVICE_SNAPSHOT.json"
    write_json(snapshot_path, snapshot)

    hash_paths = [
        "DecisionRAG_Industry_Experimental_Plan_v1_1_PRE_REGISTRATION.md",
        "Decision_Model_in_RAG_Production_System_COMPLETE_SDD.md",
        "V11_PREFLIGHT_AND_PREREG_REPORT.md",
        "V11_P0_REMEDIATION_REPORT.md",
        "V11_P0_FINAL_COHORT_VALIDATION.md",
        "V11_ACCEPTED_AMENDMENT_QST_0413.md",
        "V11_EXECUTION_READINESS_REPORT.md",
        "V11_TRACK_A_EXECUTION_MANIFEST.json",
        "V11_TRACK_B_EXECUTION_READINESS.md",
        "V11_TRACK_C_EXECUTION_READINESS.md",
        "V11_P0_FINALIZATION_REPORT.md",
        "V11_HARDWARE_SERVICE_SNAPSHOT.json",
        "configs/study.yaml",
        "configs/models.yaml",
        "configs/prompts.json",
        "scripts/v11_p0_preflight.py",
        "scripts/v11_p0_remediation.py",
        "scripts/v11_p0_qdrant_validate.py",
        "scripts/v11_p0_final_cohort_validation.py",
        "scripts/v11_p0_finalization.py",
        "results/preflight/v11/NEW_TEST_IDS.jsonl",
        "results/preflight/v11_remediation/MULTIHOP_V11R1_FINAL_COHORT_LOCK.json",
        "results/preflight/v11_remediation/MULTIHOP_SOURCE_ID_INTERSECTIONS_V11R1.jsonl",
        "results/preflight/v11_remediation/MULTIHOP_SELECTION_REPLAY_V11R1.jsonl",
        "results/preflight/v11_remediation/ENTERPRISE_IDENTITY_POLICY_V11R1.json",
        "results/preflight/v11_remediation/ENTERPRISE_INDEXING_CONTRACT_V11R1.json",
        "results/preflight/v11_remediation/ENTERPRISE_QST_0413_ACCEPTANCE_V11R1.json",
        "results/preflight/v11_remediation/ENTERPRISE_DENOMINATOR_RECOMPUTATION.json",
        "results/preflight/v11_remediation/ENTERPRISE_CHUNK_COUNT_ESTIMATE.json",
        "results/preflight/v11_remediation/ENTERPRISE_LENGTH_SOURCE_STATS.json",
        "results/preflight/v11_remediation/TRACK_C_V11R1_COHORT.jsonl",
        "results/preflight/v11_remediation/TRACK_C_V11R1_SCHEDULE.json",
        "results/preflight/v11_remediation/QDRANT_V11R1_VALIDATION.json",
        "results/preflight/v11_remediation/V11_P0_FINAL_COHORT_MACHINE_SUMMARY.json",
        "data/frozen_slates/m2_top20_slates.jsonl",
        "data/results/audit/quality/REVISED_NON_JEV_COMPLETE_REPORT_vFINAL.md",
    ]
    artifact_hashes: dict[str, str] = {}
    missing: list[str] = []
    for item in hash_paths:
        path = ROOT / item
        if path.exists():
            artifact_hashes[item] = sha256(path)
        else:
            missing.append(item)

    lock = {
        "lock_version": "decisionrag-v1.1-p0-finalization-r1",
        "freeze_date": TODAY,
        "status": "P0_LOCKED_PENDING_GPU_AUTHORIZATION",
        "inference_executed": False,
        "jev_calls": 0,
        "gpu_gate": "BLOCKED_GPU_WINDOW_NOT_AUTHORIZED",
        "cohort": {
            "new_multihop_query_count": 3767,
            "new_multihop_query_ids_sha256": id_hash,
            "new_multihop_source_record_keys_sha256": source_hash,
            "all_new_query_ids_unique": True,
            "source_record_level_disjoint_from_prior_cal_test_debug": True,
            "deterministic_replay_all_datasets": True,
            "debug_fixture_overlap": {"hotpot": 0, "musique": 0, "wiki2": 0},
            "invalid_source_records": 12,
            "track_c_query_count": 100,
            "track_c_query_ids_reused_for_track_a": False,
        },
        "enterprise": {
            "official_question_count_retained": 500,
            "qst_0413_exception": "V11.1R1-GOLD-ID-AMBIGUITY-qst_0413",
            "identity_dependent_exclusion": ["qst_0413"],
            "potentially_eligible_identity_resolved": 443,
            "future_conditional_cssr_denominator": "NOT_YET_MEASURED",
            "path_qualified_internal_identity": True,
        },
        "frozen_experimental_decisions": {
            "non_jev_policy_count": 12,
            "non_jev_policy_ids": track_a["policies"],
            "study_config": "configs/study.yaml",
            "model_config": "configs/models.yaml",
            "prompt_config": "configs/prompts.json",
            "track_c_cohort": "results/preflight/v11_remediation/TRACK_C_V11R1_COHORT.jsonl",
            "track_c_schedule": "results/preflight/v11_remediation/TRACK_C_V11R1_SCHEDULE.json",
            "old_frozen_m2_slates_unchanged": True,
            "old_result_artifacts_rewritten": False,
        },
        "artifact_hashes": artifact_hashes,
        "missing_audit_inputs": missing,
        "omitted_from_commit_by_design": {
            "raw_datasets_and_corpus": "large or source data; preserved in place and represented by existing hashes",
            "enterprise_document_identity_manifest": "275 MB generated map; not needed for reproducibility code review",
            "enterprise_gold_sidecars": "gold-bearing data; not committed",
            "new_test_gold_sidecar": "gold-bearing data; not committed",
            "model_weights_and_indexes": "large runtime artifacts; not committed",
            "credentials_and_secrets": "never committed",
        },
        "commit_scope": [
            ".gitignore",
            "DecisionRAG_Industry_Experimental_Plan_v1_1_PRE_REGISTRATION.md",
            "Decision_Model_in_RAG_Production_System_COMPLETE_SDD.md",
            "V11_PREFLIGHT_AND_PREREG_REPORT.md",
            "V11_P0_REMEDIATION_REPORT.md",
            "V11_P0_FINAL_COHORT_VALIDATION.md",
            "V11_ACCEPTED_AMENDMENT_QST_0413.md",
            "V11_P0_FINALIZATION_REPORT.md",
            "V11_EXECUTION_READINESS_REPORT.md",
            "V11_TRACK_A_EXECUTION_MANIFEST.json",
            "V11_TRACK_B_EXECUTION_READINESS.md",
            "V11_TRACK_C_EXECUTION_READINESS.md",
            "V11_HARDWARE_SERVICE_SNAPSHOT.json",
            "V11_PREREGISTRATION_LOCK_MANIFEST.json",
            "configs/study.yaml",
            "configs/models.yaml",
            "configs/prompts.json",
            "scripts/v11_p0_preflight.py",
            "scripts/v11_p0_remediation.py",
            "scripts/v11_p0_qdrant_validate.py",
            "scripts/v11_p0_final_cohort_validation.py",
            "scripts/v11_p0_finalization.py",
            "results/preflight/v11/NEW_TEST_IDS.jsonl",
            "results/preflight/v11_remediation/MULTIHOP_V11R1_FINAL_COHORT_LOCK.json",
            "results/preflight/v11_remediation/MULTIHOP_SOURCE_ID_INTERSECTIONS_V11R1.jsonl",
            "results/preflight/v11_remediation/MULTIHOP_SELECTION_REPLAY_V11R1.jsonl",
            "results/preflight/v11_remediation/ENTERPRISE_IDENTITY_POLICY_V11R1.json",
            "results/preflight/v11_remediation/ENTERPRISE_INDEXING_CONTRACT_V11R1.json",
            "results/preflight/v11_remediation/ENTERPRISE_QST_0413_ACCEPTANCE_V11R1.json",
            "results/preflight/v11_remediation/ENTERPRISE_DENOMINATOR_RECOMPUTATION.json",
            "results/preflight/v11_remediation/ENTERPRISE_CHUNK_COUNT_ESTIMATE.json",
            "results/preflight/v11_remediation/ENTERPRISE_LENGTH_SOURCE_STATS.json",
            "results/preflight/v11_remediation/TRACK_C_V11R1_COHORT.jsonl",
            "results/preflight/v11_remediation/TRACK_C_V11R1_SCHEDULE.json",
            "results/preflight/v11_remediation/QDRANT_V11R1_VALIDATION.json",
            "results/preflight/v11_remediation/V11_P0_FINAL_COHORT_MACHINE_SUMMARY.json",
            "data/frozen_slates/m2_top20_slates.jsonl",
            "data/results/audit/quality/REVISED_NON_JEV_COMPLETE_REPORT_vFINAL.md",
        ],
        "self_hash_policy": "This manifest is excluded from its own artifact_hashes map to avoid a self-referential hash cycle.",
    }
    write_json(ROOT / "V11_PREREGISTRATION_LOCK_MANIFEST.json", lock)


if __name__ == "__main__":
    main()
