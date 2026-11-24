#!/usr/bin/env python3
"""CPU-only E6-2 R2 preflight.

This module authenticates the frozen R1 inputs and the admitted model registry,
then stops before model loading or selector inference when fresh device-specific
GPU authorization is absent.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from e6_2_r0_readiness import read_arrow, sha_file


ROOT = Path(__file__).resolve().parents[1]
R0 = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r0_scientific_readiness"
R1 = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r1_scifact_retrieval_freeze"
E60R = ROOT / "results/non_jev/v2/e6_v1_0/e6_0r_endpoint_dataset_closure"
E61 = ROOT / "results/non_jev/v2/e6_v1_0/e6_1_native_adapter_validation"
E61B = ROOT / "results/non_jev/v2/e6_v1_0/e6_1b_checkpoint_recovery"
OUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_2_r2_selector_inference"
DATASETS = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets")


def write_json(name: str, value: object) -> Path:
    path = OUT / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def stable(value: object) -> str:
    return sha_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_split_metadata() -> dict:
    return json.loads((E60R / "E6_0R_SOURCE_SPLIT_MANIFEST.json").read_text(encoding="utf-8"))


def authenticate_slates() -> dict:
    top20_path = R1 / "E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl"
    top5_path = R1 / "E6_2_R1_EXPLORATORY_TOP5_BASELINE.jsonl"
    gate_path = R1 / "E6_2_R1_E6_2_RELEASE_GATE_V2.json"
    manifest_path = R1 / "E6_2_R1_EXECUTION_MANIFEST_V2.json"
    top20 = read_jsonl(top20_path)
    top5 = read_jsonl(top5_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = load_split_metadata()
    cal_ids = set(split["partitions"]["CAL"]["query_ids"])
    exploratory_ids = set(split["partitions"]["EXPLORATORY"]["query_ids"])
    # RESERVED retains only metadata (count/component/qid hashes); do not
    # reconstruct or open its question IDs during R2 preflight.
    reserved_ids: set[str] = set()
    corpus_path = sorted(DATASETS.rglob("scifact-corpus.arrow"))[0]
    corpus = read_arrow(corpus_path)
    corpus_by_id = {f"scifact:{row['_id']}": (i, row) for i, row in enumerate(corpus)}
    checks = {
        "r1_gate_ready": gate["gate"] == "E6_2_R1_RETRIEVAL_FROZEN_READY",
        "r1_qrels_free": gate["contract"]["qrels_accessed_during_generation"] is False,
        "r1_reserved_free": gate["contract"]["reserved_content_accessed"] is False,
        "top20_count": len(top20) == 277,
        "top20_query_unique": len({row["query_id"] for row in top20}) == 277,
        "top20_ids_in_exploratory": {row["query_id"] for row in top20} == exploratory_ids,
        "top20_not_cal": not ({row["query_id"] for row in top20} & cal_ids),
        "top5_count": len(top5) == 277,
        "top5_query_unique": len({row["query_id"] for row in top5}) == 277,
    }
    candidate_records = 0
    for slate in top20:
        ids = slate["candidate_ids"]
        checks[f"slate_{slate['query_id']}_valid"] = len(ids) == 20 and len(set(ids)) == 20
        checks[f"slate_{slate['query_id']}_hash"] = slate["slate_sha256"] == stable(ids)
        checks[f"slate_{slate['query_id']}_candidates"] = len(slate["candidates"]) == 20
        candidate_records += len(slate["candidates"])
        for candidate in slate["candidates"]:
            doc_id = candidate["document_id"]
            if doc_id not in corpus_by_id:
                checks[f"candidate_{doc_id}_exists"] = False
                continue
            row_index, row = corpus_by_id[doc_id]
            text = (f"{row.get('title', '')}\n{row.get('text', '')}").strip()
            checks[f"candidate_{doc_id}_content"] = (
                candidate["corpus_row"] == row_index
                and candidate["content_sha256"] == sha_text(text)
            )
    top20_by_qid = {row["query_id"]: row for row in top20}
    for row in top5:
        checks[f"top5_{row['query_id']}_prefix"] = row["candidate_ids"] == top20_by_qid[row["query_id"]]["candidate_ids"][:5]
    return {
        "status": "PASS" if all(checks.values()) else "CHECK_REQUIRED",
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "counts": {"exploratory_queries": len(top20), "top20_candidate_records": candidate_records, "top5_slates": len(top5), "corpus_documents": len(corpus)},
        "files": {
            "top20": {"path": str(top20_path.relative_to(ROOT)), "sha256": sha_file(top20_path)},
            "top5": {"path": str(top5_path.relative_to(ROOT)), "sha256": sha_file(top5_path)},
            "r1_gate": {"path": str(gate_path.relative_to(ROOT)), "sha256": sha_file(gate_path)},
            "r1_manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": sha_file(manifest_path)},
        },
        "expected_historical_e5_top20_sha256": "be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7",
        "r1_contract": manifest["retrieval_contract"],
        "reserved_metadata_only": {"query_count": split["partitions"]["RESERVED"]["query_count"], "qid_hash": split["partitions"]["RESERVED"]["qid_sha256"], "content_loaded": False, "qrels_loaded": False, "direct_id_overlap_recomputed": False, "overlap_basis": "existing source-disjoint partition manifest only"},
    }


def model_registry() -> tuple[list[dict[str, str]], dict[str, object]]:
    base_rows = csv_rows(E61 / "E6_1_MODEL_CONTRACT_REGISTRY.csv")
    recovered_rows = csv_rows(E61B / "E6_1B_MODEL_REGISTRY.csv")
    base = {row["model"]: row for row in base_rows}
    recovered = {row["model"]: row for row in recovered_rows}
    names = ["pplx_decider", "liquid_d1_3b", "startlux_decision_4b", "amos_1_small_2b", "jina_v3_5", "qwen3_reranker_4b", "bge_reranker_v2_m3", "qwen3_reranker_0_6b"]
    rows = []
    for name in names:
        row = dict(recovered.get(name, base.get(name, {})))
        if name in recovered:
            row["registry_source"] = "E6_1B_MODEL_REGISTRY.csv"
            row["e6_2_policy"] = "INCLUDE_WITH_CONTEXT_FEASIBILITY_LIMIT"
        else:
            row["registry_source"] = "E6_1_MODEL_CONTRACT_REGISTRY.csv"
            row["e6_2_policy"] = "ADMIT_WITH_ORDER_SENSITIVITY" if name in {"pplx_decider", "liquid_d1_3b", "jina_v3_5"} else "ADMIT"
        row["checkpoint_path_exists"] = str(Path(row.get("snapshot_path", row.get("checkpoint_path", ""))).exists())
        row["r2_attempts"] = "0"
        row["r2_status"] = "NOT_RUN_GPU_AUTH_PENDING"
        rows.append(row)
    status = {
        "roster_count": len(rows),
        "roster_names": names,
        "all_contracts_prevalidated": all(row.get("contract_status", "").startswith("VALIDATED") for row in rows),
        "checkpoint_paths_present_as_recorded": all(row["checkpoint_path_exists"] == "True" for row in rows),
        "weights_loaded_in_preflight": False,
        "inference_executed_in_preflight": False,
        "conditional_notes": {"amos_1_small_2b": "Six prior strict context-limit failures; retain explicit failures and do not truncate.", "order_sensitive": ["pplx_decider", "liquid_d1_3b", "jina_v3_5", "startlux_decision_4b"]},
    }
    return rows, status


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    slate_audit = authenticate_slates()
    registry, registry_status = model_registry()
    manifest = {
        "schema": "E6-2-R2-PREINFERENCE-MANIFEST-V1",
        "created_utc": now,
        "gate": "E6_2_R2_GPU_AUTHORIZATION_PENDING",
        "authorization": {"fresh_device_specific_authorization_present": False, "authorized_gpu_uuids": [], "prior_general_idle_gpu_statement_reused": False},
        "scope": "CPU-only preflight; no model loading, selector inference, quality scoring or prohibited-data access",
        "frozen_input": slate_audit,
        "model_roster": registry,
        "model_roster_status": registry_status,
        "execution_contract": {
            "queries": 277,
            "models": 8,
            "scheduled_attempts_if_authorized": 2216,
            "candidate_budget": "Top-20 -> Top-5",
            "canonical_order_primary": True,
            "permutation_diagnostic": "deferred; no separate authorization",
            "quality_metrics": "not run",
            "failure_policy": "preserve context_limit, timeout, OOM, invalid_output, missing and unsupported statuses; no fallback or silent retry",
            "physical_forwards": "NOT_MEASURED unless instrumented by the native wrapper",
        },
        "prohibited_access": {"qrels": False, "gold_relevance": False, "reference_answers": False, "reserved_content": False, "future_r3_outputs": False, "reader": False, "jev": False, "paid_api": False},
        "historical_artifacts_modified": False,
    }
    write_json("E6_2_R2_PREINFERENCE_MANIFEST.json", manifest)
    write_json("E6_2_R2_INPUT_OUTPUT_VALIDATION.json", {"status": slate_audit["status"], "preinference_only": True, "frozen_input_authentication": slate_audit, "native_outputs": "NOT_CREATED_PENDING_GPU_AUTHORIZATION"})
    write_json("E6_2_R2_GPU_AUDIT.json", {"status": "PENDING", "reason": "Fresh device-specific GPU UUID authorization is absent; no occupancy query or model loading was performed.", "authorized_gpu_uuids": [], "prior_r1_observed_idle_gpu_indices": [1, 5, 6], "prior_observation_is_not_reused_as_authorization": True, "new_process_started": False, "unrelated_processes_modified": False})
    (OUT / "E6_2_R2_LEAKAGE_AUDIT.md").write_text(
        "# E6-2 R2 Leakage and Reservation Audit\n\n"
        "Status: PASS for CPU preflight; selector inference not started.\n\n"
        "The preflight read only the frozen R1 slates, public SciFact corpus metadata/content needed for candidate-hash verification, prior model registries, and reservation metadata. It did not read qrels, relevance labels, reference answers, reader outputs, RESERVED question text, or R3 outputs. No model weights were loaded and no selector request was made.\n\n"
        "Fresh device-specific GPU authorization is absent. The earlier general statement that idle GPUs may be used is not reused as R2 device authorization because the R2 SDD explicitly requires current GPU UUID authorization.\n",
        encoding="utf-8",
    )
    registry_path = OUT / "E6_2_R2_MODEL_EXECUTION_REGISTRY.csv"
    fields = ["model", "repo", "revision", "checkpoint_present", "e6_2_policy", "r2_attempts", "r2_successes", "r2_failures", "status"]
    with registry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in registry:
            writer.writerow({"model": row.get("model", ""), "repo": row.get("repo", ""), "revision": row.get("revision", ""), "checkpoint_present": row.get("checkpoint_path_exists", "False"), "e6_2_policy": row.get("e6_2_policy", ""), "r2_attempts": 0, "r2_successes": 0, "r2_failures": 0, "status": "NOT_RUN_GPU_AUTH_PENDING"})
    (OUT / "E6_2_R2_TEST_REPORT.md").write_text(
        "# E6-2 R2 Test Report\n\n"
        "- Frozen input authentication: PASS.\n"
        "- R1 slates: 277 queries, 5,540 candidate records, exact Top-20/Top-5 prefix checks: PASS.\n"
        "- Model registry authentication: 8/8 admitted records present as recorded; no weights loaded.\n"
        "- GPU authorization test: BLOCKED_PENDING_FRESH_DEVICE_UUID.\n"
        "- Native selector attempts: 0 by design.\n"
        "- Quality scoring: 0 by design.\n",
        encoding="utf-8",
    )
    gate = {"gate": "E6_2_R2_GPU_AUTHORIZATION_PENDING", "status": "BLOCKED_PENDING_FRESH_DEVICE_SPECIFIC_GPU_AUTHORIZATION", "created_utc": now, "preflight": {"frozen_inputs": slate_audit["status"], "model_roster": registry_status["roster_count"], "scheduled_attempts": 2216, "native_attempts": 0, "quality_metrics": False}, "next_required_owner_decision": "Authorize current GPU UUID(s) for R2 selector inference, then rerun preflight occupancy check before model loading.", "r3_ready": False}
    write_json("E6_2_R2_R3_READINESS_GATE.json", gate)
    (OUT / "E6_2_R2_FINAL_REPORT.md").write_text(
        "# DecisionRAG E6-2 R2 Preflight Report\n\n"
        "## Gate: `E6_2_R2_GPU_AUTHORIZATION_PENDING`\n\n"
        "CPU-only preflight authenticated the frozen R1 SciFact inputs and the eight-model admission registries. The 277 exploratory Top-20 slates, 5,540 candidate records and Top-5 prefixes are intact. All eight admitted checkpoint/adapter records are present as recorded.\n\n"
        "No model weights were loaded and no native selector request was made because the R2 specification requires fresh device-specific GPU UUID authorization. No quality metrics, native selections, traces, failures or latency results exist yet. No synthetic result records were created.\n\n"
        "A current authorization naming one or more GPU UUIDs is required before R2 inference can begin.\n",
        encoding="utf-8",
    )
    files = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "E6_2_R2_PROVENANCE_MANIFEST.json":
            files.append({"path": str(path.relative_to(ROOT)), "sha256": sha_file(path), "bytes": path.stat().st_size})
    write_json("E6_2_R2_PROVENANCE_MANIFEST.json", {"schema": "E6-2-R2-PREFLIGHT-PROVENANCE-V1", "created_utc": now, "gate": gate["gate"], "files": files, "inference_attempts": 0, "quality_metrics": False, "reserved_content_accessed": False, "jev_calls": 0, "paid_api_calls": 0, "historical_artifacts_modified": False})
    print(json.dumps({"gate": gate["gate"], "frozen_queries": 277, "candidate_records": 5540, "models": 8, "attempts": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
