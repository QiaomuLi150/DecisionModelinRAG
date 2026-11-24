#!/usr/bin/env python3
"""CPU-only, append-only E2-P0v2 owner acceptance and decision resolution.

This phase resolves only decisions that can be established from existing local
metadata and frozen artifacts.  It deliberately does not open a proposed fresh
holdout gold sidecar, load a checkpoint, contact a service, use a GPU, rebuild an
index, or run inference.  New evidence is written below the P0v2 output tree in
an append-only resolution directory; the original P0v2 artifacts are preserved.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "results/non_jev/v2/e2_p0v2"
M0 = ROOT / "results/non_jev/v2/e2_m0"
OUT = P0 / "decision_resolution"
MISSING = "MISSING"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"non-object JSONL row {path}:{line_no}")
            rows.append(value)
    return rows


def source_record_id(row: dict) -> str:
    """Return an ID-only key; this never reads a question's answer/gold fields."""
    explicit = row.get("source_record_id")
    if explicit is not None:
        return str(explicit)
    query_id = str(row.get("query_id", ""))
    dataset = str(row.get("dataset", ""))
    debug_prefix = dataset + ":debug:"
    if query_id.startswith(debug_prefix):
        return query_id[len(debug_prefix) :]
    prefix = dataset + ":"
    if query_id.startswith(prefix):
        return query_id[len(prefix) :]
    return query_id


def id_sets() -> dict[str, dict[str, set[str]]]:
    files = {
        "cal": ROOT / "data/manifests/split_manifest.jsonl",
        "debug": ROOT / "data/manifests/debug_manifest.jsonl",
        "new_test": ROOT / "results/preflight/v11/NEW_TEST_IDS.jsonl",
    }
    result: dict[str, dict[str, set[str]]] = {
        key: defaultdict(set) for key in files
    }
    for cohort, path in files.items():
        for row in read_jsonl(path):
            dataset = str(row.get("dataset", "unknown"))
            result[cohort][dataset].add(source_record_id(row))
    return result


def count_table(groups: dict[str, dict[str, set[str]]]) -> list[dict]:
    datasets = sorted({ds for cohort in groups.values() for ds in cohort})
    rows: list[dict] = []
    for dataset in datasets:
        cal = groups["cal"].get(dataset, set())
        debug = groups["debug"].get(dataset, set())
        new_test = groups["new_test"].get(dataset, set())
        rows.append(
            {
                "dataset": dataset,
                "cal_or_prior_test_n": len(cal),
                "debug_n": len(debug),
                "new_test_n": len(new_test),
                "cal_debug_intersection_n": len(cal & debug),
                "cal_new_test_intersection_n": len(cal & new_test),
                "debug_new_test_intersection_n": len(debug & new_test),
                "all_three_intersection_n": len(cal & debug & new_test),
                "new_test_unique_n": len(new_test),
                "new_test_status": "PASS" if not (cal & new_test or debug & new_test) else "BLOCKED",
            }
        )
    return rows


def file_inventory(path: Path, role: str, inspect_content: bool = False) -> dict:
    row = {
        "path": str(path),
        "role": role,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
        "content_inspected": False,
    }
    # This branch is intentionally never enabled for proposed fresh holdout
    # releases.  Keeping the flag explicit prevents accidental label opening.
    if inspect_content:
        row["content_inspected"] = True
    return row


def verify_p0_package() -> dict:
    expected = [
        "E2_P0V2_RESEARCH_PROTOCOL.md",
        "E2_P0V2_HYPOTHESES_AND_PRIMARY_CONTRASTS.md",
        "E2_P0V2_SCORING_CONTRACTS.md",
        "E2_P0V2_MODEL_CAPABILITY_MATRIX.csv",
        "E2_P0V2_MECHANISM_EXPERIMENT_MATRIX.csv",
        "E2_P0V2_CANDIDATE_VIEW_CONTRACT.md",
        "E2_P0V2_HOLDOUT_AND_LEAKAGE_PLAN.md",
        "E2_P0V2_STATISTICAL_ANALYSIS_PLAN.md",
        "E2_P0V2_SERVING_BENCHMARK_SDD.md",
        "E2_P0V2_AUTHENTICITY_AND_TELEMETRY_SPEC.md",
        "E2_P0V2_CPU_VALIDATION_REPORT.md",
        "E2_P0V2_PROVENANCE_MANIFEST.json",
        "E2_P0V2_OWNER_DECISION_REGISTER.md",
        "E2_P0V2_OWNER_GATE.md",
    ]
    manifest_path = P0 / "E2_P0V2_PROVENANCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[dict] = []
    for name in expected:
        path = P0 / name
        checks.append({"name": f"required:{name}", "status": "PASS" if path.is_file() else "BLOCKED"})
    output_hashes = manifest.get("outputs", {})
    for name, recorded in output_hashes.items():
        path = P0 / name
        actual = sha256(path) if path.is_file() else None
        checks.append({
            "name": f"hash:{name}",
            "status": "PASS" if actual == recorded else "BLOCKED",
            "recorded": recorded,
            "actual": actual,
        })
    return {
        "manifest_final_status": manifest.get("final_status"),
        "manifest_output_count": manifest.get("output_count"),
        "expected_artifact_count": len(expected),
        "checks": checks,
        "all_required_present": all(c["status"] == "PASS" for c in checks[: len(expected)]),
        "all_recorded_hashes_match": all(c["status"] == "PASS" for c in checks[len(expected) :]),
    }


def main() -> int:
    start = datetime.now(timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)

    groups = id_sets()
    overlaps = count_table(groups)
    package = verify_p0_package()

    # Metadata-only inventory.  The 2Wiki test file is deliberately hashed but
    # not parsed because it may contain answer/gold fields for a future holdout.
    candidates = [
        (ROOT / "data/raw/vendor/musique/musique_ans_v1.0_dev.jsonl", "MuSiQue local dev release; historical source, not fresh"),
        (ROOT / "data/raw/vendor/musique/musique_ans_v1.0_train.jsonl", "MuSiQue local train release; not an approved fresh holdout"),
        (ROOT / "data/raw/vendor/hotpot/hotpot_dev_distractor_v1.parquet", "HotpotQA local dev release; historical source, not fresh"),
        (ROOT / "data/raw/vendor/hotpot/hotpot_train.parquet", "HotpotQA local train release; not an approved fresh holdout"),
        (ROOT / "data/raw/vendor/2wiki/data/dev.json", "2Wiki local dev release; historical source, not fresh"),
        (ROOT / "data/raw/vendor/2wiki/data/train.json", "2Wiki local train release; not an approved fresh holdout"),
        (ROOT / "data/raw/vendor/2wiki/data/test.json", "2Wiki local test release; metadata-only, gold not opened"),
        (ROOT / "data/raw/vendor/enterprise_rag_bench/v1.0.0/questions.jsonl", "EnterpriseRAG-Bench v1.0.0; historical/used source"),
    ]
    source_inventory = [file_inventory(path, role) for path, role in candidates]

    m0_gate_path = M0 / "E2_M0_OWNER_GATE.json"
    m0_gate = json.loads(m0_gate_path.read_text(encoding="utf-8")) if m0_gate_path.is_file() else {}
    m0_no_calls = (
        m0_gate.get("jev_calls") == 0
        and m0_gate.get("paid_api_calls") == 0
        and m0_gate.get("p0v2_authorized") is False
    )

    checks = [
        {"check": "p0v2_required_artifacts", "status": "PASS" if package["all_required_present"] else "BLOCKED"},
        {"check": "p0v2_recorded_hashes", "status": "PASS" if package["all_recorded_hashes_match"] else "BLOCKED"},
        {"check": "m0_preserved_no_paid_or_jev_calls", "status": "PASS" if m0_no_calls else "BLOCKED"},
        {"check": "id_only_overlap_audit_completed", "status": "PASS"},
        {"check": "fresh_holdout_gold_not_opened", "status": "PASS"},
        {"check": "no_inference_or_service_request", "status": "PASS"},
        {"check": "no_p0v2_scientific_freeze", "status": "PASS"},
    ]

    decision_rows = [
        {"decision": "owner_scope_acceptance", "status": "ACCEPTED_OFFLINE_ONLY", "resolution": "Owner accepted P0v2 for offline decision-resolution only; no execution or freeze authorization."},
        {"decision": "scoring_and_view_contract", "status": "RESOLVED_FOR_PLANNING", "resolution": "Use the existing explicit score-contract and candidate-view rules; native feasibility remains model-specific and unverified until authorized audit."},
        {"decision": "primary_H1_pair", "status": "OWNER_VALUE_REQUIRED", "resolution": "The existing Clef-Flash-versus-one-specialist recommendation remains a recommendation, not a frozen contrast."},
        {"decision": "H2b_policy_contrast", "status": "RESOLVED_FOR_PLANNING", "resolution": "If approved, use one globally comparable score vector with global Top-5 versus fixed authenticated-quota Top-5; no historical quartet probabilities are pooled."},
        {"decision": "fresh_confirmatory_holdout", "status": "BLOCKED", "resolution": "No locally verified, untouched release/revision/label lock is available. Local files are inventoried metadata-only; 2Wiki test gold was not opened."},
        {"decision": "noninferiority_epsilon", "status": "OWNER_VALUE_REQUIRED", "resolution": "No numeric epsilon was supplied; no margin is selected or frozen."},
        {"decision": "power_and_sample_size", "status": "BLOCKED_ON_EPSILON_AND_HOLDOUT", "resolution": "Cannot freeze power/sample size without approved epsilon, alpha, target power, discordance assumptions and holdout identity/n."},
        {"decision": "multiplicity", "status": "OWNER_VALUE_REQUIRED", "resolution": "Holm remains a recommendation only; primary family and alpha are not frozen."},
        {"decision": "numeric_p95_SLO_and_load", "status": "OWNER_VALUE_REQUIRED", "resolution": "No numeric serving SLO or load regime is frozen. The prior 2-second proposal remains non-binding."},
        {"decision": "core_model_roster", "status": "PARTIAL_EVIDENCE", "resolution": "Only four locally evidenced model revisions remain eligible for future audit planning; all additional roster entries require external verification."},
        {"decision": "GPU_execution_window", "status": "BLOCKED", "resolution": "This user authorization covers offline work only; no GPU window is opened."},
        {"decision": "Jev", "status": "NOT_AUTHORIZED", "resolution": "Jev remains last and untouched; zero calls."},
    ]

    report = f"""# E2-P0v2 Offline Decision-Resolution Report

**Owner action recorded:** `ACCEPT P0v2 FOR OFFLINE DECISION-RESOLUTION ONLY`

**Current verdict:** `E2_P0V2_PARTIAL_BLOCKED`

**Freeze marker:** `P0v2_NOT_FROZEN`

**Execution window:** CPU-only static and metadata audit, {start} through {datetime.now(timezone.utc).isoformat()} UTC.

## Scope acceptance

The owner acceptance authorizes this offline resolution phase only. It does **not** authorize checkpoint loading, model inference, GPU serving, network/API requests, Qdrant access, retrieval/index construction, training, M1/N/S/P, Jev, paid inference, or opening a proposed fresh holdout gold sidecar. The original P0v2 artifacts remain preserved; this report is an append-only derived record.

## What was resolved

- The P0v2 planning package is accepted as the working specification for offline resolution.
- The explicit score contracts, candidate-view rules, historical quartet non-pooling rule, H2a/H2b/H2c separation, and telemetry requirements are retained for planning.
- Existing P0v2 artifact presence and recorded output hashes were rechecked.
- Existing ID-only metadata was audited for overlap among the local CAL/prior-test manifest, debug fixtures, and the historical `NEW_TEST_IDS` manifest. The audit did not inspect answers or fresh holdout gold.
- Local release files were inventoried by path, size, and SHA-256 only. Availability does not establish untouched-holdout independence.

## Artifact and execution checks

| Check | Result |
|---|---|
| Required P0v2 artifacts | `{package['expected_artifact_count']} expected; {'PASS' if package['all_required_present'] else 'BLOCKED'}` |
| Existing P0v2 recorded hashes | `{'PASS' if package['all_recorded_hashes_match'] else 'BLOCKED'}` |
| M0 no-call/preservation gate | `{'PASS' if m0_no_calls else 'BLOCKED'}` |
| ID-only overlap audit | `PASS` |
| Proposed fresh holdout gold opened | `NO (PASS)` |
| Model/GPU/service/network calls | `0 (PASS)` |
| Jev/paid API calls | `0 (PASS)` |

## ID-only overlap audit

The source-record key is derived from an explicit `source_record_id`, or from the query ID after its dataset/debug namespace. This is an identity audit only; it is not a label audit.

| Dataset | CAL/prior-test | Debug | Historical new-test manifest | CAL∩debug | CAL∩new-test | Debug∩new-test | Status |
|---|---:|---:|---:|---:|---:|---:|---|
""" + "\n".join(
        f"| {row['dataset']} | {row['cal_or_prior_test_n']} | {row['debug_n']} | {row['new_test_n']} | {row['cal_debug_intersection_n']} | {row['cal_new_test_intersection_n']} | {row['debug_new_test_intersection_n']} | {row['new_test_status']} |"
        for row in overlaps
    ) + f"""

`NEW_TEST_IDS.jsonl` is a historical artifact and is not promoted to the new P0v2 confirmatory holdout. Any nonzero overlap would block a future immutable lock rather than trigger outcome-based resampling. Exact machine-readable counts are in `E2_P0V2_OFFLINE_DECISION_LOCK.json`.

## Local source inventory

The local MuSiQue, HotpotQA, 2Wiki, and EnterpriseRAG-Bench files were recorded by path/bytes/hash only. The 2Wiki `test.json` file was deliberately not parsed because it may contain answer/gold fields that must remain closed until a holdout is formally approved. No local file is therefore accepted as the fresh confirmatory holdout by this phase.

## Decision register after offline resolution

| Decision | Status | Consequence |
|---|---|---|
""" + "\n".join(
        f"| {row['decision']} | `{row['status']}` | {row['resolution']} |" for row in decision_rows
    ) + f"""

## Remaining freeze blockers

1. Approve and lock an exact untouched confirmatory holdout identity, release/revision, sample size, document/source independence, label-access protocol, and hashes.
2. Select the primary H1 pair or explicitly approve a CAL-only selection algorithm.
3. Approve an absolute noninferiority margin, alpha/power/sample-size plan, and multiplicity family.
4. Approve the core checkpoint roster/revisions and complete external capability/license/hash verification for any models beyond the four locally evidenced entries.
5. Approve a numeric p95 serving SLO, load regime, and an isolated execution window before any GPU/inference phase.

Until all mandatory items are resolved and independently validated, the required state remains **`P0v2_NOT_FROZEN`**. No scientific result, holdout, checkpoint roster, SLO, or paid phase has been frozen by this report.

## Stop gate

This phase stops here. No M1/N/S/P, model request, GPU workload, Qdrant/index operation, service restart, dataset expansion, Jev request, or paid API request was made.
"""
    report_path = OUT / "E2_P0V2_OFFLINE_DECISION_RESOLUTION.md"
    write_text(report_path, report)

    lock = {
        "schema": "e2-p0v2-offline-decision-lock-v1",
        "owner_acceptance": {
            "accepted": True,
            "scope": "offline_decision_resolution_only",
            "accepted_utc": start,
            "scientific_freeze": False,
            "freeze_marker": "P0v2_NOT_FROZEN",
        },
        "status": "E2_P0V2_PARTIAL_BLOCKED",
        "freeze_marker": "P0v2_NOT_FROZEN",
        "execution_scope": {
            "cuda_visible_devices": "",
            "gpu_calls": 0,
            "model_calls": 0,
            "checkpoint_loads": 0,
            "network_calls": 0,
            "qdrant_calls": 0,
            "retrieval_or_index_runs": 0,
            "jev_calls": 0,
            "paid_api_calls": 0,
            "fresh_holdout_gold_opened": 0,
            "note": "standard-library metadata and hash checks only",
        },
        "p0v2_package": package,
        "m0_gate_status": m0_gate.get("status"),
        "m0_no_calls_and_p0v2_not_previously_authorized": m0_no_calls,
        "source_record_overlap": overlaps,
        "source_inventory_metadata_only": source_inventory,
        "decisions": decision_rows,
        "checks": checks,
        "required_final_state": "P0v2_NOT_FROZEN",
        "stop": True,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "project_root": str(ROOT),
        },
    }
    lock_path = OUT / "E2_P0V2_OFFLINE_DECISION_LOCK.json"
    write_json(lock_path, lock)

    # A separate acceptance receipt is intentionally small and immutable in
    # meaning: it records scope acceptance without changing the original gate.
    receipt = {
        "schema": "e2-p0v2-owner-acceptance-receipt-v1",
        "owner_action": "accept_p0v2_offline_decision_resolution_only",
        "accepted_utc": start,
        "status": "E2_P0V2_PARTIAL_BLOCKED",
        "freeze_marker": "P0v2_NOT_FROZEN",
        "stop": True,
        "prohibited_actions": [
            "model_inference", "gpu_workload", "network", "qdrant", "retrieval_or_index_rebuild",
            "training", "jev", "paid_api", "fresh_holdout_gold_access", "dataset_expansion",
        ],
        "report": str(report_path),
        "lock": str(lock_path),
    }
    receipt_path = OUT / "E2_P0V2_OWNER_ACCEPTANCE_RECEIPT.json"
    write_json(receipt_path, receipt)

    # Record hashes after all resolution outputs have been written.
    # A self-hash would be circular and would become stale on serialization;
    # the lock records the immutable companion artifact hashes only.
    lock["resolution_artifacts"] = {
        report_path.name: sha256(report_path),
        receipt_path.name: sha256(receipt_path),
    }
    write_json(lock_path, lock)

    summary = {
        "status": lock["status"],
        "freeze_marker": lock["required_final_state"],
        "out_dir": str(OUT),
        "overlap_rows": len(overlaps),
        "p0v2_artifacts_present": package["all_required_present"],
        "p0v2_hashes_match": package["all_recorded_hashes_match"],
        "cpu_checks": len(checks),
        "inference_or_external_calls": 0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
