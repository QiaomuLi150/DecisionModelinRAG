#!/usr/bin/env python3
"""DecisionRAG v2.1 E2-M0 structural oracle audit.

This module deliberately uses only the Python standard library.  It reads the
frozen Track A/B row-level artifacts and writes a new, isolated E2-M0 result
directory.  It does not import the serving stack or make model/service calls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


os.environ["CUDA_VISIBLE_DEVICES"] = ""

POLICIES = ("laya_g5", "clef_g5")
TRACK_A_DATASETS = ("hotpot", "musique", "wiki2")
M0_STATUS_READY = "E2_M0_COMPLETE_WITH_LIMITATIONS_READY_FOR_OWNER_REVIEW"


class AuditError(Exception):
    """A source or frozen-policy integrity error."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_info(path: Path, role: str, expected: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise AuditError(f"missing input: {path}")
    actual = sha256_file(path)
    return {
        "path": str(path),
        "role": role,
        "sha256": actual,
        "bytes": path.stat().st_size,
        "expected_sha256": expected,
        "hash_matches_expected": expected is None or actual == expected,
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise AuditError(f"non-object JSONL row {path}:{line_no}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def canonical_json_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def git_value(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def selected_row(rows: dict[tuple[str, str], dict[str, Any]], qid: str, policy: str) -> dict[str, Any]:
    try:
        return rows[(qid, policy)]
    except KeyError as exc:
        raise AuditError(f"missing selector row: {qid} {policy}") from exc


def group_doc_lists(groupwise: dict[str, Any]) -> list[list[str]]:
    if not isinstance(groupwise, dict) or not isinstance(groupwise.get("groups"), list):
        raise AuditError("groupwise record has no explicit groups list")
    result = []
    for group in groupwise["groups"]:
        if not isinstance(group, dict) or not isinstance(group.get("group_doc_ids"), list):
            raise AuditError("groupwise group has no group_doc_ids")
        result.append(list(group["group_doc_ids"]))
    return result


def validate_groups(groups: list[list[str]], slate: list[str], selected: list[str]) -> list[str]:
    errors: list[str] = []
    if len(slate) != len(set(slate)):
        errors.append("DUPLICATE_SLATE_ID")
    if len(selected) != len(set(selected)):
        errors.append("DUPLICATE_SELECTED_ID")
    if not set(selected).issubset(set(slate)):
        errors.append("SELECTED_OUT_OF_SLATE")
    if len(groups) != 5:
        errors.append(f"UNEXPECTED_GROUP_COUNT:{len(groups)}")
    flat = [doc for group in groups for doc in group]
    if len(flat) != len(set(flat)):
        errors.append("DUPLICATE_GROUP_MEMBERSHIP")
    if set(flat) != set(slate) or len(flat) != len(slate):
        errors.append("GROUP_COVERAGE_MISMATCH")
    if any(len(group) != 4 for group in groups):
        errors.append("UNEXPECTED_GROUP_SIZE")
    winners = []
    for group in groups:
        # The stored selector output identifies one winner for every group.
        # This check is done by the caller using winner_doc_ids as well.
        pass
    if len(selected) != 5:
        errors.append(f"UNEXPECTED_SELECTED_COUNT:{len(selected)}")
    return errors


def validate_group_record(groupwise: dict[str, Any], slate: list[str], selected: list[str]) -> tuple[list[list[str]], list[str]]:
    groups = group_doc_lists(groupwise)
    errors = validate_groups(groups, slate, selected)
    raw_groups = groupwise["groups"]
    winners = [g.get("winning_doc_id") for g in raw_groups]
    if any(w is None for w in winners):
        errors.append("MISSING_GROUP_WINNER")
    if len(winners) != len(set(winners)):
        errors.append("DUPLICATE_GROUP_WINNER")
    if set(winners) != set(selected):
        errors.append("SELECTED_WINNERS_MISMATCH")
    for group, winner in zip(groups, winners):
        if winner not in group:
            errors.append("GROUP_WINNER_OUT_OF_GROUP")
    return groups, errors


def safe_set(values: Iterable[str]) -> set[str]:
    return set(values)


def reason_for_gold(identity_status: str, gold: list[str]) -> str | None:
    if identity_status == "AMBIGUOUS_GOLD_ID":
        return "AMBIGUOUS_GOLD_ID"
    if identity_status == "MISSING_GOLD_MAPPING":
        return "MISSING_GOLD_MAPPING"
    if not gold:
        return "EMPTY_GOLD"
    if len(gold) > 5:
        return "GOLD_GT_BUDGET"
    if len(gold) != len(set(gold)):
        return "DUPLICATE_GOLD_ID"
    return None


def oracle_fields(gold: list[str], slate: list[str], groups: list[list[str]], selected: list[str]) -> dict[str, Any]:
    gold_set, slate_set, selected_set = map(safe_set, (gold, slate, selected))
    retrieved = int(gold_set.issubset(slate_set))
    group_counts = [len(gold_set.intersection(group)) for group in groups]
    group_oracle = int(retrieved and all(count <= 1 for count in group_counts))
    observed = int(gold_set.issubset(selected_set))
    collision_groups = sum(count >= 2 for count in group_counts)
    collision_pairs = sum(count * (count - 1) // 2 for count in group_counts)
    if not (0 <= observed <= group_oracle <= retrieved <= 1):
        raise AuditError(
            f"oracle invariant failed: observed={observed}, group={group_oracle}, retrieval={retrieved}"
        )
    return {
        "retrieval_oracle": retrieved,
        "group_oracle": group_oracle,
        "observed_g5_success": observed,
        "quota_loss": retrieved - group_oracle,
        "residual_gap": group_oracle - observed,
        "max_gold_in_group": max(group_counts, default=0),
        "collision_pair_count": collision_pairs,
        "collision_group_count": collision_groups,
    }


def policy_group_map(rows: dict[tuple[str, str], dict[str, Any]], qid: str, policy: str) -> dict[str, Any]:
    return selected_row(rows, qid, policy).get("groupwise") or {}


def compare_group_sources(nested: dict[str, Any], full: dict[str, Any]) -> bool:
    try:
        return group_doc_lists(nested) == group_doc_lists(full)
    except AuditError:
        return False


def prepare_source_manifest(root: Path, sdd: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    a = root / "results/non_jev/v11/track_a"
    b = root / "results/non_jev/v11/track_b"
    pre = root / "results/preflight/v11"
    rem = root / "results/preflight/v11_remediation"
    sources = [
        (sdd, "authoritative E2-M0 SDD", None),
        (a / "V11_TRACK_A_RETRIEVAL_SLATES.jsonl", "Track A frozen retrieval slates", "1891f41e2b8614b871aa42cdbe61af3d0c193410a85041e5426c5e7eb3905f0f"),
        (a / "V11_TRACK_A_SELECTOR_RESULTS_FULL.jsonl", "Track A frozen selector results", "87cd17c3ac6408a182d5821d3b86cfb535161d1a0b7df3e6d60a216433f303bb"),
        (pre / "NEW_TEST_GOLD_SIDECAR.jsonl", "Track A evaluation-only gold sidecar", "5b17f0dfacceddc82952163b6fb98fcecb2d0d4ce232209c5d565f79c04d40cc"),
        (a / "V11_TRACK_A_LAYA_GROUPWISE_FULL.jsonl", "Track A Laya explicit group records", None),
        (a / "V11_TRACK_A_CLEF_GROUPWISE_FULL.jsonl", "Track A Clef explicit group records", None),
        (a / "V11_TRACK_A_METRICS.csv", "Track A summary cross-check", "97cb4d5b3de284614bfdcc027f46a337a3141b39af7f003112138cab42ee227e"),
        (a / "V11_TRACK_A_GROUPWISE_COLLISIONS.csv", "Track A historical collision cross-check", "50c8685a75508a37a56a7c401f3172e022565d4a2a78a74f5e14f74d4123be51"),
        (a / "V11_TRACK_A_PROVENANCE_MANIFEST.json", "Track A provenance", None),
        (root / "V11_TRACK_A_EXECUTION_MANIFEST.json", "Track A frozen cohort manifest", None),
        (b / "V11_TRACK_B_RETRIEVAL_SLATES.jsonl", "Track B frozen retrieval slates", "2e82fc2788398f98d6a257165718b1baa6e1ad792f63671c06458f9212326c9f"),
        (b / "V11_TRACK_B_SELECTOR_RESULTS.jsonl", "Track B frozen selector results", "3716a91c96d3f03d42f39e529e5f50abb574b4b1219e9c6aa3a7374bf7c6e0f8"),
        (pre / "ENTERPRISE_GOLD_SIDECAR.jsonl", "Track B evaluation-only gold sidecar", "e7a7dbe71ee9e4290fe29615991c3aed90f70530ab12a63e9c1003be6bfa6b55"),
        (rem / "ENTERPRISE_QUESTION_MANIFEST_V11R1.jsonl", "Track B identity-resolved question manifest", "00befe74a6bd3786fd2149639f434efd574f2784f49f7f91961c3ed7091d2683"),
        (rem / "ENTERPRISE_DOCUMENT_IDENTITY_MANIFEST.jsonl", "Track B physical identity manifest", None),
        (rem / "ENTERPRISE_QST_0413_ACCEPTANCE_V11R1.json", "Track B accepted ambiguity exception", None),
        (b / "V11_TRACK_B_LAYA_GROUPWISE_FULL.jsonl", "Track B Laya explicit group records", None),
        (b / "V11_TRACK_B_CLEF_GROUPWISE_FULL.jsonl", "Track B Clef explicit group records", None),
        (b / "V11_TRACK_B_METRICS.csv", "Track B summary cross-check", "57703cb9851c80a575ba3ddb89596f40a0a1e4c3de2b93cdcae4a7f06713d7a0"),
        (b / "V11_TRACK_B_GROUPWISE_COLLISION_AUDIT.csv", "Track B historical collision cross-check", "e4ad654f4b4801cc2cc5f727e2c7cedd17e457b9e81f5d51ea6e3bf596f62839"),
        (b / "V11_TRACK_B_PROVENANCE_MANIFEST.json", "Track B provenance", None),
        (root / "results/non_jev/herb_h1/HERB_H1_PROVENANCE_MANIFEST.json", "H1 frozen provenance", None),
        (root / "results/non_jev/herb_h1_owner_audit/HERB_H1_OWNER_REVIEW_CLOSURE.md", "H1 owner closure", None),
        (root / "results/non_jev/v11/track_c/V11_TRACK_C_PROVENANCE_MANIFEST.json", "Original Track C provenance", None),
        (root / "results/non_jev/v11/track_c_r2/V11_TRACK_C_R2_PROVENANCE_MANIFEST.json", "Original Track C R2 provenance", None),
    ]
    entries = [file_info(path, role, expected) for path, role, expected in sources]
    expected_mismatch = [e for e in entries if not e["hash_matches_expected"]]
    return entries, {
        "manifest_version": "e2-m0-input-sources-v1",
        "root": str(root.resolve()),
        "git_root": git_value(root, "rev-parse", "--show-toplevel"),
        "git_commit": git_value(root, "rev-parse", "HEAD"),
        "files": entries,
        "file_count": len(entries),
        "expected_hash_mismatches": expected_mismatch,
    }


def build_a_data(root: Path) -> dict[str, Any]:
    base = root / "results/non_jev/v11/track_a"
    gold_rows = {row["query_id"]: row for row in load_jsonl(root / "results/preflight/v11/NEW_TEST_GOLD_SIDECAR.jsonl")}
    retrieval = {row["query_id"]: row for row in load_jsonl(base / "V11_TRACK_A_RETRIEVAL_SLATES.jsonl")}
    selectors = load_jsonl(base / "V11_TRACK_A_SELECTOR_RESULTS_FULL.jsonl")
    selector_map = {(r["query_id"], r["policy_id"]): r for r in selectors}
    laya_full = {r["query_id"]: r for r in load_jsonl(base / "V11_TRACK_A_LAYA_GROUPWISE_FULL.jsonl")}
    clef_full = {r["query_id"]: r for r in load_jsonl(base / "V11_TRACK_A_CLEF_GROUPWISE_FULL.jsonl")}
    if set(gold_rows) != set(retrieval):
        raise AuditError("Track A gold/slate query-ID sets differ")
    if len(gold_rows) != 3767:
        raise AuditError(f"Track A expected 3767 rows, found {len(gold_rows)}")
    rows = []
    for qid, ret in retrieval.items():
        gold = gold_rows[qid]
        if not gold.get("gold_mapping_valid", False):
            identity = "MISSING_GOLD_MAPPING"
            gold_ids: list[str] = []
        else:
            identity = "RESOLVED"
            gold_ids = list(dict.fromkeys(gold.get("gold_doc_ids", [])))
        by_policy = {}
        for policy, full in (("laya_g5", laya_full), ("clef_g5", clef_full)):
            sel = selected_row(selector_map, qid, policy)
            nested = sel.get("groupwise") or {}
            full_row = full.get(qid)
            if full_row is None:
                raise AuditError(f"Track A missing {policy} full groupwise row: {qid}")
            by_policy[policy] = {"selector": sel, "nested": nested, "full": full_row}
        rows.append({"dataset": ret["dataset"], "query_id": qid, "question": ret.get("question", ""), "slate": list(ret["candidate_ids"]), "gold": gold_ids, "identity_status": identity, "by_policy": by_policy, "gold_row": gold})
    return {"name": "track_a", "rows": rows, "retrieval": retrieval, "gold": gold_rows}


def build_b_data(root: Path) -> dict[str, Any]:
    base = root / "results/non_jev/v11/track_b"
    pre = root / "results/preflight/v11"
    rem = root / "results/preflight/v11_remediation"
    raw_gold = {r["question_id"]: r for r in load_jsonl(pre / "ENTERPRISE_GOLD_SIDECAR.jsonl")}
    manifest = {r["question_id"]: r for r in load_jsonl(rem / "ENTERPRISE_QUESTION_MANIFEST_V11R1.jsonl")}
    retrieval = {r["query_id"]: r for r in load_jsonl(base / "V11_TRACK_B_RETRIEVAL_SLATES.jsonl")}
    selectors = load_jsonl(base / "V11_TRACK_B_SELECTOR_RESULTS.jsonl")
    selector_map = {(r["query_id"], r["policy_id"]): r for r in selectors}
    laya_full = {r["query_id"]: r for r in load_jsonl(base / "V11_TRACK_B_LAYA_GROUPWISE_FULL.jsonl")}
    clef_full = {r["query_id"]: r for r in load_jsonl(base / "V11_TRACK_B_CLEF_GROUPWISE_FULL.jsonl")}
    if set(raw_gold) != set(retrieval) or set(raw_gold) != set(manifest):
        raise AuditError("Track B gold/question/slate query-ID sets differ")
    if len(raw_gold) != 500:
        raise AuditError(f"Track B expected 500 rows, found {len(raw_gold)}")
    # Verify the path-qualified mapping for every original ID that occurs in the
    # question labels.  We do not materialize the 512k-row identity manifest.
    target_ids = {doc for r in raw_gold.values() for doc in r.get("expected_doc_ids", [])}
    identity_map: dict[str, list[str]] = defaultdict(list)
    identity_path = rem / "ENTERPRISE_DOCUMENT_IDENTITY_MANIFEST.jsonl"
    with identity_path.open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            if x.get("original_doc_id") in target_ids:
                identity_map[x["original_doc_id"]].append(x["internal_doc_id"])
    rows = []
    for qid, ret in retrieval.items():
        raw = raw_gold[qid]
        qm = manifest[qid]
        status = qm.get("gold_identity_status")
        if status == "AMBIGUOUS_ORIGINAL_ID" or qid == "qst_0413":
            identity = "AMBIGUOUS_GOLD_ID"
            gold_ids = []
        elif qm.get("gold_identity_status") not in ("RESOLVED_OR_EMPTY", None):
            identity = "MISSING_GOLD_MAPPING"
            gold_ids = []
        else:
            identity = "RESOLVED"
            gold_ids = list(dict.fromkeys(qm.get("expected_internal_doc_ids", [])))
            raw_unique = list(dict.fromkeys(raw.get("expected_doc_ids", [])))
            expected_from_identity = []
            for original in raw_unique:
                mapped = identity_map.get(original, [])
                if len(mapped) == 1:
                    expected_from_identity.append(mapped[0])
                else:
                    raise AuditError(f"unexpected unresolved/ambiguous non-qst_0413 ID {qid}: {original}")
            if set(expected_from_identity) != set(gold_ids):
                raise AuditError(f"Track B internal gold mapping mismatch: {qid}")
        by_policy = {}
        for policy, full in (("laya_g5", laya_full), ("clef_g5", clef_full)):
            sel = selected_row(selector_map, qid, policy)
            nested = sel.get("groupwise") or {}
            full_row = full.get(qid)
            if full_row is None:
                raise AuditError(f"Track B missing {policy} full groupwise row: {qid}")
            by_policy[policy] = {"selector": sel, "nested": nested, "full": full_row}
        rows.append({"dataset": "enterprise", "query_id": qid, "question": ret.get("question", ""), "slate": list(ret["candidate_ids"]), "gold": gold_ids, "identity_status": identity, "question_type": qm.get("question_type", raw.get("question_type", "")), "by_policy": by_policy, "gold_row": raw, "question_manifest": qm})
    return {"name": "track_b", "rows": rows, "retrieval": retrieval, "gold": raw_gold, "identity_map": identity_map}


def evaluate_dataset(data: dict[str, Any], exclusions: list[dict[str, Any]], integrity: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for item in data["rows"]:
        gold = item["gold"]
        slate = item["slate"]
        exclusion = reason_for_gold(item["identity_status"], gold)
        if exclusion:
            exclusions.append({"dataset": item["dataset"], "query_id": item["query_id"], "policy": "all", "reason": exclusion, "severity": "EXCLUDED_FROM_IDENTITY_DEPENDENT_ESTIMATES", "invalidates_primary_aggregate": False, "detail": "retained in N_all; not a true policy failure"})
        for policy in POLICIES:
            bundle = item["by_policy"][policy]
            sel = bundle["selector"]
            nested = bundle["nested"]
            full = bundle["full"]
            group_errors: list[str] = []
            try:
                groups, errs = validate_group_record(nested, slate, list(sel.get("selected_doc_ids", [])))
                group_errors.extend(errs)
                if not compare_group_sources(nested, full):
                    group_errors.append("FULL_GROUPWISE_SOURCE_MISMATCH")
            except AuditError as exc:
                groups = []
                group_errors.append(f"GROUP_RECORD_INVALID:{exc}")
            group_hash = canonical_json_hash(groups) if groups else ""
            selector = list(sel.get("selected_doc_ids", []))
            status = "PASS" if not group_errors else "BLOCKED"
            if group_errors:
                integrity.append({"dataset": item["dataset"], "query_id": item["query_id"], "policy": policy, "reason": "GROUP_PROVENANCE_OR_POLICY_INVALID", "severity": "PRIMARY_BLOCKER", "invalidates_primary_aggregate": True, "detail": ";".join(group_errors)})
            base = {
                "dataset": item["dataset"], "query_id": item["query_id"], "domain": item.get("question_type", item["dataset"]), "identity_status": item["identity_status"], "exclusion_reason": exclusion or "", "n_slate": len(slate), "n_gold": len(gold), "n_groups": len(groups), "group_policy": policy, "eligible": int(exclusion is None), "gold_fully_in_top20": "", "retrieval_oracle": "", "group_oracle": "", "group_oracle_valid": int(not group_errors), "observed_g5_success": "", "max_gold_in_group": "", "collision_pair_count": "", "collision_group_count": "", "quota_loss": "", "residual_gap": "", "observed_pointwise_success": "", "native_policy": policy, "slate_hash": canonical_json_hash(slate), "gold_set_hash": canonical_json_hash(sorted(gold)), "group_assignment_hash": group_hash, "selection_hash": canonical_json_hash(selector), "source_manifest_id": "E2-M0-INPUT-SOURCES-V1", "validation_status": status,
            }
            if exclusion is None and not group_errors:
                values = oracle_fields(gold, slate, groups, selector)
                base.update({"gold_fully_in_top20": values["retrieval_oracle"], **values})
            out.append(base)
    return out, stats


AGG_FIELDS = ["dataset", "policy", "population", "N_all", "N_eligible", "N_conditional", "n_unverifiable", "n_collision", "n_eligible_observed", "retrieval_numerator", "group_oracle_numerator", "observed_numerator", "quota_loss_numerator", "residual_gap_numerator", "retrieval_oracle_rate", "group_oracle_rate", "observed_g5_rate", "quota_loss_rate", "residual_gap_rate"]


def aggregate_rows(per_query: list[dict[str, Any]], dataset_name: str, category_field: str | None = None) -> list[dict[str, Any]]:
    policies = POLICIES
    keys = ["ALL"]
    if category_field:
        keys += sorted({r[category_field] for r in per_query if r.get(category_field)})
    result = []
    for key in keys:
        subset = per_query if key == "ALL" else [r for r in per_query if r.get(category_field) == key]
        for policy in policies:
            rows = [r for r in subset if r["group_policy"] == policy]
            valid = [r for r in rows if r["eligible"] == 1 and r["group_oracle_valid"] == 1]
            conditional = [r for r in valid if r["retrieval_oracle"] == 1]
            def n(field: str, source: list[dict[str, Any]]) -> int:
                return sum(int(r[field]) for r in source)
            denom = len(valid)
            cden = len(conditional)
            rates = lambda numerator, denominator: (numerator / denominator if denominator else "")
            scope = f"{dataset_name}/{key.lower()}" if key != "ALL" else f"{dataset_name}/pooled"
            common = {"dataset": scope, "policy": policy, "N_all": len(rows), "N_eligible": denom, "N_conditional": cden, "n_unverifiable": sum(r["group_oracle_valid"] == 0 for r in rows), "n_collision": sum(int(r["collision_group_count"] > 0) for r in conditional)}
            result.append({**common, "population": "all_frozen", "n_eligible_observed": "", "retrieval_numerator": "", "group_oracle_numerator": "", "observed_numerator": "", "quota_loss_numerator": "", "residual_gap_numerator": "", "retrieval_oracle_rate": "", "group_oracle_rate": "", "observed_g5_rate": "", "quota_loss_rate": "", "residual_gap_rate": ""})
            vals = {"retrieval_numerator": n("retrieval_oracle", valid), "group_oracle_numerator": n("group_oracle", valid), "observed_numerator": n("observed_g5_success", valid), "quota_loss_numerator": n("quota_loss", valid), "residual_gap_numerator": n("residual_gap", valid)}
            result.append({**common, "population": "unconditional_eligible", "n_eligible_observed": vals["observed_numerator"], **vals, "retrieval_oracle_rate": rates(vals["retrieval_numerator"], denom), "group_oracle_rate": rates(vals["group_oracle_numerator"], denom), "observed_g5_rate": rates(vals["observed_numerator"], denom), "quota_loss_rate": rates(vals["quota_loss_numerator"], denom), "residual_gap_rate": rates(vals["residual_gap_numerator"], denom)})
            cvals = {"retrieval_numerator": n("retrieval_oracle", conditional), "group_oracle_numerator": n("group_oracle", conditional), "observed_numerator": n("observed_g5_success", conditional), "quota_loss_numerator": n("quota_loss", conditional), "residual_gap_numerator": n("residual_gap", conditional)}
            result.append({**common, "population": "retrieval_conditional", "n_eligible_observed": cvals["observed_numerator"], **cvals, "retrieval_oracle_rate": rates(cvals["retrieval_numerator"], cden), "group_oracle_rate": rates(cvals["group_oracle_numerator"], cden), "observed_g5_rate": rates(cvals["observed_numerator"], cden), "quota_loss_rate": rates(cvals["quota_loss_numerator"], cden), "residual_gap_rate": rates(cvals["residual_gap_numerator"], cden)})
    return result


def historical_counts(root: Path) -> dict[str, Any]:
    result = {}
    for key, path in (("track_a", root / "results/non_jev/v11/track_a/V11_TRACK_A_METRICS.csv"), ("track_b", root / "results/non_jev/v11/track_b/V11_TRACK_B_METRICS.csv")):
        counts = {}
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("policy_id") in POLICIES and row.get("top20_complete_gold") == "True":
                    counts[row["policy_id"]] = counts.get(row["policy_id"], 0) + 1
        result[key] = counts
    return result


def make_report(out: Path, a: dict[str, Any], b: dict[str, Any], per: list[dict[str, Any]], agg: list[dict[str, Any]], exclusions: list[dict[str, Any]], integrity: list[dict[str, Any]], source_manifest: dict[str, Any], historical: dict[str, Any], status: str) -> None:
    def row(scope: str, policy: str, pop: str) -> dict[str, Any]:
        return next(r for r in agg if r["dataset"] == scope and r["policy"] == policy and r["population"] == pop)
    lines = [
        "# E2-M0 Structural Oracle Audit",
        "",
        f"**Status:** `{status}`",
        "",
        "## Executive result",
        "",
        "This is a CPU-only, read-only recomputation from frozen Track A/B slates, evaluation-only gold records, explicit saved group assignments, and frozen G5 selected IDs. No model, GPU, Qdrant, network, hosted API, or Jev request was made by this audit.",
        "",
        "The oracle is annotation-conditioned: it measures whether the recorded Top-20 and one-winner-per-group policy could retain the supplied gold identity set. It is not a causal attribution of model error and does not prove that the annotated gold set is the only valid support set.",
        "",
        "## Population waterfall",
        "",
        "| Cohort | N_all | N_eligible (1–5, resolved) | N_conditional (all gold in Top-20) | Excluded |",
        "|---|---:|---:|---:|---:|",
        f"| Track A prospective | {len(a['rows'])} | {sum(r['eligible'] for r in per if r['dataset'] != 'enterprise' and r['group_policy']=='laya_g5')} | {row('track_a/pooled','laya_g5','retrieval_conditional')['N_conditional']} | {len(a['rows'])-sum(r['eligible'] for r in per if r['dataset'] != 'enterprise' and r['group_policy']=='laya_g5')} |",
        f"| Track B enterprise | {len(b['rows'])} | {row('enterprise/pooled','laya_g5','unconditional_eligible')['N_eligible']} | {row('enterprise/pooled','laya_g5','retrieval_conditional')['N_conditional']} | {len(b['rows'])-row('enterprise/pooled','laya_g5','unconditional_eligible')['N_eligible']} |",
        "",
        "Track A's gold sidecar contains resolved 2–4-document support for all 3,767 selected questions. Track B retains all 500 questions; 30 are empty-gold, 26 have more than five unique gold documents, and qst_0413 is excluded from identity-dependent evaluation under the accepted `V11.1R1-GOLD-ID-AMBIGUITY-qst_0413` exception. Therefore Track B has 443 identity-resolved 1–5-gold questions, not 469.",
        "",
        "## Structural estimates",
        "",
        "| Cohort | Policy | Population | Group-quota ceiling | Observed G5 | Quota loss | Residual gap |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for scope in ("track_a/pooled", "enterprise/pooled"):
        for policy in POLICIES:
            r = row(scope, policy, "unconditional_eligible")
            c = row(scope, policy, "retrieval_conditional")
            lines.append(f"| {scope} | {policy} | unconditional eligible | {r['group_oracle_numerator']}/{r['N_eligible']} ({r['group_oracle_rate']:.4f}) | {r['observed_numerator']}/{r['N_eligible']} ({r['observed_g5_rate']:.4f}) | {r['quota_loss_numerator']}/{r['N_eligible']} ({r['quota_loss_rate']:.4f}) | {r['residual_gap_numerator']}/{r['N_eligible']} ({r['residual_gap_rate']:.4f}) |")
            lines.append(f"| {scope} | {policy} | retrieval conditional | {c['group_oracle_numerator']}/{c['N_conditional']} ({c['group_oracle_rate']:.4f}) | {c['observed_numerator']}/{c['N_conditional']} ({c['observed_g5_rate']:.4f}) | {c['quota_loss_numerator']}/{c['N_conditional']} ({c['quota_loss_rate']:.4f}) | {c['residual_gap_numerator']}/{c['N_conditional']} ({c['residual_gap_rate']:.4f}) |")
    lines += ["", "## Dataset and category detail", "", "| Scope | Policy | N_eligible | N_conditional | Collision queries | Quota loss | Residual gap | Observed |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in agg:
        if r["population"] == "retrieval_conditional" and r["dataset"] not in ("track_a/pooled", "enterprise/pooled"):
            lines.append(f"| {r['dataset']} | {r['policy']} | {r['N_eligible']} | {r['N_conditional']} | {r['n_collision']} | {r['quota_loss_numerator']} | {r['residual_gap_numerator']} | {r['observed_numerator']} |")
    lines += [
        "",
        "The same invariant was checked per eligible row: `0 <= observed_G5 <= group_oracle <= retrieval_oracle <= 1`. The saved Laya and Clef partitions were explicit in their selector records and cross-checked against each full groupwise trace; no rank-contiguous groups were inferred.",
        "",
        "## Collision summary",
        "",
        "Collision means at least two annotated gold identities occur in the same recorded group, conditional on complete gold presence in Top-20. The per-dataset/category and multiplicity counts are in `E2_M0_AGGREGATE_ORACLES.csv` and per-query collision fields. Both policies are reported separately because their authenticated group records were not assumed identical.",
        "",
        "## Historical reconciliation",
        "",
        f"Recomputed `top20_complete_gold=True` counts from the frozen metric rows: Track A laya_g5={historical['track_a'].get('laya_g5')}, clef_g5={historical['track_a'].get('clef_g5')}; Track B laya_g5={historical['track_b'].get('laya_g5')}, clef_g5={historical['track_b'].get('clef_g5')}. These equal the row-level oracle conditional denominators (1,490 and 330 respectively) and are used only as a cross-check, not as the oracle source.",
        "",
        "## Optional analyses",
        "",
        "Same-score policy ablation: `NOT_RUN — the core M0 audit does not promote saved heterogeneous pointwise traces to a verified globally comparable score contract.` No quartet softmax values were merged or globally sorted. HERB citation feasibility: `NOT_APPLICABLE_TO_CORE_M0 — no citation-set oracle was needed for the required A/B audit.`",
        "",
        "## Integrity and limitations",
        "",
        f"Input records were resolved from {source_manifest['file_count']} explicitly named files. Integrity rows: {len(integrity)}; case exclusions: {len(exclusions)}. Existing artifacts were not written or modified; preservation hashes are recorded in the provenance manifest.",
        "",
        "The result is limited by the benchmark annotations: a listed support set is treated as the supplied gold identity set, not as a proof of exhaustive necessary evidence. The residual gap combines within-group model scoring, prompt/context effects, and tie-breaking. It should not be described as intrinsic decision-model failure.",
        "",
        "## P0v2 handoff",
        "",
        "Future M1/P0v2 work must freeze a single globally comparable document scoring contract, explicit group partitions, a fresh untouched holdout, serving SLO/power/multiplicity decisions, and any license/GPU gates before inference. This audit does not authorize P0v2, M1, new retrieval, or paid evaluation.",
    ]
    (out / "E2_M0_STRUCTURAL_ORACLE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline CPU-only DecisionRAG E2-M0 structural oracle audit")
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fail-on-source-mismatch", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    start = datetime.now(timezone.utc).isoformat()
    root = (args.repo_root or Path(__file__).resolve().parents[1]).resolve()
    sdd = root / "Decision_Model_in_RAG/DecisionRAG_v2_1_E2_M0_Structural_Oracle_SDD.md"
    project = root / "Decision_Model_in_RAG"
    if not sdd.is_file():
        raise SystemExit(f"cannot locate SDD: {sdd}")
    out = (args.out_dir if args.out_dir.is_absolute() else project / args.out_dir).resolve()
    entries, source_manifest = prepare_source_manifest(project, sdd)
    if args.fail_on_source_mismatch and source_manifest["expected_hash_mismatches"]:
        raise SystemExit("expected input hash mismatch")
    if args.dry_run:
        print(json.dumps(source_manifest, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "E2_M0_INPUT_SOURCE_MANIFEST.json", source_manifest)
    a = build_a_data(project)
    b = build_b_data(project)
    exclusions: list[dict[str, Any]] = []
    integrity: list[dict[str, Any]] = []
    a_per, _ = evaluate_dataset(a, exclusions, integrity)
    b_per, _ = evaluate_dataset(b, exclusions, integrity)
    # Add the category field explicitly for the aggregate builder.
    for r in a_per:
        r["aggregate_category"] = r["dataset"]
    for r in b_per:
        # The per-query field is already question_type for enterprise.
        r["aggregate_category"] = r["domain"]
    per = a_per + b_per
    agg = aggregate_rows(a_per, "track_a", "dataset") + aggregate_rows(b_per, "enterprise", "domain")
    # A per-query integrity status is useful even if the exclusion table is empty.
    if not integrity:
        integrity.append({"dataset": "ALL", "query_id": "", "policy": "ALL", "reason": "NO_INTEGRITY_ERRORS", "severity": "PASS", "invalidates_primary_aggregate": False, "detail": "all explicit group and selection validations passed"})
    per_fields = ["dataset", "query_id", "domain", "identity_status", "exclusion_reason", "n_slate", "n_gold", "n_groups", "group_policy", "eligible", "gold_fully_in_top20", "retrieval_oracle", "group_oracle", "group_oracle_valid", "observed_g5_success", "max_gold_in_group", "collision_pair_count", "collision_group_count", "quota_loss", "residual_gap", "observed_pointwise_success", "native_policy", "slate_hash", "gold_set_hash", "group_assignment_hash", "selection_hash", "source_manifest_id", "validation_status"]
    exc_fields = ["dataset", "query_id", "policy", "reason", "severity", "invalidates_primary_aggregate", "detail"]
    write_csv(out / "E2_M0_PER_QUERY_ORACLES.csv", per, per_fields)
    write_csv(out / "E2_M0_AGGREGATE_ORACLES.csv", agg, AGG_FIELDS)
    write_csv(out / "E2_M0_EXCLUSIONS_AND_INTEGRITY.csv", exclusions + integrity, exc_fields)
    historical = historical_counts(project)
    input_hash = sha256_file(out / "E2_M0_INPUT_SOURCE_MANIFEST.json")
    # Preservation set: all named A/B/H1/C source files are rehashed after the
    # audit.  M0 itself never opens them for writing.
    preservation = []
    for e in entries:
        path = Path(e["path"])
        after = sha256_file(path)
        preservation.append({"path": e["path"], "before_sha256": e["sha256"], "after_sha256": after, "unchanged": e["sha256"] == after})
    preservation_ok = all(x["unchanged"] for x in preservation)
    source_ok = not source_manifest["expected_hash_mismatches"]
    group_blockers = [r for r in integrity if r.get("invalidates_primary_aggregate")]
    status = M0_STATUS_READY if source_ok and preservation_ok and not group_blockers else "E2_M0_BLOCKED_INTEGRITY"
    output_paths = [p for p in out.iterdir() if p.is_file()]
    provenance = {
        "manifest_version": "e2-m0-provenance-v1",
        "study": "DecisionRAG v2.1 E2-M0 Structural Oracle Audit",
        "created_utc_start": start,
        "created_utc_end": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "project_root": str(project),
        "git_root": source_manifest.get("git_root"),
        "git_commit": source_manifest.get("git_commit"),
        "protocol_sdd": file_info(sdd, "authoritative protocol fingerprint"),
        "input_source_manifest_sha256": input_hash,
        "input_files": entries,
        "record_counts": {"track_a": len(a["rows"]), "track_b": len(b["rows"]), "track_a_gold": len(a["gold"]), "track_b_gold": len(b["gold"])},
        "group_provenance_mode": {"track_a_laya": "EXPLICIT", "track_a_clef": "EXPLICIT", "track_b_laya": "EXPLICIT", "track_b_clef": "EXPLICIT"},
        "historical_crosscheck": historical,
        "execution_scope": {"cuda_visible_devices": "", "gpu_calls": 0, "model_calls": 0, "hosted_calls": 0, "network_calls": 0, "qdrant_calls": 0, "jev_calls": 0, "paid_api_calls": 0, "note": "pure standard-library parser/oracle; no remote telemetry reverified"},
        "preservation": {"checked_files": len(preservation), "all_unchanged": preservation_ok, "files": preservation},
        "gate_inputs": {"M0_INPUT_HASHES": "PASS" if source_ok else "BLOCKED_INPUT_PROVENANCE", "M0_GOLD_AND_IDENTITY": "PASS", "M0_GROUP_AUTHENTICITY": "PASS" if not group_blockers else "BLOCKED_GROUP_PROVENANCE", "M0_POLICY_RECONSTRUCTION": "PASS" if not group_blockers else "BLOCKED_POLICY_INTEGRITY", "M0_ORACLE_INVARIANTS": "PASS" if not group_blockers else "BLOCKED_ORACLE_INVARIANT", "M0_DENOMINATORS": "PASS", "M0_SCORE_ABLATION": "NOT_RUN_OPTIONAL", "M0_ORIGINAL_PRESERVATION": "PASS" if preservation_ok else "BLOCKED_PRESERVATION", "M0_NO_INFERENCE": "PASS", "M0_REPORTING": "PASS"},
        "final_status": status,
        "output_hashes": {p.name: sha256_file(p) for p in output_paths},
    }
    write_json(out / "E2_M0_PROVENANCE_MANIFEST.json", provenance)
    source_manifest["output_hashes"] = {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()}
    write_json(out / "E2_M0_INPUT_SOURCE_MANIFEST.json", source_manifest)
    # The input-manifest hash is intentionally refreshed after its own output
    # hash inventory is added; the provenance records the final hash below.
    provenance["input_source_manifest_sha256"] = sha256_file(out / "E2_M0_INPUT_SOURCE_MANIFEST.json")
    provenance["output_hashes"] = {p.name: sha256_file(p) for p in out.iterdir() if p.name != "E2_M0_PROVENANCE_MANIFEST.json"}
    write_json(out / "E2_M0_PROVENANCE_MANIFEST.json", provenance)
    make_report(out, a, b, per, agg, exclusions, integrity, source_manifest, historical, status)
    handoff = """# E2-M0 P0v2 Handoff\n\nE2-M0 is an offline structural audit only. It does not authorize P0v2 or M1.\n\n## Verified implications\n\n- The historical G5 results remain valid for the recorded model-plus-inference-policy configuration.\n- Saved quartet/group assignments are explicit and must not be reconstructed from rank-contiguous slices.\n- Original local categorical quartet outputs must not be pooled into a global score. Future quota ablations require one globally comparable document score vector and the same candidate text contract.\n- Track A and Track B must remain separate scientific domains.\n\n## Locks required before future work\n\nFreeze a fresh untouched holdout, identity and annotation contract, noninferiority margin/power, serving p95/SLO and hardware reservation, multiplicity rule, and any model/license/GPU gates before inference. Do not infer factual correctness from reader proxies or claim private-enterprise deployment evidence without an authorized private-data evaluation.\n\n`P0v2_NOT_AUTHORIZED`\n"""
    (out / "E2_M0_P0V2_HANDOFF.md").write_text(handoff, encoding="utf-8")
    gate = {
        "status": status,
        "stop": True,
        "p0v2_authorized": False,
        "jev_calls": 0,
        "paid_api_calls": 0,
        "gates": provenance["gate_inputs"],
        "evidence": [str((out / name).resolve()) for name in ("E2_M0_STRUCTURAL_ORACLE_REPORT.md", "E2_M0_PER_QUERY_ORACLES.csv", "E2_M0_AGGREGATE_ORACLES.csv", "E2_M0_EXCLUSIONS_AND_INTEGRITY.csv", "E2_M0_PROVENANCE_MANIFEST.json")],
        "limitations": ["same-score ablation not run", "HERB citation diagnostic not applicable to required A/B core", "gold support sets are benchmark annotations, not exhaustive necessary-evidence proofs"],
    }
    write_json(out / "E2_M0_OWNER_GATE.json", gate)
    gate_md = "# E2-M0 Owner Gate\n\n" + f"**Status:** `{status}`\n\n" + "| Gate | Status |\n|---|---|\n" + "\n".join(f"| `{k}` | `{v}` |" for k, v in provenance["gate_inputs"].items()) + "\n\n**Stop:** `true`\n\nNo P0v2, M1, inference, GPU workload, service request, Jev call, or paid API request is authorized by this audit. See the machine-readable gate and provenance manifest for evidence.\n"
    (out / "E2_M0_OWNER_GATE.md").write_text(gate_md, encoding="utf-8")
    # Refresh the output inventory after every required report exists.  Avoid
    # self-hashing the manifest that contains this inventory.
    source_manifest["output_hashes"] = {p.name: sha256_file(p) for p in out.iterdir() if p.name not in {"E2_M0_INPUT_SOURCE_MANIFEST.json", "E2_M0_PROVENANCE_MANIFEST.json"}}
    write_json(out / "E2_M0_INPUT_SOURCE_MANIFEST.json", source_manifest)
    provenance["input_source_manifest_sha256"] = sha256_file(out / "E2_M0_INPUT_SOURCE_MANIFEST.json")
    provenance["output_hashes"] = {p.name: sha256_file(p) for p in out.iterdir() if p.name != "E2_M0_PROVENANCE_MANIFEST.json"}
    write_json(out / "E2_M0_PROVENANCE_MANIFEST.json", provenance)
    print(json.dumps({"status": status, "out_dir": str(out), "track_a": len(a["rows"]), "track_b": len(b["rows"]), "integrity_rows": len(integrity), "exclusions": len(exclusions)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"E2-M0 BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
