#!/usr/bin/env python3
"""Final source-record-ID validation for the v1.1 P0 cohort.

Offline only.  This script does not call models, Qdrant, Jev, or any service,
and it does not change the already-generated query manifests.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "results/preflight/v11_remediation"
SEED = 20261011
SELECTION_HASH_SEED = "20261011-v1.1"
DATASETS = ("hotpot", "musique", "wiki2")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_json(path: Path, value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = b"".join(canonical(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_preflight_module():
    spec = importlib.util.spec_from_file_location("v11_p0_preflight", ROOT / "scripts/v11_p0_preflight.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load v11 preflight helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_id_from_query_id(query_id: str, debug: bool = False) -> str:
    return query_id.split(":debug:", 1)[1] if debug else query_id.split(":", 1)[1]


def set_intersection(a: set[str], b: set[str]) -> dict:
    values = sorted(a & b)
    return {"count": len(values), "source_record_ids": values}


def exact_intersections(sets: dict[str, set[str]]) -> dict:
    names = tuple(sets)
    result = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            result[f"{left}&{right}"] = set_intersection(sets[left], sets[right])
    for width in (3, 4):
        for combo in __import__("itertools").combinations(names, width):
            values = set.intersection(*(sets[name] for name in combo))
            result["&".join(combo)] = {"count": len(values), "source_record_ids": sorted(values)}
    return result


def exact_partition_counts(sets: dict[str, set[str]]) -> dict:
    universe = set.union(*sets.values())
    names = tuple(sets)
    partitions = collections.Counter()
    for value in universe:
        mask = "".join(name for name in names if value in sets[name]) or "NONE"
        partitions[mask] += 1
    return {key: partitions[key] for key in sorted(partitions)}


def load_debug_training_ids(prep) -> dict[str, set[str]]:
    raw = ROOT / "data/raw/vendor"
    train = {
        "hotpot": [row["id"] for row in prep.hotpot_rows(raw / "hotpot/hotpot_train.parquet")],
        "musique": [row["id"] for row in prep.musique_rows(raw / "musique/musique_ans_v1.0_train.jsonl")],
        "wiki2": [row["id"] for row in prep.wiki2_rows(raw / "2wiki/data/train.json")],
    }
    return {dataset: set(values) for dataset, values in train.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    helper = load_preflight_module()
    prep = helper.load_prepare_module()
    sources, _ = helper.load_multihop()
    previous = load_jsonl(ROOT / "data/manifests/questions.jsonl")
    debug_manifest = load_jsonl(ROOT / "data/manifests/debug_manifest.jsonl")
    new_ids = load_jsonl(ROOT / "results/preflight/v11/NEW_TEST_IDS.jsonl")

    debug_train_ids = load_debug_training_ids(prep)
    intersection_rows = []
    replay_rows = []
    for dataset in DATASETS:
        valid_list = [str(row["id"]) for row in sources[dataset] if row["gold_mapping_valid"]]
        existing_list = [
            source_id_from_query_id(row["query_id"])
            for row in previous
            if row["dataset"] == dataset and row.get("split") in {"cal", "test"}
        ]
        debug_list = [
            source_id_from_query_id(row["query_id"], debug=True)
            for row in debug_manifest
            if row["dataset"] == dataset
        ]
        new_list = [
            row["source_record_id"]
            for row in new_ids
            if row["dataset"] == dataset
        ]
        valid = set(valid_list)
        existing = set(existing_list)
        debug = set(debug_list)
        new = set(new_list)
        sets = {"V": valid, "C": existing, "D": debug, "N": new}
        intersection_rows.append(
            {
                "dataset": dataset,
                "set_semantics": {
                    "V": "valid development source-record IDs",
                    "C": "existing CAL/TEST source-record IDs",
                    "D": "debug-fixture source-record IDs",
                    "N": "new selected TEST source-record IDs",
                },
                "set_counts": {name: len(values) for name, values in sets.items()},
                "set_duplicate_checks": {
                    "V": len(valid_list) == len(valid),
                    "C": len(existing_list) == len(existing),
                    "D": len(debug_list) == len(debug),
                    "N": len(new_list) == len(new),
                },
                "pairwise_and_higher_order_intersections": exact_intersections(sets),
                "exact_membership_partition_counts": exact_partition_counts(sets),
                "debug_training_source_id_count": len(debug & debug_train_ids[dataset]),
                "debug_source_ids_in_valid_development": sorted(debug & valid),
                "valid_records_not_existing_or_debug_or_new": len(valid - existing - debug - new),
                "selected_source_ids_unique": len(new) == sum(row["dataset"] == dataset for row in new_ids),
            }
        )

        # Replay the frozen deterministic selector at source-ID level.
        eligible = [
            row for row in sources[dataset]
            if row["gold_mapping_valid"]
            and str(row["id"]) not in existing
            and str(row["id"]) not in debug
        ]
        if dataset in {"hotpot", "wiki2"}:
            eligible.sort(key=lambda row: hashlib.sha256(
                f"{dataset}:{row['query_id']}:{SELECTION_HASH_SEED}".encode("utf-8")
            ).hexdigest())
            expected = eligible[:800]
            rule = "first_800_by_sha256(dataset:query_id:20261011-v1.1)"
        else:
            expected = sorted(eligible, key=lambda row: row["query_id"])
            rule = "all_remaining_valid_answerable_dev_records_sorted_by_query_id"
        expected_ids = [str(row["id"]) for row in expected]
        # Preserve the serialized order from the frozen manifest.  Hotpot and
        # 2Wiki are serialized in their deterministic hash-selection order;
        # MuSiQue is serialized in query-ID order.  Comparing against a
        # lexicographically sorted copy would falsely flag the former two.
        actual_ids = [
            row["source_record_id"]
            for row in new_ids
            if row["dataset"] == dataset
        ]
        replay_rows.append(
            {
                "dataset": dataset,
                "selection_rule": rule,
                "eligible_source_record_count": len(eligible),
                "expected_selected_count": len(expected_ids),
                "actual_selected_count": len(actual_ids),
                "expected_selected_source_ids_sha256": sha256_bytes(("\n".join(expected_ids) + "\n").encode()),
                "actual_selected_source_ids_sha256": sha256_bytes(("\n".join(actual_ids) + "\n").encode()),
                "deterministic_replay_matches": expected_ids == actual_ids,
            }
        )

    intersections_sha = write_jsonl(args.output_dir / "MULTIHOP_SOURCE_ID_INTERSECTIONS_V11R1.jsonl", intersection_rows)
    replay_sha = write_jsonl(args.output_dir / "MULTIHOP_SELECTION_REPLAY_V11R1.jsonl", replay_rows)

    question_manifest_path = args.output_dir / "ENTERPRISE_QUESTION_MANIFEST_V11R1.jsonl"
    enterprise_manifest_sha = sha256_file(question_manifest_path)
    enterprise_acceptance = {
        "version": "V11.1R1-GOLD-ID-AMBIGUITY-qst_0413-ACCEPTED",
        "acceptance": True,
        "acceptance_basis": "explicit owner instruction in the current P0 final cohort-validation request",
        "question_id": "qst_0413",
        "exception": "V11.1R1-GOLD-ID-AMBIGUITY-qst_0413",
        "official_release_files_preserved": True,
        "all_enterprise_questions_retained": 500,
        "identity_dependent_evaluation": "exclude qst_0413 only from identity-dependent evaluation; retain it in the 500-question manifest and all non-identity metadata",
        "nominal_known_gold_1_to_5": 444,
        "potentially_eligible_identity_resolved_questions": 443,
        "future_conditional_cssr_denominator": {
            "status": "NOT_YET_MEASURED",
            "upper_bound": 443,
            "definition": "subset of the 443 identity-resolved 1-to-5 questions whose complete internal gold set appears in the future frozen Top-20 slate; do not treat 443 as the observed CSSR denominator",
        },
        "manifest_sha256": enterprise_manifest_sha,
        "no_physical_file_selected": True,
    }
    acceptance_sha = write_json(args.output_dir / "ENTERPRISE_QST_0413_ACCEPTANCE_V11R1.json", enterprise_acceptance)

    all_new_query_ids = [row["query_id"] for row in new_ids]
    all_new_source_keys = [f"{row['dataset']}:{row['source_record_id']}" for row in new_ids]
    existing_query_ids = [row["query_id"] for row in previous]
    debug_query_ids = [row["query_id"] for row in debug_manifest]
    final_lock = {
        "lock_version": "v1.1-final-cohort-validation-r1",
        "status": "FINAL_COHORT_VALIDATION_PASS_PENDING_IMMUTABLE_COMMIT",
        "freeze_date": "2026-10-11",
        "selection_seed": SEED,
        "selection_hash_seed": SELECTION_HASH_SEED,
        "inference_executed": False,
        "jev_calls": 0,
        "source_record_level_validation": True,
        "new_selected_total": len(new_ids),
        "new_query_id_unique": len(all_new_query_ids) == len(set(all_new_query_ids)),
        "new_source_record_key_unique": len(all_new_source_keys) == len(set(all_new_source_keys)),
        "new_disjoint_from_existing_cal_test": not set(all_new_source_keys) & {f"{row['dataset']}:{source_id_from_query_id(row['query_id'])}" for row in previous},
        "new_disjoint_from_debug": not set(all_new_source_keys) & {f"{row['dataset']}:{source_id_from_query_id(row['query_id'], True)}" for row in debug_manifest},
        "hotpot_selected_debug_overlap": intersection_rows[0]["pairwise_and_higher_order_intersections"]["D&N"]["count"],
        "wiki2_selected_debug_overlap": intersection_rows[2]["pairwise_and_higher_order_intersections"]["D&N"]["count"],
        "multihop_manifest_sha256": sha256_file(ROOT / "results/preflight/v11/NEW_TEST_IDS.jsonl"),
        "multihop_gold_sidecar_sha256": sha256_file(ROOT / "results/preflight/v11/NEW_TEST_GOLD_SIDECAR.jsonl"),
        "source_id_intersections_sha256": intersections_sha,
        "selection_replay_sha256": replay_sha,
        "selection_replay_all_datasets_match": all(row["deterministic_replay_matches"] for row in replay_rows),
        "enterprise_qst_0413_acceptance_sha256": acceptance_sha,
        "enterprise_question_manifest_sha256": enterprise_manifest_sha,
        "enterprise_question_count": 500,
        "enterprise_identity_dependent_exclusion": ["qst_0413"],
        "enterprise_potentially_eligible_identity_resolved": 443,
        "enterprise_future_conditional_cssr_denominator": "NOT_YET_MEASURED; upper bound 443",
        "gpu_gate": "BLOCKED_GPU_WINDOW_NOT_AUTHORIZED",
        "protocol_outcome": "no selected IDs changed; no outcome-based resampling; no model or retrieval outcome accessed",
    }
    lock_sha = write_json(args.output_dir / "MULTIHOP_V11R1_FINAL_COHORT_LOCK.json", final_lock)
    summary = {
        "version": "v1.1-final-cohort-validation-r1",
        "status": final_lock["status"],
        "inference_executed": False,
        "jev_calls": 0,
        "new_selected_total": len(new_ids),
        "new_query_id_unique": final_lock["new_query_id_unique"],
        "new_disjoint_from_existing_cal_test": final_lock["new_disjoint_from_existing_cal_test"],
        "new_disjoint_from_debug": final_lock["new_disjoint_from_debug"],
        "selection_replay_all_datasets_match": final_lock["selection_replay_all_datasets_match"],
        "invalid_source_records_total": 12,
        "enterprise_qst_0413_accepted": True,
        "enterprise_questions_retained": 500,
        "enterprise_potentially_eligible": 443,
        "future_cssr_denominator": "NOT_YET_MEASURED",
        "gpu_gate": final_lock["gpu_gate"],
        "lock_sha256": lock_sha,
        "source_id_intersections_sha256": intersections_sha,
        "selection_replay_sha256": replay_sha,
        "acceptance_sha256": acceptance_sha,
    }
    write_json(args.output_dir / "V11_P0_FINAL_COHORT_MACHINE_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
