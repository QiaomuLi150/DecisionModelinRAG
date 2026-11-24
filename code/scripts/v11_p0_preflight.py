#!/usr/bin/env python3
"""Build the DecisionRAG v1.1 P0 preflight package without inference.

This script is intentionally limited to local data inspection and deterministic
manifest construction. It never calls a model, a localhost service, or Jev.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "results/preflight/v11"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_jsonl(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = canonical_jsonl(rows)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_prepare_module():
    spec = importlib.util.spec_from_file_location("decisionrag_prepare_data", ROOT / "scripts/prepare_data.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_multihop() -> tuple[dict[str, list[dict]], dict[str, str]]:
    prep = load_prepare_module()
    raw = ROOT / "data/raw/vendor"
    sources = {
        "hotpot": list(prep.hotpot_rows(raw / "hotpot/hotpot_dev_distractor_v1.parquet")),
        "musique": list(prep.musique_rows(raw / "musique/musique_ans_v1.0_dev.jsonl")),
        "wiki2": list(prep.wiki2_rows(raw / "2wiki/data/dev.json")),
    }
    corpus: dict[tuple[str, str, str], dict] = {}
    for dataset, rows in sources.items():
        for row in rows:
            row["dataset"] = dataset
            row["query_id"] = f"{dataset}:{row['id']}"
            row["query_type"] = row.pop("type")
            prep.map_gold(row, corpus)
    source_paths = {
        "hotpot": "data/raw/vendor/hotpot/hotpot_dev_distractor_v1.parquet",
        "musique": "data/raw/vendor/musique/musique_ans_v1.0_dev.jsonl",
        "wiki2": "data/raw/vendor/2wiki/data/dev.json",
    }
    return sources, source_paths


def load_id_sets() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    old: dict[str, set[str]] = collections.defaultdict(set)
    debug: dict[str, set[str]] = collections.defaultdict(set)
    for line in (ROOT / "data/manifests/questions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        old[row["dataset"]].add(row["query_id"])
    for line in (ROOT / "data/manifests/debug_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        debug[row["dataset"]].add(row["query_id"])
    return old, debug


def build_multihop(out: Path) -> dict:
    sources, source_paths = load_multihop()
    old, debug = load_id_sets()
    selected: dict[str, list[dict]] = {}
    exclusion_summary = {}
    for dataset, rows in sources.items():
        valid = [row for row in rows if row["gold_mapping_valid"]]
        excluded = old[dataset] | debug[dataset]
        eligible = [row for row in valid if row["query_id"] not in excluded]
        if dataset in {"hotpot", "wiki2"}:
            eligible.sort(
                key=lambda row: hashlib.sha256(
                    f"{dataset}:{row['query_id']}:20261011-v1.1".encode("utf-8")
                ).hexdigest()
            )
            chosen = eligible[:800]
            rule = "first_800_by_sha256(dataset:query_id:20261011-v1.1)"
        else:
            chosen = sorted(eligible, key=lambda row: row["query_id"])
            rule = "all_remaining_valid_answerable_dev_records_sorted_by_query_id"
        selected[dataset] = chosen
        exclusion_summary[dataset] = {
            "raw_records": len(rows),
            "valid_gold_mapping_records": len(valid),
            "legacy_selected_records": len(old[dataset]),
            "debug_records": len(debug[dataset]),
            "eligible_after_exclusions": len(eligible),
            "selected_new_records": len(chosen),
            "invalid_raw_records": sum(not row["gold_mapping_valid"] for row in rows),
            "selection_rule": rule,
        }

    ids_rows: list[dict] = []
    gold_rows: list[dict] = []
    for dataset in ("hotpot", "musique", "wiki2"):
        for order, row in enumerate(selected[dataset], start=1):
            ids_rows.append(
                {
                    "dataset": dataset,
                    "query_id": row["query_id"],
                    "source_record_id": row["id"],
                    "question": row["question"],
                    "query_type": row["query_type"],
                    "hop_count": row.get("hop_count"),
                    "selection_order_within_dataset": order,
                    "selection_rule": exclusion_summary[dataset]["selection_rule"],
                    "provenance": "LOCAL_PINNED_SOURCE_DRY_RUN",
                }
            )
            gold_rows.append(
                {
                    "dataset": dataset,
                    "query_id": row["query_id"],
                    "answer": row["answer"],
                    "answer_aliases": row.get("answer_aliases", []),
                    "gold_doc_ids": row["gold_doc_ids"],
                    "context_doc_ids": row["context_doc_ids"],
                    "gold_mapping_valid": row["gold_mapping_valid"],
                    "mapping_ambiguities": row["mapping_ambiguities"],
                    "evaluation_only": True,
                }
            )

    ids_sha = write_jsonl(out / "NEW_TEST_IDS.jsonl", ids_rows)
    gold_sha = write_jsonl(out / "NEW_TEST_GOLD_SIDECAR.jsonl", gold_rows)
    source_hashes = {name: sha256_file(ROOT / rel) for name, rel in source_paths.items()}
    existing_hashes = {
        rel: sha256_file(ROOT / rel)
        for rel in (
            "configs/study.yaml",
            "configs/models.yaml",
            "configs/prompts.json",
            "data/manifests/questions.jsonl",
            "data/manifests/split_manifest.jsonl",
            "data/frozen_slates/m2_top20_slates.jsonl",
            "data/corpus/passages.jsonl",
            "DecisionRAG_Industry_Experimental_Plan_v1_1_PRE_REGISTRATION.md",
            "Decision_Model_in_RAG_Production_System_COMPLETE_SDD.md",
        )
    }
    lock = {
        "lock_status": "PROPOSED_FROZEN_PENDING_OWNER_APPROVAL_AND_IMMUTABLE_COMMIT",
        "study_id": "decisionrag-industry-v1.1",
        "protocol_revision": "DecisionRAG_Industry_Experimental_Plan_v1_1_PRE_REGISTRATION",
        "freeze_date": "2026-10-11",
        "inference_executed": False,
        "jev_calls": 0,
        "seed": 20261011,
        "sampling_hash_seed": "20261011-v1.1",
        "new_query_counts": {dataset: len(rows) for dataset, rows in selected.items()},
        "new_query_total": len(ids_rows),
        "target_counts": {"hotpot": 800, "musique": 2167, "wiki2": 800},
        "manifest_sha256": ids_sha,
        "gold_sidecar_sha256": gold_sha,
        "source_file_sha256": source_hashes,
        "preserved_artifact_sha256": existing_hashes,
        "exclusion_summary": exclusion_summary,
        "corpus_requirement": {
            "required_passages": 144650,
            "currently_frozen_passages": 144650,
            "current_corpus_sha256": existing_hashes["data/corpus/passages.jsonl"],
            "must_be_reverified_before_retrieval": True,
        },
        "frozen_settings_not_changed": {
            "policies": [
                "raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5",
                "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5",
            ],
            "bm25_top_k": 100,
            "dense_top_k": 100,
            "rrf_k": 60,
            "slate_k": 20,
            "output_k": 5,
            "reader_max_output_tokens": 64,
        },
    }
    write_json(out / "MULTIHOP_V11_LOCK.json", lock)
    return {"lock": lock, "ids_rows": ids_rows, "gold_rows": gold_rows}


def enterprise(out: Path) -> dict:
    base = ROOT / "data/raw/vendor/enterprise_rag_bench/v1.0.0"
    question_path = base / "questions.jsonl"
    archive_path = base / "all_documents.zip"
    rows = [json.loads(line) for line in question_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required_fields = {"question_id", "question_type", "question", "expected_doc_ids", "gold_answer", "answer_facts"}
    malformed = []
    for line_no, row in enumerate(rows, start=1):
        missing = sorted(required_fields - set(row))
        if missing or not isinstance(row.get("expected_doc_ids"), list) or not isinstance(row.get("answer_facts"), list):
            malformed.append({"line": line_no, "missing": missing})
    with zipfile.ZipFile(archive_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        txt_infos = [info for info in infos if info.filename.lower().endswith(".txt")]
        by_dsid: dict[str, list[str]] = collections.defaultdict(list)
        source_counts = collections.Counter()
        for info in txt_infos:
            source = Path(info.filename).parts[0]
            source_counts[source] += 1
            basename = Path(info.filename).name
            match = re.match(r"(dsid_[0-9a-f]{32})__", basename)
            if match:
                by_dsid[match.group(1)].append(info.filename)
        expected_ids = sorted({doc_id for row in rows for doc_id in row.get("expected_doc_ids", [])})
        found_expected = sorted(doc_id for doc_id in expected_ids if doc_id in by_dsid)
        duplicate_ids = {
            doc_id: sorted(paths) for doc_id, paths in by_dsid.items() if len(paths) > 1
        }
        sample_paths: list[str] = []
        by_source: dict[str, list[str]] = collections.defaultdict(list)
        for info in txt_infos:
            by_source[Path(info.filename).parts[0]].append(info.filename)
        # Cover every source type first, then add deterministic second/third
        # examples until the required 20-document audit is reached.
        for ordinal in range(3):
            for source in sorted(by_source):
                candidates = sorted(by_source[source])
                if ordinal < len(candidates):
                    sample_paths.append(candidates[ordinal])
                if len(sample_paths) == 20:
                    break
            if len(sample_paths) == 20:
                break
        sample_rows = []
        for path in sample_paths:
            data = archive.read(path)
            text = data.decode("utf-8")
            sample_rows.append(
                {
                    "path": path,
                    "source_type": Path(path).parts[0],
                    "doc_id": Path(path).name.split("__", 1)[0],
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "utf8_decoded": True,
                    "contains_nul": "\x00" in text,
                    "first_line": text.splitlines()[0] if text.splitlines() else "",
                    "audit_selection": "first_three_sorted_documents_per_source_type_truncated_to_20",
                }
            )

    category_counts = collections.Counter(row.get("question_type") for row in rows)
    raw_cardinality = collections.Counter()
    unique_cardinality = collections.Counter()
    duplicate_expected = []
    question_manifest = []
    gold_sidecar = []
    for row in rows:
        expected = row.get("expected_doc_ids", [])
        unique_expected = list(dict.fromkeys(expected))
        raw_cardinality[str(len(expected))] += 1
        condition = "empty_no_defined_gold" if not unique_expected else (
            "known_gold_1_to_5" if len(unique_expected) <= 5 else "infeasible_at_k5_gt5"
        )
        unique_cardinality["empty" if not unique_expected else "1-5" if len(unique_expected) <= 5 else ">5"] += 1
        if len(expected) != len(unique_expected):
            duplicate_expected.append(row["question_id"])
        question_manifest.append(
            {
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "source_types": row.get("source_types", []),
                "unique_expected_doc_count": len(unique_expected),
                "raw_expected_doc_count": len(expected),
                "label_condition": condition,
                "expected_doc_ids_resolve_in_archive": all(doc_id in by_dsid for doc_id in unique_expected),
                "evaluation_only_metadata": True,
            }
        )
        gold_sidecar.append(
            {
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "expected_doc_ids": expected,
                "gold_answer": row.get("gold_answer"),
                "answer_facts": row.get("answer_facts", []),
                "evaluation_only": True,
            }
        )

    selected_audit = []
    for category in sorted(category_counts):
        for row in sorted((r for r in rows if r.get("question_type") == category), key=lambda r: r["question_id"])[:3]:
            selected_audit.append(
                {
                    "question_id": row["question_id"],
                    "question_type": row["question_type"],
                    "source_types": row.get("source_types", []),
                    "unique_expected_doc_count": len(dict.fromkeys(row.get("expected_doc_ids", []))),
                    "duplicate_expected_doc_id": len(row.get("expected_doc_ids", [])) != len(set(row.get("expected_doc_ids", []))),
                    "all_referenced_ids_in_archive": all(doc_id in by_dsid for doc_id in set(row.get("expected_doc_ids", []))),
                    "audit_selection": "first_three_sorted_questions_per_category",
                }
            )

    question_sha = sha256_file(question_path)
    archive_sha = sha256_file(archive_path)
    question_manifest_sha = write_jsonl(out / "ENTERPRISE_QUESTION_MANIFEST.jsonl", question_manifest)
    gold_sha = write_jsonl(out / "ENTERPRISE_GOLD_SIDECAR.jsonl", gold_sidecar)
    annotation_audit_sha = write_jsonl(out / "ENTERPRISE_ANNOTATION_AUDIT.jsonl", selected_audit)
    corpus_sample_sha = write_jsonl(out / "ENTERPRISE_CORPUS_SAMPLE_AUDIT.jsonl", sample_rows)
    release = {
        "release_status": "AVAILABLE_LOCALLY_WITH_INTEGRITY_BLOCKER_PENDING_OWNER_ADJUDICATION",
        "release_name": "Dataset v1.0.0",
        "release_tag": "v1.0.0",
        "release_commit": "56ba6a62cb66bf0a68ff995b1c423680980bf70a",
        "published_at": "2026-03-29T08:17:18Z",
        "source_repository": "https://github.com/onyx-dot-app/EnterpriseRAG-Bench",
        "source_huggingface": "https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench",
        "secondary_huggingface_revision": "69916e31c68aa5963c00248fd7f0bc12d04fd235",
        "secondary_huggingface_revision_note": "Metadata-only pin; GitHub v1.0.0 release assets are the selected primary files.",
        "question_asset": {
            "url": "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/v1.0.0/questions.jsonl",
            "local_path": "data/raw/vendor/enterprise_rag_bench/v1.0.0/questions.jsonl",
            "bytes": question_path.stat().st_size,
            "sha256": question_sha,
            "publisher_sha256": "f9524b9157cd43aae36b99333a124738804306ea6d07f332d49faa6d3d147905",
            "hash_verified": question_sha == "f9524b9157cd43aae36b99333a124738804306ea6d07f332d49faa6d3d147905",
        },
        "corpus_asset": {
            "url": "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/v1.0.0/all_documents.zip",
            "local_path": "data/raw/vendor/enterprise_rag_bench/v1.0.0/all_documents.zip",
            "bytes": archive_path.stat().st_size,
            "sha256": archive_sha,
            "publisher_sha256": "9d1174928696ad08bc15f3f104739519de633c1605a4ec2034e0e3c0087bc5cd",
            "hash_verified": archive_sha == "9d1174928696ad08bc15f3f104739519de633c1605a4ec2034e0e3c0087bc5cd",
        },
        "question_integrity": {
            "rows": len(rows),
            "unique_question_ids": len({row.get("question_id") for row in rows}),
            "malformed_rows": len(malformed),
            "category_counts": dict(sorted(category_counts.items())),
            "raw_expected_doc_cardinality": dict(sorted(raw_cardinality.items())),
            "unique_expected_doc_cardinality": dict(unique_cardinality),
            "unique_expected_doc_ids": len(expected_ids),
            "expected_doc_ids_found_in_archive": len(found_expected),
            "missing_expected_doc_ids": len(expected_ids) - len(found_expected),
            "duplicate_expected_doc_id_questions": sorted(duplicate_expected),
        },
        "corpus_integrity": {
            "txt_document_files": len(txt_infos),
            "unique_dsid_values": len(by_dsid),
            "duplicate_dsid_values": len(duplicate_ids),
            "duplicate_dsid_details": duplicate_ids,
            "source_counts_including_root_noncorpus_files": dict(sorted(source_counts.items())),
            "sample_audit_rows": len(sample_rows),
            "sample_audit_sha256": corpus_sample_sha,
        },
        "p0_manifest_hashes": {
            "enterprise_question_manifest_sha256": question_manifest_sha,
            "enterprise_gold_sidecar_sha256": gold_sha,
            "enterprise_annotation_audit_sha256": annotation_audit_sha,
        },
        "do_not_pass_to_models": ["expected_doc_ids", "gold_answer", "answer_facts", "question_type", "label_condition"],
        "integrity_blockers": [
            "Four dsid values occur in multiple physical files with distinct contents.",
            "qst_0413 repeats the same expected_doc_id twice and that ID maps to two distinct Jira files.",
        ],
    }
    write_json(out / "ENTERPRISE_RELEASE_MANIFEST.json", release)
    write_json(out / "ENTERPRISE_CORPUS_INTEGRITY.json", {
        "release_tag": "v1.0.0",
        "txt_document_files": len(txt_infos),
        "unique_dsid_values": len(by_dsid),
        "duplicate_dsid_values": duplicate_ids,
        "expected_doc_ids": len(expected_ids),
        "expected_doc_ids_missing": sorted(set(expected_ids) - set(found_expected)),
        "duplicate_expected_doc_id_questions": sorted(duplicate_expected),
        "sample_audit_path": "results/preflight/v11/ENTERPRISE_CORPUS_SAMPLE_AUDIT.jsonl",
        "annotation_audit_path": "results/preflight/v11/ENTERPRISE_ANNOTATION_AUDIT.jsonl",
    })
    return release


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    multihop = build_multihop(args.output_dir)
    enterprise_result = enterprise(args.output_dir)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inference_executed": False,
        "jev_calls": 0,
        "multihop": {
            "counts": multihop["lock"]["new_query_counts"],
            "total": multihop["lock"]["new_query_total"],
            "ids_sha256": multihop["lock"]["manifest_sha256"],
            "lock_status": multihop["lock"]["lock_status"],
        },
        "enterprise": {
            "questions": enterprise_result["question_integrity"],
            "corpus": enterprise_result["corpus_integrity"],
            "release_status": enterprise_result["release_status"],
        },
    }
    write_json(args.output_dir / "V11_P0_MACHINE_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
