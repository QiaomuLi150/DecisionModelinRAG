#!/usr/bin/env python3
"""Build the DecisionRAG v1.1 P0 remediation package without inference.

This script is deliberately limited to deterministic local data inspection,
archive accounting, and manifest construction.  It does not call model
endpoints, Qdrant, Jev, or any paid service.  It never edits the original v11
preflight artifacts or frozen experiment results.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import re
import statistics
import zipfile
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "results/preflight/v11_remediation"
FREEZE_DATE = "2026-10-11"
SEED = 20261011
IDENTITY_RULE = "enterprise:v1.0.0:path-qualified-relative-archive-member"
POLICIES = [
    "raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5",
    "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5",
    "laya_rescue5", "clef_rescue5",
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_json(path: Path, value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = b"".join(canonical_json(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def write_jsonl_stream(path: Path, rows) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            payload = canonical_json(row) + b"\n"
            handle.write(payload)
            digest.update(payload)
            count += 1
    return count, digest.hexdigest()


def load_preflight_module():
    spec = importlib.util.spec_from_file_location("v11_p0_preflight", ROOT / "scripts/v11_p0_preflight.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the existing preflight helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_multihop() -> dict[str, list[dict]]:
    module = load_preflight_module()
    sources, _ = module.load_multihop()
    return sources


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_id_from_query_id(query_id: str, debug: bool = False) -> str:
    if debug:
        return query_id.split(":debug:", 1)[1]
    return query_id.split(":", 1)[1]


def multihop_remediation(out: Path) -> dict:
    sources = load_multihop()
    old_rows = load_jsonl(ROOT / "data/manifests/questions.jsonl")
    debug_rows = load_jsonl(ROOT / "data/manifests/debug_manifest.jsonl")
    selected_rows = load_jsonl(ROOT / "results/preflight/v11/NEW_TEST_IDS.jsonl")
    selected_gold = load_jsonl(ROOT / "results/preflight/v11/NEW_TEST_GOLD_SIDECAR.jsonl")
    corpus_ids = {row["doc_id"] for row in load_jsonl(ROOT / "data/corpus/passages.jsonl")}

    old_source_ids: dict[str, set[str]] = collections.defaultdict(set)
    for row in old_rows:
        old_source_ids[row["dataset"]].add(source_id_from_query_id(row["query_id"]))
    debug_source_ids: dict[str, set[str]] = collections.defaultdict(set)
    for row in debug_rows:
        debug_source_ids[row["dataset"]].add(source_id_from_query_id(row["query_id"], debug=True))

    overlap_rows: list[dict] = []
    for dataset in ("hotpot", "musique", "wiki2"):
        source_lookup = {str(row["id"]): row for row in sources[dataset]}
        debug_ids = sorted(debug_source_ids[dataset])
        overlap = sorted(set(debug_ids) & old_source_ids[dataset])
        overlap_rows.append(
            {
                "dataset": dataset,
                "debug_fixture_count": len(debug_ids),
                "debug_source_record_ids": debug_ids,
                "debug_source_records_found_in_pinned_dev": sum(item in source_lookup for item in debug_ids),
                "overlap_with_previous_cal_test_source_records": overlap,
                "overlap_count": len(overlap),
                "proof": "compare exact underlying source-record IDs, not decorated debug query IDs",
            }
        )

    invalid_by_dataset = {}
    invalid_records = []
    for dataset, rows in sources.items():
        bad = [row for row in rows if not row["gold_mapping_valid"]]
        invalid_by_dataset[dataset] = len(bad)
        invalid_records.extend(
            {
                "dataset": dataset,
                "source_record_id": row["id"],
                "query_id": row["query_id"],
                "mapping_ambiguities": row["mapping_ambiguities"],
            }
            for row in bad
        )

    gold_by_qid = {row["query_id"]: row for row in selected_gold}
    coverage_by_dataset: dict[str, dict] = {}
    missing_rows = []
    for dataset in ("hotpot", "musique", "wiki2"):
        rows = [row for row in selected_rows if row["dataset"] == dataset]
        total_questions = len(rows)
        all_present = 0
        any_present = 0
        gold_instances = 0
        missing_instances = 0
        for row in rows:
            gold = gold_by_qid[row["query_id"]]
            gold_ids = list(dict.fromkeys(gold.get("gold_doc_ids", [])))
            present = [doc_id in corpus_ids for doc_id in gold_ids]
            gold_instances += len(gold_ids)
            missing = [doc_id for doc_id, ok in zip(gold_ids, present) if not ok]
            missing_instances += len(missing)
            if present:
                any_present += int(any(present))
                all_present += int(all(present))
            if missing:
                missing_rows.append(
                    {
                        "dataset": dataset,
                        "query_id": row["query_id"],
                        "missing_gold_doc_ids": missing,
                        "gold_doc_count": len(gold_ids),
                    }
                )
        coverage_by_dataset[dataset] = {
            "new_questions": total_questions,
            "gold_doc_instances": gold_instances,
            "questions_with_any_gold_in_corpus": any_present,
            "questions_with_all_gold_in_corpus": all_present,
            "any_gold_coverage_rate": any_present / total_questions if total_questions else None,
            "complete_gold_coverage_rate": all_present / total_questions if total_questions else None,
            "missing_gold_doc_instances": missing_instances,
        }

    corrected_counts = {
        dataset: {
            "raw_records": len(sources[dataset]),
            "valid_gold_mapping_records": sum(row["gold_mapping_valid"] for row in sources[dataset]),
            "invalid_source_records": invalid_by_dataset[dataset],
            "previous_cal_test_records": len(old_source_ids[dataset]),
            "debug_fixtures": len(debug_source_ids[dataset]),
            "debug_overlap_with_previous_cal_test": len(
                set(debug_source_ids[dataset]) & old_source_ids[dataset]
            ),
            "new_selected_records": sum(row["dataset"] == dataset for row in selected_rows),
        }
        for dataset in ("hotpot", "musique", "wiki2")
    }
    lock = {
        "lock_version": "v1.1-remediation-r1",
        "status": "PROPOSED_FROZEN_PENDING_OWNER_REVIEW_AND_IMMUTABLE_COMMIT",
        "freeze_date": FREEZE_DATE,
        "inference_executed": False,
        "jev_calls": 0,
        "raw_records_total": sum(len(rows) for rows in sources.values()),
        "valid_records_total": sum(sum(row["gold_mapping_valid"] for row in rows) for rows in sources.values()),
        "invalid_source_records_total": len(invalid_records),
        "invalid_source_records_by_dataset": invalid_by_dataset,
        "previous_cal_test_total": len(old_rows),
        "debug_fixture_total": len(debug_rows),
        "debug_overlap_total": sum(row["overlap_count"] for row in overlap_rows),
        "new_selected_total": len(selected_rows),
        "new_selected_ids_sha256": sha256_file(ROOT / "results/preflight/v11/NEW_TEST_IDS.jsonl"),
        "new_gold_sidecar_sha256": sha256_file(ROOT / "results/preflight/v11/NEW_TEST_GOLD_SIDECAR.jsonl"),
        "eligibility_rule": "valid source mapping AND source-record ID absent from previous CAL/TEST AND source-record ID absent from debug fixtures; no retrieval/model outcome filtering",
        "per_dataset": corrected_counts,
        "gold_in_corpus_coverage": coverage_by_dataset,
        "missing_gold_rows": len(missing_rows),
        "preserve_all_selected_questions": len(selected_rows) == 3767,
        "protocol_change": False,
    }
    write_jsonl(out / "MULTIHOP_DEBUG_OVERLAP_AUDIT.jsonl", overlap_rows)
    write_jsonl(out / "MULTIHOP_INVALID_SOURCE_RECORDS.jsonl", sorted(invalid_records, key=lambda row: (row["dataset"], row["source_record_id"])))
    write_jsonl(out / "MULTIHOP_NEW_GOLD_COVERAGE_EXCEPTIONS.jsonl", missing_rows)
    write_json(out / "MULTIHOP_V11R1_LOCK.json", lock)
    return {"lock": lock, "overlap": overlap_rows, "invalid": invalid_records, "coverage": coverage_by_dataset}


def internal_doc_id(path: str) -> str:
    return f"{IDENTITY_RULE}:" + quote(path, safe="")


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def enterprise_remediation(out: Path) -> dict:
    base = ROOT / "data/raw/vendor/enterprise_rag_bench/v1.0.0"
    question_path = base / "questions.jsonl"
    archive_path = base / "all_documents.zip"
    questions = load_jsonl(question_path)
    with zipfile.ZipFile(archive_path) as archive:
        infos = sorted(
            [info for info in archive.infolist() if not info.is_dir() and info.filename.lower().endswith(".txt")],
            key=lambda item: item.filename,
        )
        by_original: dict[str, list[dict]] = collections.defaultdict(list)
        source_lengths: dict[str, dict[str, list[int] | int]] = collections.defaultdict(
            lambda: {"count": 0, "bytes": [], "chars": [], "words": [], "lines": [], "utf8_decode_errors": 0}
        )
        for info in infos:
            path = info.filename
            match = re.match(r"(dsid_[0-9a-f]{32})__", Path(path).name)
            if not match:
                continue
            original = match.group(1)
            record = {
                "internal_doc_id": internal_doc_id(path),
                "original_doc_id": original,
                "archive_path": path,
                "source_type": Path(path).parts[0],
                "archive_member_crc32": f"{info.CRC:08x}",
                "uncompressed_bytes": info.file_size,
                "identity_rule": IDENTITY_RULE,
            }
            by_original[original].append(record)
            stats = source_lengths[record["source_type"]]
            stats["count"] += 1
            stats["bytes"].append(info.file_size)
            try:
                text = archive.read(path).decode("utf-8")
            except UnicodeDecodeError:
                stats["utf8_decode_errors"] += 1
                text = archive.read(path).decode("utf-8", errors="replace")
            stats["chars"].append(len(text))
            stats["words"].append(len(text.split()))
            stats["lines"].append(len(text.splitlines()))

        identity_count, identity_sha = write_jsonl_stream(
            out / "ENTERPRISE_DOCUMENT_IDENTITY_MANIFEST.jsonl",
            (row for original in sorted(by_original) for row in sorted(by_original[original], key=lambda item: item["archive_path"])),
        )

        expected_ids = sorted({doc_id for row in questions for doc_id in row.get("expected_doc_ids", [])})
        collision_ids = sorted(original for original, paths in by_original.items() if len(paths) > 1)
        collision_rows = []
        for original in collision_ids:
            for record in sorted(by_original[original], key=lambda item: item["archive_path"]):
                data = archive.read(record["archive_path"])
                text = data.decode("utf-8", errors="replace")
                collision_rows.append(
                    {
                        **record,
                        "content_bytes": len(data),
                        "content_sha256": sha256_bytes(data),
                        "first_line": text.splitlines()[0] if text.splitlines() else "",
                        "distinct_content_from_same_original_id": True,
                    }
                )
        collision_sha = write_jsonl(out / "ENTERPRISE_COLLISION_IDENTITY_AUDIT.jsonl", collision_rows)

        qst_0413 = next(row for row in questions if row["question_id"] == "qst_0413")
        qst_0413_physical = []
        for original in dict.fromkeys(qst_0413.get("expected_doc_ids", [])):
            for record in sorted(by_original.get(original, []), key=lambda item: item["archive_path"]):
                data = archive.read(record["archive_path"])
                text = data.decode("utf-8", errors="replace")
                qst_0413_physical.append(
                    {
                        **record,
                        "content_sha256": sha256_bytes(data),
                        "first_lines": text.splitlines()[:8],
                        "text_prefix_800": text[:800],
                    }
                )

        question_manifest = []
        ambiguous_questions = []
        for row in questions:
            original_expected = list(row.get("expected_doc_ids", []))
            unique_expected = list(dict.fromkeys(original_expected))
            resolutions = []
            all_internal = []
            for original in unique_expected:
                matches = sorted(by_original.get(original, []), key=lambda item: item["archive_path"])
                internal_ids = [item["internal_doc_id"] for item in matches]
                all_internal.extend(internal_ids)
                resolutions.append(
                    {
                        "original_doc_id": original,
                        "internal_doc_ids": internal_ids,
                        "archive_paths": [item["archive_path"] for item in matches],
                        "status": "MISSING" if not matches else "UNIQUE" if len(matches) == 1 else "AMBIGUOUS_ORIGINAL_ID",
                    }
                )
            ambiguous = any(item["status"] == "AMBIGUOUS_ORIGINAL_ID" for item in resolutions)
            missing = any(item["status"] == "MISSING" for item in resolutions)
            condition = (
                "empty_no_defined_gold" if not unique_expected else
                "known_gold_1_to_5" if len(unique_expected) <= 5 else
                "infeasible_at_k5_gt5"
            )
            if not unique_expected:
                eligibility = "NOT_APPLICABLE_EMPTY_GOLD"
            elif len(unique_expected) > 5:
                eligibility = "NOT_APPLICABLE_GT5_GOLD"
            elif ambiguous:
                eligibility = "NOT_APPLICABLE_GOLD_IDENTITY_AMBIGUOUS"
            elif missing:
                eligibility = "NOT_APPLICABLE_MISSING_GOLD_ID"
            else:
                eligibility = "APPLICABLE"
            manifest_row = {
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "source_types": row.get("source_types", []),
                "question": row["question"],
                "expected_doc_ids": original_expected,
                "unique_expected_doc_ids": unique_expected,
                "expected_internal_doc_ids": sorted(set(all_internal)),
                "expected_document_resolution": resolutions,
                "raw_expected_doc_count": len(original_expected),
                "unique_expected_doc_count": len(unique_expected),
                "label_condition": condition,
                "cssr_denominator_status": eligibility,
                "gold_identity_status": "AMBIGUOUS_ORIGINAL_ID" if ambiguous else "MISSING_ORIGINAL_ID" if missing else "RESOLVED_OR_EMPTY",
                "gold_identity_exception": "V11.1R1-GOLD-ID-AMBIGUITY-qst_0413" if row["question_id"] == "qst_0413" and ambiguous else None,
                "evaluation_only_metadata": True,
            }
            question_manifest.append(manifest_row)
            if ambiguous or missing:
                ambiguous_questions.append(row["question_id"])

        question_manifest_sha = write_jsonl(out / "ENTERPRISE_QUESTION_MANIFEST_V11R1.jsonl", question_manifest)

    source_stats = {}
    for source, values in sorted(source_lengths.items()):
        source_stats[source] = {
            "documents": values["count"],
            "utf8_decode_errors": values["utf8_decode_errors"],
            "bytes": {"min": min(values["bytes"]), "median": int(statistics.median(values["bytes"])), "p95": percentile(values["bytes"], .95), "max": max(values["bytes"])},
            "characters": {"min": min(values["chars"]), "median": int(statistics.median(values["chars"])), "p95": percentile(values["chars"], .95), "max": max(values["chars"])},
            "whitespace_words": {"min": min(values["words"]), "median": int(statistics.median(values["words"])), "p95": percentile(values["words"], .95), "max": max(values["words"])},
            "lines": {"min": min(values["lines"]), "median": int(statistics.median(values["lines"])), "p95": percentile(values["lines"], .95), "max": max(values["lines"])},
        }
    length_stats = {
        "release": "EnterpriseRAG-Bench-v1.0.0",
        "archive_sha256": sha256_file(archive_path),
        "text_files": len(infos),
        "unique_original_doc_ids": len(by_original),
        "duplicate_original_doc_ids": len([value for value in by_original.values() if len(value) > 1]),
        "source_type_counts": {source: values["documents"] for source, values in sorted(source_stats.items())},
        "source_type_length_stats": source_stats,
        "measurement": "all .txt archive members; UTF-8 strict decode with replacement only for the reported decode-error count; no model tokenization or inference",
    }
    write_json(out / "ENTERPRISE_LENGTH_SOURCE_STATS.json", length_stats)

    categories = collections.Counter(row["question_type"] for row in questions)
    raw_cardinality = collections.Counter(len(row.get("expected_doc_ids", [])) for row in questions)
    unique_cardinality = collections.Counter(len(dict.fromkeys(row.get("expected_doc_ids", []))) for row in questions)
    applicable = sum(row["cssr_denominator_status"] == "APPLICABLE" for row in question_manifest)
    denom = {
        "all_questions": len(questions),
        "empty_gold": sum(not row["unique_expected_doc_ids"] for row in question_manifest),
        "known_unique_gold_1_to_5_nominal": sum(1 <= row["unique_expected_doc_count"] <= 5 for row in question_manifest),
        "known_unique_gold_1_to_5_applicable_after_identity_resolution": applicable,
        "known_unique_gold_1_to_5_blocked_by_identity_ambiguity": sum(row["cssr_denominator_status"] == "NOT_APPLICABLE_GOLD_IDENTITY_AMBIGUOUS" for row in question_manifest),
        "gt5_gold": sum(row["unique_expected_doc_count"] > 5 for row in question_manifest),
        "missing_gold_id": sum(row["cssr_denominator_status"] == "NOT_APPLICABLE_MISSING_GOLD_ID" for row in question_manifest),
        "gold_identity_ambiguity_exception": {
            "version": "V11.1R1",
            "question_ids": [row["question_id"] for row in question_manifest if row["gold_identity_exception"]],
            "decision": "retain all questions; do not select either physical file; exclude only the ambiguous row from an identity-resolved CSSR denominator",
        },
        "raw_expected_cardinality": {str(key): value for key, value in sorted(raw_cardinality.items())},
        "unique_expected_cardinality": {str(key): value for key, value in sorted(unique_cardinality.items())},
        "category_counts": dict(sorted(categories.items())),
    }
    write_json(out / "ENTERPRISE_DENOMINATOR_RECOMPUTATION.json", denom)
    write_json(out / "ENTERPRISE_QST_0413_SOURCE_INSPECTION.json", {
        "question_id": "qst_0413",
        "question": qst_0413["question"],
        "original_expected_doc_ids": qst_0413["expected_doc_ids"],
        "answer_facts": qst_0413.get("answer_facts", []),
        "physical_candidates": qst_0413_physical,
        "finding": "the original expected ID maps to two distinct physical Jira files with different content hashes; source mapping does not uniquely resolve the intended gold file",
        "exception": "V11.1R1-GOLD-ID-AMBIGUITY-qst_0413",
    })
    identity_policy = {
        "version": "enterprise-identity-v1.1r1",
        "status": "FROZEN_FOR_ENTERPRISE_INDEX_PREPARATION; GOLD_AMBIGUITY_REMAINS_BLOCKED",
        "official_files_preserved": True,
        "internal_id": IDENTITY_RULE + ":<percent-encoded-relative-archive-member-path>",
        "original_id_metadata_field": "original_doc_id",
        "archive_path_metadata_field": "archive_path",
        "collision_policy": "retain every physical .txt member; never deduplicate colliding original IDs; use path-qualified internal_doc_id",
        "qst_0413_policy": "retain all 500 questions, preserve raw expected_doc_ids, do not arbitrarily choose a physical file, version the ambiguity exception",
        "identity_manifest": "results/preflight/v11_remediation/ENTERPRISE_DOCUMENT_IDENTITY_MANIFEST.jsonl",
        "collision_audit": "results/preflight/v11_remediation/ENTERPRISE_COLLISION_IDENTITY_AUDIT.jsonl",
    }
    write_json(out / "ENTERPRISE_IDENTITY_POLICY_V11R1.json", identity_policy)
    adapter = {
        "version": "enterprise-index-adapter-v1.1r1",
        "status": "FROZEN_BEFORE_INDEXING; EXECUTION_BLOCKED_UNTIL_OWNER_ACCEPTS_IDENTITY_EXCEPTION",
        "source_documents": "all official v1.0.0 .txt archive members; no gold-derived filtering",
        "normalization": {"encoding": "UTF-8 with strict validation", "unicode": "NFKC", "line_endings": "LF", "title": "first non-empty line", "content": "remaining normalized text", "empty_title": "empty string"},
        "long_document_handling": {"unit": "whitespace-token chunks", "chunk_size": 384, "overlap": 64, "token_definition": "repeated non-whitespace spans after normalization", "short_document": "one chunk", "deterministic_order": "archive path then chunk index"},
        "candidate_identity": {"public_candidate_id": "internal_doc_id", "intermediate_chunk_id": "internal_doc_id#chunk-{zero_based_index}", "original_doc_id_metadata": True, "archive_path_metadata": True},
        "chunk_to_document_aggregation": {"per_modality": "best (minimum) chunk rank from that modality", "fusion": "RRF(k=60) over document ranks", "document_tie_break": "lexicographic internal_doc_id", "representative_chunk": "best fused chunk rank, then lexicographic chunk ID"},
        "deduplication": "deduplicate exact repeated chunks within one physical document by normalized text, first occurrence retained; do not deduplicate across physical files even when original IDs or content collide",
        "selector_visible_text": {"pointwise": "title plus first 128 tokens of the representative chunk", "groupwise": "title plus first 64 tokens of the representative chunk", "question_cap": 64, "reader_cap": 256, "views": "reuse the frozen v1.1 view builder; no prompt/checkpoint/ranking-policy change"},
        "retrieval_contract": {"bm25_top_k": 100, "dense_top_k": 100, "rrf_k": 60, "frozen_slate_k": 20, "output_k": 5, "qrels_accessed_during_generation": False},
        "multi_hop_protocol": "unchanged; this adapter applies only to EnterpriseRAG-Bench",
    }
    write_json(out / "ENTERPRISE_INDEXING_CONTRACT_V11R1.json", adapter)
    return {
        "questions": questions,
        "question_manifest": question_manifest,
        "by_original": by_original,
        "identity_count": identity_count,
        "identity_sha256": identity_sha,
        "collision_sha256": collision_sha,
        "question_manifest_sha256": question_manifest_sha,
        "length_stats": length_stats,
        "denominator": denom,
        "qst_0413": qst_0413_physical,
    }


def track_c_cohort(out: Path, multihop: dict, enterprise: dict) -> dict:
    new_rows = load_jsonl(ROOT / "results/preflight/v11/NEW_TEST_IDS.jsonl")
    enterprise_rows = enterprise["questions"]
    selected = []
    multi_targets = {"hotpot": 17, "musique": 17, "wiki2": 16}
    for dataset, target in multi_targets.items():
        candidates = [row for row in new_rows if row["dataset"] == dataset]
        candidates.sort(key=lambda row: hashlib.sha256(f"track-c-v11r1:{SEED}:{row['query_id']}".encode()).hexdigest())
        for row in candidates[:target]:
            selected.append({
                "track": "A",
                "dataset": dataset,
                "query_id": row["query_id"],
                "stratum": dataset,
                "selection_rule": "first stable-hash IDs within dataset from already frozen new-query manifest",
                "gold_or_model_outcome_used": False,
            })
    categories = sorted({row["question_type"] for row in enterprise_rows})
    for category in categories:
        candidates = [row for row in enterprise_rows if row["question_type"] == category]
        candidates.sort(key=lambda row: hashlib.sha256(f"track-c-v11r1:{SEED}:{row['question_id']}".encode()).hexdigest())
        for row in candidates[:5]:
            selected.append({
                "track": "B",
                "dataset": "enterprise_rag_bench_v1.0.0",
                "query_id": row["question_id"],
                "stratum": row["question_type"],
                "selection_rule": "first five stable-hash IDs within each published question_type",
                "gold_or_model_outcome_used": False,
            })
    selected.sort(key=lambda row: (row["track"], row["stratum"], row["query_id"]))
    for index, row in enumerate(selected, start=1):
        row["cohort_position"] = index
        row["sampling_seed"] = SEED
    cohort_sha = write_jsonl(out / "TRACK_C_V11R1_COHORT.jsonl", selected)
    blocks = []
    for block_index in range(10):
        query_ids = [row["query_id"] for row in selected[block_index * 10:(block_index + 1) * 10]]
        order = sorted(POLICIES, key=lambda policy: hashlib.sha256(f"track-c-policy:{SEED}:{block_index}:{policy}".encode()).hexdigest())
        blocks.append({"block_index": block_index, "query_ids": query_ids, "policy_order": order})
    schedule = {
        "version": "track-c-v1.1r1",
        "status": "COHORT_FROZEN_EXECUTION_BLOCKED_BY_GPU_WINDOW_AND_ENTERPRISE_GOLD_EXCEPTION",
        "seed": SEED,
        "cohort_sha256": cohort_sha,
        "cohort_rows": len(selected),
        "track_counts": {"A_multihop": sum(row["track"] == "A" for row in selected), "B_enterprise": sum(row["track"] == "B" for row in selected)},
        "strata": collections.Counter(row["stratum"] for row in selected),
        "policies": POLICIES,
        "concurrency": 10,
        "cache_disabled": True,
        "blocks": blocks,
        "warmup": "No timed warmup is authorized in this P0 package; any future warmup must be frozen before execution and separately traced",
        "serving_inference_executed": False,
        "jev_calls": 0,
    }
    schedule["strata"] = dict(sorted(schedule["strata"].items()))
    schedule_sha = write_json(out / "TRACK_C_V11R1_SCHEDULE.json", schedule)
    return {"cohort": selected, "cohort_sha256": cohort_sha, "schedule": schedule, "schedule_sha256": schedule_sha}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    multihop = multihop_remediation(args.output_dir)
    enterprise = enterprise_remediation(args.output_dir)
    track_c = track_c_cohort(args.output_dir, multihop, enterprise)
    summary = {
        "version": "v11.1-remediation-r1",
        "freeze_date": FREEZE_DATE,
        "inference_executed": False,
        "jev_calls": 0,
        "multihop_lock_sha256": sha256_file(args.output_dir / "MULTIHOP_V11R1_LOCK.json"),
        "multihop_debug_overlap_sha256": sha256_file(args.output_dir / "MULTIHOP_DEBUG_OVERLAP_AUDIT.jsonl"),
        "multihop_invalid_source_records_total": multihop["lock"]["invalid_source_records_total"],
        "multihop_new_selected_total": multihop["lock"]["new_selected_total"],
        "enterprise_identity_manifest_rows": enterprise["identity_count"],
        "enterprise_identity_manifest_sha256": enterprise["identity_sha256"],
        "enterprise_collision_audit_sha256": enterprise["collision_sha256"],
        "enterprise_question_manifest_rows": len(enterprise["question_manifest"]),
        "enterprise_question_manifest_sha256": enterprise["question_manifest_sha256"],
        "enterprise_denominator_sha256": sha256_file(args.output_dir / "ENTERPRISE_DENOMINATOR_RECOMPUTATION.json"),
        "track_c_cohort_rows": len(track_c["cohort"]),
        "track_c_cohort_sha256": track_c["cohort_sha256"],
        "track_c_schedule_sha256": track_c["schedule_sha256"],
    }
    write_json(args.output_dir / "V11_P0_REMEDIATION_MACHINE_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
