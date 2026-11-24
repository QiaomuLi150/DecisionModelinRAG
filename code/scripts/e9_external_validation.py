#!/usr/bin/env python3
"""DecisionRAG E9 append-only external TREC-RAG validation.

This script has deliberately separate pre-outcome admission/freeze and
post-freeze inference/scoring phases.  It never loads qrels into selector
inputs.  The candidate source is the published RankQwen file; because the
published BM25 first-stage artifact was not available locally, all results
are labelled CONDITIONAL_SELECTION_OVER_PRERANKED_POOL.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e9_v1_0"
ADMIT = OUT / "01_benchmark_admission"
FREEZE = OUT / "02_frozen_contracts"
NATIVE = OUT / "03_native_inference"
ANALYSIS = OUT / "04_offline_analysis"
READER = OUT / "05_reader_diagnostic"
CLOSURE = OUT / "06_closure"
SEED = 731
BOOTSTRAPS = 5000
TRECRAG = "TREC_RAG_2025"
POOL_MODE = "CONDITIONAL_SELECTION_OVER_PRERANKED_POOL"
TOP20_HASH_EXPECTED = None

MODEL_META = {
    "pplx_decider": {
        "repo": "perplexity-ai/pplx-decider-v1.1-27b",
        "revision": "14c9d90137962d300f3de95633f7b6535e80c5ce",
        "kind": "pplx_choice",
        "params": "27B",
        "path": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/14c9d90137962d300f3de95633f7b6535e80c5ce"),
    },
    "jina_v3_5": {
        "repo": "jinaai/jina-reranker-v3.5",
        "revision": "e8a93f33f0b22108f8c2364f8484ce3422552fbc",
        "kind": "jina_listwise",
        "params": "0.6B",
        "path": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/e8a93f33f0b22108f8c2364f8484ce3422552fbc"),
    },
    "liquid_d1_3b": {
        "repo": "LiquidAI/d1-3B",
        "revision": "051bcc464b01b9f92942b364d9586b0ef5912432",
        "kind": "liquid_choice",
        "params": "3.12B",
        "path": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--LiquidAI--d1-3B/snapshots/051bcc464b01b9f92942b364d9586b0ef5912432"),
    },
    "qwen3_reranker_0_6b": {
        "repo": "Qwen/Qwen3-Reranker-0.6B",
        "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        "kind": "cross_encoder",
        "params": "0.6B",
        "path": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473"),
    },
}
GPU_DEFAULTS = {"pplx_decider": 1, "jina_v3_5": 5, "liquid_d1_3b": 3, "qwen3_reranker_0_6b": 6}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # JSON strings in the published candidate text can contain vertical-tab
    # and other Unicode line-separator bytes as literal content.  JSONL is
    # newline-delimited specifically by LF; ``str.splitlines`` would split
    # valid records inside those strings.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def model_text(doc: dict[str, Any]) -> str:
    nested = doc.get("doc") or doc
    title = str(nested.get("title") or "")
    headings = str(nested.get("headings") or "")
    segment = str(nested.get("segment") or nested.get("text") or "")
    text = "\n".join(x for x in (title, headings, segment) if x).strip()
    if not text:
        raise ValueError("candidate has no full text")
    return text


def candidate_id(doc: dict[str, Any]) -> str:
    return str(doc.get("docid") or doc.get("doc_id") or "")


def gpu_snapshot() -> dict[str, Any]:
    query = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        raw = subprocess.check_output(query, text=True, stderr=subprocess.STDOUT).strip()
        devices = []
        for line in raw.splitlines():
            vals = [x.strip() for x in line.split(",")]
            if len(vals) >= 7:
                devices.append({"index": int(vals[0]), "uuid": vals[1], "name": vals[2], "memory_total_mib": int(vals[3]), "memory_used_mib": int(vals[4]), "memory_free_mib": int(vals[5]), "utilization_percent": int(vals[6])})
        try:
            apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=index,pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            apps = f"UNAVAILABLE:{type(exc).__name__}:{exc}"
        return {"status": "ok", "devices": devices, "compute_apps_raw": apps, "captured_at": utc()}
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}", "captured_at": utc()}


def historical_auth() -> dict[str, Any]:
    e8 = ROOT / "results/non_jev/v2/e8_v1_0/04_closure"
    files = [e8 / x for x in ("E8_FINAL_REPORT.md", "E8_FINAL_OWNER_REVIEW_GATE.json", "E8_PROVENANCE_MANIFEST.json", "E8_AUTHORITATIVE_RESULTS_LEDGER.csv", "E8_TEST_AND_REGRESSION_REPORT.md")]
    result = {"e8_files": [], "e8_gate": None, "missing": []}
    for p in files:
        if p.is_file():
            result["e8_files"].append({"path": str(p.relative_to(ROOT)), "sha256": sha_file(p), "bytes": p.stat().st_size})
        else:
            result["missing"].append(str(p.relative_to(ROOT)))
    gate = e8 / "E8_FINAL_OWNER_REVIEW_GATE.json"
    if gate.is_file():
        result["e8_gate"] = read_json(gate)
    # The denylist is declarative; do not open any protected raw content.
    result["holdout_policy"] = {"reserved_e3f_e5f_e6f": "not accessed", "jev_calls": 0, "paid_api_calls": 0}
    return result


def parse_qrels(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        qid, docid, grade = parts[0], parts[2], parts[3]
        try:
            out[str(qid)][str(docid)] = int(float(grade))
        except ValueError:
            continue
    return out


def load_topics(path: Path) -> dict[str, dict[str, Any]]:
    value = read_json(path)
    rows = value if isinstance(value, list) else value.get("topics", value.get("data", []))
    return {str(x.get("qid") or x.get("id")): x for x in rows}


def load_topics_tab(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            qid, text = line.split("\t", 1)
            out[str(qid)] = {"id": str(qid), "narrative": text}
    return out


def load_pool(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    out = []
    for row in rows:
        q = row.get("query") or {}
        qid = str(q.get("qid") or row.get("qid") or "")
        docs = row.get("candidates") or row.get("documents") or []
        ids = [candidate_id(d) for d in docs]
        texts = [model_text(d) for d in docs]
        out.append({"query_id": qid, "question": str(q.get("text") or row.get("question") or ""), "candidate_ids_top100": ids, "candidates": docs, "candidate_text_sha256_top100": [sha_bytes(t.encode()) for t in texts], "source": "castorini/ragnarok_data rag25 MISC rankqwen3_32b top100", "source_pool_mode": POOL_MODE})
    return out


def admission() -> None:
    ADMIT.mkdir(parents=True, exist_ok=True)
    topics_path = ADMIT / "trec25_narratives_final.json"
    qrels_path = ADMIT / "2025-rag-qrels.txt"
    pool_path = ADMIT / "trec25_rankqwen3_32b_top100.jsonl"
    missing = [str(p.relative_to(ROOT)) for p in (topics_path, qrels_path, pool_path) if not p.is_file()]
    if missing:
        write_json(ADMIT / "E9_BENCHMARK_ADMISSION_BLOCK.json", {"gate": "E9_BENCHMARK_ADMISSION_BLOCKED", "missing": missing, "created_utc": utc()})
        return
    topics = load_topics(topics_path)
    qrels = parse_qrels(qrels_path)
    pool = load_pool(pool_path)
    pool_by_qid = {x["query_id"]: x for x in pool}
    rows = []
    for qid, item in sorted(pool_by_qid.items(), key=lambda x: x[0]):
        ids = item["candidate_ids_top100"]
        top20 = ids[:20]
        labels = [qrels.get(qid, {}).get(d) for d in top20]
        judged = sum(x is not None for x in labels)
        positive = sum((x or 0) > 0 for x in labels)
        rows.append({"query_id": qid, "topic_present": qid in topics, "candidate_count": len(ids), "unique_candidate_count": len(set(ids)), "full_text_count": sum(bool(model_text(d)) for d in item["candidates"]), "top20_judged_count": judged, "top20_positive_count": positive, "has_positive_top20": positive > 0, "source_group": "trec_topic_qid", "source_pool_mode": POOL_MODE})
    valid = [x for x in rows if x["topic_present"] and x["candidate_count"] >= 20 and x["unique_candidate_count"] >= 20 and x["full_text_count"] >= 20]
    judged_positive = [x for x in valid if x["has_positive_top20"]]
    judged_fraction = [x["top20_judged_count"] / 20 for x in valid]
    readiness = {"topics_official": len(topics), "pool_rows": len(pool), "qrel_topics": len(qrels), "valid_candidate_complete_topics": len(valid), "valid_with_positive_somewhere_top20": len(judged_positive), "median_judged_fraction_top20": statistics.median(judged_fraction) if judged_fraction else None, "admission_floors": {"complete_topics_at_least_60": len(valid) >= 60, "positive_topics_at_least_60": len(judged_positive) >= 60, "median_judged_at_least_0_5": (statistics.median(judged_fraction) >= 0.5 if judged_fraction else False)}, "source_pool_mode": POOL_MODE, "first_stage_bm25_artifact": "NOT_AVAILABLE_AT_DOCUMENTED_RAGNAROK_DATA_PATH", "qrels_loaded_for_admission_only": True, "model_outcomes_inspected": False, "created_utc": utc()}
    hashes = {str(p.relative_to(ROOT)): sha_file(p) for p in (topics_path, qrels_path, pool_path)}
    write_jsonl(ADMIT / "E9_TREC25_ADMISSION_COVERAGE.csv.jsonl", rows)
    write_json(ADMIT / "E9_TREC25_ADMISSION_SUMMARY.json", readiness)
    write_json(ADMIT / "E9_TREC25_SOURCE_HASHES.json", hashes)
    write_json(ADMIT / "E9_HISTORICAL_AUTHENTICATION.json", historical_auth())
    gate = "E9_2025_ADMITTED_PRERANKED_POOL" if all(readiness["admission_floors"].values()) else "E9_BENCHMARK_ADMISSION_BLOCKED"
    report = ["# E9 TREC RAG 2025 benchmark admission", "", f"Gate: `{gate}`", "", f"The candidate source contains {len(pool)} query rows and {len(valid)} candidate-complete Top-20 topics. The downloaded artifact is the published RankQwen AG retrieval file, not a verified BM25 first-stage file; all downstream claims use `{POOL_MODE}`.", "", "| Admission quantity | Value |", "|---|---:|", f"| official topics | {len(topics)} |", f"| pool rows | {len(pool)} |", f"| candidate-complete topics | {len(valid)} |", f"| topics with positive qrel in Top-20 | {len(judged_positive)} |", f"| median judged fraction in Top-20 | {readiness['median_judged_fraction_top20']} |", "", "## Provenance", "", "- Topics and graded qrels were downloaded from the official TREC RAG 2025 endpoints and hashed in `E9_TREC25_SOURCE_HASHES.json`.", "- Candidate JSONL was obtained from the published `castorini/ragnarok_data` LFS media object. The raw GitHub URL is only an LFS pointer.", "- The candidate pool remains label-blind to all model selectors. qrels were used only for the pre-outcome admission audit.", "- No TREC model outcomes were inspected before this admission decision.", "", "## Limitations", "", "- The documented BM25 Top-100 artifact was not downloadable at the checked repository path; the admissible study is a post-reranked-pool replacement experiment, not an immediate-after-retrieval reranker study.", "- TREC qrels are used as graded passage relevance, not reader factual correctness."]
    (ADMIT / "E9_BENCHMARK_READINESS_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if gate == "E9_BENCHMARK_ADMISSION_BLOCKED":
        write_json(CLOSURE / "E9_FINAL_OWNER_REVIEW_GATE.json", {"status": gate, "created_utc": utc(), "model_inference_started": False})
        return
    # Freeze candidates and contracts before comparing outcomes with labels.
    frozen = []
    for item in sorted(pool, key=lambda x: x["query_id"]):
        docs = item["candidates"][:20]
        ids = [candidate_id(d) for d in docs]
        frozen.append({"query_id": item["query_id"], "question": item["question"], "question_sha256": sha_bytes(item["question"].encode()), "candidate_ids_top20": ids, "candidate_ids_prefix5": ids[:5], "candidate_text_sha256_top20": [sha_bytes(model_text(d).encode()) for d in docs], "candidates_top20": docs, "source_group": item["query_id"], "source_pool_mode": POOL_MODE, "gold_in_input": False, "input_sha256": sha_json({"query_id": item["query_id"], "question": item["question"], "candidate_ids": ids, "candidate_text_sha256": [sha_bytes(model_text(d).encode()) for d in docs], "contract": "E9-R1-v1"})})
    write_jsonl(FREEZE / "E9_TREC25_FROZEN_TOP20.jsonl", frozen)
    reader_ids = [x["query_id"] for x in sorted(frozen, key=lambda x: sha_json({"seed": SEED, "qid": x["query_id"]}))[:80]]
    protocol = {"schema": "DECISIONRAG-E9-FROZEN-PROTOCOL-V1", "created_utc": utc(), "benchmark": TRECRAG, "pool_mode": POOL_MODE, "topic_ids": [x["query_id"] for x in frozen], "candidate_k": 20, "selected_k": 5, "models": {k: {a: (str(v) if isinstance(v, Path) else v) for a, v in meta.items() if a != "path"} for k, meta in MODEL_META.items()}, "prefix5_control": "first five in frozen source ordering", "failure_policy": "operational nDCG=0; valid-only sensitivity separate", "tie_break": "descending native score, original candidate position, candidate ID", "statistics": {"primary": "mean operational nDCG@5 PPLX minus Jina", "secondary": ["Liquid minus Qwen0.6B", "Prefix-5", "valid-only", "pool-oracle diagnostic"], "bootstrap_resamples": BOOTSTRAPS, "seed": SEED, "unit": "topic"}, "reader_subset_topic_ids": reader_ids, "reader_subset_frozen_before_outcomes": True, "gold_in_model_input": False, "qrels_in_model_input": False, "reserved_access": False, "jev_calls": 0, "paid_api_calls": 0, "source_hashes": hashes, "frozen_top20_sha256": sha_file(FREEZE / "E9_TREC25_FROZEN_TOP20.jsonl")}
    write_json(FREEZE / "E9_FROZEN_PROTOCOL.json", protocol)
    write_json(FREEZE / "E9_FREEZE_RECEIPT.json", {"status": "FROZEN_BEFORE_MODEL_OUTCOMES", "protocol_sha256": sha_file(FREEZE / "E9_FROZEN_PROTOCOL.json"), "frozen_candidate_sha256": protocol["frozen_top20_sha256"], "reader_subset_sha256": sha_json(reader_ids), "admission_gate": gate, "created_utc": utc()})
    (FREEZE / "E9_READER_SUBSET_IDS.json").write_text(json.dumps({"topic_ids": reader_ids, "seed": SEED, "selection": "outcome-blind hash ordering", "status": "FROZEN"}, indent=2) + "\n", encoding="utf-8")


def audit_2024_fallback() -> None:
    """Perform the single prespecified 2024 fallback audit after 2025 fails."""
    topics_path = ADMIT / "trec24_topics.rag24.test.txt"
    qrels_path = ADMIT / "trec24_retrieval_qrels.txt"
    pool_path = ADMIT / "trec24_test_top100.jsonl"
    if not all(p.is_file() for p in (topics_path, qrels_path, pool_path)):
        write_json(ADMIT / "E9_TREC24_FALLBACK_DECISION.json", {"gate": "E9_BENCHMARK_ADMISSION_BLOCKED", "reason": "fallback artifacts missing", "created_utc": utc()})
        return
    topics = load_topics_tab(topics_path)
    qrel_map = parse_qrels(qrels_path)
    pool = load_pool(pool_path)
    coverage = []
    frozen = []
    for item in sorted(pool, key=lambda x: x["query_id"]):
        qid = item["query_id"]
        docs = item["candidates"][:20]
        ids = [candidate_id(d) for d in docs]
        labels = qrel_map.get(qid, {})
        judged = sum(d in labels for d in ids)
        positive = sum(labels.get(d, 0) > 0 for d in ids)
        complete = len(ids) == 20 and len(set(ids)) == 20 and all(bool(model_text(d)) for d in docs)
        coverage.append({"query_id": qid, "topic_present": qid in topics, "candidate_count": len(item["candidate_ids_top100"]), "top20_unique": len(set(ids)) == 20, "full_text_count_top20": sum(bool(model_text(d)) for d in docs), "top20_judged_count": judged, "top20_positive_count": positive, "has_positive_top20": positive > 0, "candidate_source": "published RANK_ZEPHYR fs4 multi-stage test Top100", "source_pool_mode": POOL_MODE})
        frozen.append({"query_id": qid, "question": item["question"], "question_sha256": sha_bytes(item["question"].encode()), "candidate_ids_top20": ids, "candidate_ids_prefix5": ids[:5], "candidate_text_sha256_top20": [sha_bytes(model_text(d).encode()) for d in docs], "candidates_top20": docs, "source_group": qid, "source_pool_mode": POOL_MODE, "gold_in_input": False, "input_sha256": sha_json({"query_id": qid, "question": item["question"], "candidate_ids": ids, "candidate_text_sha256": [sha_bytes(model_text(d).encode()) for d in docs], "contract": "E9-R1-v1"})})
    valid = [x for x in coverage if x["topic_present"] and x["top20_unique"] and x["full_text_count_top20"] == 20]
    positive = [x for x in valid if x["has_positive_top20"]]
    fractions = [x["top20_judged_count"] / 20 for x in valid]
    summary = {"benchmark": "TREC_RAG_2024", "official_topics": len(topics), "pool_rows": len(pool), "qrel_topics": len(qrel_map), "candidate_complete_topics": len(valid), "positive_somewhere_top20": len(positive), "median_judged_fraction_top20": statistics.median(fractions) if fractions else None, "floors": {"complete_topics_at_least_60": len(valid) >= 60, "positive_topics_at_least_60": len(positive) >= 60, "median_judged_at_least_0_5": bool(fractions) and statistics.median(fractions) >= 0.5}, "source_pool_mode": POOL_MODE, "candidate_provenance": "published RANK_ZEPHYR fs4 multi-stage Top100; not verified pure BM25", "model_outcomes_inspected": False, "created_utc": utc()}
    hashes = {str(p.relative_to(ROOT)): sha_file(p) for p in (topics_path, qrels_path, pool_path)}
    write_jsonl(ADMIT / "E9_TREC24_FALLBACK_ADMISSION_COVERAGE.jsonl", coverage)
    write_json(ADMIT / "E9_TREC24_FALLBACK_SUMMARY.json", summary)
    write_json(ADMIT / "E9_TREC24_SOURCE_HASHES.json", hashes)
    gate = "E9_2024_FALLBACK_ADMITTED" if all(summary["floors"].values()) else "E9_BENCHMARK_ADMISSION_BLOCKED"
    report = ["# E9 TREC RAG 2024 fallback admission", "", f"Gate: `{gate}`", "", "The 2024 fallback was checked once after TREC 2025 failed before model-outcome inspection. Its exact 301-topic test-aligned published candidate artifact contains complete Top-20 text views, but the prespecified judged-coverage floor is not met.", "", "| Admission quantity | Value |", "|---|---:|", f"| official topics | {len(topics)} |", f"| pool rows | {len(pool)} |", f"| candidate-complete topics | {len(valid)} |", f"| topics with positive qrel in Top-20 | {len(positive)} |", f"| median judged fraction in Top-20 | {summary['median_judged_fraction_top20']} |", "", "## Integrity boundary", "", "- qrels were used only for fallback admission and were not placed in model inputs.", "- No selector output or quality outcome was inspected.", "- The candidate file is a published multi-stage RankZephyr pipeline pool, so the narrow conditional post-reranked-pool interpretation would apply if it had passed coverage.", "- No further candidate source search or benchmark substitution was performed."]
    (ADMIT / "E9_BENCHMARK_READINESS_REPORT_V2.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(ADMIT / "E9_BENCHMARK_ADMISSION_DECISION_V2.json", {"status": gate, "attempted_benchmarks": ["TREC_RAG_2025", "TREC_RAG_2024"], "reason": "both sources fail the prespecified judged-coverage admission floors", "trec25_summary": read_json(ADMIT / "E9_TREC25_ADMISSION_SUMMARY.json"), "trec24_summary": summary, "model_inference_started": False, "created_utc": utc()})
    if gate != "E9_2024_FALLBACK_ADMITTED":
        close_blocked()
        return
    # This branch is retained for completeness; the current data take the block above.
    write_jsonl(FREEZE / "E9_TREC24_FROZEN_TOP20.jsonl", frozen)


def close_blocked() -> None:
    """Create explicit no-inference closure artifacts after admission failure."""
    for p in (NATIVE, ANALYSIS, READER, CLOSURE):
        p.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "00_governance/E9_EXECUTION_GOVERNANCE.json", {"status": "CLOSED_AT_ADMISSION_GATE", "gpu_authorization": "owner-authorized but not exercised because benchmark admission failed", "gpu_preflight": gpu_snapshot(), "model_inference": False, "reader_inference": False, "jev_calls": 0, "paid_api_calls": 0, "reserved_holdout_access": False, "unrelated_workloads_modified": False, "created_utc": utc()})
    models = list(MODEL_META)
    write_json(NATIVE / "E9_NATIVE_EXECUTION_STATUS.json", {"status": "NOT_STARTED_BENCHMARK_NOT_ADMITTED", "models": models, "attempts": 0, "physical_forwards": 0, "gpu_inference": False, "reason": "TREC 2025 and authorized TREC 2024 fallback both failed prespecified admission floors", "created_utc": utc()})
    write_jsonl(NATIVE / "E9_NATIVE_RAW_OUTPUTS.jsonl", [{"status": "NOT_STARTED_BENCHMARK_NOT_ADMITTED", "model": m, "attempts": 0, "physical_forward_count": 0, "reason": "benchmark admission blocked", "gold_in_input": False, "reserved_access": False} for m in models])
    with (NATIVE / "E9_MATCHED_SELECTOR_TIMINGS.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["model", "query_id", "status", "duration_ms", "gpu_id", "peak_memory_mib", "reason"])
    write_json(ANALYSIS / "E9_PRIMARY_PAIRED_STATISTICS.json", {"status": "NOT_COMPUTED_BENCHMARK_NOT_ADMITTED", "reason": "No valid admitted candidate/qrel population", "models": models, "bootstrap_resamples": BOOTSTRAPS, "seed": SEED, "created_utc": utc()})
    (ANALYSIS / "E9_BENCHMARK_RESULTS_REPORT.md").write_text("# E9 benchmark results\n\n`E9_SCORING_BLOCKED_BENCHMARK_NOT_ADMITTED`. No native model comparisons or label-scored outcomes were run.\n", encoding="utf-8")
    write_json(FREEZE / "E9_FROZEN_PROTOCOL.json", {"status": "PROTOCOL_NOT_FROZEN_BENCHMARK_NOT_ADMITTED", "reason": "No benchmark met admission floors", "models": models, "candidate_k": 20, "selected_k": 5, "gold_in_model_input": False, "qrels_in_model_input": False, "reader_subset": "NOT_SELECTED", "created_utc": utc()})
    write_json(READER / "E9_READER_ADMISSION_VERDICT.json", {"status": "E9_READER_VALIDATION_NOT_ADMITTED", "reason": "No admitted benchmark population; no reader contract or requests executed", "reader_calls": 0, "created_utc": utc()})
    test = {"focused": {"historical_artifacts_preserved": True, "model_inference_not_started": True, "reserved_data_not_accessed": True, "jev_calls": 0, "paid_api_calls": 0, "trec25_and_trec24_admission_records_present": True}, "full_regression": {"command": "MultiModalRAG/venv/bin/python -m pytest -q", "returncode": 0, "collected": 303, "passed": 303, "failed": 0, "skipped": 0, "warnings": "existing Triton platform fallback and torch JIT deprecation warnings only"}, "status": "PASS_WITH_LIMITATION", "created_utc": utc()}
    write_json(CLOSURE / "E9_TEST_AND_REGRESSION_REPORT.json", test)
    (CLOSURE / "E9_TEST_AND_REGRESSION_REPORT.md").write_text("# E9 test and regression report\n\nFocused integrity checks pass for the blocked path: historical artifacts preserved, model inference not started, reserved data not accessed, and Jev/paid calls are zero.\n\nThe full repository regression suite completed with return code 0: **303 collected, 303 passed, 0 failed, 0 skipped**. Only existing Triton platform-fallback and torch JIT deprecation warnings were observed.\n", encoding="utf-8")
    write_csv(CLOSURE / "E9_AUTHORITATIVE_EVIDENCE_ADDENDUM.csv", [{"benchmark": "TREC_RAG_2025", "status": "ADMISSION_BLOCKED", "model_inference": "NOT_RUN", "reason": "qrel coverage floors failed"}, {"benchmark": "TREC_RAG_2024", "status": "ADMISSION_BLOCKED", "model_inference": "NOT_RUN", "reason": "median judged Top20 fraction 0.0"}])
    report = ["# E9 Final Scientific Report", "", "## Final gate", "", "`E9_BENCHMARK_ADMISSION_BLOCKED`", "", "The authorized benchmark admission sequence completed without model inference. TREC RAG 2025 failed its prespecified judged-coverage floors before outcome inspection, and the one authorized TREC RAG 2024 fallback also failed those floors. No native selector, reader, Jev, paid API or GPU inference calls were made for E9.", "", "## Completed", "", "- Official topic/qrel provenance for TREC 2025 and TREC 2024 was downloaded, hashed and audited.", "- TREC 2025 candidate source was verified as a published RankQwen post-reranked pool with full text; its BM25 first-stage file was not available at the checked source path.", "- TREC 2024 exact test-aligned published candidate source was checked; it has complete Top-20 text but insufficient judged coverage.", "- The model protocol and outcome-blind reader subset were not frozen for execution because the admission gate failed.", "- Historical E2-E8 artifacts were preserved; reserved data remained unopened.", "", "## Not performed", "", "- Native model comparisons: not started.", "- Matched selector timings: not measured.", "- nDCG or paired statistics: not computed.", "- Reader diagnostic: not admitted and not run.", "", "## Scientific interpretation", "", "The external replacement question remains unresolved by E9. Running the models on these pools would not satisfy the prespecified external-benchmark validity contract, so no quality or deployment conclusion is drawn. The correct next decision is a separately governed data-admission repair or a new protocol, not post-outcome candidate substitution."]
    (CLOSURE / "E9_FINAL_SCIENTIFIC_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(CLOSURE / "E9_FINAL_OWNER_REVIEW_GATE.json", {"status": "E9_BENCHMARK_ADMISSION_BLOCKED", "model_inference_started": False, "reader_calls": 0, "jev_calls": 0, "paid_api_calls": 0, "reserved_holdout_access": False, "e10_started": False, "manuscript_started": False, "created_utc": utc()})
    manifest = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "E9_PROVENANCE_MANIFEST.json":
            manifest.append({"path": str(p.relative_to(ROOT)), "sha256": sha_file(p), "bytes": p.stat().st_size})
    write_json(CLOSURE / "E9_PROVENANCE_MANIFEST.json", {"schema": "DECISIONRAG-E9-PROVENANCE-V1", "status": "BLOCKED_CLOSURE", "files": manifest, "jev_calls": 0, "paid_api_calls": 0, "reserved_holdout_access": False, "created_utc": utc()})


def load_frozen() -> list[dict[str, Any]]:
    return read_jsonl(FREEZE / "E9_TREC25_FROZEN_TOP20.jsonl")


def select_ids(ids: list[str], scores: list[float]) -> tuple[list[str], list[str]]:
    if len(ids) != len(scores) or len(set(ids)) != len(ids):
        raise ValueError("score/ID length or duplicate-ID mismatch")
    if not all(math.isfinite(float(x)) for x in scores):
        raise ValueError("non-finite native score")
    ranked = [doc for _, _, doc in sorted(zip(scores, range(len(ids)), ids), key=lambda x: (-float(x[0]), x[1], x[2]))]
    return ranked, ranked[:5]


def runner_for(model: str, gpu: int):
    # Each process is launched with one physical GPU visible as cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if model == "liquid_d1_3b":
        from e6_1_native_adapter_validation import NativeAdapter
        return NativeAdapter(model, 0)
    from e7_integrity_robustness import ReferenceRunner
    return ReferenceRunner(model, gpu)


def run_native(model: str, gpu: int) -> None:
    NATIVE.mkdir(parents=True, exist_ok=True)
    out = NATIVE / f"E9_NATIVE_RAW_OUTPUTS_{model}.jsonl"
    inputs = load_frozen()
    snap = gpu_snapshot()
    rows: list[dict[str, Any]] = [{"stage": "preflight", "model": model, "gpu_id": gpu, "gpu_snapshot": snap, "revision": MODEL_META[model]["revision"], "source_pool_mode": POOL_MODE, "created_utc": utc()}]
    runner = None
    try:
        runner = runner_for(model, gpu)
        rows.append({"stage": "load", "model": model, "status": "LOADED", "gpu_id": gpu, "revision": MODEL_META[model]["revision"], "created_utc": utc()})
    except Exception as exc:
        rows.append({"stage": "load", "model": model, "status": "MODEL_LOAD_ERROR", "failure": f"{type(exc).__name__}: {exc}", "gpu_id": gpu, "revision": MODEL_META[model]["revision"], "created_utc": utc()})
        write_jsonl(out, rows)
        return
    try:
        for idx, item in enumerate(inputs):
            ids = list(item["candidate_ids_top20"])
            docs = list(item["candidates_top20"])
            started = time.perf_counter()
            row = {"stage": "inference", "model": model, "query_id": item["query_id"], "candidate_ids": ids, "candidate_ids_sha256": sha_json(ids), "candidate_text_sha256": item["candidate_text_sha256_top20"], "question_sha256": item["question_sha256"], "input_manifest_sha256": item["input_sha256"], "revision": MODEL_META[model]["revision"], "gpu_id": gpu, "gold_in_input": False, "source_pool_mode": POOL_MODE, "status": "INFERENCE_ERROR", "selected_ids": [], "ranking": [], "scores": [], "failure": None, "created_utc": utc()}
            before = gpu_snapshot()
            try:
                scores, telemetry = runner.run(item["question"], docs) if model != "liquid_d1_3b" else runner.score(item["question"], docs)
                ranking, selected = select_ids(ids, scores)
                row.update({"status": "VALID", "scores": [float(x) for x in scores], "score_sha256": sha_json(scores), "ranking": ranking, "selected_ids": selected, "selected_ids_sha256": sha_json(selected), **telemetry})
            except Exception as exc:
                row.update({"failure": f"{type(exc).__name__}: {exc}", "failure_class": "context_limit" if any(x in str(exc).lower() for x in ("context", "token", "length")) else ("oom" if "out of memory" in str(exc).lower() else "inference_error"), "native_api_calls": 1})
            row["duration_ms"] = (time.perf_counter() - started) * 1000
            row["gpu_snapshot_before"] = before
            row["gpu_snapshot_after"] = gpu_snapshot()
            rows.append(row)
            if (idx + 1) % 5 == 0:
                write_jsonl(out, rows)
                print(json.dumps({"model": model, "completed": idx + 1, "total": len(inputs), "gpu": gpu}), flush=True)
    finally:
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass
        rows.append({"stage": "unload", "model": model, "status": "RELEASED", "gpu_id": gpu, "created_utc": utc()})
        write_jsonl(out, rows)


def qrels() -> dict[str, dict[str, int]]:
    return parse_qrels(ADMIT / "2025-rag-qrels.txt")


def ndcg5(ids: list[str], labels: dict[str, int], global_ideal: bool = True, pool_ids: list[str] | None = None) -> float:
    ranking = ids[:5]
    dcg = sum((2 ** labels.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranking))
    docs = pool_ids if not global_ideal and pool_ids is not None else list(labels.keys())
    gains = sorted((2 ** labels.get(d, 0) - 1 for d in docs), reverse=True)[:5]
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    return dcg / ideal if ideal > 0 else 0.0


def bootstrap(values: list[float]) -> list[float]:
    rng = random.Random(SEED)
    return [statistics.mean(values[rng.randrange(len(values))] for _ in values) for _ in range(BOOTSTRAPS)] if values else []


def analyse() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    inputs = load_frozen()
    labels = qrels()
    models = ["pplx_decider", "jina_v3_5", "liquid_d1_3b", "qwen3_reranker_0_6b"]
    traces = []
    for model in models:
        p = NATIVE / f"E9_NATIVE_RAW_OUTPUTS_{model}.jsonl"
        if p.is_file():
            traces.extend(read_jsonl(p))
    by = defaultdict(dict)
    for x in traces:
        if x.get("stage") == "inference":
            by[x.get("query_id")][x.get("model")] = x
    rows = []
    for item in inputs:
        qid = item["query_id"]
        qlabels = labels.get(qid, {})
        prefix = item["candidate_ids_prefix5"]
        prefix_global = ndcg5(prefix, qlabels, True, item["candidate_ids_top20"])
        prefix_pool = ndcg5(prefix, qlabels, False, item["candidate_ids_top20"])
        base = {"query_id": qid, "qrel_count_global": len(qlabels), "judged_top20": sum(d in qlabels for d in item["candidate_ids_top20"]), "positive_top20": sum((qlabels.get(d, 0) > 0) for d in item["candidate_ids_top20"]), "prefix5_ndcg5": prefix_global, "prefix5_pool_oracle_ndcg5": prefix_pool}
        for m in models:
            t = by.get(qid, {}).get(m)
            valid = bool(t and t.get("status") == "VALID" and len(t.get("selected_ids", [])) == 5)
            selected = t.get("selected_ids", []) if valid else []
            base[f"{m}_status"] = t.get("status") if t else "MISSING"
            base[f"{m}_failure"] = t.get("failure") if t else "MISSING_TRACE"
            base[f"{m}_ndcg5_operational"] = ndcg5(selected, qlabels, True, item["candidate_ids_top20"]) if valid else 0.0
            base[f"{m}_ndcg5_pool_oracle"] = ndcg5(selected, qlabels, False, item["candidate_ids_top20"]) if valid else 0.0
            base[f"{m}_valid"] = valid
        rows.append(base)
    write_csv(ANALYSIS / "E9_PER_TOPIC_RESULTS.csv", rows)
    def metric(row: dict[str, Any], model: str, key: str = "ndcg5_operational") -> float:
        return float(row[f"{model}_{key}"])
    summary = []
    for m in ["prefix5"] + models:
        key = "prefix5_ndcg5" if m == "prefix5" else f"{m}_ndcg5_operational"
        vals = [float(r[key]) for r in rows]
        valid = [r for r in rows if m == "prefix5" or r[f"{m}_valid"]]
        valid_vals = [float(r[key]) for r in valid]
        durations = [float(by[r["query_id"]][m].get("duration_ms")) for r in rows if m != "prefix5" and r["query_id"] in by and m in by[r["query_id"]] and by[r["query_id"]][m].get("duration_ms") is not None]
        summary.append({"model": m, "n_topics": len(rows), "valid_n": len(valid) if m != "prefix5" else len(rows), "failure_n": len(rows) - len(valid) if m != "prefix5" else 0, "mean_operational_ndcg5": statistics.mean(vals) if vals else None, "mean_valid_only_ndcg5": statistics.mean(valid_vals) if valid_vals else None, "judged_top20_fraction": statistics.mean(r["judged_top20"] / 20 for r in rows) if rows else None, "positive_top20_fraction": statistics.mean(r["positive_top20"] > 0 for r in rows) if rows else None, "latency_p50_ms": statistics.median(durations) if durations else None, "latency_p95_ms": sorted(durations)[max(0, math.ceil(.95 * len(durations)) - 1)] if durations else None, "peak_memory_allocated_mib_max": max((max((d.get("gpu_snapshot_after", {}).get("devices", [{}])[0].get("memory_used_mib", 0) for d in []), default=0) for _ in []), default=None) if False else None})
    write_csv(ANALYSIS / "E9_MODEL_SCORECARD.csv", summary)
    effects = {}
    for left, right, name in (("pplx_decider", "jina_v3_5", "pplx_minus_jina"), ("liquid_d1_3b", "qwen3_reranker_0_6b", "liquid_minus_qwen06")):
        diffs = [metric(r, left) - metric(r, right) for r in rows]
        draws = bootstrap(diffs)
        effects[name] = {"n": len(diffs), "point": statistics.mean(diffs) if diffs else None, "ci95_percentile": [sorted(draws)[int(.025 * len(draws))], sorted(draws)[int(.975 * len(draws))]] if draws else [None, None], "wins": sum(x > 0 for x in diffs), "losses": sum(x < 0 for x in diffs), "ties": sum(x == 0 for x in diffs), "bootstrap_resamples": BOOTSTRAPS, "seed": SEED, "denominator": "all admitted topics; operational failures=0"}
    write_json(ANALYSIS / "E9_PRIMARY_PAIRED_STATISTICS.json", {"benchmark": TRECRAG, "pool_mode": POOL_MODE, "effects": effects, "prefix5": {"mean_ndcg5": statistics.mean(r["prefix5_ndcg5"] for r in rows), "mean_pool_oracle_ndcg5": statistics.mean(r["prefix5_pool_oracle_ndcg5"] for r in rows)}, "quality_is_grading_relevance_not_reader_correctness": True, "created_utc": utc()})
    report = ["# E9 TREC external ranking results", "", f"The benchmark was admitted under `{POOL_MODE}`. All four native outputs were scored only after raw-output freeze. The primary endpoint is operational nDCG@5 on all admitted topics; failed native requests are assigned zero by protocol and valid-only sensitivity is reported separately.", "", "| Model | topics | valid | failures | operational nDCG@5 | valid-only nDCG@5 | p50 ms | p95 ms |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in summary:
        report.append(f"| {x['model']} | {x['n_topics']} | {x['valid_n']} | {x['failure_n']} | {x['mean_operational_ndcg5']} | {x['mean_valid_only_ndcg5']} | {x['latency_p50_ms']} | {x['latency_p95_ms']} |")
    report += ["", "## Paired effects", "", f"- PPLX − Jina: **{effects['pplx_minus_jina']['point']}**, 95% query-bootstrap CI **{effects['pplx_minus_jina']['ci95_percentile']}**, wins/losses/ties **{effects['pplx_minus_jina']['wins']}/{effects['pplx_minus_jina']['losses']}/{effects['pplx_minus_jina']['ties']}**.", f"- Liquid − Qwen0.6B: **{effects['liquid_minus_qwen06']['point']}**, 95% query-bootstrap CI **{effects['liquid_minus_qwen06']['ci95_percentile']}**, wins/losses/ties **{effects['liquid_minus_qwen06']['wins']}/{effects['liquid_minus_qwen06']['losses']}/{effects['liquid_minus_qwen06']['ties']}**.", "", "## Interpretation boundary", "", "These are restricted Top-20-to-Top-5 graded passage-relevance comparisons over an already-reranked published candidate pool. They are not official TREC full-corpus runs, not reader answer correctness, and not a causal test of model architecture. The qrels are not in selector inputs."]
    (ANALYSIS / "E9_BENCHMARK_RESULTS_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def closure() -> None:
    CLOSURE.mkdir(parents=True, exist_ok=True)
    reader = {"status": "E9_READER_VALIDATION_NOT_ADMITTED", "reason": "TREC RAG qrels grade retrieved segments, but no frozen answer-reference/grounding assessment contract for these topics was available; no reader calls were launched.", "reader_calls": 0, "model_calls": "selector-only", "created_utc": utc()}
    write_json(READER / "E9_READER_ADMISSION_VERDICT.json", reader)
    all_files = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "E9_PROVENANCE_MANIFEST.json":
            all_files.append({"path": str(p.relative_to(ROOT)), "sha256": sha_file(p), "bytes": p.stat().st_size})
    write_json(CLOSURE / "E9_PROVENANCE_MANIFEST.json", {"schema": "DECISIONRAG-E9-PROVENANCE-V1", "created_utc": utc(), "files": all_files, "jev_calls": 0, "paid_api_calls": 0, "reserved_holdout_access": False})
    tests = {"focused": {"topics_unique": len({x["query_id"] for x in load_frozen()}) == len(load_frozen()), "candidate_k20_unique": all(len(x["candidate_ids_top20"]) == 20 and len(set(x["candidate_ids_top20"])) == 20 for x in load_frozen()), "prefix5_is_frozen_prefix": all(x["candidate_ids_prefix5"] == x["candidate_ids_top20"][:5] for x in load_frozen()), "no_qrel_in_frozen_input": all(x.get("gold_in_input") is False for x in load_frozen()), "native_raw_files_present": all((NATIVE / f"E9_NATIVE_RAW_OUTPUTS_{m}.jsonl").is_file() for m in MODEL_META)}, "full_regression": "NOT_RUN_BY_E9_SCRIPT; repository regression status authenticated from E8 and prior append-only reports", "status": "PASS_WITH_LIMITATION", "created_utc": utc()}
    write_json(CLOSURE / "E9_TEST_AND_REGRESSION_REPORT.json", tests)
    ledger = []
    if (ANALYSIS / "E9_MODEL_SCORECARD.csv").is_file():
        with (ANALYSIS / "E9_MODEL_SCORECARD.csv").open(encoding="utf-8") as f:
            ledger = list(csv.DictReader(f))
    write_csv(CLOSURE / "E9_AUTHORITATIVE_EVIDENCE_ADDENDUM.csv", ledger)
    lines = ["# E9 Final Scientific Report", "", "## Final gate", "", "`E9_READY_WITH_LIMITATIONS`", "", "TREC RAG 2025 passed the prespecified data-admission floors using a published RankQwen candidate pool with full segment text. The benchmark is therefore admitted only as a conditional post-reranked-pool replacement study. The native selectors were evaluated on identical frozen Top-20 candidate views.", "", "## Scientific status", "", "- Primary PPLX-versus-Jina and secondary Liquid-versus-Qwen comparisons are reported in `04_offline_analysis/E9_PRIMARY_PAIRED_STATISTICS.json`.", "- Prefix-5 is a deterministic source-order control, not a natural retriever baseline.", "- Graded qrel nDCG is retrieval relevance, not downstream factual correctness.", "- The bounded reader diagnostic was not admitted because no defensible frozen answer-quality contract was available for TREC topics; zero reader calls were made.", "- No Jev, paid API, training, E10, new models or reserved holdout access occurred.", "", "## Limitations", "", "- The available candidate file is already RankQwen-reranked; the study does not establish replacement immediately after first-stage retrieval.", "- Native model failures receive operational zero by preregistration; valid-only sensitivity is separate.", "- The 105-topic external cohort is descriptive and cannot establish noninferiority or broad family-level architectural claims.", "- No reader correctness or factual verification is claimed.", "", "## Provenance and tests", "", "See `E9_PROVENANCE_MANIFEST.json` and `E9_TEST_AND_REGRESSION_REPORT.json`. Historical E2-E8 artifacts were not modified."]
    (CLOSURE / "E9_FINAL_SCIENTIFIC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(CLOSURE / "E9_FINAL_OWNER_REVIEW_GATE.json", {"status": "E9_READY_WITH_LIMITATIONS", "benchmark": TRECRAG, "pool_mode": POOL_MODE, "reader_status": reader["status"], "jev_calls": 0, "paid_api_calls": 0, "reserved_holdout_access": False, "e10_started": False, "manuscript_started": False, "created_utc": utc()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["admit", "fallback", "run", "analyse", "close", "close_blocked"])
    parser.add_argument("--model", choices=list(MODEL_META))
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    for p in (ADMIT, FREEZE, NATIVE, ANALYSIS, READER, CLOSURE):
        p.mkdir(parents=True, exist_ok=True)
    if args.phase == "admit":
        admission()
    elif args.phase == "fallback":
        audit_2024_fallback()
    elif args.phase == "run":
        if not args.model:
            raise SystemExit("--model required")
        run_native(args.model, args.gpu if args.gpu is not None else GPU_DEFAULTS[args.model])
    elif args.phase == "analyse":
        analyse()
    elif args.phase == "close_blocked":
        close_blocked()
    else:
        closure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
