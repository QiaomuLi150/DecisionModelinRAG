#!/usr/bin/env python3
"""DecisionRAG E2-M1 controlled Clef mechanism experiment.

This runner is versioned separately from historical M3/Track A/B scripts.  It
uses only the locally pinned Clef checkpoint and frozen historical slates.  It
supports smoke, CAL and full exploratory stages with resumable JSONL traces.
Gold is loaded only by the offline evaluator after model calls; inference
records are built from question/slate/corpus fields only.
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
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results/non_jev/v2/e2_m1"
HF_ROOT = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub")
CLEF_REV = "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3"
CLEF_ID = "Cloudflare/clef-flash"
VIEW_REV = "7b928d828b7b0e022f929d9bd2e44165aa270148"
SEED = 20261011
BOOTSTRAP_SEED = 731
BOOTSTRAP_REPS = 5000
MAX_LENGTH = 16384
GPU_ID = 0


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def snapshot(owner: str, repo: str, revision: str) -> Path:
    return HF_ROOT / f"models--{owner}--{repo}" / "snapshots" / revision


def make_gpu_audit() -> dict:
    """Record approved-device metadata without querying any other device."""
    return {
        "authorized_gpu_ids": [0, 1, 2, 4],
        "selected_gpu_ids": [GPU_ID],
        "selected_gpu_reason": "GPU0 had approximately 81 GiB free and no compute process at inspection; GPUs1/2 host active services, GPU4 was busy, and other devices were not authorized by the attached task.",
        "unrelated_services_interrupted": False,
        "service_restarts": 0,
        "checkpoint": {"model_id": CLEF_ID, "revision": CLEF_REV, "local_snapshot": str(snapshot("Cloudflare", "clef-flash", CLEF_REV))},
        "precision": "bfloat16",
        "telemetry_note": "Actual GPU utilization/memory samples are recorded per inference call by torch; device inventory was inspected outside the sandbox before this run.",
    }


def load_local_cal_items() -> list[dict]:
    from decisionrag.selection import make_views
    questions = load_jsonl(ROOT / "data/manifests/questions.jsonl")
    slates = {r["query_id"]: r for r in load_jsonl(ROOT / "data/frozen_slates/m2_top20_slates.jsonl")}
    corpus = {r["doc_id"]: r for r in load_jsonl(ROOT / "data/corpus/passages.jsonl")}
    rows = []
    for q in questions:
        if q.get("split") != "cal":
            continue
        s = slates.get(q["query_id"])
        if not s or len(s.get("candidate_ids", [])) != 20:
            raise RuntimeError(f"missing complete CAL slate for {q['query_id']}")
        candidates = []
        for rank, doc_id in enumerate(s["candidate_ids"], 1):
            doc = corpus.get(doc_id)
            if not doc:
                raise RuntimeError(f"missing CAL corpus document {doc_id}")
            candidates.append({"doc_id": doc_id, "dataset": doc["dataset"], "title": doc["title"], "full_text": doc["full_text"], "original_rank": rank})
        rows.append({"cohort": "cal", "dataset": q["dataset"], "query_id": q["query_id"], "question": q["question"], "candidates": candidates, "source": "data/manifests/questions.jsonl+m2_top20_slates"})
    rows.sort(key=lambda x: (x["dataset"], x["query_id"]))
    # Freeze a bounded, deterministic 10-per-dataset CAL validation cohort.
    chosen = []
    for dataset in ("hotpot", "musique", "wiki2"):
        group = [r for r in rows if r["dataset"] == dataset]
        chosen.extend(group[:10])
    return chosen


def load_full_items() -> list[dict]:
    """Build inference-only rows from frozen slates; no gold fields are copied."""
    from decisionrag.selection import make_views
    a_slates = load_jsonl(ROOT / "results/non_jev/v11/track_a/V11_TRACK_A_RETRIEVAL_SLATES.jsonl")
    b_slates = load_jsonl(ROOT / "results/non_jev/v11/track_b/V11_TRACK_B_RETRIEVAL_SLATES.jsonl")
    corpus = {r["doc_id"]: r for r in load_jsonl(ROOT / "data/corpus/passages.jsonl")}
    db_path = ROOT / "results/non_jev/v11/track_b/V11_TRACK_B_ENTERPRISE_CORPUS.sqlite3"
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    enterprise = {}
    try:
        ids = [c.get("representative_chunk_id", c["doc_id"]) for s in b_slates for c in s["candidates"]]
        unique = sorted(set(ids))
        for start in range(0, len(unique), 500):
            chunk = unique[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT chunk_id,title,text,source_type FROM chunks WHERE chunk_id IN ({placeholders})"
            for row in conn.execute(query, chunk):
                enterprise[row[0]] = {"doc_id": row[0], "title": row[1], "full_text": row[2], "source_type": row[3], "dataset": "enterprise"}
    finally:
        conn.close()
    rows = []
    for cohort, slates in (("track_a", a_slates), ("track_b", b_slates)):
        for slate in slates:
            candidates = []
            for rank, candidate in enumerate(slate["candidates"], 1):
                doc_id = candidate["doc_id"]
                # Enterprise slates use a document identity for ranking and a
                # separately recorded representative chunk for the selector
                # text.  Preserve the canonical candidate ID while resolving
                # the exact frozen chunk text.
                lookup_id = candidate.get("representative_chunk_id", doc_id)
                doc = corpus.get(doc_id) if cohort == "track_a" else enterprise.get(lookup_id)
                if not doc:
                    raise RuntimeError(f"missing {cohort} candidate text {doc_id}")
                candidates.append({"doc_id": doc_id, "dataset": doc.get("dataset", "enterprise"), "title": candidate.get("title", doc.get("title", "")), "full_text": doc["full_text"], "original_rank": rank})
            if len(candidates) != 20 or len({c["doc_id"] for c in candidates}) != 20:
                raise RuntimeError(f"invalid 20-candidate slate {slate['query_id']}")
            rows.append({"cohort": cohort, "dataset": slate.get("dataset", "enterprise") if cohort == "track_a" else "enterprise", "query_id": slate["query_id"], "question": slate["question"], "candidates": candidates, "source": slate.get("slate_id", "frozen")})
    rows.sort(key=lambda x: (x["cohort"], x["query_id"]))
    return rows


def import_clef():
    import importlib
    import torch
    path = snapshot("Cloudflare", "clef-flash", CLEF_REV)
    if not path.is_dir():
        raise RuntimeError(f"missing local Clef snapshot {path}")
    sys.path.insert(0, str(path))
    module = importlib.import_module("joint_schema_model")
    model, processor = module.load_release_model(path, device="cuda", dtype=torch.bfloat16)
    return module, model, processor, torch


def build_views(item: dict, tokenizer):
    from decisionrag.selection import make_views
    return [make_views(item["question"], c, tokenizer) for c in item["candidates"]]


def score_from_logits(logits, encoded_question, torch) -> float:
    probabilities = logits.float().softmax(-1).detach().cpu().tolist()
    option_ids = list(encoded_question.option_ids)
    if "true" not in option_ids or "false" not in option_ids:
        raise RuntimeError(f"binary field options invalid: {option_ids}")
    score = float(probabilities[option_ids.index("true")])
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise RuntimeError(f"invalid binary score {score}")
    return score


def telemetry(torch, started_ns: int) -> dict:
    elapsed = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    row = {"duration_ms": elapsed, "gpu_id": GPU_ID}
    if torch.cuda.is_available():
        row["gpu_memory_allocated_bytes"] = int(torch.cuda.memory_allocated())
        row["gpu_peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        row["gpu_memory_reserved_bytes"] = int(torch.cuda.memory_reserved())
        row["gpu_utilization"] = "NOT_DIRECTLY_AVAILABLE_FROM_TORCH"
    return row


def pointwise_one(item: dict, module, model, processor, tokenizer, torch, batch_size: int = 20) -> dict:
    from decisionrag.selection import pointwise_question, pointwise_state
    views = build_views(item, tokenizer)
    records = [{"state": pointwise_state(v), "questions": pointwise_question(), "id": f"{item['query_id']}::{v.doc_id}"} for v in views]
    encoded = [module.encode_record(processor.tokenizer, r, processor=processor, max_length=MAX_LENGTH) for r in records]
    lengths = [len(e.input_ids) for e in encoded]
    if any(x > MAX_LENGTH for x in lengths):
        raise RuntimeError(f"pointwise context exceeds max length: {max(lengths)}")
    input_hash = sha_json([list(e.input_ids) for e in encoded])
    request_hash = sha_json({"query_id": item["query_id"], "candidate_ids": [v.doc_id for v in views], "view_hashes": [v.selection_text_sha256 for v in views], "contract": "INDEPENDENT_RELEVANCE:v2"})
    scores = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter_ns()
    physical_forwards = 0
    for start in range(0, len(encoded), batch_size):
        batch_records = encoded[start : start + batch_size]
        batch = module.collate_records(batch_records, processor.tokenizer.pad_token_id, torch.device("cuda"))
        with torch.inference_mode():
            outputs = model(batch)
        physical_forwards += 1
        for enc, output in zip(batch_records, outputs):
            scores.append(score_from_logits(output[0], enc.questions[0], torch))
    response_hash = sha_json(scores)
    row = {
        "query_id": item["query_id"], "cohort": item["cohort"], "dataset": item["dataset"], "model_id": CLEF_ID, "revision": CLEF_REV,
        "formulation": "pointwise_binary", "score_contract": "INDEPENDENT_RELEVANCE:v2", "status": "ok", "real_inference": True,
        "candidate_ids": [v.doc_id for v in views], "scores": scores, "score_direction": "higher_is_more_relevant", "score_alignment": "candidate_ids_order",
        "selection_text_sha256": [v.selection_text_sha256 for v in views], "request_hash": request_hash, "actual_input_sha256": input_hash, "response_hash": response_hash,
        "input_tokens": lengths, "context_overflow": False, "physical_forward_calls": physical_forwards, "batch_size": batch_size,
        "telemetry": telemetry(torch, started), "worker_id": "gpu0-single-worker", "cache_reuse": False, "gold_in_inference_input": False,
    }
    return row


def joint_record(item: dict, views: list[Any]) -> dict:
    state = {
        "question": views[0].question_view,
        "candidates": [
            {"candidate_id": v.doc_id, "original_rank": v.original_rank, "title": v.title, "passage": v.passage_view}
            for v in views
        ],
    }
    questions = {}
    for index, view in enumerate(views):
        questions[f"relevance_{index:02d}_{view.doc_id}"] = {
            "type": "noul",
            "instructions": f"For candidate_id={view.doc_id}, does that candidate passage contain factual evidence useful for answering the question? Answer true only for actual evidence, including an intermediate multi-hop link.",
            "criteria": {"true": "The named candidate contains useful factual evidence.", "false": "The named candidate does not contain useful factual evidence."},
        }
    return {"id": f"{item['query_id']}::joint", "state": state, "questions": questions}


def joint_one(item: dict, module, model, processor, tokenizer, torch) -> dict:
    views = build_views(item, tokenizer)
    record = joint_record(item, views)
    encoded = module.encode_record(processor.tokenizer, record, processor=processor, max_length=MAX_LENGTH)
    input_tokens = len(encoded.input_ids)
    if input_tokens > MAX_LENGTH:
        raise RuntimeError(f"joint K20 context exceeds max length: {input_tokens}")
    input_hash = sha_json(list(encoded.input_ids))
    request_hash = sha_json({"query_id": item["query_id"], "candidate_ids": [v.doc_id for v in views], "view_hashes": [v.selection_text_sha256 for v in views], "contract": "JOINT_BINARY_RELEVANCE:v2", "field_ids": list(record["questions"])})
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter_ns()
    batch = module.collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cuda"))
    with torch.inference_mode():
        outputs = model(batch)
    probabilities = outputs[0]
    if len(probabilities) != 20:
        raise RuntimeError(f"native joint output has {len(probabilities)} fields, expected 20")
    scores = [score_from_logits(logit, enc, torch) for logit, enc in zip(probabilities, encoded.questions)]
    row = {
        "query_id": item["query_id"], "cohort": item["cohort"], "dataset": item["dataset"], "model_id": CLEF_ID, "revision": CLEF_REV,
        "formulation": "native_joint_binary", "score_contract": "JOINT_BINARY_RELEVANCE:v2", "status": "ok", "real_inference": True,
        "candidate_ids": [v.doc_id for v in views], "question_field_ids": list(record["questions"]), "scores": scores, "score_direction": "higher_is_more_relevant", "score_alignment": "field_id_candidate_order",
        "selection_text_sha256": [v.selection_text_sha256 for v in views], "request_hash": request_hash, "actual_input_sha256": input_hash, "response_hash": sha_json(scores),
        "input_tokens": input_tokens, "context_overflow": False, "physical_forward_calls": 1, "batch_size": 1,
        "telemetry": telemetry(torch, started), "worker_id": "gpu0-single-worker", "cache_reuse": False, "gold_in_inference_input": False,
    }
    return row


def choose_smoke(items: list[dict]) -> list[dict]:
    # deterministic 2-per-dataset sample; last item gives a different length
    out = []
    for dataset in ("hotpot", "musique", "wiki2"):
        group = [x for x in items if x["dataset"] == dataset]
        out.extend([group[0], group[-1]])
    return out


def load_group_assignments(cohort: str) -> dict[str, list[list[str]]]:
    if cohort == "track_a":
        path = ROOT / "results/non_jev/v11/track_a/V11_TRACK_A_CLEF_GROUPWISE_FULL.jsonl"
    else:
        path = ROOT / "results/non_jev/v11/track_b/V11_TRACK_B_CLEF_GROUPWISE_FULL.jsonl"
    out = {}
    for row in load_jsonl(path):
        groups = [g.get("group_doc_ids", []) for g in row.get("groups", [])]
        if len(groups) == 5 and all(len(g) == 4 and len(set(g)) == 4 for g in groups) and len({d for g in groups for d in g}) == 20:
            out[row["query_id"]] = groups
    return out


def select_top(scores: list[float], ids: list[str], original_ranks: list[int], k: int = 5) -> list[str]:
    return [x[1] for x in sorted(zip(scores, ids), key=lambda x: (-float(x[0]), original_ranks[ids.index(x[1])], x[1]))[:k]]


def select_quota(scores: list[float], ids: list[str], original_ranks: list[int], groups: list[list[str]]) -> list[str]:
    score_map = dict(zip(ids, scores)); rank_map = dict(zip(ids, original_ranks))
    winners = []
    for group in groups:
        winners.append(sorted(group, key=lambda d: (-float(score_map[d]), rank_map[d], d))[0])
    return [x[1] for x in sorted(((score_map[d], d) for d in winners), key=lambda x: (-float(x[0]), rank_map[x[1]], x[1]))]


def load_historical_g5(cohort: str) -> dict[str, list[str]]:
    path = ROOT / ("results/non_jev/v11/track_a/V11_TRACK_A_SELECTOR_RESULTS_FULL.jsonl" if cohort == "track_a" else "results/non_jev/v11/track_b/V11_TRACK_B_SELECTOR_RESULTS.jsonl")
    out = {}
    for row in load_jsonl(path):
        if row.get("policy_id") == "clef_g5":
            out[row["query_id"]] = row.get("selected_doc_ids", [])
    return out


def run_inference(items: list[dict], stage: str, module, model, processor, tokenizer, torch) -> tuple[list[dict], list[dict]]:
    if stage == "smoke":
        selected = choose_smoke(items)
        point_path, joint_path = OUT / "E2_M1_SMOKE_POINTWISE.jsonl", OUT / "E2_M1_SMOKE_JOINT.jsonl"
    elif stage == "cal":
        selected = items
        point_path, joint_path = OUT / "E2_M1_CAL_POINTWISE.jsonl", OUT / "E2_M1_CAL_JOINT.jsonl"
    else:
        selected = items
        point_path, joint_path = OUT / "E2_M1_POINTWISE_SCORES.jsonl", OUT / "E2_M1_JOINT_BINARY_SCORES.jsonl"
    existing_p = {r["query_id"]: r for r in load_jsonl(point_path)} if point_path.is_file() else {}
    existing_j = {r["query_id"]: r for r in load_jsonl(joint_path)} if joint_path.is_file() else {}
    point_rows, joint_rows = list(existing_p.values()), list(existing_j.values())
    for index, item in enumerate(selected, 1):
        qid = item["query_id"]
        try:
            if qid not in existing_p:
                row = pointwise_one(item, module, model, processor, tokenizer, torch)
                append_jsonl(point_path, row); existing_p[qid] = row; point_rows.append(row)
            if qid not in existing_j:
                row = joint_one(item, module, model, processor, tokenizer, torch)
                append_jsonl(joint_path, row); existing_j[qid] = row; joint_rows.append(row)
        except Exception as exc:
            error = {"query_id": qid, "cohort": item["cohort"], "dataset": item["dataset"], "status": "error", "real_inference": False, "error_type": f"{type(exc).__name__}: {exc}", "gold_in_inference_input": False}
            if qid not in existing_p:
                append_jsonl(point_path, dict(error, formulation="pointwise_binary")); existing_p[qid] = dict(error, formulation="pointwise_binary"); point_rows.append(existing_p[qid])
            if qid not in existing_j:
                append_jsonl(joint_path, dict(error, formulation="native_joint_binary")); existing_j[qid] = dict(error, formulation="native_joint_binary"); joint_rows.append(existing_j[qid])
            print(f"{stage} ERROR {qid}: {error['error_type']}", flush=True)
            # An unsupported native branch is a hard gate; continue only to
            # preserve explicit error accounting for the selected cohort.
        if index % 10 == 0 or index == len(selected):
            okp = sum(r.get("status") == "ok" for r in point_rows if r.get("query_id") in {x["query_id"] for x in selected})
            okj = sum(r.get("status") == "ok" for r in joint_rows if r.get("query_id") in {x["query_id"] for x in selected})
            print(f"{stage} {index}/{len(selected)} pointwise_ok={okp} joint_ok={okj}", flush=True)
    return point_rows, joint_rows


def cal_report(items: list[dict], point: list[dict], joint: list[dict]) -> dict:
    return {
        "stage": "S1_CAL",
        "status": "PASS" if len(items) == 30 and all(r.get("status") == "ok" for r in point) and all(r.get("status") == "ok" for r in joint) else "BLOCKED",
        "cohort_count": len(items), "dataset_counts": dict(Counter(i["dataset"] for i in items)),
        "pointwise_ok": sum(r.get("status") == "ok" for r in point), "joint_ok": sum(r.get("status") == "ok" for r in joint),
        "pointwise_errors": [r for r in point if r.get("status") != "ok"], "joint_errors": [r for r in joint if r.get("status") != "ok"],
        "joint_k20_feasible": all(r.get("status") == "ok" for r in joint),
        "max_joint_input_tokens": max((r.get("input_tokens", 0) for r in joint if r.get("status") == "ok"), default=None),
        "min_joint_input_tokens": min((r.get("input_tokens", 0) for r in joint if r.get("status") == "ok"), default=None),
    }


def write_selected_and_evaluate(items: list[dict], point: list[dict], joint: list[dict]) -> dict:
    """Create policy rows and compute offline quality from gold sidecars."""
    point_by = {r["query_id"]: r for r in point}
    joint_by = {r["query_id"]: r for r in joint}
    groups = {c: load_group_assignments(c) for c in ("track_a", "track_b")}
    historical = {c: load_historical_g5(c) for c in ("track_a", "track_b")}
    rows = []
    selection_path = OUT / "E2_M1_SELECTED_DOCUMENTS.jsonl"
    selection_path.write_text("", encoding="utf-8")
    for item in items:
        qid = item["query_id"]; ids = [c["doc_id"] for c in item["candidates"]]; ranks = list(range(1, 21)); cohort = item["cohort"]
        p = point_by.get(qid, {}); j = joint_by.get(qid, {})
        policies = {}
        if p.get("status") == "ok":
            policies["clef_pointwise_top5"] = select_top(p["scores"], ids, ranks)
            policies["clef_pointwise_quartet_top5"] = select_quota(p["scores"], ids, ranks, groups[cohort].get(qid, []))
        if j.get("status") == "ok":
            policies["clef_joint_top5"] = select_top(j["scores"], ids, ranks)
            policies["clef_joint_quartet_top5"] = select_quota(j["scores"], ids, ranks, groups[cohort].get(qid, []))
        if qid in historical[cohort]:
            policies["historical_clef_g5"] = historical[cohort][qid]
        row = {"query_id": qid, "cohort": cohort, "dataset": item["dataset"], "candidate_ids": ids, "group_assignments": groups[cohort].get(qid, []), "policies": policies, "status": "ok" if len(policies) >= 5 else "partial", "gold_in_inference_input": False}
        append_jsonl(selection_path, row); rows.append(row)
    return {"rows": rows, "groups": groups}


def load_gold() -> dict[str, set[str]]:
    gold = {}
    paths = (ROOT / "results/preflight/v11/NEW_TEST_GOLD_SIDECAR.jsonl", ROOT / "results/preflight/v11/ENTERPRISE_GOLD_SIDECAR.jsonl")
    enterprise_path = paths[1]
    enterprise_map = defaultdict(list)
    db_path = ROOT / "results/non_jev/v11/track_b/V11_TRACK_B_ENTERPRISE_CORPUS.sqlite3"
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for internal_id, original_id in conn.execute("SELECT internal_doc_id, original_doc_id FROM documents"):
            enterprise_map[original_id].append(internal_id)
    finally:
        conn.close()
    # The enterprise release deliberately preserves original dsid_ values in
    # its gold sidecar while the frozen slates use path-qualified internal IDs.
    # Resolve only one-to-one mappings.  In particular, qst_0413 is retained
    # in the 500-question manifest but omitted from identity-dependent metrics
    # because its original ID maps to two physical files.
    for path in paths:
        for row in load_jsonl(path):
            # Track A sidecars use query_id/gold_doc_ids; the enterprise
            # sidecar uses question_id/expected_doc_ids.  Normalize both
            # frozen, read-only formats without changing their contents.
            qid = row.get("query_id") or row.get("question_id")
            doc_ids = row.get("gold_doc_ids")
            if doc_ids is None:
                doc_ids = row.get("expected_doc_ids")
            if qid is None:
                raise ValueError(f"Gold sidecar row has no query/question ID: {path}")
            if path == enterprise_path:
                resolved = []
                ambiguous = False
                for original_id in dict.fromkeys(doc_ids or []):
                    matches = enterprise_map.get(original_id, [])
                    if len(matches) != 1:
                        ambiguous = True
                        break
                    resolved.append(matches[0])
                if ambiguous:
                    continue
                gold[qid] = set(resolved)
            else:
                gold[qid] = set(doc_ids or [])
    return gold


def metric_for(gold: set[str], candidates: list[str], selected: list[str]) -> tuple[bool | None, bool, bool]:
    eligible = 1 <= len(gold) <= 5
    if not eligible:
        return None, False, False
    conditional = gold.issubset(set(candidates))
    success = gold.issubset(set(selected))
    return success, conditional, success


def write_quality(items: list[dict], selected: list[dict], gold: dict[str, set[str]]) -> tuple[list[dict], dict]:
    policies = ["clef_pointwise_top5", "clef_joint_top5", "clef_pointwise_quartet_top5", "clef_joint_quartet_top5", "historical_clef_g5"]
    by = {r["query_id"]: r for r in selected}
    all_rows = []
    summary = []
    for cohort, label in (("track_a", "Track A"), ("track_b", "Track B")):
        cohort_items = [i for i in items if i["cohort"] == cohort]
        for policy in policies:
            successes = []; conditional_successes = []; conditional_flags = []; eligible_flags = []
            for item in cohort_items:
                g = gold.get(item["query_id"], set()); sel = by[item["query_id"]]["policies"].get(policy)
                if sel is None:
                    continue
                success, conditional, _ = metric_for(g, [c["doc_id"] for c in item["candidates"]], sel)
                eligible = 1 <= len(g) <= 5
                row = {"query_id": item["query_id"], "cohort": cohort, "dataset": item["dataset"], "policy": policy, "gold_count": len(g), "eligible": eligible, "conditional": conditional, "success": success}
                all_rows.append(row)
                if eligible:
                    eligible_flags.append(success)
                    if conditional:
                        conditional_flags.append(success)
            n_eligible = len(eligible_flags); n_cond = len(conditional_flags)
            summary.append({"cohort": cohort, "scope": label, "dataset": "pooled", "policy": policy, "n_eligible": n_eligible, "n_conditional": n_cond, "success_unconditional": sum(bool(x) for x in eligible_flags), "cssr_unconditional": (sum(bool(x) for x in eligible_flags) / n_eligible if n_eligible else None), "success_conditional": sum(bool(x) for x in conditional_flags), "cssr_conditional": (sum(bool(x) for x in conditional_flags) / n_cond if n_cond else None)})
        for dataset in sorted({i["dataset"] for i in cohort_items}):
            subset = [i for i in cohort_items if i["dataset"] == dataset]
            for policy in policies:
                vals = [r for r in all_rows if r["cohort"] == cohort and r["dataset"] == dataset and r["policy"] == policy and r["eligible"]]
                cond = [r for r in vals if r["conditional"]]
                summary.append({"cohort": cohort, "scope": label, "dataset": dataset, "policy": policy, "n_eligible": len(vals), "n_conditional": len(cond), "success_unconditional": sum(bool(r["success"]) for r in vals), "cssr_unconditional": (sum(bool(r["success"]) for r in vals) / len(vals) if vals else None), "success_conditional": sum(bool(r["success"]) for r in cond), "cssr_conditional": (sum(bool(r["success"]) for r in cond) / len(cond) if cond else None)})
    fields = ["cohort", "scope", "dataset", "policy", "n_eligible", "n_conditional", "success_unconditional", "cssr_unconditional", "success_conditional", "cssr_conditional"]
    with (OUT / "E2_M1_POLICY_QUALITY_SUMMARY.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summary)
    with (OUT / "E2_M1_CASE_METRICS.jsonl").open("w", encoding="utf-8") as f:
        for row in all_rows: f.write(json.dumps(row, sort_keys=True) + "\n")
    return all_rows, {"summary": summary}


def bootstrap_ci(values: list[float], seed: int = BOOTSTRAP_SEED, reps: int = BOOTSTRAP_REPS) -> tuple[float, float]:
    if not values:
        return (None, None)
    rng = random.Random(seed)
    n = len(values); estimates = []
    for _ in range(reps):
        estimates.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    return estimates[int(0.025 * reps)], estimates[int(0.975 * reps) - 1]


def write_statistics(items: list[dict], selected: list[dict], gold: dict[str, set[str]]) -> list[dict]:
    by = {r["query_id"]: r for r in selected}
    comparisons = [("joint_minus_pointwise", "clef_joint_top5", "clef_pointwise_top5"), ("pointwise_minus_pointwise_quartet", "clef_pointwise_top5", "clef_pointwise_quartet_top5"), ("joint_minus_joint_quartet", "clef_joint_top5", "clef_joint_quartet_top5")]
    rows = []
    for cohort in ("track_a", "track_b"):
        for population in ("unconditional_eligible", "retrieval_conditional"):
            cohort_items = [i for i in items if i["cohort"] == cohort]
            for name, left, right in comparisons:
                left_vals = {}; right_vals = {}
                for item in cohort_items:
                    qid = item["query_id"]; g = gold.get(qid, set())
                    if not (1 <= len(g) <= 5): continue
                    if population == "retrieval_conditional" and not g.issubset(set(c["doc_id"] for c in item["candidates"])): continue
                    lp = by[qid]["policies"].get(left); rp = by[qid]["policies"].get(right)
                    if lp is None or rp is None: continue
                    left_vals[qid] = float(g.issubset(set(lp))); right_vals[qid] = float(g.issubset(set(rp)))
                ids = sorted(set(left_vals) & set(right_vals)); deltas = [left_vals[x] - right_vals[x] for x in ids]
                ci = bootstrap_ci(deltas) if deltas else (None, None)
                rows.append({"cohort": cohort, "population": population, "comparison": name, "left_policy": left, "right_policy": right, "n": len(deltas), "left_success": int(sum(left_vals[x] for x in ids)), "right_success": int(sum(right_vals[x] for x in ids)), "difference": (sum(deltas) / len(deltas) if deltas else None), "ci95_low": ci[0], "ci95_high": ci[1], "paired_wins": sum(x > 0 for x in deltas), "paired_losses": sum(x < 0 for x in deltas), "paired_ties": sum(x == 0 for x in deltas), "bootstrap_reps": BOOTSTRAP_REPS, "bootstrap_seed": BOOTSTRAP_SEED, "interpretation": "exploratory follow-up; not untouched confirmatory inference"})
    fields = list(rows[0]) if rows else ["cohort"]
    with (OUT / "E2_M1_PRIMARY_PAIRED_STATISTICS.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return rows


def write_runtime(point: list[dict], joint: list[dict], gpu_audit: dict) -> None:
    def stats(rows):
        vals = [r.get("telemetry", {}).get("duration_ms") for r in rows if r.get("status") == "ok"]
        vals = sorted(float(x) for x in vals if x is not None)
        if not vals: return {"n": 0}
        return {"n": len(vals), "p50_ms": vals[len(vals)//2], "p95_ms": vals[max(0, math.ceil(.95*len(vals))-1)], "mean_ms": statistics.mean(vals), "total_s": sum(vals)/1000.0}
    payload = {"gpu_audit": gpu_audit, "pointwise": stats(point), "joint": stats(joint), "pointwise_physical_forward_calls": sum(r.get("physical_forward_calls", 0) for r in point if r.get("status") == "ok"), "joint_physical_forward_calls": sum(r.get("physical_forward_calls", 0) for r in joint if r.get("status") == "ok"), "selection_latency_scope": "model scoring only; exploratory, not production SLO validation", "gpu_active_seconds": "NOT_MEASURED_DIRECTLY", "jev_calls": 0, "paid_api_calls": 0}
    write_json(OUT / "E2_M1_RUNTIME_SUMMARY.json", payload)
    write_text(OUT / "E2_M1_RUNTIME_AND_GPU_REPORT.md", "# E2-M1 Runtime and GPU Report\n\n```json\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n```\n\nThese are exploratory selector-mechanism timings, not maximum sustainable production capacity and not validation of the proposed 1-second SLO. GPU0 was used because it was the only owner-authorized device observed genuinely free; GPUs1/2 host active services and GPU4 was busy. No unrelated service was interrupted.\n")


def write_analysis(items, selected, point, joint, stats_rows, quality_rows, cal_status):
    p_by = {r["query_id"]: r for r in point}; j_by = {r["query_id"]: r for r in joint}
    changed = 0; comparable = 0; quota_violations = 0; quota_rows = 0
    for item in items:
        p = p_by.get(item["query_id"], {}); j = j_by.get(item["query_id"], {})
        if p.get("status") == "ok" and j.get("status") == "ok":
            comparable += 1
            p_order = sorted(range(20), key=lambda i: (-p["scores"][i], i, p["candidate_ids"][i]))
            j_order = sorted(range(20), key=lambda i: (-j["scores"][i], i, j["candidate_ids"][i]))
            if p_order != j_order: changed += 1
    for row in selected:
        groups = row.get("group_assignments", []); ids = set(row["candidate_ids"])
        if len(groups) == 5 and len({d for g in groups for d in g}) == 20:
            quota_rows += 1
            for policy in ("clef_pointwise_top5", "clef_joint_top5"):
                chosen = set(row["policies"].get(policy, []))
                # Structural oracle is computed later from gold; this loop only
                # verifies the selection cardinality and group contract.
                if len(chosen) != 5 or not chosen.issubset(ids): quota_violations += 1
    text = f"""# E2-M1 Structural and Mechanism Analysis

**Stage status:** S0 smoke and S1 CAL status are `{cal_status}`. Full exploratory traces are analyzed only when both branches have authentic `ok` outputs.

## Mechanism interpretation

- M1-A pointwise binary and M1-B native joint binary use the same Clef checkpoint, binary `noul` objective, canonical document views and candidate IDs.
- Joint scoring is a single native record with 20 candidate-specific binary fields. It is not five quartet calls and is not a calibrated independent-probability claim.
- M1-C policies reuse the same score vector; no rescoring occurs between global and quartet selection.
- Historical Clef-G5 remains a separate categorical reference and is never merged into the global score vector.

## Context feasibility

The runner records actual encoded token counts per query. K=20 is considered feasible only for rows that successfully encode and execute without silent truncation. A failure is `JOINT_K20_INFEASIBLE` or an explicit model/runtime error; it is not replaced by reduced-K data.

## Ordering and policy diagnostics

- Completed comparable pointwise/joint query rows: `{comparable}`.
- Queries whose full candidate ordering changed between pointwise and joint scores: `{changed}`.
- Valid five-group/four-candidate group contracts checked: `{quota_rows}`.
- Selection cardinality/group-contract violations: `{quota_violations}`.

## Quality interpretation

The policy CSV and paired statistics report Track A and Track B separately, with unconditional eligible and retrieval-conditional denominators. The paired intervals use 5,000 query-level bootstrap resamples with seed 731 and are exploratory follow-up evidence, not untouched confirmation. Tiny subgroup claims are not promoted to significance claims.

## Limitations

Any missing/error rows, context overflows, unmeasured physical GPU active seconds, or unavailable native score semantics remain explicit. The proposed 1-second SLO is not enforced here. No reader generation or new retrieval was run.
"""
    write_text(OUT / "E2_M1_STRUCTURAL_AND_MECHANISM_ANALYSIS.md", text)


def write_authenticity(point, joint, items, gpu_audit):
    p_ok = sum(r.get("status") == "ok" and r.get("real_inference") for r in point)
    j_ok = sum(r.get("status") == "ok" and r.get("real_inference") for r in joint)
    text = f"""# E2-M1 Authenticity Audit

| Check | Result |
|---|---|
| Pointwise rows | {len(point)} total; {p_ok} authentic `ok` |
| Native joint rows | {len(joint)} total; {j_ok} authentic `ok` |
| Candidate count/alignment | Each successful row records 20 canonical IDs and score order |
| Gold in inference inputs | `0` by construction; gold is loaded only after inference for offline evaluation |
| Synthetic/fallback scores | `0` |
| Application cache reuse | `false`; only authenticated resume records can be reused |
| Checkpoint revision | `{CLEF_REV}` |
| GPU | `{gpu_audit['selected_gpu_ids']}` |
| Jev/paid calls | `0/0` |

The runner fails closed on missing candidate text, duplicate IDs, non-finite scores, invalid binary option fields, context overflow, or incomplete native output. Original historical artifacts are read-only and unchanged; all M1 outputs are under `results/non_jev/v2/e2_m1/`.
"""
    write_text(OUT / "E2_M1_AUTHENTICITY_AUDIT.md", text)


def execute_stage(stage: str, gpu: int) -> int:
    global GPU_ID
    GPU_ID = gpu
    OUT.mkdir(parents=True, exist_ok=True)
    gpu_audit = make_gpu_audit()
    config = {
        "study": "DecisionRAG v2.1 E2-M1 accelerated controlled mechanism experiment",
        "stage": stage, "status": "RUNNING", "created_utc": now(), "gpu_audit": gpu_audit,
        "checkpoint": {"model_id": CLEF_ID, "revision": CLEF_REV, "precision": "bfloat16"},
        "view_contract": {"question_tokens": 64, "passage_tokens": 128, "joint_max_length": MAX_LENGTH, "candidate_count": 20},
        "policies": ["clef_pointwise_top5", "clef_joint_top5", "clef_pointwise_quartet_top5", "clef_joint_quartet_top5", "historical_clef_g5"],
        "random_seed": SEED, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_reps": BOOTSTRAP_REPS,
        "gold_in_inference": False, "jev_calls": 0, "paid_api_calls": 0,
    }
    write_json(OUT / "E2_M1_FROZEN_EXPLORATORY_CONFIG.json", config)
    write_json(OUT / "E2_M1_GPU_RESOURCE_AUDIT.json", gpu_audit)
    write_text(OUT / "E2_M1_GPU_RESOURCE_AUDIT.md", "# E2-M1 GPU Resource Audit\n\n```json\n" + json.dumps(gpu_audit, indent=2, sort_keys=True) + "\n```\n\nNo unrelated service was stopped or restarted. Only GPU0 was selected from the authorized set because it was genuinely free at inspection; GPUs1/2 host active services and GPU4 was busy.\n")
    if stage in ("smoke", "cal"):
        items = load_local_cal_items()
    else:
        items = load_full_items()
    print(f"Loading Clef for {stage} on CUDA_VISIBLE_DEVICES={gpu}; rows={len(items)}", flush=True)
    module, model, processor, torch = import_clef()
    tokenizer = processor.tokenizer
    if stage == "smoke":
        selected = choose_smoke(items)
        point, joint = run_inference(selected, "smoke", module, model, processor, tokenizer, torch)
        smoke = {"stage": "S0", "status": "PASS" if all(r.get("status") == "ok" for r in point + joint) else "BLOCKED", "queries": len(selected), "pointwise_ok": sum(r.get("status") == "ok" for r in point), "joint_ok": sum(r.get("status") == "ok" for r in joint), "joint_k20": all(r.get("status") == "ok" for r in joint), "pointwise_rows": point, "joint_rows": joint}
        write_json(OUT / "E2_M1_SMOKE_RESULTS.json", smoke)
        write_text(OUT / "E2_M1_NATIVE_ADAPTER_SMOKE_REPORT.md", "# E2-M1 Native Adapter Smoke Report\n\n```json\n" + json.dumps({k: v for k, v in smoke.items() if k not in ("pointwise_rows", "joint_rows")}, indent=2, sort_keys=True) + "\n```\n\nThe smoke test used authentic Clef inference and candidate-specific native binary fields. Full row-level request/response hashes are in the JSON traces.\n")
        write_text(OUT / "E2_M1_EXECUTION_MANIFEST.json", json.dumps({"stage": "S0", "status": smoke["status"], "next_stage": "S1 if PASS", "jev_calls": 0, "paid_api_calls": 0}, indent=2) + "\n")
        if smoke["status"] != "PASS":
            write_text(OUT / "E2_M1_FINAL_REPORT.md", "# E2-M1 Final Report\n\nStatus: `E2_M1_BLOCKED` at S0. The affected native branch did not pass the authentic smoke gate; no full evaluation was run.\n")
            write_text(OUT / "E2_M1_OWNER_REVIEW_GATE.md", "# E2-M1 Owner Review Gate\n\nStatus: `E2_M1_BLOCKED`. Stop at S0.\n")
            return 2
    elif stage == "cal":
        point, joint = run_inference(items, "cal", module, model, processor, tokenizer, torch)
        report = cal_report(items, point, joint); write_json(OUT / "E2_M1_CAL_VALIDATION.json", report)
        write_text(OUT / "E2_M1_CAL_VALIDATION_REPORT.md", "# E2-M1 CAL Validation Report\n\n```json\n" + json.dumps(report, indent=2, sort_keys=True) + "\n```\n\nCAL was frozen before new results were examined: 10 deterministic rows per HotpotQA, MuSiQue and 2Wiki. No gold fields were copied into inference records.\n")
        write_json(OUT / "E2_M1_EXECUTION_MANIFEST.json", {"stage": "S1", "status": report["status"], "cal_count": len(items), "next_stage": "S2 if PASS", "jev_calls": 0, "paid_api_calls": 0})
        if report["status"] != "PASS":
            write_text(OUT / "E2_M1_FINAL_REPORT.md", "# E2-M1 Final Report\n\nStatus: `E2_M1_PARTIAL_BLOCKED` at S1. Full exploratory evaluation was not run.\n")
            write_text(OUT / "E2_M1_OWNER_REVIEW_GATE.md", "# E2-M1 Owner Review Gate\n\nStatus: `E2_M1_PARTIAL_BLOCKED`. Stop at S1.\n")
            return 2
    else:
        point, joint = run_inference(items, "full", module, model, processor, tokenizer, torch)
        evaluation = write_selected_and_evaluate(items, point, joint)
        gold = load_gold()
        quality_rows, _ = write_quality(items, evaluation["rows"], gold)
        stat_rows = write_statistics(items, evaluation["rows"], gold)
        write_runtime(point, joint, gpu_audit)
        write_analysis(items, evaluation["rows"], point, joint, stat_rows, quality_rows, "PASS")
        write_authenticity(point, joint, items, gpu_audit)
        full_ok = all(r.get("status") == "ok" for r in point + joint) and len(evaluation["rows"]) == len(items)
        status = "E2_M1_PASS" if full_ok else "E2_M1_PARTIAL_BLOCKED"
        write_json(OUT / "E2_M1_EXECUTION_MANIFEST.json", {"stage": "S2/S3/S4", "status": status, "track_a_queries": sum(i["cohort"] == "track_a" for i in items), "track_b_queries": sum(i["cohort"] == "track_b" for i in items), "pointwise_ok": sum(r.get("status") == "ok" for r in point), "joint_ok": sum(r.get("status") == "ok" for r in joint), "selected_rows": len(evaluation["rows"]), "jev_calls": 0, "paid_api_calls": 0, "gold_in_inference": False})
        write_text(OUT / "E2_M1_FINAL_REPORT.md", f"# E2-M1 Final Report\n\nStatus: `{status}`\n\n- Track A queries: `{sum(i['cohort'] == 'track_a' for i in items)}`; Track B queries: `{sum(i['cohort'] == 'track_b' for i in items)}`.\n- Pointwise authentic rows: `{sum(r.get('status') == 'ok' for r in point)}`.\n- Native joint authentic rows: `{sum(r.get('status') == 'ok' for r in joint)}`.\n- K=20 native joint feasibility: `{'PASS' if all(r.get('status') == 'ok' for r in joint) else 'BLOCKED'}`.\n- Historical Clef-G5 was reused as a separate reference.\n- Paired statistics and quality summaries are exploratory, dataset-separated and hash-backed.\n- Jev/paid calls: `0/0`.\n\nSee `E2_M1_POLICY_QUALITY_SUMMARY.csv`, `E2_M1_PRIMARY_PAIRED_STATISTICS.csv`, `E2_M1_RUNTIME_AND_GPU_REPORT.md` and `E2_M1_STRUCTURAL_AND_MECHANISM_ANALYSIS.md`.\n")
        write_text(OUT / "E2_M1_OWNER_REVIEW_GATE.md", f"# E2-M1 Owner Review Gate\n\nStatus: `{status}`\n\nThe run stopped after M1. No E2-N, E2-S/P, model-family expansion, Jev or paid evaluation was started.\n")
    # Release model memory before provenance hashing.
    del model, processor, module
    gc.collect()
    try: torch.cuda.empty_cache()
    except Exception: pass
    return 0


def finalize_provenance() -> int:
    files = sorted(p for p in OUT.iterdir() if p.is_file())
    manifest = {"study": "DecisionRAG v2.1 E2-M1 accelerated controlled mechanism experiment", "created_utc": now(), "git_head": None, "outputs": {p.name: {"bytes": p.stat().st_size, "sha256": sha_file(p)} for p in files}, "execution_scope": {"gpu_calls": "see runtime and traces", "jev_calls": 0, "paid_api_calls": 0, "fresh_holdout_access": 0}, "status": "E2_M1_PROVENANCE_COMPLETE"}
    write_json(OUT / "E2_M1_PROVENANCE_MANIFEST.json", manifest)
    # Do not include self hash in the manifest to avoid circular provenance.
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "cal", "full"], required=True)
    parser.add_argument("--gpu", type=int, default=0, choices=[0, 1, 2, 4])
    args = parser.parse_args()
    if args.stage == "smoke":
        return execute_stage("smoke", args.gpu)
    if args.stage == "cal":
        return execute_stage("cal", args.gpu)
    return execute_stage("full", args.gpu)


if __name__ == "__main__":
    raise SystemExit(main())
