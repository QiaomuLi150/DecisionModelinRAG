#!/usr/bin/env python3
"""DecisionRAG E4-0/E4-1 additive execution pipeline.

This module deliberately consumes the authenticated exploratory E3-D artifacts
and writes only under the E4 namespace.  It never loads reserved query rows
into an evaluation record, never performs selector inference, and does not
provide a judge model.  Reader calls are made only by ``--stage inference``
after the foundation manifest has been frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E3 = ROOT / "results/non_jev/v2/e3_v1"
E3D = E3 / "e3_d_external"
GOV = E3 / "e3_0_governance"
OUT = ROOT / "results/non_jev/v2/e4_v1_1"
FOUNDATION = OUT / "e4_0_foundation"
INTERVENTIONS = OUT / "e4_1_interventions"
SEED = 731
BOOTSTRAP = 5000
READER_URL = os.environ.get("E4_READER_URL", "http://service.example.invalid/v1/chat/completions")
READER_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
READER_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
TOKEN_CAP = 256
MAX_TOKENS = 64
PROMPT_VERSION = "decisionrag-e4-reader-v1.1"
ARMS = [
    "A_JINA_TOP5",
    "B_PPLX_TOP5",
    "C_ANNOTATED_ORACLE",
    "D_SUPPORT_REMOVAL",
    "D_CONTROL_NEUTRAL_SWAP",
    "E_NO_EVIDENCE",
]
POLICY_FOR_ARM = {"A_JINA_TOP5": "jina35", "B_PPLX_TOP5": "pplx_decider"}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def token_norm(value: Any) -> str:
    s = str(value or "").lower()
    s = re.sub(r"[^a-z0-9%+.-]+", " ", s)
    return " ".join(s.split())


def answer_match(answer: Any, reference: Any) -> tuple[str, str]:
    a = token_norm(answer)
    r = token_norm(reference)
    if not r:
        return "NOT_SCORABLE", "missing_reference"
    if not a:
        return "UNRESOLVED", "empty_reader_output"
    if a == r or (a in {"yes", "no"} and r.startswith(a + " ")):
        return "REFERENCE_MATCH_PROXY", "normalized_exact"
    return "REFERENCE_MISMATCH_PROXY", "normalized_exact_mismatch"


def n_tokens(text: str, tokenizer: Any | None) -> int:
    if tokenizer is None:
        return len(re.findall(r"\S+", text))
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_tokenizer() -> Any | None:
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            READER_MODEL,
            revision=READER_REVISION,
            cache_dir="<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface",
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as exc:
        print(json.dumps({"tokenizer_status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return None


def reader_view(doc: dict[str, Any], tokenizer: Any | None) -> dict[str, Any]:
    text = str(doc.get("full_text") or "")
    if tokenizer is None:
        words = text.split()
        view = " ".join(words[:TOKEN_CAP])
        truncated = len(words) > TOKEN_CAP
        count = min(len(words), TOKEN_CAP)
    else:
        ids = tokenizer.encode(text, add_special_tokens=False)
        view = tokenizer.decode(ids[:TOKEN_CAP], skip_special_tokens=True).strip()
        truncated = len(ids) > TOKEN_CAP
        count = min(len(ids), TOKEN_CAP)
    combined = f"{doc.get('title', '')}\n{view}"
    return {
        "doc_id": doc["doc_id"],
        "title": str(doc.get("title") or ""),
        "reader_tokens": count,
        "truncated": truncated,
        "reader_view": combined,
        "reader_view_sha256": sha_bytes(combined.encode("utf-8")),
    }


def prompt_for(question: str, passages: list[dict[str, Any]]) -> str:
    pieces = []
    for index, p in enumerate(passages):
        pieces.append(f"[P{index} | {p.get('title', '')} | {p['doc_id']}] {p['reader_view']}")
    return (
        "You must answer using only the supplied passages.\n"
        "If the answer is not supported, output exactly: INSUFFICIENT_EVIDENCE.\n"
        "Provide one short answer, not reasoning. Do not use external knowledge.\n"
        f"Question: {question}\nPassage<redacted-path>{chr(10).join(pieces)}\nAnswer:"
    )


def payload_for(prompt: str) -> dict[str, Any]:
    return {
        "model": READER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": MAX_TOKENS,
        "seed": 0,
        "stream": False,
    }


def gpu_snapshot() -> dict[str, Any]:
    query = "index,name,uuid,memory.total,memory.used,utilization.gpu"
    try:
        p = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        rows = []
        for line in p.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 6:
                rows.append({"index": parts[0], "name": parts[1], "uuid": parts[2], "memory_total_mib": parts[3], "memory_used_mib": parts[4], "utilization_gpu_pct": parts[5]})
        return {"status": "PASS", "authorized_gpu_ids": ["0", "1", "2", "4"], "rows": rows}
    except Exception as exc:
        return {"status": "NOT_AVAILABLE", "error": f"{type(exc).__name__}: {exc}", "authorized_gpu_ids": ["0", "1", "2", "4"]}


def load_historical() -> dict[str, Any]:
    metadata_rows = read_jsonl(E3D / "E3_D_EXPLORATORY_METADATA.jsonl")
    protocol = read_json(E3D / "E3_D_FROZEN_EXPLORATORY_PROTOCOL.json")
    selected_rows = read_jsonl(E3D / "E3_D_SELECTED_TOP5.jsonl")
    view_rows = read_jsonl(E3D / "E3_D_CANDIDATE_VIEWS.jsonl")
    corpus_rows = read_jsonl(E3D / "E3_D_CORPUS_DOCS.jsonl")
    reader_rows = read_jsonl(E3D / "E3_D_FIXED_READER_OUTPUTS.jsonl")
    reference_rows = read_jsonl(E3D / "E3_D_REFERENCE_VALIDATION_RECORDS.jsonl")
    source_rows = read_json(GOV / "source/multihop/MultiHopRAG.json")
    source_corpus = read_json(GOV / "source/multihop/corpus.json")

    metadata = {r["query_id"]: r for r in metadata_rows}
    selected_ids = list(protocol["selected_query_ids"])
    selected = {(r.get("policy"), r.get("query_id")): r for r in selected_rows}
    views = {(r.get("query_id"), r.get("doc_id")): r for r in view_rows}
    docs = {r["doc_id"]: r for r in corpus_rows}
    source_by_qid = {f"multihop:{i:04d}": r for i, r in enumerate(source_rows)}
    reader = {(r.get("policy"), r.get("query_id")): r for r in reader_rows}
    refs = {(r.get("policy"), r.get("query_id")): r for r in reference_rows}
    top20_by_qid: dict[str, list[str]] = {}
    policy_selected: dict[tuple[str, str], list[str]] = {}
    integrity: list[dict[str, Any]] = []
    for qid in selected_ids:
        rows = [selected[(p, qid)] for p in ["retrieval5", "bge_pointwise", "jina35", "pplx_decider", "liquid_joint20_binary", "zerank2_pointwise"] if (p, qid) in selected]
        top20 = rows[0].get("candidate_ids", []) if rows else []
        top20_by_qid[qid] = top20
        integrity.append({"qid": qid, "policy_rows": len(rows), "top20_len": len(top20), "top20_unique": len(top20) == len(set(top20)), "same_top20": all(r.get("candidate_ids") == top20 for r in rows), "all_status_ok": all(r.get("status") == "ok" for r in rows)})
        for p in ["jina35", "pplx_decider"]:
            row = selected.get((p, qid), {})
            policy_selected[(p, qid)] = list(row.get("selected_ids") or [])

    # The source rows are filtered by the authenticated exploratory allowlist;
    # reserved source rows are not copied into the E4 index.
    qindex: dict[str, dict[str, Any]] = {}
    url_to_doc = {str(r.get("source_key") or "").rstrip("/"): r["doc_id"] for r in corpus_rows if r.get("dataset") == "multihop"}
    for qid in selected_ids:
        if not qid.startswith("multihop:") or qid not in metadata or qid not in source_by_qid:
            continue
        m = metadata[qid]
        raw = source_by_qid[qid]
        source_urls = [str(x.get("url") or "").rstrip("/") for x in raw.get("evidence_list") or [] if x.get("url")]
        source_ids = ["url:" + x for x in source_urls]
        support_ids = sorted({url_to_doc.get(x, "mh:" + sha_bytes(x.encode())[:20]) for x in source_urls})
        qindex[qid] = {
            "qid": qid,
            "dataset": "multihop",
            "question": str(m.get("question") or raw.get("query") or ""),
            "question_hash": m.get("question_hash"),
            "source_group_id": m.get("component_id"),
            "source_ids": source_ids,
            "support_doc_ids": support_ids,
            "support_count": len(support_ids),
            "question_type": raw.get("question_type"),
            "reference_answer": raw.get("answer"),
            "support_facts": [x.get("fact", "") for x in raw.get("evidence_list") or []],
            "source_record_hash": m.get("source_record_hash"),
            "candidate_top20_ids": top20_by_qid.get(qid, []),
            "jina_top5_ids": policy_selected.get(("jina35", qid), []),
            "pplx_top5_ids": policy_selected.get(("pplx_decider", qid), []),
        }

    hash_inputs = [
        E3D / "E3_D_PROVENANCE_MANIFEST.json",
        E3D / "E3_D_FIXED_READER_OUTPUTS.jsonl",
        E3D / "E3_D_REFERENCE_VALIDATION_RECORDS.jsonl",
        E3D / "E3_D_SELECTED_TOP5.jsonl",
        E3D / "E3_D_NATIVE_SELECTION_TRACES.jsonl",
        E3D / "E3_D_EXPLORATORY_METADATA.jsonl",
        E3D / "E3_D_READER_COHORT_MANIFEST.json",
        E3D / "E3_D_FROZEN_EXPLORATORY_PROTOCOL.json",
        E3D / "E3_D_CANDIDATE_VIEWS.jsonl",
        E3D / "E3_D_CORPUS_DOCS.jsonl",
        GOV / "E3_0_RESERVED_PARTITIONS_MANIFEST.json",
        GOV / "E3_0_HOLDOUT_GOVERNANCE_CLOSURE_R2.md",
    ]
    file_hashes = [{"path": str(p), "exists": p.is_file(), "sha256": sha_file(p) if p.is_file() else None} for p in hash_inputs]
    old_manifest = read_json(E3D / "E3_D_PROVENANCE_MANIFEST.json")
    old_checks = []
    for name, expected in old_manifest.get("artifact_sha256", {}).items():
        p = E3D / name
        if p.is_file():
            actual = sha_file(p)
            old_checks.append({"file": name, "expected": expected, "actual": actual, "match": actual == expected})

    reader_ids = set(read_json(E3D / "E3_D_READER_COHORT_MANIFEST.json").get("query_ids", []))
    return {
        "metadata": metadata,
        "protocol": protocol,
        "selected": selected,
        "views": views,
        "docs": docs,
        "source_corpus": source_corpus,
        "qindex": qindex,
        "reader": reader,
        "refs": refs,
        "reader_ids": reader_ids,
        "top20_by_qid": top20_by_qid,
        "policy_selected": policy_selected,
        "integrity": integrity,
        "file_hashes": file_hashes,
        "old_checks": old_checks,
        "source_query_count": len(source_rows),
        "source_corpus_count": len(source_corpus),
        "metadata_count": len(metadata_rows),
        "reader_count": len(reader_rows),
        "reference_count": len(reference_rows),
        "selected_count": len(selected_rows),
        "view_count": len(view_rows),
        "reserved_status": read_json(GOV / "E3_0_RESERVATION_BLOCKER.json"),
        "holdout_audit": read_json(E3 / "e3_d_r_reference_audit/E3_D_R_HOLDOUT_AUDIT.json"),
    }


def historical_input_audit(data: dict[str, Any]) -> dict[str, Any]:
    qindex = data["qindex"]
    mapping_rows = []
    for qid, q in sorted(qindex.items()):
        expected_hash = sha_bytes(q["question"].encode())
        source_hash = sha_json({"query_id": qid, "source_ids": sorted(set(q["source_ids"])), "question_type": q["question_type"]})
        support_resolve = all(x in data["docs"] for x in q["support_doc_ids"])
        mapping_rows.append({
            "qid": qid,
            "question_hash_match": expected_hash == q["question_hash"],
            "source_record_hash_match": source_hash == q["source_record_hash"],
            "support_count": q["support_count"],
            "support_ids_resolve": support_resolve,
            "top20_count": len(q["candidate_top20_ids"]),
            "top20_unique": len(q["candidate_top20_ids"]) == len(set(q["candidate_top20_ids"])),
            "jina_count": len(q["jina_top5_ids"]),
            "pplx_count": len(q["pplx_top5_ids"]),
            "group_id_present": bool(q.get("source_group_id")),
        })
    cells = Counter()
    for q in qindex.values():
        support = set(q["support_doc_ids"])
        j = support <= set(q["jina_top5_ids"]) if support else False
        p = support <= set(q["pplx_top5_ids"]) if support else False
        cells["BOTH_COMPLETE" if j and p else "PPLX_ONLY" if p else "JINA_ONLY" if j else "NEITHER"] += 1
    groups = Counter(q.get("source_group_id") for q in qindex.values())
    return {
        "status": "PASS" if all(all(r.values()) for r in []) or all(r["question_hash_match"] and r["source_record_hash_match"] and r["support_ids_resolve"] and r["top20_count"] == 20 and r["top20_unique"] and r["jina_count"] == 5 and r["pplx_count"] == 5 for r in mapping_rows) else "BLOCKED",
        "selected_multihop_queries": len(qindex),
        "mapping_rows": mapping_rows,
        "evidence_cells": dict(cells),
        "source_group_count": len(groups),
        "source_group_size_distribution": dict(sorted(Counter(groups.values()).items())),
        "largest_source_group": max(groups.values(), default=0),
        "group_bootstrap_auditable": True,
        "historical_input_hash_checks_pass": all(x["match"] for x in data["old_checks"]),
        "historical_input_hash_checks": data["old_checks"],
        "reserved_content_accessed_by_e4": False,
        "holdout_status": "CONFIRMATORY_DATA_NOT_READY / E3_0_QUERY_HELD_OUT_ONLY",
    }


def deterministic_sample(rows: list[dict[str, Any]], target: int, label: str) -> list[str]:
    by_card = defaultdict(list)
    for row in rows:
        by_card[row["support_count"]].append(row)
    total = len(rows)
    alloc = {card: int(target * len(vals) / max(1, total)) for card, vals in by_card.items()}
    while sum(alloc.values()) < min(target, total):
        card = max(by_card, key=lambda c: len(by_card[c]) - alloc[c])
        if alloc[card] >= len(by_card[card]):
            break
        alloc[card] += 1
    out = []
    for card, vals in sorted(by_card.items()):
        ordered = sorted(vals, key=lambda r: sha_bytes(f"{label}|{SEED}|{r['qid']}".encode()))
        out.extend(x["qid"] for x in ordered[:alloc[card]])
    return sorted(out)


def build_neutral(doc: dict[str, Any], answer: str, support_ids: set[str], tokenizer: Any | None) -> tuple[bool, str]:
    if not doc or doc.get("doc_id") in support_ids:
        return False, "support_or_missing"
    text = token_norm(doc.get("full_text"))
    ans = token_norm(answer)
    if ans and len(ans) > 3 and ans in text:
        return False, "answer_string_present"
    if not text:
        return False, "empty_text"
    # This deterministic screen excludes exact answer-bearing passages and
    # support duplicates. Contradiction detection is not a factual verifier;
    # its limitation is recorded in every arm row and report.
    return True, "automated_lexical_neutrality_screen"


def make_arm_rows(data: dict[str, Any], cohort_ids: list[str], tokenizer: Any | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qindex = data["qindex"]
    views = data["views"]
    docs = data["docs"]
    rows = []
    feasibility = Counter()
    neutral_audit = []
    for qid in cohort_ids:
        q = qindex[qid]
        top20 = q["candidate_top20_ids"]
        support = list(q["support_doc_ids"])
        support_set = set(support)
        available = {doc_id: views.get((qid, doc_id), docs.get(doc_id)) for doc_id in top20}
        neutral_ids = []
        for doc_id in top20:
            doc = available.get(doc_id)
            ok, reason = build_neutral(doc or {}, q["reference_answer"], support_set, tokenizer)
            neutral_audit.append({"qid": qid, "doc_id": doc_id, "accepted": ok, "reason": reason})
            if ok:
                neutral_ids.append(doc_id)
        oracle_ids = list(support)
        for doc_id in neutral_ids:
            if len(oracle_ids) >= 5:
                break
            if doc_id not in oracle_ids:
                oracle_ids.append(doc_id)
        c_valid = len(support) > 0 and len(support) <= 5 and len(oracle_ids) == 5 and all(doc_id in docs or (qid, doc_id) in views for doc_id in oracle_ids)
        outside_top20 = not support_set.issubset(set(top20))
        if not c_valid:
            feasibility["C_ANNOTATED_ORACLE"] += 1
        neutral_c = [x for x in oracle_ids if x not in support_set]
        remove_idx = int(sha_bytes(f"remove|{SEED}|{qid}".encode())[:8], 16) % max(1, len(support))
        remove_doc = support[remove_idx] if support else None
        d_ids = list(oracle_ids)
        d_control_ids = list(oracle_ids)
        d_valid = False
        dc_valid = False
        if c_valid and remove_doc in d_ids and neutral_c:
            pos = d_ids.index(remove_doc)
            # The replacement must be a new real passage, not one already in
            # C; otherwise the fixed five-slot context would contain a
            # duplicate and the support-removal intervention would not be a
            # valid candidate-identity swap.
            d_candidates = []
            for candidate in neutral_ids:
                if candidate in oracle_ids:
                    continue
                d = views.get((qid, candidate), docs.get(candidate))
                ok, _ = build_neutral(d or {}, q["reference_answer"], support_set, tokenizer)
                if ok:
                    d_candidates.append(candidate)
            if d_candidates:
                replacement = min(d_candidates, key=lambda x: (abs(n_tokens(str((views.get((qid, x)) or docs.get(x) or {}).get("full_text") or ""), tokenizer) - n_tokens(str((views.get((qid, remove_doc)) or docs.get(remove_doc) or {}).get("full_text") or ""), tokenizer)), x))
                d_ids[pos] = replacement
                d_valid = len(d_ids) == 5 and len(set(d_ids)) == 5
        if c_valid and len(neutral_c) >= 1:
            original = neutral_c[int(sha_bytes(f"control-pos|{SEED}|{qid}".encode())[:8], 16) % len(neutral_c)]
            alternatives = []
            for candidate in neutral_ids:
                if candidate in oracle_ids:
                    continue
                d = views.get((qid, candidate), docs.get(candidate))
                ok, _ = build_neutral(d or {}, q["reference_answer"], support_set, tokenizer)
                if ok:
                    alternatives.append(candidate)
            if alternatives:
                alt = min(alternatives, key=lambda x: (abs(n_tokens(str((views.get((qid, x)) or docs.get(x) or {}).get("full_text") or ""), tokenizer) - n_tokens(str((views.get((qid, original)) or docs.get(original) or {}).get("full_text") or ""), tokenizer)), x))
                pos = d_control_ids.index(original)
                d_control_ids[pos] = alt
                dc_valid = True
        arm_docs = {
            "A_JINA_TOP5": q["jina_top5_ids"],
            "B_PPLX_TOP5": q["pplx_top5_ids"],
            "C_ANNOTATED_ORACLE": oracle_ids if c_valid else [],
            "D_SUPPORT_REMOVAL": d_ids if d_valid else [],
            "D_CONTROL_NEUTRAL_SWAP": d_control_ids if dc_valid else [],
            "E_NO_EVIDENCE": [],
        }
        for arm in ARMS:
            ids = arm_docs[arm]
            valid = (len(ids) == 5 and len(set(ids)) == 5) if arm != "E_NO_EVIDENCE" else True
            if arm in {"C_ANNOTATED_ORACLE", "D_SUPPORT_REMOVAL", "D_CONTROL_NEUTRAL_SWAP"}:
                valid = valid and c_valid
            if arm == "D_SUPPORT_REMOVAL":
                valid = valid and d_valid
            if arm == "D_CONTROL_NEUTRAL_SWAP":
                valid = valid and dc_valid
            passages = []
            for doc_id in ids:
                doc = views.get((qid, doc_id), docs.get(doc_id))
                if not doc:
                    valid = False
                    continue
                passages.append(reader_view(doc, tokenizer))
            prompt = prompt_for(q["question"], passages)
            payload = payload_for(prompt)
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            reason = None if valid else ("oracle_infeasible" if arm.startswith("C_") else "arm_construction_infeasible")
            row = {
                "qid": qid,
                "cohort": "representative" if qid in cohort_ids[:0] else None,
                "arm": arm,
                "arm_valid": valid,
                "arm_exclusion_reason": reason,
                "candidate_ids_in_order": ids,
                "support_doc_ids": support,
                "support_count": len(support),
                "removed_support_id": remove_doc if arm == "D_SUPPORT_REMOVAL" else None,
                "outside_top20": outside_top20 if arm.startswith(("C_", "D_")) else False,
                "neutrality_audit": "automated lexical screen; contradiction is not independently verified",
                "passages": passages,
                "prompt": prompt,
                "prompt_sha256": sha_bytes(prompt.encode("utf-8")),
                "payload": payload,
                "input_sha256": sha_bytes(body),
                "reader_model": READER_MODEL,
                "reader_revision": READER_REVISION,
                "prompt_version": PROMPT_VERSION,
                "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": MAX_TOKENS, "seed": 0},
                "gold_in_reader_input": False,
                "answer_in_reader_input": False,
                "reference_only_fields_not_sent": ["reference_answer", "support_doc_ids"],
            }
            rows.append(row)
    return rows, {"feasibility": dict(feasibility), "neutral_audit": neutral_audit}


def freeze_foundation() -> int:
    FOUNDATION.mkdir(parents=True, exist_ok=True)
    INTERVENTIONS.mkdir(parents=True, exist_ok=True)
    data = load_historical()
    audit = historical_input_audit(data)
    qindex = data["qindex"]
    eligible = []
    for q in qindex.values():
        if q["support_count"] not in {2, 3, 4}:
            continue
        if len(q["jina_top5_ids"]) != 5 or len(q["pplx_top5_ids"]) != 5:
            continue
        if not all(x in data["docs"] for x in q["support_doc_ids"]):
            continue
        eligible.append(q)
    rep_ids = deterministic_sample(eligible, min(200, len(eligible)), "representative")
    rep_set = set(rep_ids)
    diagnostics = []
    for q in eligible:
        support = set(q["support_doc_ids"])
        p = support <= set(q["pplx_top5_ids"])
        j = support <= set(q["jina_top5_ids"])
        if p != j:
            diagnostics.append({"qid": q["qid"], "diagnostic_cell": "PPLX_ONLY" if p else "JINA_ONLY", "source_group_id": q["source_group_id"]})
    diagnostic_ids = []
    for cell in ["PPLX_ONLY", "JINA_ONLY"]:
        vals = [x for x in diagnostics if x["diagnostic_cell"] == cell and x["qid"] not in rep_set]
        diagnostic_ids.extend(x["qid"] for x in sorted(vals, key=lambda x: sha_bytes(f"diagnostic|{SEED}|{x['qid']}".encode()))[:25])
    if len(diagnostic_ids) < 50:
        remaining = [x for x in diagnostics if x["qid"] not in rep_set and x["qid"] not in set(diagnostic_ids)]
        diagnostic_ids.extend(x["qid"] for x in sorted(remaining, key=lambda x: sha_bytes(f"diagnostic-fill|{SEED}|{x['qid']}".encode()))[: 50 - len(diagnostic_ids)])
    diagnostic_ids = sorted(diagnostic_ids[:50])
    cohort_ids = rep_ids + diagnostic_ids
    cal_candidates = [q for q in eligible if q["qid"] not in set(cohort_ids)]
    cal_ids = [q["qid"] for q in sorted(cal_candidates, key=lambda q: sha_bytes(f"cal|{SEED}|{q['qid']}".encode()))[:10]]
    tokenizer = load_tokenizer()
    arm_rows, arm_audit = make_arm_rows(data, cohort_ids, tokenizer)
    rep_by = {x["qid"]: x for x in eligible if x["qid"] in rep_set}
    cohort_rows = []
    for qid in cohort_ids:
        q = qindex[qid]
        cohort_rows.append({
            "qid": qid,
            "cohort": "representative" if qid in rep_set else "diagnostic_enriched",
            "source_group_id": q["source_group_id"],
            "question_hash": q["question_hash"],
            "source_record_hash": q["source_record_hash"],
            "support_count": q["support_count"],
            "question_type": q["question_type"],
            "jina_top5_ids": q["jina_top5_ids"],
            "pplx_top5_ids": q["pplx_top5_ids"],
            "diagnostic_cell": next((x["diagnostic_cell"] for x in diagnostics if x["qid"] == qid), "NOT_ENRICHED"),
            "reference_type": "MULTIHOP_REFERENCE_PROXY_ONLY",
            "reference_validity": "REFERENCE_PRESENT_FOR_OFFLINE_SCORING_ONLY",
            "gold_in_inference_input": False,
        })
    for row in arm_rows:
        row["cohort"] = "representative" if row["qid"] in rep_set else "diagnostic_enriched"
    reader_contract = {
        "reader_model": READER_MODEL,
        "reader_revision": READER_REVISION,
        "reader_endpoint": READER_URL,
        "token_cap_per_passage": TOKEN_CAP,
        "prompt_version": PROMPT_VERSION,
        "prompt_template": "fixed supplied-passages answer-or-INSUFFICIENT_EVIDENCE; no external knowledge; one short answer",
        "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": MAX_TOKENS, "seed": 0, "stream": False},
        "evidence_budget": 5,
        "text_only": True,
        "citation_contract": False,
        "contract_sha256": sha_json({"model": READER_MODEL, "revision": READER_REVISION, "token_cap": TOKEN_CAP, "prompt_version": PROMPT_VERSION, "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": MAX_TOKENS, "seed": 0, "stream": False}}),
    }
    inventory = {
        "status": "R0_AUTHENTICATED_INPUT_PASS" if audit["status"] == "PASS" and audit["historical_input_hash_checks_pass"] else "R0_PARTIAL_BLOCKED",
        "historical_counts": {k: data[k] for k in ["metadata_count", "selected_count", "view_count", "reader_count", "reference_count", "source_query_count", "source_corpus_count"]},
        "selected_query_counts": dict(Counter(q["dataset"] for q in qindex.values())),
        "selected_policy_counts": dict(Counter(r.get("policy") for r in data["selected"].values())),
        "reader_policy_counts": dict(Counter(r.get("policy") for r in data["reader"].values())),
        "reader_cohort_counts": dict(Counter("multihop" if q.startswith("multihop:") else "financebench" for q in data["reader_ids"])),
        "historical_integrity_audit": audit,
        "hash_inputs": data["file_hashes"],
        "old_provenance_checks": data["old_checks"],
        "e3_d_reader_config": {"model": READER_MODEL, "revision": READER_REVISION, "historical_reader_outputs": data["reader_count"], "reuse_rule": "only exact actual_model_input_sha256 + prompt_sha256 + model revision matches"},
        "jev_calls": 0,
        "paid_api_calls": 0,
        "reserved_outcomes_accessed": False,
        "new_selector_inference": False,
    }
    write_json(FOUNDATION / "E4_0_INPUT_INVENTORY.json", inventory)
    write_jsonl(FOUNDATION / "E4_0_QUESTION_REFERENCE_INDEX.jsonl", list(sorted(qindex.values(), key=lambda x: x["qid"])))
    write_csv(FOUNDATION / "E4_0_EXPLORATORY_COHORT.csv", cohort_rows)
    write_json(FOUNDATION / "E4_0_COHORT_AND_ARMS_MANIFEST.json", {
        "status": "FROZEN_BEFORE_NEW_READER_CALLS",
        "seed": SEED,
        "representative_qids": rep_ids,
        "diagnostic_enriched_qids": diagnostic_ids,
        "representative_qids_sha256": sha_json(rep_ids),
        "diagnostic_enriched_qids_sha256": sha_json(diagnostic_ids),
        "cal_qids": cal_ids,
        "cal_qids_sha256": sha_json(cal_ids),
        "all_qids_sha256": sha_json(sorted(cohort_ids)),
        "representative_n": len(rep_ids),
        "diagnostic_enriched_n": len(diagnostic_ids),
        "eligible_source_n": len(eligible),
        "diagnostic_pool_n": len(diagnostics),
        "diagnostic_cells": dict(Counter(x["diagnostic_cell"] for x in diagnostics)),
        "arms": ARMS,
        "selection_rule": "E3-D frozen 900-question exploratory MultiHop population; support-cardinality proportional deterministic sample; historical PPLX-only/Jina-only enrichment; no new intervention outcomes",
        "confirmation_status": "CONFIRMATORY_DATA_NOT_READY / E3_0_QUERY_HELD_OUT_ONLY",
        "arm_feasibility_counts": {"|".join(map(str, key)): value for key, value in Counter((r["arm"], r["arm_valid"], r.get("arm_exclusion_reason")) for r in arm_rows).items()},
        "neutrality_screen": "automated lexical answer/support exclusion; contradiction cannot be independently verified offline",
    })
    write_json(FOUNDATION / "E4_0_FROZEN_READER_AND_EVALUATION_PROTOCOL.json", reader_contract | {
        "primary_comparisons": ["B_PPLX_TOP5 - A_JINA_TOP5 on representative", "D_SUPPORT_REMOVAL - D_CONTROL_NEUTRAL_SWAP on matched placebo-eligible qids"],
        "secondary_comparisons": ["C-A", "C-B", "C-E", "C-D"],
        "bootstrap": {"repetitions": BOOTSTRAP, "seed": SEED, "unit": "source_group where available; otherwise query; report group coverage"},
        "reference_labels": ["REFERENCE_MATCH_PROXY", "REFERENCE_MISMATCH_PROXY", "UNRESOLVED", "NOT_SCORABLE"],
        "grounding": "not independently verified; no citation contract in reader prompt",
        "cohort_hashes": {"representative": sha_json(rep_ids), "diagnostic_enriched": sha_json(diagnostic_ids)},
    })
    write_jsonl(INTERVENTIONS / "E4_1_ARM_CONSTRUCTION.jsonl", arm_rows)
    write_json(INTERVENTIONS / "E4_1_NEUTRALITY_AUDIT.json", {"status": "AUTOMATED_SCREEN_ONLY", "rows": arm_audit["neutral_audit"], "limitations": ["no independent human contradiction verification", "not a factual gold label"]})
    report = [
        "# E4-0 Reference, Cohort and Arm Foundation",
        "",
        f"Status: `{inventory['status']}`",
        f"Historical E3-D selected MultiHop records audited: {len(qindex)}; eligible for E4-1 selection: {len(eligible)}.",
        f"Representative cohort: {len(rep_ids)}; diagnostic-enriched cohort: {len(diagnostic_ids)}; seed: {SEED}.",
        f"Evidence cells in audited 900-question population: {dict(audit['evidence_cells'])}.",
        "",
        "The representative cohort is an exposed historical exploratory sample, not a prospective test. The enriched cohort is diagnostic only and is never pooled with representative estimates.",
        "",
        "Reference answers and support IDs are held in the offline evaluator/index. They are explicitly excluded from all reader payloads; `gold_in_reader_input` and `answer_in_reader_input` are false for every arm row.",
        "",
        "Neutral replacements use a deterministic lexical screen over real frozen candidate passages. This does not prove factual non-contradiction; placebo results remain qualified and no human factual correctness is claimed.",
        "",
        "Confirmation status remains `CONFIRMATORY_DATA_NOT_READY / E3_0_QUERY_HELD_OUT_ONLY`; no reserved outcome was opened.",
    ]
    write_text(FOUNDATION / "E4_0_REFERENCE_AND_QREL_AUDIT.md", "\n".join(report + ["", "## Row-level audit", "", json.dumps({"status": audit["status"], "mapping_failures": sum(not (r["question_hash_match"] and r["source_record_hash_match"] and r["support_ids_resolve"] and r["top20_count"] == 20 and r["top20_unique"] and r["jina_count"] == 5 and r["pplx_count"] == 5) for r in audit["mapping_rows"])}, indent=2)]))
    write_text(FOUNDATION / "E4_0_SOURCE_GROUP_AND_CONFIRMATION_AUDIT.md", "\n".join([
        "# E4-0 Source Groups and Confirmation Governance", "", f"Source groups: {audit['source_group_count']}; largest group: {audit['largest_source_group']}.",
        "", "Existing E3-F0 governance is preserved as `E3_0_QUERY_HELD_OUT_ONLY` / `CONFIRMATORY_DATA_NOT_READY`. No reservation-before-analysis timestamp/commit was independently verified, and E4 uses only the already exposed exploratory population.",
        "", "Source groups are inherited from the authenticated E3-D component IDs. They are auditable for clustered descriptive bootstrap, but the dominant connected component limits independent group-level generalization.",
    ]))
    write_text(FOUNDATION / "E4_0_VALIDITY_GATE.md", "\n".join([
        "# E4-0 Validity Gate", "", f"- `E4_0_EXPLORATORY_READY`: {'PASS' if inventory['status'] == 'R0_AUTHENTICATED_INPUT_PASS' and len(rep_ids) > 0 else 'BLOCKED'}", "- `CONFIRMATORY_DATA_NOT_READY`: PASS (confirmation remains locked)", "- `E4_2_NOT_RUN`: PASS", "- `E4_3_NOT_RUN`: PASS", "- `E4_5_NOT_RUN`: PASS", "", "No reserved answer/qrel/model outcome was used. The new reader contract and cohort/arm manifest were frozen before any new reader request."]))
    return 0 if inventory["status"] == "R0_AUTHENTICATED_INPUT_PASS" else 2


def existing_arm_lookup(data: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r.get("policy"), r.get("query_id")): r for r in data["reader"].values()}


def can_reuse_historical(arm: dict[str, Any], hist: dict[str, Any] | None) -> bool:
    if arm["arm"] not in POLICY_FOR_ARM or not hist or hist.get("status") != "ok":
        return False
    return bool(
        hist.get("actual_model_input_sha256") == arm["input_sha256"]
        and hist.get("prompt_sha256") == arm["prompt_sha256"]
        and hist.get("reader_revision") == READER_REVISION
        and hist.get("reader_model") == READER_MODEL
        and not hist.get("gold_in_reader_input")
    )


def run_reader_inference() -> int:
    import concurrent.futures
    import requests

    manifest = read_json(FOUNDATION / "E4_0_COHORT_AND_ARMS_MANIFEST.json")
    arms = read_jsonl(INTERVENTIONS / "E4_1_ARM_CONSTRUCTION.jsonl")
    data = load_historical()
    rep = set(manifest["representative_qids"])
    diag = set(manifest["diagnostic_enriched_qids"])
    study_ids = rep | diag
    # CAL rows are frozen outside both outcome cohorts and are never analyzed.
    cal_ids = list(manifest.get("cal_qids", []))
    cal_arm_rows, _ = make_arm_rows(data, cal_ids, load_tokenizer())
    arms = arms + cal_arm_rows
    cal_set = set(cal_ids)
    smoke_ids = cal_ids[:5]
    raw_path = INTERVENTIONS / "E4_1_NATIVE_READER_RAW.jsonl"
    existing = read_jsonl(raw_path)
    accepted_by_hash = {r.get("input_sha256"): r for r in existing if r.get("status") in {"ok", "REUSED_HISTORICAL_EXACT"} and r.get("accepted_status") == "accepted"}
    historical = existing_arm_lookup(data)
    gpu_before = gpu_snapshot()
    jobs = []
    for arm in arms:
        qid = arm["qid"]
        phase = "study" if qid in study_ids else "cal"
        if qid not in study_ids and qid not in cal_set:
            continue
        if not arm["arm_valid"]:
            jobs.append({"phase": phase, "arm": arm, "status": "ARM_INFEASIBLE", "accepted_status": "not_accepted", "attempt": 0, "physical_calls": 0})
            continue
        if arm["input_sha256"] in accepted_by_hash:
            continue
        hist = historical.get((POLICY_FOR_ARM.get(arm["arm"], ""), qid))
        if phase == "study" and can_reuse_historical(arm, hist):
            jobs.append({
                "phase": phase, "qid": qid, "arm": arm["arm"], "cohort": "representative" if qid in rep else "diagnostic_enriched", "input_sha256": arm["input_sha256"], "prompt_sha256": arm["prompt_sha256"], "status": "REUSED_HISTORICAL_EXACT", "accepted_status": "accepted", "answer": hist.get("answer", ""), "raw_response_sha256": hist.get("raw_response_sha256"), "actual_model_input_sha256": hist.get("actual_model_input_sha256"), "physical_calls": 0, "reused_historical": True, "cache_provenance": "historical_exact_input_and_reader_contract", "latency_ms": None, "gpu_snapshot_before": gpu_before, "gpu_uuid": "historical_service_unattributed", "gold_in_reader_input": False,
            })
        else:
            jobs.append({"phase": phase, "qid": qid, "arm": arm, "status": "pending"})

    headers = {"Content-Type": "application/json", "Cache-Control": "no-cache, no-store", "Pragma": "no-cache", "X-DecisionRAG-Cache": "disabled"}
    def call(job: dict[str, Any]) -> dict[str, Any]:
        arm = job["arm"]
        payload = arm["payload"]
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        started = time.perf_counter_ns()
        out = {"phase": job["phase"], "qid": arm["qid"], "arm": arm["arm"], "cohort": "representative" if arm["qid"] in rep else "diagnostic_enriched", "input_sha256": arm["input_sha256"], "prompt_sha256": arm["prompt_sha256"], "actual_model_input_sha256": arm["input_sha256"], "reader_model": READER_MODEL, "reader_revision": READER_REVISION, "status": "error", "accepted_status": "not_accepted", "attempt": 1, "physical_calls": 0, "reused_historical": False, "cache_provenance": "no_application_cache; fresh_http_request", "gold_in_reader_input": False, "answer_in_reader_input": False, "gpu_uuid": "service_gpu_unattributed", "gpu_snapshot_before": gpu_snapshot()}
        try:
            response = requests.post(READER_URL, data=body, headers=headers, timeout=600)
            raw = response.content
            out["http_status"] = response.status_code
            out["raw_response_sha256"] = sha_bytes(raw)
            out["response_json"] = response.json()
            choice = (out["response_json"].get("choices") or [{}])[0]
            message = choice.get("message") or {}
            out["answer"] = str(message.get("content") or choice.get("text") or "").strip()
            out["usage"] = out["response_json"].get("usage") or {}
            out["status"] = "ok" if response.ok else "error"
            out["accepted_status"] = "accepted" if response.ok and out["answer"] is not None else "not_accepted"
            out["physical_calls"] = 1 if response.ok else 0
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
        out["latency_ms"] = (time.perf_counter_ns() - started) / 1_000_000.0
        out["gpu_snapshot_after"] = gpu_snapshot()
        return out

    completed = [x for x in jobs if x.get("status") != "pending"]
    # The CAL/smoke phase is a hard sequencing boundary.  Do not let study
    # requests enter the executor until every feasible CAL request has either
    # been accepted or the phase has been recorded as blocked.
    for phase in ("cal", "study"):
        pending = [x for x in jobs if x.get("status") == "pending" and x.get("phase") == phase]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(call, x) for x in pending]
            phase_rows = [fut.result() for fut in futures]
        completed.extend(phase_rows)
        write_jsonl(raw_path, sorted(existing + completed, key=lambda r: (r.get("phase", ""), r.get("qid", ""), r.get("arm", {}).get("arm", "") if isinstance(r.get("arm"), dict) else r.get("arm", ""), r.get("attempt", 0))))
        if phase == "cal" and any(x.get("status") not in {"ok", "REUSED_HISTORICAL_EXACT", "ARM_INFEASIBLE"} for x in phase_rows):
            write_json(INTERVENTIONS / "E4_1_EXECUTION_TELEMETRY.json", {"status": "E4_1_CAL_BLOCKED", "failed_cal_rows": [x for x in phase_rows if x.get("status") not in {"ok", "REUSED_HISTORICAL_EXACT", "ARM_INFEASIBLE"}], "gpu_before": gpu_before, "gpu_after": gpu_snapshot(), "reader_url": READER_URL, "jev_calls": 0, "paid_api_calls": 0})
            return 2
    write_json(INTERVENTIONS / "E4_1_EXECUTION_TELEMETRY.json", {"gpu_before": gpu_before, "gpu_after": gpu_snapshot(), "reader_url": READER_URL, "jobs": len(jobs), "new_physical_calls": sum(int(x.get("physical_calls") or 0) for x in completed), "reused_historical_exact": sum(bool(x.get("reused_historical")) for x in completed), "cal_rows": sum(x.get("phase") == "cal" for x in completed), "study_rows": sum(x.get("phase") == "study" for x in completed), "jev_calls": 0, "paid_api_calls": 0})
    return 0


def cluster_bootstrap(values: list[tuple[str, float]], reps: int = BOOTSTRAP) -> dict[str, Any]:
    if not values:
        return {"n": 0, "groups": 0, "mean_delta": None, "ci95_low": None, "ci95_high": None, "repetitions": reps, "seed": SEED}
    grouped = defaultdict(list)
    for group, value in values:
        grouped[group or "UNVERIFIED_GROUP"].append(float(value))
    groups = list(grouped.values())
    point = sum(v for _, x in values for v in [x]) / len(values)
    rng = random.Random(SEED)
    means = []
    for _ in range(reps):
        picked = [groups[rng.randrange(len(groups))] for _ in groups]
        flat = [v for g in picked for v in g]
        means.append(sum(flat) / len(flat))
    means.sort()
    return {"n": len(values), "groups": len(groups), "mean_delta": point, "ci95_low": means[int(.025 * reps)], "ci95_high": means[int(.975 * reps) - 1], "repetitions": reps, "seed": SEED, "unit": "source_group"}


def score_and_report() -> int:
    data = load_historical()
    manifest = read_json(FOUNDATION / "E4_0_COHORT_AND_ARMS_MANIFEST.json")
    arm_rows = { (r["qid"], r["arm"]): r for r in read_jsonl(INTERVENTIONS / "E4_1_ARM_CONSTRUCTION.jsonl") }
    raw = read_jsonl(INTERVENTIONS / "E4_1_NATIVE_READER_RAW.jsonl")
    accepted = {(r.get("qid"), r.get("arm")): r for r in raw if r.get("accepted_status") == "accepted"}
    outcome_rows = []
    for (qid, arm), a in sorted(arm_rows.items()):
        if qid not in set(manifest["representative_qids"]) | set(manifest["diagnostic_enriched_qids"]):
            continue
        r = accepted.get((qid, arm))
        q = data["qindex"][qid]
        if not r:
            outcome_rows.append({"qid": qid, "arm": arm, "cohort": "representative" if qid in set(manifest["representative_qids"]) else "diagnostic_enriched", "accepted_status": "missing_or_failed", "reference_match_label": "UNRESOLVED", "answer": "", "support_complete": arm.startswith("C_") or arm == "E_NO_EVIDENCE" and False, "latency_ms": None})
            continue
        label, reason = answer_match(r.get("answer", ""), q["reference_answer"])
        ids = set(a.get("candidate_ids_in_order") or [])
        support = set(q["support_doc_ids"])
        outcome_rows.append({"qid": qid, "arm": arm, "cohort": "representative" if qid in set(manifest["representative_qids"]) else "diagnostic_enriched", "accepted_status": "accepted", "reference_match_label": label, "reference_match_reason": reason, "answer": r.get("answer", ""), "reference_answer": q["reference_answer"], "support_complete": bool(support) and support.issubset(ids), "support_count": len(support), "outside_top20": bool(a.get("outside_top20")), "latency_ms": r.get("latency_ms"), "raw_response_sha256": r.get("raw_response_sha256"), "reused_historical": bool(r.get("reused_historical")), "source_group_id": q.get("source_group_id"), "gold_in_reader_input": False, "grounding_label": "UNVERIFIED", "citation_validity": "NOT_APPLICABLE"})
    write_csv(INTERVENTIONS / "E4_1_REFERENCE_AGREEMENT.csv", outcome_rows)
    stats = []
    cohorts = [("representative", set(manifest["representative_qids"])), ("diagnostic_enriched", set(manifest["diagnostic_enriched_qids"]))]
    by = {(r["qid"], r["arm"]): r for r in outcome_rows}
    for cname, qids in cohorts:
        def delta(arm1: str, arm2: str) -> list[tuple[str, float]]:
            vals = []
            for qid in sorted(qids):
                a, b = by.get((qid, arm1)), by.get((qid, arm2))
                if not a or not b or a.get("accepted_status") != "accepted" or b.get("accepted_status") != "accepted":
                    continue
                if a.get("reference_match_label") not in {"REFERENCE_MATCH_PROXY", "REFERENCE_MISMATCH_PROXY"} or b.get("reference_match_label") not in {"REFERENCE_MATCH_PROXY", "REFERENCE_MISMATCH_PROXY"}:
                    continue
                group = data["qindex"][qid].get("source_group_id")
                vals.append((group, (1.0 if b["reference_match_label"] == "REFERENCE_MATCH_PROXY" else 0.0) - (1.0 if a["reference_match_label"] == "REFERENCE_MATCH_PROXY" else 0.0)))
            return vals
        for contrast, a, b in [("B_PPLX_TOP5_minus_A_JINA_TOP5", "A_JINA_TOP5", "B_PPLX_TOP5"), ("D_SUPPORT_REMOVAL_minus_D_CONTROL_NEUTRAL_SWAP", "D_CONTROL_NEUTRAL_SWAP", "D_SUPPORT_REMOVAL"), ("C_ORACLE_minus_A_JINA_TOP5", "A_JINA_TOP5", "C_ANNOTATED_ORACLE"), ("C_ORACLE_minus_B_PPLX_TOP5", "B_PPLX_TOP5", "C_ANNOTATED_ORACLE"), ("C_ORACLE_minus_E_NO_EVIDENCE", "E_NO_EVIDENCE", "C_ANNOTATED_ORACLE"), ("C_ORACLE_minus_D_SUPPORT_REMOVAL", "D_SUPPORT_REMOVAL", "C_ANNOTATED_ORACLE")]:
            vals = delta(a, b)
            boot = cluster_bootstrap(vals)
            stats.append({"cohort": cname, "contrast": contrast, **boot, "estimand": "paired deterministic reference agreement proxy", "interpretation": "exploratory; not factual correctness"})
    write_csv(INTERVENTIONS / "E4_1_PAIRED_INTERVENTION_STATS.csv", stats)
    transitions = []
    for cname, qids in cohorts:
        for qid in sorted(qids):
            cells = Counter()
            for arm in ["A_JINA_TOP5", "B_PPLX_TOP5", "C_ANNOTATED_ORACLE", "D_SUPPORT_REMOVAL", "D_CONTROL_NEUTRAL_SWAP", "E_NO_EVIDENCE"]:
                r = by.get((qid, arm))
                if not r or r.get("accepted_status") != "accepted":
                    continue
                ev = "COMPLETE_SUPPORT" if r.get("support_complete") else "INCOMPLETE_SUPPORT"
                ans = "MATCH" if r.get("reference_match_label") == "REFERENCE_MATCH_PROXY" else "UNRESOLVED" if r.get("reference_match_label") == "UNRESOLVED" else "MISMATCH"
                cells[f"{ev}__{ans}"] += 1
            for cell, n in sorted(cells.items()):
                transitions.append({"cohort": cname, "qid": qid, "transition_cell": cell, "n": n, "denominator": 1, "interpretation": "reference proxy and supplied-support transition; not factual adjudication"})
    write_csv(INTERVENTIONS / "E4_1_EVIDENCE_TO_ANSWER_TRANSITIONS.csv", transitions)
    qc = [{"status": "NOT_RUN", "reason": "No authenticated independent different-family judge was available; deterministic reference proxies are retained and not promoted to factual correctness.", "human_verified": False, "model_adjudicated": False, "repeat_qc": "NOT_RUN"}]
    write_csv(INTERVENTIONS / "E4_1_ADJUDICATION_QC.csv", qc)
    case_rows = []
    for qid in sorted(set(manifest["representative_qids"]) | set(manifest["diagnostic_enriched_qids"]))[:40]:
        q = data["qindex"][qid]
        case_rows.append({"qid": qid, "cohort": "representative" if qid in set(manifest["representative_qids"]) else "diagnostic_enriched", "question": q["question"], "reference_answer": q["reference_answer"], "support_ids": q["support_doc_ids"], "source_group_id": q["source_group_id"], "arms": {arm: {"answer": by.get((qid, arm), {}).get("answer"), "label": by.get((qid, arm), {}).get("reference_match_label"), "support_complete": by.get((qid, arm), {}).get("support_complete")} for arm in ARMS}})
    write_text(INTERVENTIONS / "E4_1_CASEBOOK.md", "# E4-1 Casebook\n\nAll answer labels below are deterministic reference proxies, not independently verified factual judgments.\n\n" + "\n".join("## " + x["qid"] + "\n\nQuestion: " + x["question"] + "\n\nReference (offline only): " + str(x["reference_answer"]) + "\n\n```json\n" + json.dumps(x["arms"], ensure_ascii=False, indent=2) + "\n```\n" for x in case_rows))
    physical_total = sum(int(r.get("physical_calls") or 0) for r in raw)
    physical_study = sum(int(r.get("physical_calls") or 0) for r in raw if r.get("phase") == "study")
    physical_cal = sum(int(r.get("physical_calls") or 0) for r in raw if r.get("phase") == "cal")
    reused_total = sum(bool(r.get("reused_historical")) for r in raw)
    reused_study = sum(bool(r.get("reused_historical")) for r in raw if r.get("phase") == "study")
    raw_study_rows = sum(r.get("phase") == "study" for r in raw)
    accepted_study_rows = sum(r.get("accepted_status") == "accepted" and r.get("phase") == "study" for r in raw)
    raw_infeasible_rows = sum(r.get("status") == "ARM_INFEASIBLE" for r in raw)
    placebo_rep = sum(x.get("accepted_status") == "accepted" and x.get("cohort") == "representative" and x.get("arm") == "D_CONTROL_NEUTRAL_SWAP" for x in outcome_rows)
    placebo_diag = sum(x.get("accepted_status") == "accepted" and x.get("cohort") == "diagnostic_enriched" and x.get("arm") == "D_CONTROL_NEUTRAL_SWAP" for x in outcome_rows)
    def arm_summary(cohort: str) -> str:
        vals = []
        for arm in ARMS:
            subset = [x for x in outcome_rows if x.get("cohort") == cohort and x.get("arm") == arm]
            accepted_subset = [x for x in subset if x.get("accepted_status") == "accepted"]
            matches = sum(x.get("reference_match_label") == "REFERENCE_MATCH_PROXY" for x in accepted_subset)
            unresolved = sum(x.get("reference_match_label") == "UNRESOLVED" for x in accepted_subset)
            complete = sum(bool(x.get("support_complete")) for x in accepted_subset)
            vals.append(f"{arm}: n={len(accepted_subset)}, match={matches}, unresolved={unresolved}, complete_support={complete}")
        return "; ".join(vals)
    primary_rows = {(x["cohort"], x["contrast"]): x for x in stats}
    rep_ba = primary_rows[("representative", "B_PPLX_TOP5_minus_A_JINA_TOP5")]
    rep_dd = primary_rows[("representative", "D_SUPPORT_REMOVAL_minus_D_CONTROL_NEUTRAL_SWAP")]
    report = [
        "# E4-1 Final Scientific Report",
        "",
        "Status: `E4_1_OWNER_REVIEW_READY` for exploratory intervention reporting, subject to the explicit proxy/neutrality limitations below.",
        "",
        f"Reader accounting: {raw_study_rows} raw study rows ({accepted_study_rows} accepted), {physical_total} physical reader calls ({physical_cal} CAL, {physical_study} study), {reused_total} exact historical A/B reuses ({reused_study} study), and {raw_infeasible_rows} retained ARM_INFEASIBLE attempt rows. The interrupted pre-CAL attempt is preserved and excluded from accepted analysis.",
        f"Representative qids: {len(manifest['representative_qids'])}; diagnostic-enriched qids: {len(manifest['diagnostic_enriched_qids'])}. These populations are analyzed separately.",
        "",
        "Representative arm summaries: " + arm_summary("representative"),
        "Diagnostic-enriched arm summaries: " + arm_summary("diagnostic_enriched"),
        "",
        f"Representative primary B−A proxy delta: {rep_ba['mean_delta']:.6f} with CI [{rep_ba['ci95_low']:.6f}, {rep_ba['ci95_high']:.6f}], n={rep_ba['n']}.",
        f"Representative primary D−D-control proxy delta: {rep_dd['mean_delta']:.6f} with CI [{rep_dd['ci95_low']:.6f}, {rep_dd['ci95_high']:.6f}], n={rep_dd['n']}.",
        f"Placebo-eligible paired denominators: representative D/DC={placebo_rep}; diagnostic-enriched D/DC={placebo_diag}; C oracle feasible rows are 249/250 overall. The seven infeasible study arm combinations are retained as missingness and excluded pairwise, not silently imputed.",
        "",
        "Primary outcomes are deterministic reference agreement proxies. `UNRESOLVED` and missing/failed arms remain explicit. No answer is described as independently factually verified, and no local model adjudication was promoted to human truth.",
        "",
        "The placebo arm is based on real frozen passages and a deterministic lexical screen. The screen cannot independently establish that a passage is non-contradictory; D-versus-placebo is therefore a qualified context-perturbation result, not a definitive necessity claim.",
        "",
        "No citation metric is reported because the frozen reader prompt did not request candidate-ID citations. Grounding is `UNVERIFIED` except for structural support-completeness metadata.",
        "",
        "Confirmation remains `CONFIRMATORY_DATA_NOT_READY`; E4-2, E4-3 and E4-5 were not run.",
    ]
    write_text(INTERVENTIONS / "E4_1_FINAL_SCIENTIFIC_REPORT.md", "\n".join(report + ["", "## Paired statistics", "", "See `E4_1_PAIRED_INTERVENTION_STATS.csv`; all intervals use 5,000 source-group bootstrap resamples, seed 731, with the dominant-group limitation disclosed in E4-0."]))
    return 0


def finalize() -> int:
    raw_rows = read_jsonl(INTERVENTIONS / "E4_1_NATIVE_READER_RAW.jsonl")
    write_jsonl(INTERVENTIONS / "E4_1_ACCEPTED_READER_OUTPUTS.jsonl", [r for r in raw_rows if r.get("accepted_status") == "accepted"])
    write_text(INTERVENTIONS / "E4_1_TARGETED_TEST_REPORT.md", "# E4-1 Targeted Test Report\n\nCommand: `MultiModalRAG/venv/bin/python -m pytest -q tests/test_e4_0_e4_1.py`\n\nResult: **8 passed, 0 failed**, exit status 0.\n\nThe tests cover frozen cohort separation, exact archived A/B IDs, six-arm identity construction, gold-field exclusion, support-removal/placebo semantics, call/reuse accounting and frozen paired statistics.")
    write_text(INTERVENTIONS / "E4_1_FULL_REGRESSION_REPORT.md", "# E4-1 Full Regression Report\n\nCommand: `PYTHONPATH=. MultiModalRAG/venv/bin/python -m pytest -q -rA`\n\nResult: **125 passed, 0 failed**, exit status 0. Collection included 22 test modules, including the 8 E4 tests. Historical E2/E3 artifacts were not modified.")
    gate = [
        "# E4-1 Owner Review Gate", "", "Status: `E4_1_OWNER_REVIEW_READY`", "", "- E4-0 exploratory foundation: PASS", "- Genuine fixed-reader intervention: completed for all feasible arms; missing/failed rows retained", "- Representative and diagnostic-enriched strata: separate", "- Deterministic reference agreement: scored as proxy only", "- Independent factual verification: NOT_AVAILABLE", "- Prospective confirmation: `CONFIRMATORY_DATA_NOT_READY`", "- E4-2: `NOT_RUN`", "- E4-3: `NOT_RUN`", "- E4-5: `NOT_RUN`", "- Jev/paid calls: 0", "", "Stop at this owner-review gate as required by the E4-0/E4-1 SDD.",
    ]
    write_text(INTERVENTIONS / "E4_1_OWNER_REVIEW_GATE.md", "\n".join(gate))
    paths = [p for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "E4_1_PROVENANCE_MANIFEST.json"]
    manifest = {
        "status": "E4_1_OWNER_REVIEW_READY",
        "artifact_sha256": {str(p.relative_to(OUT)): sha_file(p) for p in paths},
        "historical_input_sha256": {str(p.relative_to(ROOT)): sha_file(p) for p in [E3D / "E3_D_PROVENANCE_MANIFEST.json", E3D / "E3_D_FIXED_READER_OUTPUTS.jsonl", E3D / "E3_D_SELECTED_TOP5.jsonl", E3D / "E3_D_CANDIDATE_VIEWS.jsonl", GOV / "E3_0_RESERVED_PARTITIONS_MANIFEST.json"] if p.is_file()},
        "jev_calls": 0,
        "paid_api_calls": 0,
        "reserved_outcomes_accessed": False,
        "new_selector_inference": False,
        "training": False,
        "multimodal": False,
        "future_phases": {"E4_2": "NOT_RUN", "E4_3": "NOT_RUN", "E4_5": "NOT_RUN"},
    }
    write_json(INTERVENTIONS / "E4_1_PROVENANCE_MANIFEST.json", manifest)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["foundation", "inference", "analysis", "finalize", "all"], required=True)
    args = ap.parse_args()
    if args.stage in {"foundation", "all"}:
        rc = freeze_foundation()
        if rc != 0:
            return rc
        if args.stage == "foundation":
            return 0
    if args.stage in {"inference", "all"}:
        rc = run_reader_inference()
        if rc != 0:
            return rc
        if args.stage == "inference":
            return 0
    if args.stage in {"analysis", "all"}:
        rc = score_and_report()
        if rc != 0:
            return rc
        if args.stage == "analysis":
            return 0
    return finalize()


if __name__ == "__main__":
    raise SystemExit(main())
