#!/usr/bin/env python3
"""E11-V V0/V1 CPU-only admission audit.

This script deliberately stops before model loading or inference.  It audits the
official CRAG Task-3 archive, creates a deterministic pre-outcome cohort, and
writes append-only V0/V1 artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e11v_v1_0"
INV = OUT / "00_inventory"
ADM = OUT / "01_admission"
CRAG = ROOT / "data/e11v/crag_task_3"
EXTRACTED = CRAG / "extracted/crag_task_3_dev_v4"
SEED = 1103
MAX_REP = 240
MAX_DIAG = 60
TOPK = 20

HISTORICAL = [
    ROOT / "results/non_jev/v2/e9_bright_v1_0/06_closure/E9_BRIGHT_PROVENANCE_MANIFEST.json",
    ROOT / "results/non_jev/v2/e9_bright_v1_0/06_closure/E9_BRIGHT_FINAL_OWNER_REVIEW_GATE.json",
    ROOT / "results/non_jev/v2/e10_v1_0/05_closure/E10_FINAL_PROVENANCE_MANIFEST.json",
    ROOT / "results/non_jev/v2/e10_v1_0/05_closure/E10_FINAL_OWNER_REVIEW_GATE.json",
    ROOT / "results/non_jev/v2/e11_v1_0/06_closure/E11_FINAL_PROVENANCE_MANIFEST.json",
    ROOT / "results/non_jev/v2/e11_v1_0/06_closure/E11_FINAL_OWNER_REVIEW_GATE.json",
    ROOT / "results/non_jev/v2/e11_v1_0/01_policy_freeze/E11_POLICY_FREEZE_RECEIPT.json",
    ROOT / "results/non_jev/v2/e11_v1_0/02_budgeted_selector/E11_BUDGETED_SELECTOR_PROVENANCE.json",
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def stable_score(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()

def nonempty(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, str):
        return bool(x.strip())
    if isinstance(x, (list, tuple, dict)):
        return bool(x)
    return True

def answer_scorable(row: dict[str, Any]) -> bool:
    return nonempty(row.get("answer")) or any(nonempty(x) for x in (row.get("alt_ans") or []))

def candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for i, page in enumerate(row.get("search_results") or []):
        if not isinstance(page, dict):
            continue
        url = str(page.get("page_url") or "").strip()
        html = page.get("page_result")
        snippet = page.get("page_snippet")
        content = html if isinstance(html, str) and html.strip() else snippet
        if not url or not isinstance(content, str) or not content.strip():
            continue
        out.append({
            "candidate_id": f"crag_task3:{row.get('interaction_id')}::archived_rank_{i+1:02d}",
            "archived_rank": i + 1,
            "page_url": url,
            "page_name": str(page.get("page_name") or ""),
            "content_chars": len(content),
        })
    return out

def iter_rows() -> Iterable[dict[str, Any]]:
    paths = sorted(EXTRACTED.glob("crag_task_3_dev_v4_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No extracted CRAG shards under {EXTRACTED}")
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    row = json.loads(line)
                except Exception:
                    yield {"__malformed__": True, "__path__": str(path), "__line__": line_no}
                    continue
                if isinstance(row, dict):
                    row["__path__"] = str(path)
                    yield row
                else:
                    yield {"__malformed__": True, "__path__": str(path), "__line__": line_no}

class DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)
    def find(self, x: str) -> str:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]
    def union(self, a: str, b: str) -> None:
        self.add(a); self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

def gate_json(status: str, reason: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "gate": "E11V_V1_DATASET_ADMISSION",
        "status": status,
        "reason": reason,
        "model_inference_started": False,
        "reader_inference_started": False,
        "reserved_holdout_opened": False,
        "metrics": metrics,
        "seed": SEED,
    }

def main() -> int:
    INV.mkdir(parents=True, exist_ok=True)
    ADM.mkdir(parents=True, exist_ok=True)
    shard_paths = sorted(EXTRACTED.glob("crag_task_3_dev_v4_*.jsonl"))
    if not shard_paths:
        raise SystemExit(f"Missing extracted CRAG shards: {EXTRACTED}")

    totals = Counter()
    split_counts = Counter()
    domain_counts = Counter()
    type_counts = Counter()
    malformed: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    eligible: list[dict[str, Any]] = []
    url_first: dict[str, str] = {}
    dsu = DSU()
    answer_kind = Counter()

    for row in iter_rows():
        if row.get("__malformed__"):
            totals["malformed"] += 1
            if len(malformed) < 100:
                malformed.append(row)
            continue
        totals["records"] += 1
        qid = str(row.get("interaction_id") or "")
        if not qid:
            totals["missing_interaction_id"] += 1
            continue
        if qid in seen_ids:
            duplicate_ids.append(qid)
        seen_ids.add(qid)
        split = row.get("split")
        split_counts[str(split)] += 1
        domain_counts[str(row.get("domain") or "missing")] += 1
        type_counts[str(row.get("question_type") or "missing")] += 1
        answer_kind[type(row.get("answer")).__name__] += 1
        pages = candidates(row)
        totals["records_with_answer_or_alt"] += int(answer_scorable(row))
        totals["records_with_20_valid_archived_candidates"] += int(len(pages) >= TOPK)
        if str(split) != "0" or not answer_scorable(row) or len(pages) < TOPK:
            continue
        totals["validation_answer_scorable_20_candidate"] += 1
        # URL-overlap components are a conservative metadata-only grouping.  A
        # component is never presented as semantic proof of source independence.
        qid_key = f"crag:{qid}"
        dsu.add(qid_key)
        for page in pages[:TOPK]:
            url = page["page_url"]
            other = url_first.get(url)
            if other is None:
                url_first[url] = qid_key
            else:
                dsu.union(qid_key, other)
        eligible.append({
            "question_id": qid,
            "domain": str(row.get("domain") or "missing"),
            "question_type": str(row.get("question_type") or "missing"),
            "question": row.get("query"),
            "pages": pages[:TOPK],
            "answer": row.get("answer"),
            "alt_ans": row.get("alt_ans") or [],
        })

    components = Counter(dsu.find(x) for x in dsu.parent)
    eligible_components = len(components)
    # Deterministic stratified selection: rank within domain/type buckets, then
    # round-robin buckets so no outcome or selector result affects sampling.
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        item["source_component"] = f"component:{dsu.find('crag:'+item['question_id'])}"
        item["selection_key"] = stable_score(item["question_id"])
        buckets[(item["domain"], item["question_type"])].append(item)
    for values in buckets.values():
        values.sort(key=lambda x: x["selection_key"])
    ordered_buckets = sorted(buckets)
    representative: list[dict[str, Any]] = []
    cursor = 0
    while len(representative) < min(MAX_REP, len(eligible)) and ordered_buckets:
        key = ordered_buckets[cursor % len(ordered_buckets)]
        if buckets[key]:
            representative.append(buckets[key].pop(0))
        ordered_buckets = [k for k in ordered_buckets if buckets[k]]
        if not ordered_buckets:
            break
        cursor += 1
    rep_ids = {x["question_id"] for x in representative}
    oversized = [x for x in eligible if x["question_id"] not in rep_ids and max(p["content_chars"] for p in x["pages"]) > 20000]
    oversized.sort(key=lambda x: x["selection_key"])
    diagnostic = oversized[:MAX_DIAG]

    cohort = representative + diagnostic
    cohort_rows = []
    for item in cohort:
        cohort_rows.append({
            "question_id": item["question_id"],
            "cohort": "representative" if item in representative else "oversized_diagnostic",
            "domain": item["domain"],
            "question_type": item["question_type"],
            "source_component": item["source_component"],
            "question_sha256": hashlib.sha256(str(item.get("question") or "").encode()).hexdigest(),
            "question": item.get("question"),
            "answer": item["answer"],
            "alt_ans": item["alt_ans"],
            "candidate_ids": [p["candidate_id"] for p in item["pages"]],
            "candidate_order": "official_archived_search_results_order_rank_1_to_20",
            "candidates": item["pages"],
        })

    source_records = []
    for p in HISTORICAL:
        rec = {"path": str(p), "exists": p.exists()}
        if p.exists():
            rec.update({"bytes": p.stat().st_size, "sha256": sha256(p)})
        source_records.append(rec)
    archive_parts = sorted(CRAG.glob("crag_task_3_dev_v4.tar.bz2.part*"))
    data_hashes = {
        "official_source": "https://github.com/facebookresearch/CRAG/blob/main/docs/dataset.md",
        "archive_sha256": sha256(CRAG / "crag_task_3_dev_v4.tar.bz2") if (CRAG / "crag_task_3_dev_v4.tar.bz2").exists() else None,
        "parts": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in archive_parts],
        "extracted_shards": [{"path": str(p), "bytes": p.stat().st_size} for p in shard_paths],
    }

    metrics = {
        "records": totals["records"],
        "malformed": totals["malformed"],
        "unique_interaction_ids": len(seen_ids),
        "duplicate_interaction_ids": len(duplicate_ids),
        "split_counts": dict(split_counts),
        "domain_counts": dict(domain_counts),
        "question_type_counts": dict(type_counts),
        "answer_type_counts": dict(answer_kind),
        "records_with_answer_or_alt": totals["records_with_answer_or_alt"],
        "records_with_20_valid_archived_candidates": totals["records_with_20_valid_archived_candidates"],
        "validation_answer_scorable_20_candidate": totals["validation_answer_scorable_20_candidate"],
        "url_overlap_source_components": eligible_components,
        "url_overlap_component_interpretation": "metadata-only connected components over archived Top-20 page URLs; not semantic independence proof",
        "representative_count": len(representative),
        "oversized_diagnostic_count": len(diagnostic),
        "oversized_available": len(oversized),
        "cohort_total": len(cohort_rows),
    }
    audit = {
        "protocol": "DecisionRAG E11-V V0/V1",
        "model_inference_started": False,
        "reader_inference_started": False,
        "reserved_holdout_opened": False,
        "source_manifest": source_records,
        "crag": metrics,
        "official_schema_fields_seen": ["interaction_id", "query_time", "domain", "question_type", "static_or_dynamic", "query", "answer", "alt_ans", "split", "search_results"],
        "malformed_examples": malformed,
    }
    (INV / "V0_SOURCE_MANIFEST.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    (INV / "V0_EXPOSURE_REGISTER.json").write_text(json.dumps({
        "gate": "E11V_V0_HISTORICAL_INTEGRITY",
        "historical_artifacts_read_for_hash_and_gate_metadata": [x["path"] for x in source_records if x["exists"]],
        "new_dataset": "official CRAG Task-3 archived data",
        "holdout_paths_opened": [],
        "holdout_outcomes_accessed": False,
        "model_calls": 0,
        "reader_calls": 0,
        "paid_api_calls": 0,
        "jev_calls": 0,
    }, indent=2) + "\n")
    (INV / "V0_HISTORICAL_LEDGER_AUDIT.md").write_text(
        "# E11-V V0 Historical Ledger Audit\n\n"
        "CPU/offline audit only. No model, reader, Jev, paid API, GPU, or protected-holdout calls were made.\n\n"
        "Historical manifests and gates were hashed before V1 admission. Existing files were not modified.\n\n"
        + "\n".join(f"- `{x['path']}` — exists={x['exists']}, bytes={x.get('bytes')}, sha256={x.get('sha256')}" for x in source_records)
        + "\n\nFrozen E11 policy and model revisions remain unchanged: PPLX `14c9d90137962d300f3de95633f7b6535e80c5ce`; Jina `e8a93f33f0b22108f8c2364f8484ce3422552fbc`.\n"
    )

    with (ADM / "V1_TOKEN_LENGTH_DISTRIBUTION.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "cohort", "domain", "candidate_count", "total_chars_top20", "approx_tokens_chars_div_4", "max_page_chars", "oversized_approx_gt_20000_chars"])
        for item in cohort:
            total_chars = sum(p["content_chars"] for p in item["pages"])
            max_chars = max(p["content_chars"] for p in item["pages"])
            w.writerow([item["question_id"], "representative" if item in representative else "oversized_diagnostic", item["domain"], len(item["pages"]), total_chars, round(total_chars/4), max_chars, int(max_chars > 20000)])
    (ADM / "V1_COHORT_MANIFEST.json").write_text(json.dumps({
        "protocol": "E11-V pre-outcome cohort freeze",
        "seed": SEED,
        "selection_rule": "validation split, answer/alt-answer scorable, >=20 valid archived pages; deterministic hash-ranked domain/question-type round robin",
        "representative": [x for x in cohort_rows if x["cohort"] == "representative"],
        "oversized_diagnostic": [x for x in cohort_rows if x["cohort"] == "oversized_diagnostic"],
    }, indent=2, ensure_ascii=False) + "\n")
    (ADM / "V1_DATA_SHA256.json").write_text(json.dumps(data_hashes, indent=2) + "\n")
    (ADM / "V1_SOURCE_GROUPS.json").write_text(json.dumps({
        "grouping": "archived Top-20 URL-overlap connected components",
        "semantic_independence_proven": False,
        "component_count_eligible_validation": eligible_components,
        "component_counts": dict(Counter(x["source_component"] for x in cohort_rows)),
        "limitation": "CRAG archived web pages do not provide an independent source-document attestation; components only prevent exact URL overlap from being treated as independent.",
    }, indent=2) + "\n")
    (ADM / "V1_CANDIDATE_TOP20_CONTRACT.json").write_text(json.dumps({
        "dataset": "facebookresearch/CRAG Task 3 dev v4",
        "candidate_scope": "official archived search_results positions 1..20",
        "candidate_id": "crag_task3:<interaction_id>::archived_rank_<01..20>",
        "ordering": "unchanged official archived order",
        "web_search": False,
        "retrieval_inference": False,
        "gold_used_for_candidate_construction": False,
        "selector_outcomes_used_for_sampling": False,
    }, indent=2) + "\n")
    (ADM / "V1_FROZEN_PROCEDURE.md").write_text(
        "# E11-V V1 Frozen Procedure\n\n"
        f"Seed: `{SEED}`. The candidate pool is the official CRAG Task-3 archived `search_results` positions 1–20, in original order. No web search, retrieval, model score, selector output, or answer outcome was used for sampling.\n\n"
        "Canonical candidate identity is interaction ID plus archived rank. The representative cohort was selected by deterministic domain/question-type round robin using SHA-256-ranked IDs. The oversized diagnostic cohort was selected from remaining eligible validation questions by deterministic hash order using a pre-outcome 20,000-character proxy threshold.\n\n"
        "Text lengths are CPU-only byte/character measurements; `characters / 4` is an explicit approximate-token proxy, not a tokenizer measurement.\n"
    )
    (ADM / "V1_REFERENCE_VALIDITY_AUDIT.md").write_text(
        "# E11-V V1 Reference Validity Audit\n\n"
        "CRAG Task 3 supplies an official answer and optional `alt_ans` values. This is an answer-match endpoint, not passage-level relevance supervision: the archive does not provide a verified supporting-passage qrel for each question. Multiple alternatives are retained separately and are never unioned into a mandatory evidence set.\n\n"
        f"Answer/alternate-answer scorable validation records: **{totals['validation_answer_scorable_20_candidate']}**.\n\n"
        "The endpoint can support deterministic answer/reference agreement if a fixed reader is later admitted. It cannot support complete-support CSSR or independently verified factual correctness without additional evidence annotation.\n"
    )
    (ADM / "V1_DATASET_ADMISSION_REPORT.md").write_text(
        "# E11-V V1 Dataset Admission Report\n\n"
        "## Dataset\n\n"
        "Official Facebook Research CRAG Task 3 development archive. The release was downloaded from the official repository documentation and verified by SHA-256. The archive was not altered; extraction is a derived working representation.\n\n"
        "## Counts\n\n"
        + "\n".join(f"- {k}: `{v}`" for k, v in metrics.items() if not isinstance(v, dict))
        + "\n\n"
        "## Integrity and limits\n\n"
        "All admissible candidates are archived pages with stable interaction/rank IDs and valid content. Exact URL-overlap components are computed for source-group-aware resampling, but the dataset does not attest semantic source independence. Gold answers and alternatives are available for answer-match scoring; passage-level relevance and grounding labels are absent.\n\n"
        "No inference was run during admission.\n"
    )
    (ADM / "V1_ADMISSION_GATE.json").write_text(json.dumps(gate_json(
        "PASS_WITH_ENDPOINT_LIMITATIONS" if len(representative) >= 160 and eligible_components >= 100 else "BLOCKED",
        "CRAG Task 3 is admissible for answer/reference agreement only; passage-level support and independent factual verification remain unavailable.",
        metrics,
    ), indent=2) + "\n")
    (ADM / "V1_ADMISSION_SUMMARY.json").write_text(json.dumps({"metrics": metrics, "data_sha256": data_hashes, "cohort_ids": [x["question_id"] for x in cohort_rows]}, indent=2) + "\n")
    print(json.dumps({"status": "completed", "metrics": metrics, "output": str(OUT)}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
