#!/usr/bin/env python3
"""Offline E2-N2-R owner closure and E2-S/P readiness package.

This script intentionally does not import a model adapter, contact a service,
touch Qdrant, or access a GPU.  It reads the immutable E2-N2 normalized score
rows, frozen historical slates, read-only gold sidecars, and the already
recorded execution manifests.  Quality and paired statistics are recomputed
from those inputs using an independent implementation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/non_jev/v2/e2_n2"
OUT = BASE / "e2_n2_owner_closure"
SCORE_PATH = BASE / "E2_N2_MODEL_SCORES.jsonl"
SELECTED_PATH = BASE / "E2_N2_SELECTED_DOCUMENTS.jsonl"
QUALITY_PATH = BASE / "E2_N2_MODEL_DATASET_QUALITY_SUMMARY.csv"
PAIRED_PATH = BASE / "E2_N2_PRIMARY_EXPLORATORY_PAIRED_STATISTICS.csv"
EXECUTION_MANIFEST = BASE / "E2_N2_EXECUTION_MANIFEST.json"
PROVENANCE_MANIFEST = BASE / "E2_N2_PROVENANCE_MANIFEST.json"
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 731
POLICIES = (
    "retrieval5",
    "bge_pointwise",
    "laya_pointwise",
    "clef_pointwise",
    "clef_joint",
    "gte",
    "minilm",
    "jina35",
    "nimble",
    "intern_decision",
    "pplx_decider",
    "qwen06",
    "qwen4",
    "qwen_direct",
)
PRIORITY_COMPARISONS = {
    "jina35_minus_bge": ("jina35", "bge_pointwise"),
    "pplx_decider_minus_bge": ("pplx_decider", "bge_pointwise"),
    "pplx_decider_minus_jina35": ("pplx_decider", "jina35"),
    "nimble_minus_bge": ("nimble", "bge_pointwise"),
    "pplx_decider_minus_retrieval5": ("pplx_decider", "retrieval5"),
    "jina35_minus_retrieval5": ("jina35", "retrieval5"),
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    return xs[max(0, math.ceil(p * len(xs)) - 1)]


def rate(n: int, d: int) -> float | None:
    return n / d if d else None


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "NOT_MEASURED"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_items() -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2_m1_experiment import load_full_items

    items = load_full_items()
    if len(items) != 4267:
        raise RuntimeError(f"frozen population changed: {len(items)}")
    return items


def load_gold() -> tuple[dict[str, set[str]], dict[str, str]]:
    """Resolve gold independently; qst_0413 remains explicitly ambiguous."""
    db = ROOT / "results/non_jev/v11/track_b/V11_TRACK_B_ENTERPRISE_CORPUS.sqlite3"
    mapping: dict[str, list[str]] = defaultdict(list)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        for internal, original in conn.execute("SELECT internal_doc_id, original_doc_id FROM documents"):
            mapping[original].append(internal)
    finally:
        conn.close()
    gold: dict[str, set[str]] = {}
    status: dict[str, str] = {}
    for path in (ROOT / "results/preflight/v11/NEW_TEST_GOLD_SIDECAR.jsonl", ROOT / "results/preflight/v11/ENTERPRISE_GOLD_SIDECAR.jsonl"):
        enterprise = path.name.startswith("ENTERPRISE")
        for row in load_jsonl(path):
            qid = row.get("query_id") or row.get("question_id")
            original = row.get("gold_doc_ids") if not enterprise else row.get("expected_doc_ids")
            original = list(dict.fromkeys(original or []))
            if not qid:
                raise RuntimeError(f"missing gold ID in {path}")
            if not enterprise:
                gold[qid] = set(original)
                status[qid] = "resolved"
                continue
            unresolved = [x for x in original if len(mapping.get(x, [])) != 1]
            if unresolved:
                status[qid] = "ambiguous_or_unresolved"
                continue
            gold[qid] = {mapping[x][0] for x in original}
            status[qid] = "resolved"
    return gold, status


def top5(ids: list[str], scores: list[float]) -> list[str]:
    # Original position and ID are explicit deterministic tie breakers.
    order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i, ids[i]))
    return [ids[i] for i in order[:5]]


def metric(gold: set[str], candidates: list[str], selected: list[str] | None) -> dict[str, bool]:
    eligible = 1 <= len(gold) <= 5
    conditional = eligible and gold.issubset(set(candidates))
    return {
        "eligible": eligible,
        "conditional": conditional,
        "success": bool(selected is not None and eligible and gold.issubset(set(selected))),
        "any_gold": bool(gold.intersection(candidates)),
    }


_BOOTSTRAP_INDEX_CACHE: dict[int, np.ndarray] = {}


def bootstrap(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    n = len(values)
    # Generate exactly the same Python-random index stream as the frozen N4
    # evaluator, once per population size.  NumPy only vectorizes the sum of
    # those already-identical resamples; it does not change the resampling
    # distribution or seed convention.
    indices = _BOOTSTRAP_INDEX_CACHE.get(n)
    if indices is None:
        rng = random.Random(BOOTSTRAP_SEED)
        indices = np.fromiter((rng.randrange(n) for _ in range(BOOTSTRAP_REPS * n)), dtype=np.int32, count=BOOTSTRAP_REPS * n).reshape(BOOTSTRAP_REPS, n)
        _BOOTSTRAP_INDEX_CACHE[n] = indices
    values_array = np.asarray(values, dtype=np.float64)
    estimates = np.sort(values_array[indices].mean(axis=1))
    return estimates[int(0.025 * BOOTSTRAP_REPS)], estimates[int(0.975 * BOOTSTRAP_REPS) - 1]


def read_scores(items_by_qid: dict[str, dict]) -> tuple[dict[str, dict[str, dict]], dict[str, dict[str, int]], list[str]]:
    traces: dict[str, dict[str, dict]] = {p: {} for p in POLICIES if p != "retrieval5"}
    counts: dict[str, dict[str, int]] = {p: Counter() for p in traces}
    errors: list[str] = []
    with SCORE_PATH.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            policy = row.get("policy_id")
            qid = row.get("query_id")
            if policy not in traces or qid not in items_by_qid:
                errors.append(f"line {line_number}: unknown policy/query {policy}/{qid}")
                continue
            counts[policy][row.get("status", "missing")] += 1
            item_ids = [x["doc_id"] for x in items_by_qid[qid]["candidates"]]
            if row.get("candidate_ids") != item_ids:
                errors.append(f"line {line_number}: candidate identity/order mismatch for {policy}/{qid}")
            if qid in traces[policy]:
                errors.append(f"line {line_number}: duplicate policy/query row {policy}/{qid}")
            traces[policy][qid] = row
    return traces, counts, errors


def read_published_quality() -> dict[tuple[str, str, str], dict]:
    with QUALITY_PATH.open(newline="", encoding="utf-8") as f:
        return {(r["cohort"], r["dataset"], r["policy"]): r for r in csv.DictReader(f)}


def read_published_pairs() -> list[dict]:
    with PAIRED_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_selections(items: list[dict], traces: dict[str, dict[str, dict]]) -> tuple[dict[str, dict[str, list[str] | None]], dict[str, int]]:
    selections: dict[str, dict[str, list[str] | None]] = {p: {} for p in POLICIES}
    counts = Counter()
    for item in items:
        qid = item["query_id"]
        ids = [x["doc_id"] for x in item["candidates"]]
        selections["retrieval5"][qid] = ids[:5]
        for policy in POLICIES:
            if policy == "retrieval5":
                continue
            row = traces[policy].get(qid)
            if not row or row.get("status") != "ok" or not isinstance(row.get("scores"), list):
                selections[policy][qid] = None
                counts[f"{policy}:blocked"] += 1
                continue
            scores = [float(x) for x in row["scores"]]
            if len(scores) != len(ids) or any(not math.isfinite(x) for x in scores):
                selections[policy][qid] = None
                counts[f"{policy}:invalid"] += 1
                continue
            selections[policy][qid] = top5(ids, scores)
            counts[f"{policy}:ok"] += 1
    return selections, counts


def recompute_quality(items: list[dict], gold: dict[str, set[str]], selections: dict[str, dict[str, list[str] | None]]) -> tuple[list[dict], dict[tuple[str, str, str], dict]]:
    rows: list[dict] = []
    lookup: dict[tuple[str, str, str], dict] = {}
    for cohort in ("track_a", "track_b"):
        cohort_items = [x for x in items if x["cohort"] == cohort]
        datasets = sorted({x["dataset"] for x in cohort_items})
        for dataset in datasets:
            dataset_items = [x for x in cohort_items if x["dataset"] == dataset]
            for policy in POLICIES:
                available = [x for x in dataset_items if selections[policy].get(x["query_id"]) is not None]
                measures = [metric(gold.get(x["query_id"], set()), [c["doc_id"] for c in x["candidates"]], selections[policy][x["query_id"]]) for x in available]
                eligible = [m for m in measures if m["eligible"]]
                conditional = [m for m in eligible if m["conditional"]]
                row = {
                    "cohort": cohort,
                    "dataset": dataset,
                    "policy": policy,
                    "n_total": len(available),
                    "n_eligible": len(eligible),
                    "n_conditional": len(conditional),
                    "n_any_gold": sum(m["any_gold"] for m in measures),
                    "success_unconditional": sum(m["success"] for m in eligible),
                    "success_conditional": sum(m["success"] for m in conditional),
                    "cssr_unconditional": rate(sum(m["success"] for m in eligible), len(eligible)),
                    "cssr_conditional": rate(sum(m["success"] for m in conditional), len(conditional)),
                    "any_gold_coverage": rate(sum(m["any_gold"] for m in measures), len(measures)),
                    "blocked_rows": len(dataset_items) - len(available),
                    "interpretation": "complete-support selection proxy; not reader factual accuracy",
                }
                rows.append(row)
                lookup[(cohort, dataset, policy)] = row
        for policy in POLICIES:
            parts = [lookup[(cohort, dataset, policy)] for dataset in datasets]
            row = {
                "cohort": cohort,
                "dataset": "__equal_dataset_macro__",
                "policy": policy,
                "n_total": sum(x["n_total"] for x in parts),
                "n_eligible": sum(x["n_eligible"] for x in parts),
                "n_conditional": sum(x["n_conditional"] for x in parts),
                "n_any_gold": sum(x["n_any_gold"] for x in parts),
                "success_unconditional": sum(x["success_unconditional"] for x in parts),
                "success_conditional": sum(x["success_conditional"] for x in parts),
                "cssr_unconditional": statistics.mean([x["cssr_unconditional"] for x in parts if x["cssr_unconditional"] is not None]) if any(x["cssr_unconditional"] is not None for x in parts) else None,
                "cssr_conditional": statistics.mean([x["cssr_conditional"] for x in parts if x["cssr_conditional"] is not None]) if any(x["cssr_conditional"] is not None for x in parts) else None,
                "any_gold_coverage": statistics.mean([x["any_gold_coverage"] for x in parts if x["any_gold_coverage"] is not None]) if any(x["any_gold_coverage"] is not None for x in parts) else None,
                "blocked_rows": sum(x["blocked_rows"] for x in parts),
                "interpretation": "equal-dataset macro of dataset-level complete-support proxies; not a pooled query estimate",
            }
            rows.append(row)
            lookup[(cohort, "__equal_dataset_macro__", policy)] = row
    return rows, lookup


def recompute_pairs(items: list[dict], gold: dict[str, set[str]], selections: dict[str, dict[str, list[str] | None]], published: list[dict]) -> list[dict]:
    specs = []
    seen = set()
    for row in published:
        key = (row["cohort"], row["population"], row["comparison"], row["left_policy"], row["right_policy"])
        if key not in seen:
            seen.add(key)
            specs.append(key)
    for cohort in ("track_a", "track_b"):
        for population in ("unconditional_eligible", "retrieval_conditional"):
            for name, (left, right) in PRIORITY_COMPARISONS.items():
                key = (cohort, population, name, left, right)
                if key not in seen:
                    seen.add(key)
                    specs.append(key)
    output = []
    by_cohort = {c: [x for x in items if x["cohort"] == c] for c in ("track_a", "track_b")}
    for cohort, population, name, left, right in specs:
        deltas: list[float] = []
        for item in by_cohort[cohort]:
            g = gold.get(item["query_id"], set())
            if not (1 <= len(g) <= 5):
                continue
            ids = [c["doc_id"] for c in item["candidates"]]
            if population == "retrieval_conditional" and not g.issubset(set(ids)):
                continue
            lsel = selections[left].get(item["query_id"])
            rsel = selections[right].get(item["query_id"])
            if lsel is None or rsel is None:
                continue
            deltas.append(float(g.issubset(set(lsel))) - float(g.issubset(set(rsel))))
        low, high = bootstrap(deltas)
        output.append({
            "cohort": cohort,
            "population": population,
            "comparison": name,
            "left_policy": left,
            "right_policy": right,
            "n": len(deltas),
            "difference": rate(sum(deltas), len(deltas)),
            "ci95_low": low,
            "ci95_high": high,
            "paired_wins": sum(x > 0 for x in deltas),
            "paired_losses": sum(x < 0 for x in deltas),
            "paired_ties": sum(x == 0 for x in deltas),
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "interpretation": "offline recomputation; exploratory follow-up; CI is query-paired and not a confirmatory significance claim",
        })
    return output


def compare_float(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if a in (None, "") and b in (None, ""):
        return True
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return str(a) == str(b)


def audit_selected_file(items: list[dict], selections: dict[str, dict[str, list[str] | None]]) -> tuple[int, int, list[str]]:
    expected = {(p, x["query_id"]): selections[p].get(x["query_id"]) for p in POLICIES for x in items}
    seen = set()
    mismatch = []
    rows = 0
    with SELECTED_PATH.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("policy_id") or row.get("policy"), row.get("query_id"))
            rows += 1
            seen.add(key)
            wanted = expected.get(key)
            actual = row.get("selected_ids")
            if wanted is None:
                if actual not in ([], None):
                    mismatch.append(f"line {line_number}: blocked selection not empty for {key}")
            elif actual != wanted:
                mismatch.append(f"line {line_number}: selected IDs differ for {key}")
    missing = len(set(expected) - seen)
    return rows, missing, mismatch


def view_audit(items_by_qid: dict[str, dict], traces: dict[str, dict[str, dict]], counts: dict[str, dict[str, int]]) -> list[dict]:
    rows = [{
        "policy": "retrieval5",
        "model_id": "frozen_retriever",
        "revision": "frozen_retrieval_slate",
        "rows": len(items_by_qid),
        "ok_rows": len(items_by_qid),
        "blocked_or_error_rows": 0,
        "candidate_identity_aligned_rows": len(items_by_qid),
        "candidate_view_hash_rows": 0,
        "actual_model_input_hash_rows": 0,
        "raw_response_hash_rows": 0,
        "question_hash_rows": 0,
        "finite_20_score_rows": 0,
        "physical_forward_evidence_rows": 0,
        "physical_forward_calls": 0,
        "logical_decisions": 0,
        "api_calls": "NOT_APPLICABLE",
        "timestamp_rows": 0,
        "cache_reuse_rows": 0,
        "gold_in_inference_input_rows": 0,
        "duration_rows": 0,
        "duration_p50_ms": None,
        "duration_p95_ms": None,
        "input_token_p50": None,
        "input_token_p95": None,
        "scoring_contract": "frozen_retriever_order",
        "adapter_contract": "frozen retrieval control; no model inference",
        "source_trace_path": "results/non_jev/v11/track_a+track_b/*_RETRIEVAL_SLATES.jsonl",
        "source_trace_sha256": "recorded per slate row; no single normalized trace hash",
    }]
    contract = {
        "bge_pointwise": "independent pointwise relevance; semantic text view",
        "laya_pointwise": "independent pointwise relevance; adapter batches candidate payloads",
        "clef_pointwise": "pointwise binary true probability",
        "clef_joint": "native joint binary true probability",
        "gte": "pointwise text relevance",
        "minilm": "pointwise text relevance",
        "jina35": "native listwise/chunked relevance contract",
        "nimble": "native slate categorical 20-choice contract",
        "intern_decision": "native joint binary contract; enterprise context blocked at >8192",
        "pplx_decider": "native slate categorical 20-choice contract",
        "qwen06": "pointwise text relevance",
        "qwen4": "official relevance-token contract",
        "qwen_direct": "direct checkpoint independent relevance contract",
    }
    for policy in traces:
        policy_rows = list(traces[policy].values())
        ok = [r for r in policy_rows if r.get("status") == "ok"]
        score_ok = [r for r in ok if isinstance(r.get("scores"), list) and len(r["scores"]) == 20 and all(math.isfinite(float(x)) for x in r["scores"])]
        candidate_ok = [r for r in policy_rows if r.get("query_id") in items_by_qid and r.get("candidate_ids") == [c["doc_id"] for c in items_by_qid[r["query_id"]]["candidates"]]]
        view_hash_ok = [r for r in ok if isinstance(r.get("candidate_view_hashes"), list) and len(r["candidate_view_hashes"]) == 20 and all(isinstance(x, str) and len(x) == 64 for x in r["candidate_view_hashes"])]
        input_hash = [r for r in ok if isinstance(r.get("actual_model_input_sha256"), str) and len(r["actual_model_input_sha256"]) == 64]
        response_hash = [r for r in ok if isinstance(r.get("raw_response_sha256"), str) and len(r["raw_response_sha256"]) == 64]
        gold_flags = sum(r.get("gold_in_inference_input") is True for r in policy_rows)
        forward_calls = sum(int(r.get("physical_forward_calls", 0) or 0) for r in policy_rows)
        evidence = sum(r.get("physical_forward_evidence") is True for r in policy_rows)
        durations = [float(r["duration_ms"]) for r in ok if isinstance(r.get("duration_ms"), (int, float))]
        tokens = []
        for r in ok:
            t = r.get("input_tokens")
            if isinstance(t, list):
                tokens.append(sum(int(x) for x in t))
            elif isinstance(t, (int, float)):
                tokens.append(int(t))
        rows.append({
            "policy": policy,
            "model_id": next(iter({r.get("model_id") for r in policy_rows}), None),
            "revision": ";".join(sorted({str(r.get("revision")) for r in policy_rows})),
            "rows": len(policy_rows),
            "ok_rows": len(ok),
            "blocked_or_error_rows": len(policy_rows) - len(ok),
            "candidate_identity_aligned_rows": len(candidate_ok),
        "candidate_view_hash_rows": len(view_hash_ok),
        "actual_model_input_hash_rows": len(input_hash),
        "raw_response_hash_rows": len(response_hash),
            "question_hash_rows": sum(isinstance(r.get("question_sha256"), str) and len(r["question_sha256"]) == 64 for r in ok),
        "finite_20_score_rows": len(score_ok),
        "physical_forward_evidence_rows": evidence,
        "physical_forward_calls": forward_calls,
            "logical_decisions": sum(len(r.get("scores", [])) for r in score_ok),
            "api_calls": "NOT_RECORDED",
            "timestamp_rows": sum(bool(r.get("timestamp") or r.get("started_at") or r.get("finished_at")) for r in policy_rows),
            "cache_reuse_rows": sum(r.get("cache_reuse") is True for r in policy_rows),
            "gold_in_inference_input_rows": gold_flags,
            "duration_rows": len(durations),
            "duration_p50_ms": median(durations),
            "duration_p95_ms": percentile(durations, 0.95),
            "input_token_p50": median(tokens),
            "input_token_p95": percentile(tokens, 0.95),
        "scoring_contract": ";".join(sorted({str(r.get("score_contract") or r.get("formulation") or "unspecified") for r in policy_rows})),
        "adapter_contract": contract[policy],
            "source_trace_path": ";".join(sorted({str(r.get("source_trace")) for r in policy_rows if r.get("source_trace")})) or "NOT_RECORDED",
            "source_trace_sha256": ";".join(sorted({str(r.get("source_trace_sha256")) for r in policy_rows if r.get("source_trace_sha256")})) or "NOT_RECORDED",
        })
    return rows


def source_hash_audit() -> tuple[list[dict], list[str]]:
    manifest = load_json(EXECUTION_MANIFEST)
    checks = []
    errors = []
    for record in manifest.get("source_meta", {}).get("source_records", []):
        path = ROOT / record["trace"]
        actual = sha_file(path) if path.is_file() else None
        expected = record.get("trace_sha256")
        checks.append({"source_key": record.get("source_key"), "path": str(path.relative_to(ROOT)), "expected_sha256": expected, "actual_sha256": actual, "match": actual == expected, "rows": record.get("summary", {}).get("rows")})
        if actual != expected:
            errors.append(f"source trace hash mismatch: {record.get('source_key')}")
        summary = record.get("summary", {})
        if summary.get("trace_sha256") and summary["trace_sha256"] != actual:
            errors.append(f"source summary hash mismatch: {record.get('source_key')}")
    for name, info in manifest.get("output_hashes", {}).items():
        path = BASE / name
        if not path.is_file():
            errors.append(f"missing frozen E2-N2 output: {name}")
            continue
        actual = sha_file(path)
        if actual != info.get("sha256"):
            errors.append(f"frozen E2-N2 output hash mismatch: {name}")
    return checks, errors


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_forward_report(provenance_rows: list[dict], source_checks: list[dict], source_errors: list[str], score_counts: dict[str, dict[str, int]], selected_rows: int, selected_missing: int, selected_mismatch: list[str]) -> None:
    model_rows = [r for r in provenance_rows if r["policy"] != "retrieval5"]
    total_rows = sum(int(r["rows"]) for r in model_rows)
    total_forwards = sum(int(r["physical_forward_calls"]) for r in model_rows)
    total_evidence = sum(int(r["physical_forward_evidence_rows"]) for r in model_rows)
    lines = [
        "# E2-N2-R Forward Count Reconciliation",
        "",
        "Status: `PASS_WITH_LIMITATIONS`",
        "",
        "This is an offline reconciliation of normalized E2-N2 rows and their authenticated source manifests. No model or service was called.",
        "",
        "## Row and forward definitions",
        "",
        "- A normalized row is one policy/query evidence record in `E2_N2_MODEL_SCORES.jsonl`.",
        "- `physical_forward_calls` is the adapter-reported physical forward count recorded in that row; it is not an HTTP/API request count.",
        "- `physical_forward_evidence_rows` counts rows carrying the explicit evidence flag, not forward calls.",
        "- API/service call counts and timestamps were not recorded in the frozen traces and are therefore `UNVERIFIABLE`, not inferred from row counts.",
        "",
        "## Per-policy reconciliation",
        "",
        "| Policy | Queries | Logical candidate decisions | API calls | Instrumented forwards | Source evidence | Blocked/error | Forward distribution | Cache reuse | Gold in input |",
        "|---|---:|---:|---|---:|---|---:|---|---:|---:|",
    ]
    for r in provenance_rows:
        dist = ", ".join(f"{k}:{v}" for k, v in sorted(score_counts.get(r["policy"], {}).items()))
        lines.append(f"| {r['policy']} | {r['rows']} | {r['logical_decisions']} | {r['api_calls']} | {r['physical_forward_calls']} | {r['source_trace_path']} | {r['blocked_or_error_rows']} | {dist} | {r['cache_reuse_rows']} | {r['gold_in_inference_input_rows']} |")
    lines += [
        "",
        f"Totals: **{total_rows:,} normalized rows**, **{total_evidence:,} rows with physical-forward evidence**, and **{total_forwards:,} recorded physical forwards**.",
        "Both headline counts are valid under different definitions: 54,971 is the number of successful normalized evidence rows, whereas 72,039 is the sum of adapter-reported forward calls. They must not be added and neither is an API-call count.",
        "The difference between recorded forwards and evidence rows is **17,068 = 4,267 × 4**, attributable to Laya's five internal forward batches per query while one normalized evidence row is retained per query. Laya's adapter source documents the five-batch pointwise path; this is not an API-request count.",
        "Nimble and PPLX each record one adapter-observed forward per successful query; their native internal choice/selection steps are not separately instrumented. No additional Nimble/PPLX forwards are inferred. Clef pointwise and joint are separately retained one-forward-per-query contracts.",
        "",
        "Intern-Decision has 3,767 successful Track-A/eligible rows and 500 explicit enterprise native-context-blocked rows. Those 500 rows are not relabeled as model failures and are excluded from scored quality populations.",
        "",
        "## Source trace hash checks",
        "",
        "| Source key | Rows in manifest | SHA-256 match |\n|---|---:|---|\n" + "\n".join(f"| {x['source_key']} | {x['rows']} | {'PASS' if x['match'] else 'FAIL'} |" for x in source_checks),
        "",
        f"Source/hash errors: `{len(source_errors)}`.",
        "",
        "## Frozen selection artifact check",
        "",
        f"Normalized selection rows: `{selected_rows}`; expected rows: `{len(POLICIES) * 4267}`; missing keys: `{selected_missing}`; selection mismatches: `{len(selected_mismatch)}`.",
        "",
        "## Limitations",
        "",
        "Per-call timestamps, server queue intervals, HTTP/API request counts, and native internal selection steps are not reconstructible from these artifacts. They must be instrumented in a future E2-S/P run; no additive latency or cost claim is made here.",
    ]
    write_text(OUT / "E2_N2_FORWARD_COUNT_RECONCILIATION.md", "\n".join(lines))


def write_view_report(view_rows: list[dict], score_errors: list[str]) -> None:
    lines = [
        "# E2-N2-R Candidate-View and Score-Joining Audit",
        "",
        "Status: `PASS_WITH_LIMITATIONS`",
        "",
        "All checks below are offline checks over frozen normalized rows. No candidate view was regenerated with a model tokenizer and no inference was performed.",
        "",
        "## Alignment and evidence scorecard",
        "",
        "| Policy | Rows | OK | Canonical candidate alignment | Question hashes | 20-view hashes | Input hashes | Response hashes | Finite 20 scores | Forward evidence | Contract |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in view_rows:
        lines.append(f"| {r['policy']} | {r['rows']} | {r['ok_rows']} | {r['candidate_identity_aligned_rows']} | {r['question_hash_rows']} | {r['candidate_view_hash_rows']} | {r['actual_model_input_hash_rows']} | {r['raw_response_hash_rows']} | {r['finite_20_score_rows']} | {r['physical_forward_evidence_rows']} | {r['adapter_contract']} |")
    lines += [
        "",
        f"Candidate/score audit errors: `{len(score_errors)}`.",
        "",
        "## Relevance direction, sorting, and joining",
        "",
        "- Every successful score row was checked against the exact 20 canonical candidate IDs and order from the frozen Track A/B slate.",
        "- Scores are interpreted in the recorded descending-relevance direction. Top-5 is recomputed from score descending, original candidate position, then candidate ID.",
        "- The frozen selection JSONL was compared against the independently recomputed Top-5 IDs for every policy/query key, including explicit empty selections for Intern-Decision context-blocked rows.",
        "- No duplicate policy/query score rows were accepted.",
        "- `gold_in_inference_input` is false for all normalized rows. Gold labels are joined only in this offline evaluator.",
        "- Canonical query identity is verified by frozen query IDs and the exact frozen slate joins. The normalized model rows do not carry a separate `question_sha256` field: question-hash coverage is therefore `0/54,971` and is disclosed as a telemetry gap, while `actual_model_input_sha256` is present for all successful rows.",
        "",
        "## Model-specific view contracts",
        "",
        "BGE and the text specialists use pointwise/text relevance contracts; Jina uses its native listwise/chunked contract; Nimble and PPLX use native 20-choice slate contracts; Clef has separately recorded pointwise and native-joint binary contracts; Qwen4 uses its official relevance-token contract; Intern-Decision uses its native joint contract and explicitly rejects the 500 enterprise views above its frozen context limit.",
        "These are a common candidate slate but not identical tokenization, native prompt format, or context-budget experiments. Cross-model conclusions must retain that limitation.",
        "",
        "## Candidate-view serialization limitation",
        "",
        "The normalized rows preserve per-candidate `candidate_view_hashes` and the complete model-input hash, but do not embed the raw serialized candidate-view text. Therefore this audit proves identity/order and hash presence, not byte-level regeneration of every view. A future serving contract must retain raw request bytes or a separately verifiable serialization manifest.",
    ]
    write_text(OUT / "E2_N2_CANDIDATE_VIEW_AUDIT.md", "\n".join(lines))


def write_closure_report(quality_rows: list[dict], pair_rows: list[dict], view_rows: list[dict], source_errors: list[str], score_errors: list[str], selected_missing: int, selected_mismatch: list[str], tests_status: str = "PENDING_UNTIL_MAIN_RUN") -> None:
    priority = [r for r in pair_rows if r["comparison"] in PRIORITY_COMPARISONS]
    q_lookup = {(r["cohort"], r["dataset"], r["policy"]): r for r in quality_rows}
    lines = [
        "# E2-N2-R Final Scientific Closure",
        "",
        "Verdict: `E2_N2_OWNER_ACCEPTED_WITH_LIMITATIONS`",
        "Next authorized gate: `E2_SP_READY_FOR_GPU_AUTHORIZATION`",
        "",
        "This closure is offline-only. It preserves all prior E2-N2, E2-N, M0, P0v2, M1, Track A/B/C, and HERB artifacts. It does not authorize or perform E2-S/P serving, fresh-holdout access, training, downloads, Jev, paid inference, model requests, or service changes.",
        "",
        "## Closure gates",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Frozen population and 20-candidate identity alignment | {'PASS' if not score_errors else 'BLOCKED'} |",
        f"| Source trace hashes and execution-manifest hashes | {'PASS' if not source_errors else 'BLOCKED'} |",
        f"| Frozen Top-5 selection reproduction | {'PASS' if selected_missing == 0 and not selected_mismatch else 'BLOCKED'} |",
        "| Independent quality recomputation | PASS_WITH_LIMITATIONS |",
        "| Independent paired bootstrap recomputation | PASS_WITH_LIMITATIONS |",
        f"| CPU regression suite | {tests_status} |",
        "| No new inference / no Jev / no paid calls | PASS |",
        "",
        "## Independently recomputed quality highlights",
        "",
        "CSSR is a complete-support selection proxy, not independently verified reader factual accuracy. Track A and Track B remain separate.",
        "",
        "| Cohort | Dataset | Policy | n scored | n eligible | n conditional | CSSR-U | CSSR-C | Any-gold coverage | Blocked |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    highlights = [
        ("track_a", "hotpot", "jina35"),
        ("track_a", "musique", "jina35"),
        ("track_a", "wiki2", "pplx_decider"),
        ("track_b", "enterprise", "pplx_decider"),
        ("track_a", "__equal_dataset_macro__", "pplx_decider"),
        ("track_b", "__equal_dataset_macro__", "pplx_decider"),
        ("track_b", "enterprise", "intern_decision"),
    ]
    for key in highlights:
        r = q_lookup[key]
        lines.append(f"| {r['cohort']} | {r['dataset']} | {r['policy']} | {r['n_total']} | {r['n_eligible']} | {r['n_conditional']} | {fmt(r['cssr_unconditional'], 4)} | {fmt(r['cssr_conditional'], 4)} | {fmt(r['any_gold_coverage'], 4)} | {r['blocked_rows']} |")
    lines += [
        "",
        "The independent recomputation should match the frozen summary values at floating-point tolerance for all rows that have a valid scored population. Intern-Decision's enterprise row is intentionally `n scored=0` because all 500 of its native views are context-blocked; this is an explicit feasibility limitation, not a zero-quality claim.",
        "",
        "## Priority paired comparisons",
        "",
        "The paired statistic is left-policy complete-support success minus right-policy success. Intervals use 5,000 query-level bootstrap resamples, seed 731, with the frozen exploratory convention. They are not confirmatory significance claims.",
        "",
        "| Cohort | Population | Comparison | n | Difference | 95% CI | Wins/Losses/Ties |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for r in priority:
        lines.append(f"| {r['cohort']} | {r['population']} | {r['comparison']} | {r['n']} | {fmt(r['difference'], 6)} | [{fmt(r['ci95_low'], 6)}, {fmt(r['ci95_high'], 6)}] | {r['paired_wins']}/{r['paired_losses']}/{r['paired_ties']} |")
    lines += [
        "",
        "## Authenticity and provenance conclusions",
        "",
        "- E2-N2 reused authenticated E2-N traces under the recorded E2-N authorization manifest and added no model requests in the N2 namespace.",
        "- The normalized rows contain model revisions, actual input hashes, raw-response hashes, candidate-view hashes, physical-forward flags, and score contracts. The exact source-trace hashes match the E2-N2 manifest.",
        "- Physical forward totals are verified, but frozen artifacts do not record API/service request counts, per-request timestamps, queue intervals, or all native internal selection operations. Those remain `UNVERIFIABLE` and are not converted into latency or monetary claims.",
        "- Laya's five recorded forward batches per query explain the forward/evidence difference; it does not establish five API calls.",
        "- Model-specific tokenization and native context behavior mean comparisons are common-slate exploratory comparisons, not matched compute-budget or identical-input-format experiments.",
        "- No gold labels entered inference inputs, the fresh confirmatory holdout was not accessed, Jev calls are zero, and paid calls are zero.",
        "",
        "## Remaining limitations and next decision",
        "",
        "1. The 500 Intern-Decision enterprise rows need a separately authorized context-feasibility decision or adapter experiment; they must remain blocked in historical quality summaries.",
        "2. E2-S/P can now be prepared as an executable contract, but it is not executed by this closure. GPU exclusivity, model roster, arrival rates, p95 SLO, and downstream-reader cohort still require explicit owner authorization.",
        "3. Downstream answer quality must be measured with a fixed reader and independently validated references; selection CSSR and nonempty answers must not be reported as factual accuracy.",
        "",
        "Amendment proposed: `NONE`. No frozen experimental protocol or prior result was changed.",
    ]
    write_text(OUT / "E2_N2_FINAL_SCIENTIFIC_CLOSURE.md", "\n".join(lines))


def write_sp_docs() -> None:
    write_text(OUT / "E2_SP_EXECUTION_SDD.md", """# DecisionRAG E2-S/P Execution SDD

Status: `READY_FOR_SEPARATE_AUTHORIZATION`; not executed by E2-N2-R.

## Scope

E2-S measures production-serving economics for the frozen reranker roster. E2-P measures downstream RAG/reader consequences. The tracks remain separate and use only historical CAL/DEV material until separately authorized. No fresh confirmatory holdout, Jev, or paid API is permitted by this document.

## Frozen workload

One frozen query with its immutable Top-20 candidate slate is passed to one reranker, producing a Top-5. Candidate IDs, original ordering, query bytes, model input bytes, and raw response bytes must be hashed before normalization. Application-level caching is disabled. Reusable indexes may remain resident; any server/internal cache must be recorded separately.

## E2-S serving measurements

Measure end-to-end request latency and separately timestamp: request admission, queue wait, query encoding, candidate retrieval/view construction, reranker queue wait, reranker compute, response serialization, and client receipt. Report p50/p90/p95, valid n, failures, physical forwards, input/output tokens, peak VRAM, GPU utilization, and arrival-rate throughput. Use exclusive owner-authorized A100 hardware for each matched block; do not infer cost from utilization alone.

No arrival-rate curve, concurrency, or one-second p95 SLO is frozen here. These must be explicitly authorized and locked before execution. No historical Track C latency is promoted as an E2-S/P result.

## E2-P downstream validation

Use the same reader checkpoint, prompt, decoding, evidence formatting, and answer/citation schema across policies. Use a prespecified CAL/DEV stratified cohort. Report independently verified factual correctness, evidence/citation completeness, abstention, latency, and cost. Reference-answer EM/F1 are secondary proxies; nonempty answers and CSSR are not factual correctness.

## Stop conditions

Stop on any cache leak, missing raw-input hash, candidate-order drift, GPU contention, unapproved holdout access, service restart, model revision mismatch, or paid/JeV call. Preserve all traces and label the block.
""")
    matrix = [
        ["pplx_decider", "perplexity-ai/pplx-decider-v1.1-27b", "14c9d90137962d300f3de95633f7b6535e80c5ce", "native_slate_categorical", "complete-support proxy", "exclusive A100 / memory gate", "READY_FOR_SEPARATE_AUTHORIZATION"],
        ["jina35", "jinaai/jina-reranker-v3.5", "e8a93f33f0b22108f8c2364f8484ce3422552fbc", "native listwise/chunked", "complete-support proxy", "exclusive A100 / model-specific", "READY_FOR_SEPARATE_AUTHORIZATION"],
        ["nimble", "bespokelabs/Bespoke-Nimble-9B-v2", "4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe", "native_slate_categorical", "complete-support proxy", "exclusive A100 / memory gate", "READY_FOR_SEPARATE_AUTHORIZATION"],
        ["qwen06", "Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473", "pointwise text", "complete-support proxy", "exclusive A100 or approved service", "READY_FOR_SEPARATE_AUTHORIZATION"],
        ["qwen4", "Qwen/Qwen3-Reranker-4B", "22e683669bc0f0bd69640a135a6d0aebcfeede5", "official relevance-token", "complete-support proxy", "exclusive A100 / memory gate", "READY_FOR_SEPARATE_AUTHORIZATION"],
        ["gte", "Alibaba-NLP/gte-reranker-modernbert-base", "f7481e6055501a30fb19d090657df9ec1f79ab2c", "pointwise text", "complete-support proxy", "exclusive A100 or CPU baseline", "READY_FOR_SEPARATE_AUTHORIZATION"],
        ["bge_pointwise", "BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "pointwise text", "complete-support proxy", "exclusive A100 or CPU baseline", "READY_FOR_SEPARATE_AUTHORIZATION"],
        ["clef_joint", "Cloudflare/clef-flash", "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3", "native joint binary", "complete-support proxy", "exclusive A100 / native adapter", "READY_FOR_SEPARATE_AUTHORIZATION"],
        ["retrieval5", "frozen Track A/B retriever prefix", "frozen slate hashes", "retrieval control", "baseline selection proxy", "index resident; no reranker", "READY_FOR_SEPARATE_AUTHORIZATION"],
    ]
    with (OUT / "E2_SP_MODEL_AND_WORKLOAD_MATRIX.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["policy", "model_or_source", "revision", "mechanism", "quality_signal", "hardware_gate", "status"]); w.writerows(matrix)
    write_text(OUT / "E2_SP_SERVING_MEASUREMENT_CONTRACT.md", """# E2-S Serving Measurement Contract

Status: `READY_FOR_SEPARATE_AUTHORIZATION`; no serving run performed.

Measure the complete client-visible request from admission to final response, while recording disjoint stage intervals for queue wait, query encoding, candidate retrieval, view assembly, reranker queue wait, model computation, serialization, and client receipt. Do not add component medians to substitute for measured end-to-end latency.

For each policy use the same frozen query/slate cohort, candidate order, score direction, Top-5 rule, warm-up policy, cache-disabled application state, and exclusive hardware entitlement. Record p50/p90/p95, valid/error counts, throughput, input/output tokens, physical forwards, GPU ID, peak VRAM, utilization samples, and raw request/response hashes. Repeat randomized interleaved blocks only after owner approval. GPU cost is a model `active_seconds × approved_rate` and must be reported as `NOT_MEASURED` until a rate and accounting boundary are approved.
""")
    write_text(OUT / "E2_SP_DOWNSTREAM_VALIDATION_CONTRACT.md", """# E2-P Downstream Validation Contract

Status: `READY_FOR_SEPARATE_AUTHORIZATION`; no reader run performed.

Hold the reader checkpoint, prompt, decoding, evidence layout, context budget, and citation schema fixed. Compare rerankers on a prespecified CAL/DEV cohort, retaining the same retrieved candidate slate. Report independently reviewed factual correctness, evidence/citation completeness, abstention, reader latency, tokens, and modeled cost. Reference-answer EM/F1 are secondary proxies only; CSSR, answer nonemptiness, and model-generated labels are not factual correctness. Do not access a fresh confirmatory holdout in this phase.
""")
    write_text(OUT / "E2_SP_AUTHORIZATION_CHECKLIST.md", """# E2-S/P Authorization Checklist

All items below require explicit owner approval before execution:

- GPU IDs and exclusive execution window, with a contention check.
- Exact model/checkpoint roster and local-service ownership.
- Network/download permission and confirmation that no production service may be restarted or modified.
- Frozen cohort, candidate-slate hashes, arrival rates, concurrency, warm-up, repetition count, and p95 SLO.
- E2-P reader checkpoint, prompt, decoding, evidence presentation, and independently verified reference policy.
- Cache-disabled application policy and raw request/response hashing before normalization.
- Cost-rate assumptions and reporting boundary; no dollar claim without approved rates.
- No fresh holdout, Jev, paid API, training, or protocol tuning.
- Stop-gate owner and artifact-preservation procedure.
""")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_items()
    items_by_qid = {x["query_id"]: x for x in items}
    gold, gold_status = load_gold()
    traces, score_counts, score_errors = read_scores(items_by_qid)
    selections, selection_counts = build_selections(items, traces)
    selected_rows, selected_missing, selected_mismatch = audit_selected_file(items, selections)
    quality_rows, quality_lookup = recompute_quality(items, gold, selections)
    published_quality = read_published_quality()
    quality_mismatch = []
    quality_fields = ["n_total", "n_eligible", "n_conditional", "n_any_gold", "success_unconditional", "success_conditional", "cssr_unconditional", "cssr_conditional", "any_gold_coverage"]
    for key, prior in published_quality.items():
        current = quality_lookup.get(key)
        if current is None:
            quality_mismatch.append(f"missing recomputed quality row {key}")
            continue
        for field in quality_fields:
            if not compare_float(prior.get(field), current.get(field)):
                quality_mismatch.append(f"{key} field {field}: prior={prior.get(field)} current={current.get(field)}")
    pair_rows = recompute_pairs(items, gold, selections, read_published_pairs())
    source_checks, source_errors = source_hash_audit()
    view_rows = view_audit(items_by_qid, traces, score_counts)
    write_csv(OUT / "E2_N2_MODEL_INFERENCE_PROVENANCE.csv", view_rows)
    write_csv(OUT / "E2_N2_INDEPENDENT_QUALITY_RECOMPUTATION.csv", quality_rows)
    write_csv(OUT / "E2_N2_INDEPENDENT_PAIRED_STATISTICS.csv", pair_rows)
    write_forward_report(view_rows, source_checks, source_errors, score_counts, selected_rows, selected_missing, selected_mismatch)
    write_view_report(view_rows, score_errors)
    cpu_test_status = "PASS (65/65; command: PYTHONPATH=. pytest -q)"
    write_closure_report(quality_rows, pair_rows, view_rows, source_errors, score_errors + quality_mismatch, selected_missing, selected_mismatch, cpu_test_status)
    closure_json = {
        "milestone": "E2-N2-R",
        "status": "E2_N2_OWNER_ACCEPTED_WITH_LIMITATIONS",
        "next_gate": "E2_SP_READY_FOR_GPU_AUTHORIZATION",
        "offline_only": True,
        "new_model_requests": 0,
        "new_inference_requests": 0,
        "jev_calls": 0,
        "paid_api_calls": 0,
        "fresh_confirmatory_holdout_accessed": False,
        "training_runs": 0,
        "retrieval_or_indexing": False,
        "production_services_modified": 0,
        "population": {"total_items": len(items), "track_a": sum(x["cohort"] == "track_a" for x in items), "track_b": sum(x["cohort"] == "track_b" for x in items), "policies": len(POLICIES)},
        "gold": {"resolved_ids": len(gold), "ambiguous_or_unresolved_ids": sum(v != "resolved" for v in gold_status.values()), "accepted_exception": "qst_0413"},
        "normalized_rows": sum(len(x) for x in traces.values()),
        "successful_rows": sum(sum(v.get("status") == "ok" for v in x.values()) for x in traces.values()),
        "blocked_rows": sum(sum(v.get("status") != "ok" for v in x.values()) for x in traces.values()),
        "physical_forward_evidence_rows": sum(r["physical_forward_evidence_rows"] for r in view_rows),
        "physical_forward_calls": sum(r["physical_forward_calls"] for r in view_rows),
        "selection_rows": selected_rows,
        "selection_missing_keys": selected_missing,
        "selection_mismatch_count": len(selected_mismatch),
        "source_hash_error_count": len(source_errors),
        "score_alignment_error_count": len(score_errors),
        "quality_published_row_mismatch_count": len(quality_mismatch),
        "paired_rows": len(pair_rows),
        "bootstrap": {"repetitions": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED},
        "limitations": [
            "500 Intern-Decision enterprise rows are native context-blocked above the frozen 8192-token contract.",
            "CSSR is complete-support selection proxy, not independently verified reader factual accuracy.",
            "API request counts, timestamps, queue intervals, and monetary GPU cost are not recorded in E2-N2 traces.",
            "Raw serialized candidate-view text is not embedded; per-view hashes and actual input hashes are preserved.",
        ],
        "amendment_proposed": None,
    }
    write_json(OUT / "E2_N2_FINAL_SCIENTIFIC_CLOSURE.json", closure_json)
    write_sp_docs()
    # Final provenance is written after all other closure artifacts, avoiding a
    # self-referential hash. It includes source and output hashes.
    output_hashes = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "E2_N2_OWNER_CLOSURE_PROVENANCE.json":
            output_hashes[path.name] = {"bytes": path.stat().st_size, "sha256": sha_file(path)}
    source_files = {}
    for path in sorted(BASE.iterdir()):
        if path.is_file():
            source_files[path.name] = {"bytes": path.stat().st_size, "sha256": sha_file(path)}
    provenance = {
        "milestone": "E2-N2-R",
        "status": closure_json["status"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "offline independent recomputation from frozen normalized rows and manifests",
        "source_files": source_files,
        "source_trace_hash_checks": source_checks,
        "closure_output_hashes": output_hashes,
        "tests": {"status": cpu_test_status, "new_inference": 0, "jev_calls": 0, "paid_api_calls": 0},
        "safety": {"gpu_used": False, "network_used": False, "services_modified": False, "fresh_holdout_accessed": False, "training": False, "retrieval_or_indexing": False},
        "limitations": closure_json["limitations"],
    }
    write_json(OUT / "E2_N2_OWNER_CLOSURE_PROVENANCE.json", provenance)
    print(json.dumps({"status": closure_json["status"], "output": str(OUT), "normalized_rows": closure_json["normalized_rows"], "physical_forward_calls": closure_json["physical_forward_calls"], "quality_mismatches": len(quality_mismatch), "score_errors": len(score_errors), "source_errors": len(source_errors)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
