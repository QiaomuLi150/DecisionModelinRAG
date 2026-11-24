#!/usr/bin/env python3
"""DecisionRAG v1.5 C2: CPU-only latency and fallback reconciliation.

This script consumes already-frozen historical traces.  It performs no model
loading, network access, inference, or mutation of historical artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pathlib
import time
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v1_5/c2_latency_uncertainty/export"
OUT.mkdir(parents=True, exist_ok=True)


SOURCE_FILES = [
    ("r1_v2", "pplx_budget", "results/v1_3/r1_serving_gate_v2/raw/pplx_budget_R1_RAW_V2.jsonl"),
    ("r1_v2", "pplx_two_stage", "results/v1_3/r1_serving_gate_v2/raw/pplx_two_stage_R1_RAW_V2.jsonl"),
    ("r1_v2", "jina_full", "results/v1_3/r1_serving_gate_v2/raw/jina_full_R1_RAW_V2.jsonl"),
    ("r1_v2", "jina_pplx_budget", "results/v1_3/r1_serving_gate_v2/raw/jina_pplx_budget_R1_RAW_V2.jsonl"),
    ("e12_o3", "jina_v3_5", "results/non_jev/v2/e12_o_final_v1_0/03_serving/e12o_o3_attempt_01/jina_v3_5_SERVING_RAW_V2.jsonl"),
    ("e12_o3", "winnow12", "results/non_jev/v2/e12_o_final_v1_0/03_serving/e12o_o3_attempt_01/winnow12_SERVING_RAW_V2.jsonl"),
    ("e12_o3", "drex_dlm8", "results/non_jev/v2/e12_o_final_v1_0/03_serving/e12o_o3_attempt_01/drex_dlm8_SERVING_RAW_V2.jsonl"),
    ("e12_o3", "kev9", "results/non_jev/v2/e12_o_final_v1_0/03_serving/e12o_o3_attempt_01/kev9_SERVING_RAW_V2.jsonl"),
    ("e12_o3", "qwen3_reranker_0_6b", "results/non_jev/v2/e12_o_final_v1_0/03_serving/e12o_o3_attempt_01/qwen3_reranker_0_6b_SERVING_RAW_V2.jsonl"),
    ("e12_o3", "mxbai_v2", "results/non_jev/v2/e12_o_final_v1_0/03_serving/e12o_o3_attempt_01/mxbai_v2_SERVING_RAW_V2.jsonl"),
    ("reviewer_liquid", "liquid_d1_3b", "results/non_jev/v2/reviewer_response_final_v1_1_blocker_resolution/02_LIQUID_SERVING_RAW_ALL_ATTEMPTS.jsonl"),
]

BRIGHT_SLATES = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def relative(path: pathlib.Path) -> str:
    return path.relative_to(ROOT).as_posix()


def status_ok(row: dict[str, Any]) -> bool:
    return str(row.get("status", "")).upper() in {"PASS", "OK", "SUCCESS", "COMPLETED"}


def error_class(row: dict[str, Any]) -> str:
    err = row.get("error") or row.get("failure") or row.get("error_class")
    if not err:
        return ""
    return str(err).split(":", 1)[0]


def get_latency(row: dict[str, Any]) -> float | None:
    value = row.get("elapsed_ms", row.get("duration_ms"))
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def get_alloc(row: dict[str, Any]) -> float | None:
    value = row.get("max_memory_allocated_mib")
    if value is None:
        value = row.get("peak_memory_allocated_mib", row.get("memory_allocated_mib"))
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def attempt_value(row: dict[str, Any]) -> int:
    value = row.get("repeat", row.get("rep", row.get("attempt", 1)))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def normalize(session: str, policy: str, row: dict[str, Any], source: pathlib.Path) -> dict[str, Any]:
    return {
        "session": session,
        "policy": policy,
        "canonical_id": str(row.get("canonical_id", row.get("query_id", ""))),
        "attempt": attempt_value(row),
        "status": "ok" if status_ok(row) else "failed",
        "error_class": error_class(row),
        "latency_ms": get_latency(row),
        "alloc_peak_mib": get_alloc(row),
        "_source": relative(source),
    }


def percentile(values: Iterable[float], q: float) -> float | None:
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return None
    return float(np.percentile(vals, q, method="linear"))


def write_csv(path: pathlib.Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_fallback() -> tuple[dict[tuple[str, int], float], dict[str, Any]]:
    slates = {
        f"{row.get('domain', '')}:{row.get('query_id', '')}": row
        for row in read_jsonl(BRIGHT_SLATES)
    }
    r1_path = ROOT / SOURCE_FILES[0][2]
    r1_rows = read_jsonl(r1_path)
    out_rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, int], float] = {}
    missing = 0
    mismatched = 0
    for row in r1_rows:
        cid = str(row.get("canonical_id", ""))
        start = time.perf_counter_ns()
        slate = slates.get(cid)
        if slate is None:
            candidate_ids = list(row.get("candidate_ids") or [])[:5]
            missing += 1
        else:
            candidate_ids = [str(x.get("candidate_id")) for x in slate.get("candidates", [])[:5]]
            if list(row.get("stage_selected_ids") or []) and not row.get("candidate_ids"):
                # R1 stores its view candidates in stage_views rather than a
                # top-level list; this is only an identity audit, not scoring.
                pass
        _ = tuple(candidate_ids)
        elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
        key = (cid, attempt_value(row))
        lookup[key] = elapsed
        out_rows.append({
            "canonical_id": cid,
            "attempt": attempt_value(row),
            "candidate_count": len(candidate_ids),
            "fallback_ms": elapsed,
            "status": "ok" if len(candidate_ids) >= 5 else "invalid",
        })
        if slate is not None and list(row.get("candidate_ids") or []):
            if list(row["candidate_ids"]) != list(slate.get("candidate_ids") or []):
                mismatched += 1
    write_csv(OUT / "fallback_cost.csv", ["canonical_id", "attempt", "candidate_count", "fallback_ms", "status"], out_rows)
    return lookup, {
        "source": relative(BRIGHT_SLATES),
        "source_sha256": sha256(BRIGHT_SLATES),
        "n_requests": len(out_rows),
        "n_unique_questions": len({r["canonical_id"] for r in out_rows}),
        "missing_slate": missing,
        "candidate_order_mismatches": mismatched,
        "timing_definition": "CPU list extraction and first-five packaging from frozen BM25 Top-20; no model inference",
    }


def p95_ratio_bootstrap(
    values_a: dict[str, list[float]],
    values_b: dict[str, list[float]],
    draws: int = 10_000,
    seed: int = 731,
) -> tuple[float | None, float | None, float | None, float | None, float | None, int]:
    queries = sorted(set(values_a) & set(values_b))
    if not queries:
        return (None, None, None, None, None, 0)
    a = [v for q in queries for v in values_a[q]]
    b = [v for q in queries for v in values_b[q]]
    if not a or not b:
        return (None, None, None, None, None, len(queries))
    point = percentile(a, 95.0)
    base = percentile(b, 95.0)
    ratio = point / base if point is not None and base else None
    # Pad each query's repeated-request values and use a vectorized bootstrap.
    # This keeps the prespecified 10,000 query-cluster draws reproducible while
    # avoiding a Python loop over every request in every draw.
    max_a = max(len(values_a[q]) for q in queries)
    max_b = max(len(values_b[q]) for q in queries)
    mat_a = np.full((len(queries), max_a), np.nan, dtype=float)
    mat_b = np.full((len(queries), max_b), np.nan, dtype=float)
    for i, q in enumerate(queries):
        mat_a[i, : len(values_a[q])] = values_a[q]
        mat_b[i, : len(values_b[q])] = values_b[q]
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(queries), size=(draws, len(queries)))
    sampled_a = mat_a[sampled]
    sampled_b = mat_b[sampled]
    boot_a = np.nanpercentile(sampled_a, 95.0, axis=(1, 2))
    boot_b = np.nanpercentile(sampled_b, 95.0, axis=(1, 2))
    valid = np.isfinite(boot_a) & np.isfinite(boot_b) & (boot_b != 0)
    if not np.any(valid):
        return (ratio, None, None, None, None, len(queries))
    arr = boot_a[valid] / boot_b[valid]
    return (
        ratio,
        float(np.percentile(arr, 2.5, method="linear")),
        float(np.percentile(arr, 97.5, method="linear")),
        float(np.mean(arr <= 2.0)),
        float(np.median(arr)),
        len(queries),
    )


def main() -> None:
    normalized: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    for session, policy, relpath in SOURCE_FILES:
        path = ROOT / relpath
        rows = read_jsonl(path)
        source_manifest.append({"path": relpath, "sha256": sha256(path), "records": len(rows)})
        normalized.extend(normalize(session, policy, row, path) for row in rows)

    write_csv(
        OUT / "latency_perrequest.csv",
        ["session", "policy", "canonical_id", "attempt", "status", "error_class", "latency_ms", "alloc_peak_mib"],
        ({k: ("" if v is None else v) for k, v in row.items()} for row in normalized),
    )

    fallback_lookup, fallback_meta = build_fallback()
    fallback_by_query = defaultdict(dict)
    for (cid, attempt), ms in fallback_lookup.items():
        fallback_by_query[cid][attempt] = ms

    effective: list[dict[str, Any]] = []
    for row in normalized:
        item = dict(row)
        fb = fallback_lookup.get((row["canonical_id"], row["attempt"]))
        item["fallback_ms"] = fb
        if row["latency_ms"] is None:
            item["effective_latency_ms"] = None
        elif row["status"] == "ok":
            item["effective_latency_ms"] = row["latency_ms"]
        else:
            item["effective_latency_ms"] = row["latency_ms"] + (fb or 0.0)
        effective.append(item)

    summary: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in effective:
        grouped[(row["session"], row["policy"])].append(row)
    for (session, policy), rows in sorted(grouped.items()):
        timed = [r["effective_latency_ms"] for r in rows if r["effective_latency_ms"] is not None]
        completed = [r["effective_latency_ms"] for r in rows if r["status"] == "ok" and r["effective_latency_ms"] is not None]
        summary.append({
            "session": session,
            "policy": policy,
            "n_requests": len(rows),
            "n_completed": sum(r["status"] == "ok" for r in rows),
            "n_failed": sum(r["status"] != "ok" for r in rows),
            "n_timed": len(timed),
            "n_untimed": len(rows) - len(timed),
            "p50_ms_all_request_timed": percentile(timed, 50),
            "p95_ms_all_request_timed": percentile(timed, 95),
            "p50_ms_completed": percentile(completed, 50),
            "p95_ms_completed": percentile(completed, 95),
            "failure_accounting": "failed timed requests include latency plus BM25 fallback; untimed failures are retained and excluded from percentiles",
        })
    write_csv(
        OUT / "latency_policy_summary.csv",
        ["session", "policy", "n_requests", "n_completed", "n_failed", "n_timed", "n_untimed", "p50_ms_all_request_timed", "p95_ms_all_request_timed", "p50_ms_completed", "p95_ms_completed", "failure_accounting"],
        summary,
    )

    ratios: list[dict[str, Any]] = []
    baselines = {"r1_v2": "jina_full", "e12_o3": "jina_v3_5"}
    for session, baseline in baselines.items():
        policies = sorted({r["policy"] for r in effective if r["session"] == session})
        for policy in policies:
            if policy == baseline:
                continue
            for basis in ("completed", "all_request"):
                va: dict[str, list[float]] = defaultdict(list)
                vb: dict[str, list[float]] = defaultdict(list)
                for row in effective:
                    if row["session"] != session:
                        continue
                    use = basis == "all_request" or row["status"] == "ok"
                    val = row["effective_latency_ms"]
                    if not use or val is None:
                        continue
                    (va if row["policy"] == policy else vb if row["policy"] == baseline else defaultdict(list))[row["canonical_id"]].append(float(val))
                point, lo, hi, prob, median, nq = p95_ratio_bootstrap(va, vb)
                ratios.append({
                    "session": session,
                    "policy": policy,
                    "baseline": baseline,
                    "basis": basis,
                    "n_common_queries": nq,
                    "p50_ms_policy": percentile([x for q in va.values() for x in q], 50),
                    "p95_ms_policy": percentile([x for q in va.values() for x in q], 95),
                    "p50_ms_baseline": percentile([x for q in vb.values() for x in q], 50),
                    "p95_ms_baseline": percentile([x for q in vb.values() for x in q], 95),
                    "p95_ratio": point,
                    "ratio_lo95": lo,
                    "ratio_hi95": hi,
                    "prob_ratio_le_2": prob,
                    "bootstrap_median_ratio": median,
                    "bootstrap_draws": 10000,
                    "bootstrap_seed": 731,
                    "note": "query-cluster bootstrap; untimed records are not imputed",
                })
    write_csv(
        OUT / "latency_ratios.csv",
        ["session", "policy", "baseline", "basis", "n_common_queries", "p50_ms_policy", "p95_ms_policy", "p50_ms_baseline", "p95_ms_baseline", "p95_ratio", "ratio_lo95", "ratio_hi95", "prob_ratio_le_2", "bootstrap_median_ratio", "bootstrap_draws", "bootstrap_seed", "note"],
        ratios,
    )

    input_meta = {
        "status": "COMPLETE_OFFLINE",
        "inference_calls": 0,
        "source_traces": source_manifest,
        "frozen_slate": fallback_meta,
        "normalized_records": len(normalized),
        "policies": sorted({(r["session"], r["policy"]) for r in normalized}),
        "untimed_policy_records": sum(r["latency_ms"] is None for r in normalized),
        "fallback_rule": "failed timed request effective latency = recorded request latency + measured CPU BM25 Top-5 fallback; no latency is imputed for untimed requests",
        "historical_artifacts_unchanged": True,
    }
    (OUT / "C2_INPUT_INTEGRITY.json").write_text(json.dumps(input_meta, indent=2, sort_keys=True) + "\n")
    (OUT / "C2_RAW_TRACE_MANIFEST.json").write_text(json.dumps({"records": source_manifest}, indent=2, sort_keys=True) + "\n")
    (OUT / "C2_RUN_RECEIPT.json").write_text(json.dumps({
        "status": "COMPLETE_OFFLINE",
        "script": "scripts/v1_5_c2_offline.py",
        "inference_calls": 0,
        "gpu_calls": 0,
        "outputs": ["latency_perrequest.csv", "latency_policy_summary.csv", "latency_ratios.csv", "fallback_cost.csv"],
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
