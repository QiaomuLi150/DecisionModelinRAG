#!/usr/bin/env python3
"""Freeze model-neutral Top-5 selections from the completed E2-S/P traces.

This is an offline operation.  It does not load a model, read qrels, or make
any inference request.  Every policy receives the exact frozen Top-20 slate;
the only model-dependent operation is deterministic sorting of the already
recorded score vector.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
E2 = ROOT / "results/non_jev/v2/e2_sp"
OUT = E2 / "gpu5"
COHORT_PATH = E2 / "E2_SP_READER_COHORT_MANIFEST.json"
SELECTION_PATH = OUT / "E2_SP_S2_SELECTION_RESULTS.jsonl"
POLICY_PATH = OUT / "E2_SP_GPU5_POLICY_MANIFEST.json"
LATENCY_PATH = OUT / "E2_SP_SELECTION_LATENCY_SUMMARY.csv"
CAPACITY_PATH = OUT / "E2_SP_CAPACITY_AND_COST_STATUS.csv"

TRACE_FILES = {
    "bge_pointwise": "E2_SP_S2_FULL_BGE_POINTWISE.jsonl",
    "qwen06": "E2_SP_S2_FULL_QWEN06.jsonl",
    "qwen4": "E2_SP_S2_FULL_QWEN4.jsonl",
    "jina35": "E2_SP_S2_FULL_JINA35.jsonl",
    "gte": "E2_SP_S2_FULL_GTE.jsonl",
    "minilm": "E2_SP_S2_FULL_MINILM.jsonl",
    "clef_joint": "E2_SP_S2_FULL_CLEF_JOINT.jsonl",
    "nimble": "E2_SP_S2_FULL_NIMBLE.jsonl",
    "pplx_decider": "E2_SP_S2_FULL_PPLX_DECIDER.jsonl",
    "qwen_direct": "E2_SP_S2_FULL_QWEN_DIRECT.jsonl",
}
POLICIES = ["retrieval5", *TRACE_FILES.keys()]
SLOS_MS = (1000.0, 2000.0, 5000.0)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * p / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def duration_ms(row: dict[str, Any]) -> float | None:
    value = row.get("duration_ms")
    if value is None and isinstance(row.get("telemetry"), dict):
        value = row["telemetry"].get("duration_ms")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def trace_model(row: dict[str, Any]) -> str | None:
    return row.get("model_id")


def trace_revision(row: dict[str, Any]) -> str | None:
    return row.get("revision")


def selected_ids_for_scores(candidate_ids: list[str], scores: list[Any]) -> tuple[list[str], list[float]]:
    if len(candidate_ids) != 20 or len(set(candidate_ids)) != 20:
        raise ValueError("candidate slate must contain exactly 20 unique IDs")
    if len(scores) != len(candidate_ids):
        raise ValueError("score vector is not aligned with candidate IDs")
    numeric: list[float] = []
    for score in scores:
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("non-finite score in completed trace")
        numeric.append(value)
    original_rank = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    order = sorted(
        range(len(candidate_ids)),
        key=lambda index: (-numeric[index], original_rank[candidate_ids[index]], candidate_ids[index]),
    )
    indices = order[:5]
    return [candidate_ids[index] for index in indices], [numeric[index] for index in indices]


def read_cohort() -> list[dict[str, Any]]:
    manifest = json.loads(COHORT_PATH.read_text(encoding="utf-8"))
    records = manifest["query_records"]
    if len(records) != 300 or len({record["query_id"] for record in records}) != 300:
        raise ValueError("frozen cohort is not exactly 300 unique queries")
    return records


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = read_cohort()
    cohort_by_id = {record["query_id"]: record for record in cohort}
    cohort_ids = set(cohort_by_id)

    traces: dict[str, list[dict[str, Any]]] = {}
    trace_hashes: dict[str, str] = {}
    for policy, filename in TRACE_FILES.items():
        path = OUT / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = load_jsonl(path)
        if len(rows) != 300 or {row.get("query_id") for row in rows} != cohort_ids:
            raise ValueError(f"{policy} trace is not a complete exact cohort")
        if any(row.get("status") != "ok" for row in rows):
            raise ValueError(f"{policy} contains non-ok rows; selection freeze is blocked")
        traces[policy] = {row["query_id"]: row for row in rows}
        trace_hashes[policy] = sha_file(path)

    selections: list[dict[str, Any]] = []
    for record in cohort:
        query_id = record["query_id"]
        candidate_ids = list(record["top20_candidate_ids"])
        baseline_ids = candidate_ids[:5]
        base = {
            "query_id": query_id,
            "dataset": record["dataset"],
            "cohort": record["cohort"],
            "candidate_ids": candidate_ids,
            "candidate_order": "frozen_top20_original_order",
            "selected_ids": baseline_ids,
            "selected_ids_sha256": sha_json(baseline_ids),
            "score_type": "retrieval_order",
            "score_formulation": "No reranker score; frozen Top-20 prefix.",
            "scores": None,
            "status": "ok",
            "policy": "retrieval5",
            "source_trace_sha256": None,
            "source_row_input_sha256": None,
            "source_row_response_sha256": None,
            "tie_breaking": "not_applicable",
            "gold_in_selection_input": False,
            "cache_reuse": False,
        }
        selections.append(base)

        for policy in TRACE_FILES:
            row = traces[policy][query_id]
            selected, selected_scores = selected_ids_for_scores(candidate_ids, row["scores"])
            score_contract = row.get("score_contract") or row.get("score_direction") or "native recorded score vector"
            selections.append({
                "query_id": query_id,
                "dataset": record["dataset"],
                "cohort": record["cohort"],
                "candidate_ids": candidate_ids,
                "candidate_order": "frozen_top20_original_order",
                "selected_ids": selected,
                "selected_ids_sha256": sha_json(selected),
                "score_type": "native_recorded_score",
                "score_formulation": score_contract,
                "scores": selected_scores,
                "status": "ok",
                "policy": policy,
                "source_trace_sha256": trace_hashes[policy],
                "source_row_input_sha256": row.get("actual_model_input_sha256"),
                "source_row_response_sha256": row.get("raw_response_sha256"),
                "tie_breaking": "descending_score; original_frozen_rank; canonical_candidate_id",
                "gold_in_selection_input": bool(row.get("gold_in_inference_input", False)),
                "cache_reuse": bool(row.get("cache_reuse", False)),
            })

    if len(selections) != 300 * len(POLICIES):
        raise AssertionError("selection row count mismatch")
    with SELECTION_PATH.open("w", encoding="utf-8") as handle:
        for row in selections:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    policy_manifest = {
        "manifest_version": "E2-SP-GPU5-POLICY-v1",
        "status": "FROZEN_OFFLINE_SELECTIONS",
        "cohort_manifest": str(COHORT_PATH.relative_to(ROOT)),
        "cohort_manifest_sha256": sha_file(COHORT_PATH),
        "query_count": 300,
        "policy_count": len(POLICIES),
        "policies": POLICIES,
        "selection_contract": "exactly five unique IDs from each immutable Top-20; descending native score; original rank then canonical ID tie-break",
        "retrieval5_contract": "first five IDs of the immutable Top-20; model selection latency NOT_MEASURED",
        "trace_sha256": trace_hashes,
        "selection_results_sha256": sha_file(SELECTION_PATH),
        "row_count": len(selections),
        "gold_accessed": False,
        "cache_reuse": False,
        "jev_calls": 0,
        "paid_api_calls": 0,
    }
    POLICY_PATH.write_text(json.dumps(policy_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    latency_rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        if policy == "retrieval5":
            row = {
                "policy": policy,
                "n": 300,
                "n_ok": 300,
                "n_error": 0,
                "p50_ms": "NOT_MEASURED",
                "p90_ms": "NOT_MEASURED",
                "p95_ms": "NOT_MEASURED",
                "p99_ms": "NOT_MEASURED",
                "mean_ms": "NOT_MEASURED",
                "min_ms": "NOT_MEASURED",
                "max_ms": "NOT_MEASURED",
                "slo_1000_hit_rate": "NOT_MEASURED",
                "slo_2000_hit_rate": "NOT_MEASURED",
                "slo_5000_hit_rate": "NOT_MEASURED",
                "mean_input_tokens": "NOT_MEASURED",
                "physical_forward_rows": "NOT_APPLICABLE",
                "source_trace_sha256": "NOT_APPLICABLE",
                "measurement_boundary": "frozen retrieval prefix only; no selection model call was timed",
            }
        else:
            rows = list(traces[policy].values())
            values = [value for row0 in rows if (value := duration_ms(row0)) is not None]
            token_values = []
            for row0 in rows:
                try:
                    value = float(row0.get("input_tokens"))
                    if math.isfinite(value):
                        token_values.append(value)
                except (TypeError, ValueError):
                    pass
            physical = sum(1 for row0 in rows if row0.get("physical_forward_evidence") is True and int(row0.get("physical_forward_calls", 0)) > 0)
            row = {
                "policy": policy,
                "n": len(rows),
                "n_ok": sum(row0.get("status") == "ok" for row0 in rows),
                "n_error": sum(row0.get("status") != "ok" for row0 in rows),
                "p50_ms": percentile(values, 50),
                "p90_ms": percentile(values, 90),
                "p95_ms": percentile(values, 95),
                "p99_ms": percentile(values, 99),
                "mean_ms": statistics.fmean(values) if values else None,
                "min_ms": min(values) if values else None,
                "max_ms": max(values) if values else None,
                "slo_1000_hit_rate": sum(value <= 1000.0 for value in values) / len(values) if values else None,
                "slo_2000_hit_rate": sum(value <= 2000.0 for value in values) / len(values) if values else None,
                "slo_5000_hit_rate": sum(value <= 5000.0 for value in values) / len(values) if values else None,
                "mean_input_tokens": statistics.fmean(token_values) if token_values else None,
                "physical_forward_rows": physical,
                "source_trace_sha256": trace_hashes[policy],
                "measurement_boundary": "native selection request; serialized single-client trace duration",
            }
        latency_rows.append(row)

    with LATENCY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(latency_rows[0]))
        writer.writeheader()
        writer.writerows(latency_rows)

    capacity_rows = []
    for policy in POLICIES:
        capacity_rows.append({
            "policy": policy,
            "capacity_sweep_status": "NOT_MEASURED",
            "arrival_rates_rps": "NOT_MEASURED",
            "stable_window_throughput_rps": "NOT_MEASURED",
            "queue_p95_ms": "NOT_MEASURED",
            "gpu_active_seconds": "NOT_MEASURED",
            "gpu_cost_usd": "NOT_MEASURED",
            "cost_basis": "No monetary rate or open-loop sweep was established in S2; do not infer cost from single-client duration.",
            "note": "S2 completed serialized quality-selection traces only; capacity/cost remains an explicit follow-up gate.",
        })
    with CAPACITY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(capacity_rows[0]))
        writer.writeheader()
        writer.writerows(capacity_rows)

    print(json.dumps({
        "status": "PASS",
        "selection_results": str(SELECTION_PATH),
        "policy_manifest": str(POLICY_PATH),
        "latency_summary": str(LATENCY_PATH),
        "capacity_status": str(CAPACITY_PATH),
        "rows": len(selections),
        "policies": POLICIES,
    }, indent=2))


if __name__ == "__main__":
    main()
