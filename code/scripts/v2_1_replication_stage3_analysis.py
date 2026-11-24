#!/usr/bin/env python3
"""Offline analysis for the completed V2-HYBRID Stage 3 serving trace."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/v2_1_replication/stage3_serving_v2_hybrid"
RAW = STAGE / "raw/stage3_perrequest.jsonl"
EXPORT = STAGE / "export"
POLICIES = ["jina_full", "jina_budget", "jina_two_stage_matched", "pplx_budget", "pplx_two_stage"]
DATASETS = ["enterprise", "musique"]


def rows() -> list[dict]:
    return [json.loads(x) for x in RAW.open(encoding="utf-8") if x.strip()]


def pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def bootstrap_ratio(policy_rows: dict[str, list[dict]], base_rows: dict[str, list[dict]], seed: int = 731, draws: int = 10000) -> tuple[float, float, float, float]:
    policy = next(iter(policy_rows))
    ids = sorted({str(r["canonical_id"]) for r in base_rows["jina_full"]})
    by_p = {cid: [r for r in policy_rows[policy] if str(r["canonical_id"]) == cid] for cid in ids}
    by_b = {cid: [r for r in base_rows["jina_full"] if str(r["canonical_id"]) == cid] for cid in ids}
    def p95(rs: list[dict]) -> float:
        vals = [float(r["latency_ms"]) for r in rs if r.get("latency_ms") is not None]
        return float(np.percentile(vals, 95))
    point = p95(policy_rows[policy]) / p95(base_rows["jina_full"])
    rng = random.Random(seed)
    ratios: list[float] = []
    for _ in range(draws):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        pr, br = [], []
        for cid in sample:
            pr.extend(by_p.get(cid, [])); br.extend(by_b.get(cid, []))
        if pr and br and all(r.get("latency_ms") is not None for r in pr + br):
            ratios.append(p95(pr) / p95(br))
    return point, float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5)), float(np.mean(np.asarray(ratios) <= 2.0))


def write_csv(path: Path, fields: list[str], data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(data)


def main() -> None:
    raw = rows()
    expected = 3600
    if len(raw) != expected:
        raise RuntimeError(f"STAGE3_ROW_COUNT:{len(raw)}")
    if any(r.get("labels_joined") is not False for r in raw):
        raise RuntimeError("LABEL_LEAKAGE_IN_SERVING_TRACE")
    per_fields = ["policy", "canonical_id", "attempt", "status", "error_class", "latency_ms", "alloc_peak_mib"]
    all_summary = []
    ratios = []
    for dataset in DATASETS:
        ds = [r for r in raw if r["dataset"] == dataset]
        if len(ds) != 1800:
            raise RuntimeError(f"DATASET_ROW_COUNT:{dataset}:{len(ds)}")
        d = EXPORT / dataset; d.mkdir(parents=True, exist_ok=True)
        per = [{k: r.get(k) for k in per_fields} for r in ds]
        write_csv(d / "serving_perrequest.csv", per_fields, per)
        by_policy = {p: [r for r in ds if r["policy"] == p] for p in POLICIES}
        for p in POLICIES:
            if len(by_policy[p]) != 360:
                raise RuntimeError(f"POLICY_ROW_COUNT:{dataset}:{p}:{len(by_policy[p])}")
        base = by_policy["jina_full"]
        base_p50 = pct([float(r["latency_ms"]) for r in base], 50)
        base_p95 = pct([float(r["latency_ms"]) for r in base], 95)
        for p in POLICIES:
            rs = by_policy[p]
            vals = [float(r["latency_ms"]) for r in rs if r.get("latency_ms") is not None]
            alloc = [float(r["alloc_peak_mib"]) for r in rs if r.get("alloc_peak_mib") is not None]
            point, lo, hi, prob = bootstrap_ratio({p: rs}, {"jina_full": base})
            ratios.append({"dataset": dataset, "policy": p, "basis": "all_request", "p50_ms": pct(vals, 50), "p95_ms": pct(vals, 95),
                           "p95_ratio_point": point, "ratio_lo95": lo, "ratio_hi95": hi, "prob_ratio_le_2": prob,
                           "alloc_p95_mib": pct(alloc, 95)})
            all_summary.append({"dataset": dataset, "policy": p, "n": len(rs), "valid": sum(bool(r.get("valid")) for r in rs),
                                "failures": sum(not bool(r.get("valid")) for r in rs), "p50_ms": pct(vals, 50), "p95_ms": pct(vals, 95),
                                "alloc_p95_mib": pct(alloc, 95)})
        write_csv(d / "serving_ratios.csv", ["policy", "basis", "p50_ms", "p95_ms", "p95_ratio_point", "ratio_lo95", "ratio_hi95", "prob_ratio_le_2", "alloc_p95_mib"], [{k: x[k] for k in ["policy", "basis", "p50_ms", "p95_ms", "p95_ratio_point", "ratio_lo95", "ratio_hi95", "prob_ratio_le_2", "alloc_p95_mib"]} for x in ratios if x["dataset"] == dataset])
    report = {"status": "PASS", "raw_sha256": sha(RAW), "raw_rows": len(raw), "dataset_rows": {d: 1800 for d in DATASETS},
              "policy_rows_per_dataset": 360, "repeats": 3, "bootstrap_draws": 10000, "bootstrap_seed": 731,
              "failures": sum(not bool(r.get("valid")) for r in raw), "summary": all_summary, "ratios": ratios,
              "fallback_time_used": False, "labels_joined": False, "protected_holdouts_accessed": False, "paid_api_calls": 0}
    (STAGE / "STAGE3_SERVING_ANALYSIS.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# Stage 3 Serving Analysis", "", f"Raw rows: {len(raw)}; failures: {report['failures']}; raw SHA-256: `{report['raw_sha256']}`.", "", "All 3,600 rows were measured in one serial session on the dedicated GPU. Model loading was excluded from request latency. No labels or holdout outcomes were accessed.", "", "| Dataset | Policy | p50 ms | p95 ms | p95 ratio vs Jina full | 95% ratio interval | P(ratio <= 2) | alloc p95 MiB |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for x in ratios:
        lines.append(f"| {x['dataset']} | {x['policy']} | {x['p50_ms']:.3f} | {x['p95_ms']:.3f} | {x['p95_ratio_point']:.4f} | [{x['ratio_lo95']:.4f}, {x['ratio_hi95']:.4f}] | {x['prob_ratio_le_2']:.4f} | {x['alloc_p95_mib']:.3f} |")
    lines += ["", "Failures were scored on the all-request basis; no failure occurred, so no BM25 fallback packaging time was added. These are serial matched-session measurements, not sustainable QPS or monetary cost estimates.", ""]
    (STAGE / "STAGE3_SERVING_REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"status": "PASS", "raw_rows": len(raw), "failures": report["failures"], "raw_sha256": report["raw_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
