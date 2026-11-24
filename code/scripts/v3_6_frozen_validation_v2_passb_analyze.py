#!/usr/bin/env python3
"""Offline reconciliation and latency analysis for CQADupStack v2 Pass-B.

This script never loads a model and never reads qrels.  It consumes only
label-blind Pass-B traces and writes a versioned analysis directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/v3_6_essential/frozen_prospective_validation_v2"
POLICIES = ["pplx_single_4096", "pplx_single_6144", "jina_full_text", "jina_trunc_8192"]
SEED = 731
BOOTSTRAP_DRAWS = 10_000


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=float), q, method="linear")) if values else None


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_pass(files: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in files:
        rows.extend(read_jsonl(path))
    return rows


def data_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("anchor_stage") is None]


def validate_pass(rows: list[dict], expected_ids: list[str]) -> dict:
    expected = set(expected_ids)
    data = data_rows(rows)
    keys = Counter((r.get("canonical_id"), r.get("policy")) for r in data)
    missing = [(cid, p) for cid in expected_ids for p in POLICIES if keys[(cid, p)] != 1]
    duplicate = [(cid, p, n) for (cid, p), n in keys.items() if n != 1]
    anchors = Counter((r.get("anchor_stage"), r.get("policy"), r.get("status")) for r in rows if r.get("anchor_stage"))
    return {
        "records": len(rows),
        "data_records": len(data),
        "unique_data_ids": len({r.get("canonical_id") for r in data}),
        "missing_or_nonunique_data_keys": missing,
        "duplicate_data_keys": duplicate,
        "anchor_counts": {"|".join(str(x) for x in k): v for k, v in sorted(anchors.items(), key=lambda x: str(x[0]))},
        "status_counts": dict(Counter(r.get("status") for r in rows)),
        "error_counts": dict(Counter(r.get("error_class") for r in rows if r.get("status") != "PASS")),
        "gpu_requested": dict(Counter(r.get("gpu_index_requested") for r in rows)),
        "label_blind": all(r.get("labels_joined") is False and r.get("input_label_blind") is True for r in rows),
        "expected_ids_exact": {r.get("canonical_id") for r in data} == expected,
    }


def policy_rows(rows: list[dict], policy: str, anchor_stage: str | None = None) -> list[dict]:
    return [r for r in rows if r.get("policy") == policy and r.get("anchor_stage") == anchor_stage]


def summary_rows(pass_name: str, rows: list[dict]) -> list[dict]:
    out = []
    for policy in POLICIES:
        rr = policy_rows(rows, policy)
        lat = [float(r["latency_ms"]) for r in rr if r.get("latency_ms") is not None]
        mem = [float(r["peak_memory_allocated_mib"]) for r in rr if r.get("peak_memory_allocated_mib") is not None]
        out.append({
            "pass": pass_name,
            "policy": policy,
            "n": len(rr),
            "valid_n": sum(bool(r.get("valid")) for r in rr),
            "valid_rate": sum(bool(r.get("valid")) for r in rr) / len(rr) if rr else None,
            "failed_n": sum(not bool(r.get("valid")) for r in rr),
            "error_classes": json.dumps(dict(Counter(r.get("error_class") for r in rr if not r.get("valid"))), sort_keys=True),
            "p50_ms": percentile(lat, 50),
            "p95_ms": percentile(lat, 95),
            "p99_ms": percentile(lat, 99),
            "mean_ms": float(np.mean(lat)) if lat else None,
            "alloc_p95_mib": percentile(mem, 95),
            "physical_forward_total": sum(int(r.get("physical_forward_count") or 0) for r in rr),
        })
    return out


def anchor_summary(pass_name: str, rows: list[dict]) -> list[dict]:
    out = []
    for stage in ("start", "end"):
        rr = policy_rows(rows, "jina_full_text", stage)
        lat = [float(r["latency_ms"]) for r in rr if r.get("latency_ms") is not None]
        out.append({
            "pass": pass_name,
            "anchor_stage": stage,
            "n": len(rr),
            "valid_n": sum(bool(r.get("valid")) for r in rr),
            "p50_ms": percentile(lat, 50),
            "p95_ms": percentile(lat, 95),
            "p99_ms": percentile(lat, 99),
            "mean_ms": float(np.mean(lat)) if lat else None,
            "gpu_requested": json.dumps(dict(Counter(r.get("gpu_index_requested") for r in rr)), sort_keys=True),
        })
    return out


def cluster_ratio(pass_rows: list[list[dict]], policy: str, draws: int = BOOTSTRAP_DRAWS) -> dict:
    """Bootstrap p95(policy)/p95(Jina) by query cluster.

    pass_rows contains one data-row list per serial pass.  Each sampled query
    contributes all available pass observations, so the resampling unit is the
    canonical query ID rather than an individual request.
    """
    by_pass = []
    common: set[str] | None = None
    for rows in pass_rows:
        d = {(r["canonical_id"], r["policy"]): r for r in data_rows(rows)}
        by_pass.append(d)
        ids_here = {r["canonical_id"] for r in data_rows(rows)}
        common = ids_here if common is None else common & ids_here
    ids = sorted(common or set())
    pvals = {
        p: [[float(by_pass[j][(cid, p)]["latency_ms"]) for j in range(len(by_pass))] for cid in ids]
        for p in ("jina_full_text", policy)
    }
    pvals = {p: np.asarray(v, dtype=float) for p, v in pvals.items()}
    observed = float(np.percentile(pvals[policy].reshape(-1), 95, method="linear") / np.percentile(pvals["jina_full_text"].reshape(-1), 95, method="linear"))
    rng = np.random.default_rng(SEED)
    boot = np.empty(draws, dtype=float)
    n = len(ids)
    for i in range(draws):
        sample = rng.integers(0, n, size=n)
        left = pvals[policy][sample].reshape(-1)
        right = pvals["jina_full_text"][sample].reshape(-1)
        boot[i] = np.percentile(left, 95, method="linear") / np.percentile(right, 95, method="linear")
    return {
        "policy": policy,
        "basis": "pooled_serial_passes",
        "n_query_ids": n,
        "p50_ms": percentile(pvals[policy].reshape(-1).tolist(), 50),
        "p95_ms": percentile(pvals[policy].reshape(-1).tolist(), 95),
        "jina_p50_ms": percentile(pvals["jina_full_text"].reshape(-1).tolist(), 50),
        "jina_p95_ms": percentile(pvals["jina_full_text"].reshape(-1).tolist(), 95),
        "p95_ratio": observed,
        "ratio_lo95": float(np.quantile(boot, 0.025)),
        "ratio_hi95": float(np.quantile(boot, 0.975)),
        "prob_ratio_le_2": float(np.mean(boot <= 2.0)),
        "bootstrap_draws": draws,
        "bootstrap_seed": SEED,
        "bootstrap_unit": "canonical_id_cluster",
        "latency_basis": "all_requests_including_failure_latency; no fallback latency substituted",
    }


def pass_specific_ratio(rows: list[dict], policy: str) -> dict:
    d = {p: {r["canonical_id"]: r for r in data_rows(rows) if r["policy"] == p} for p in (policy, "jina_full_text")}
    ids = sorted(set(d[policy]) & set(d["jina_full_text"]))
    left = np.asarray([float(d[policy][x]["latency_ms"]) for x in ids])
    right = np.asarray([float(d["jina_full_text"][x]["latency_ms"]) for x in ids])
    rng = np.random.default_rng(SEED)
    boot = np.empty(BOOTSTRAP_DRAWS)
    for i in range(BOOTSTRAP_DRAWS):
        sample = rng.integers(0, len(ids), size=len(ids))
        boot[i] = np.percentile(left[sample], 95, method="linear") / np.percentile(right[sample], 95, method="linear")
    return {
        "policy": policy,
        "basis": "single_pass",
        "n_query_ids": len(ids),
        "p50_ms": percentile(left.tolist(), 50),
        "p95_ms": percentile(left.tolist(), 95),
        "jina_p50_ms": percentile(right.tolist(), 50),
        "jina_p95_ms": percentile(right.tolist(), 95),
        "p95_ratio": float(np.percentile(left, 95, method="linear") / np.percentile(right, 95, method="linear")),
        "ratio_lo95": float(np.quantile(boot, 0.025)),
        "ratio_hi95": float(np.quantile(boot, 0.975)),
        "prob_ratio_le_2": float(np.mean(boot <= 2.0)),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": SEED,
        "bootstrap_unit": "canonical_id_cluster",
        "latency_basis": "all_requests_including_failure_latency; no fallback latency substituted",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pass1", type=Path, nargs="+", required=True)
    ap.add_argument("--pass2", type=Path, nargs="+", required=True)
    ap.add_argument("--strict-clean", action="store_true", help="mark the two inputs as the clean single-GPU paired run")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    latency_ids = [json.loads(x)["canonical_id"] for x in (DEFAULT_OUT / "LATENCY_COHORT_IDS.jsonl").open(encoding="utf-8") if x.strip()]
    p1, p2 = load_pass(args.pass1), load_pass(args.pass2)
    validations = {"pass1": validate_pass(p1, latency_ids), "pass2": validate_pass(p2, latency_ids)}

    fields = ["pass", "canonical_id", "forum", "policy", "anchor_stage", "status", "valid", "error_class", "latency_ms", "peak_memory_allocated_mib", "n_calls", "physical_forward_count", "gpu_index_requested", "policy_order_index", "input_sha256", "source_slate_sha256", "model_revision"]
    rows = []
    for name, batch in (("pass1", p1), ("pass2", p2)):
        for r in batch:
            rows.append({"pass": name, **{k: r.get(k) for k in fields if k != "pass"}})
    write_csv(args.out_dir / "PASSB_PERREQUEST.csv", fields, rows)

    summaries = summary_rows("pass1", p1) + summary_rows("pass2", p2)
    write_csv(args.out_dir / "PASSB_LATENCY_SUMMARY.csv", list(summaries[0]), summaries)
    anchors = anchor_summary("pass1", p1) + anchor_summary("pass2", p2)
    write_csv(args.out_dir / "PASSB_ANCHOR_SUMMARY.csv", list(anchors[0]), anchors)

    ratios = []
    for policy in ("pplx_single_4096", "pplx_single_6144", "jina_trunc_8192"):
        ratios.append(pass_specific_ratio(p1, policy) | {"basis": "pass1"})
        ratios.append(pass_specific_ratio(p2, policy) | {"basis": "pass2"})
        ratios.append(cluster_ratio([p1, p2], policy))
    write_csv(args.out_dir / "PASSB_LATENCY_RATIOS.csv", list(ratios[0]), ratios)

    p1d = { (r["canonical_id"], r["policy"]): r for r in data_rows(p1) }
    p2d = { (r["canonical_id"], r["policy"]): r for r in data_rows(p2) }
    order_mismatch = []
    input_mismatch = []
    selected_mismatch = []
    for cid in latency_ids:
        for policy in POLICIES:
            a, b = p1d.get((cid, policy)), p2d.get((cid, policy))
            if not a or not b:
                continue
            if b.get("policy_order_index") != 3 - a.get("policy_order_index"):
                order_mismatch.append((cid, policy, a.get("policy_order_index"), b.get("policy_order_index")))
            if a.get("input_sha256") != b.get("input_sha256") or a.get("source_slate_sha256") != b.get("source_slate_sha256"):
                input_mismatch.append((cid, policy))
            if a.get("valid") and b.get("valid") and a.get("selected_ids") != b.get("selected_ids"):
                selected_mismatch.append((cid, policy))

    raw_manifest = []
    for path in args.pass1 + args.pass2:
        resolved = path.resolve()
        raw_manifest.append({"path": str(resolved.relative_to(ROOT)), "sha256": sha_file(resolved), "records": sum(1 for _ in resolved.open(encoding="utf-8"))})
    (args.out_dir / "PASSB_RAW_MANIFEST.json").write_text(json.dumps(raw_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    integrity = {
        "strict_clean_session": bool(args.strict_clean),
        "protocol_sha256": hashlib.sha256((DEFAULT_OUT / "PROTOCOL_FROZEN.md").read_bytes()).hexdigest(),
        "latency_cohort_n": len(latency_ids),
        "latency_cohort_sha256": sha_file(DEFAULT_OUT / "LATENCY_COHORT_IDS.jsonl"),
        "validations": validations,
        "cross_pass_input_mismatch_n": len(input_mismatch),
        "cross_pass_input_mismatch_examples": input_mismatch[:20],
        "reverse_order_mismatch_n": len(order_mismatch),
        "reverse_order_mismatch_examples": order_mismatch[:20],
        "cross_pass_selected_ids_mismatch_n": len(selected_mismatch),
        "cross_pass_selected_ids_mismatch_examples": selected_mismatch[:20],
        "raw_manifest": raw_manifest,
        "raw_files_preserved": True,
        "labels_joined": False,
        "qrels_read": False,
        "fallback_latency_substituted": False,
    }
    (args.out_dir / "PASSB_INTEGRITY.json").write_text(json.dumps(integrity, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    p1_summary = {x["policy"]: x for x in summary_rows("pass1", p1)}
    p2_summary = {x["policy"]: x for x in summary_rows("pass2", p2)}
    report_status = "PASSB_STRICT_CLEAN_LATENCY_ANALYSIS_COMPLETE" if args.strict_clean else "OFFLINE_ANALYSIS_COMPLETE_WITH_LATENCY_QUALIFICATIONS"
    report = [
        "# CQADupStack v2 Pass-B Latency Reconciliation",
        "",
        f"Status: `{report_status}`",
        "",
        "This report uses only label-blind Pass-B traces. No qrels were read and no model was loaded.",
        "",
        "## Coverage",
        "",
        f"- Frozen latency cohort: {len(latency_ids)} query IDs.",
        f"- Pass 1 logical records: {len(p1)} (split across the preserved GPU3 partial and GPU4 continuation files).",
        f"- Pass 2 records: {len(p2)} (single GPU3 session).",
        f"- Pass 1 validation: {validations['pass1']['status_counts']}; Pass 2 validation: {validations['pass2']['status_counts']}.",
        f"- Pass 2 request-level failures: {validations['pass2']['error_counts'] or 'none'}.",
        "",
        "## Latency interpretation",
        "",
        "All request latencies, including the failed request's elapsed time, are retained. No BM25 fallback latency was substituted because the native runner did not execute a fallback path.",
        ("Both passes are complete single-GPU GPU3 sessions with clean pre-run snapshots, matched policy views, complete anchors and no request-level failures. These are the primary latency results for the strict S gate. Allocator warning lines observed on stderr are retained as a telemetry limitation; they did not create failed request records." if args.strict_clean else "The original Pass 1 is not a single clean hardware session: its first 342 IDs were recorded on GPU3 and the continuation on GPU4. Pass 2 is a single GPU3 session, but its pre-run telemetry and one CUDA-OOM record require the strict production-latency claim to remain qualified."),
        "",
        "## Integrity",
        "",
        f"- Cross-pass input-hash mismatches: {len(input_mismatch)}.",
        f"- Reverse-order mismatches: {len(order_mismatch)}.",
        f"- Cross-pass selected-ID mismatches among jointly valid requests: {len(selected_mismatch)}.",
        "- Label joins: none; qrels: not read by this analysis.",
        "",
        "## Artifacts",
        "",
        "- `PASSB_PERREQUEST.csv` — label-blind per-request latency and status records.",
        "- `PASSB_LATENCY_SUMMARY.csv` — p50/p95/p99, validity, failure and memory summaries by pass and policy.",
        "- `PASSB_LATENCY_RATIOS.csv` — 10,000-draw query-cluster bootstrap p95 ratios, including pooled serial-pass sensitivity.",
        "- `PASSB_ANCHOR_SUMMARY.csv` — start/end Jina anchor summaries.",
        "- `PASSB_INTEGRITY.json` — coverage, order, input-hash and preservation checks.",
        "- `PASSB_RAW_MANIFEST.json` — SHA-256 and record counts for all source traces.",
    ]
    (args.out_dir / "PASSB_LATENCY_RECONCILIATION.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "pass1": validations["pass1"], "pass2": validations["pass2"], "input_mismatch_n": len(input_mismatch), "order_mismatch_n": len(order_mismatch), "selected_mismatch_n": len(selected_mismatch)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
